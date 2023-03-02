import os
import cv2
import time
import math
import glob
from tqdm import tqdm
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import Dataset
from core.loss import AdversarialLoss


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:    ## debug模式
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train',  debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model'])  ## 动态导入sttn或者vis模块 两个模块内功能函数名称一样 且只会导入一个 所以采取动态导入
        self.netG = net.InpaintGenerator()  ## 生成器模型
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(   ## 判别器模型
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load()    ## 加载模型 要么是预训练的或者上一轮训练的 没有就算了

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']): ## 非分布式运算时  直接保存loss曲线
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):  ## 学习率衰减函数
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):  ## 如果latest.ckpt文件存在的话 从该文件加载
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]  ## 返回文件的最后一行
        else:  ## 否则加载pth文件
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]  ##os.path.basename(i)返回路径i最后的文件名 所以 ckpts就是预训练模型目录下所有pth文件
            # 不带扩展名的纯文件名列表
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None ## 如果有的话返回最后一个 没有就 None
        if latest_epoch is not None:  ## 有预训练模型的话就加载
            gen_path = os.path.join(  ## 生成器模型
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))  ## 还是熟悉的右对齐左补0凑5位数
            dis_path = os.path.join(  ## 判别器模型
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(  ## 优化器模型（其实就是超参数集）
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:  ## 没有那就从0开始
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):  ## 训练函数实体
        pbar = range(int(self.train_args['iterations']))  ## 新的迭代次数
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        ## dynamic_ncols=True持续改变进度条宽度   已存储的迭代次数作为初始值
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)  ## 训练一轮次函数
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']

        for frames, masks in self.train_loader:
            self.adjust_learning_rate()  ## 调整学习率
            self.iteration += 1

            frames, masks = frames.to(device), masks.to(device) ## frames五个维度 [个数，时间帧个数，通道数，高，宽]
            b, t, c, h, w = frames.size()
            masked_frame = (frames * (1 - masks).float())  ## masks是除了遮罩区域外全黑 两者相乘后遮罩区域黑，其余区域正常
            pred_img = self.netG(masked_frame, masks)
            frames = frames.view(b*t, c, h, w)
            masks = masks.view(b*t, 1, h, w)
            comp_img = frames*(1.-masks) + masks*pred_img  ## 将生成的图像的遮罩部分补到masked_frame上 得到尝试修补的图像

            gen_loss = 0
            dis_loss = 0

            # discriminator adversarial loss
            real_vid_feat = self.netD(frames)
            fake_vid_feat = self.netD(comp_img.detach())
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # generator adversarial loss
            gen_vid_feat = self.netD(comp_img)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_summary(
                self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_img*masks, frames*masks)
            hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            gen_loss += hole_loss 
            self.add_summary(
                self.gen_writer, 'loss/hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))
            valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss 
            self.add_summary(
                self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"
                    f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}")
                )

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

