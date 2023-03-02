import json
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from core.dataset import Dataset


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:  ## debug模式
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train', debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=False,
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

    # train entry
    def train(self):  ## 训练函数实体
        pbar = range(100)
        ## dynamic_ncols=True持续改变进度条宽度   已存储的迭代次数作为初始值
        while True:
            self.epoch += 1
            self._train_epoch(pbar)  ## 训练一轮次函数
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        for frames, masks in self.train_loader:
            self.adjust_learning_rate()  ## 调整学习率
            self.iteration += 1
            # print('train:', frames.size())
            # print(masks.size())

            # frames, masks = frames.to(device), masks.to(device)
            # b, t, c, h, w = frames.size()
            # masked_frame = (frames * (1 - masks).float())
            # pred_img = self.netG(masked_frame, masks)
            # frames = frames.view(b * t, c, h, w)
            # masks = masks.view(b * t, 1, h, w)
            #
            break

from core.utils import ZipReader, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

if __name__ == '__main__':
    config = json.load(open("configs/davis.json"))
    config['world_size'] = 1
    a = Trainer(config)
    ######
    # a.train()
    ######
    # b = a.train_dataset
    # print(b[0])

    z = ZipReader()

