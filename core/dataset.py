import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
from core.utils import ZipReader, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


class Dataset(torch.utils.data.Dataset):  ## 重写了DataLoader需要的Dataset类
    def __init__(self, args: dict, split='train', debug=False):  ## 参数列表： 1 参数字典 2 数据集名称 3 debug开关
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])  ## 参数字典提供视频分辨率 长度 数据路径 文件名

        with open(os.path.join(args['data_root'], args['name'], split + '.json'), 'r') as f:
            self.video_dict = json.load(f)  ## 加载数据
        self.video_names = list(self.video_dict.keys())  ##  获取视频名称列表
        if debug or split != 'train':  ## 如果不是训练模式
            self.video_names = self.video_names[:100]  ## 名称列表只展开到前100个

        self._to_tensors = transforms.Compose([  ## 数据预处理
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):  ## 重写取值函数 取不到时报错并取第一个
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):  ## 具体取数据操作

        video_name = self.video_names[index]  ## 用视频名称取出视频每一帧名称都是5位数字
        ## str.zfill(int)  指定字符串长度，右对齐，左边填充0
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]  ## 获取该压缩文件中每个视频帧的文件名
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)  ## 获取了所有视频帧的遮罩 可能为固定遮罩，也可能是移动遮罩
        ## 在某个压缩包内随机挑选sample_length个倒霉蛋（默认5个） 也就是一段视频中默认有5个视频帧是有mask的
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), video_name, all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)  ## 倒霉蛋本蛋
            masks.append(all_masks[idx])  ## 抽到的倒霉蛋的遮罩
        if self.split == 'train':  ## 如果是训练模式
            frames = GroupRandomHorizontalFlip()(frames)  ## 额外执行一个 随机水平翻转
        # To tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0  ## ?为什么
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)  ## 在给定长度上随机获取一串间断下标
        # sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机唯一的元素
        ref_index.sort()  ## 排序
    else:
        pivot = random.randint(0, length - sample_length)  ##  反正就是不同的随机下标
        ref_index = [pivot + i for i in range(sample_length)]  ## 唯一不同是这边是连续的
    return ref_index


# if __name__ == '__main__':
#     config = json.load(open("../configs/davis.json"))
#
#     a = Dataset(config['data_loader'], split='train')
#
#     index = 0
#     # print(len(a.video_names))
#     video_name = a.video_names[index]  ## 用视频名称取出视频每一帧名称都是5位数字
#     print(video_name)
#     ## str.zfill(int)  指定字符串长度，右对齐，左边填充0
#     all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(a.video_dict[video_name])]  ## 获取该压缩文件中每个视频帧的文件名
#     all_masks = create_random_shape_with_random_motion(
#         len(all_frames), imageHeight=a.h, imageWidth=a.w)  ## 获取了所有视频帧的遮罩 可能为固定遮罩，也可能是移动遮罩
#     print(all_frames)
#     # for i in range(5):
#     #     Image._show(all_masks[i])
#
#     # ## 在某个压缩包内随机挑选sample_length个倒霉蛋（默认5个） 也就是一段视频中默认有5个视频帧是有mask的
#     ref_index = get_ref_index(len(all_frames), a.sample_length)
#     print(ref_index)
#     # # read video frames
#     frames = []
#     masks = []
#     for idx in ref_index:
#         print(all_frames[idx])
#         img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
#             a.args['data_root'], a.args['name'], video_name), all_frames[idx]).convert('RGB')
        # print(img)
        # img = img.resize(a.size)
        # frames.append(img)  ## 倒霉蛋本蛋
        # masks.append(all_masks[idx])  ## 抽到的倒霉蛋的遮罩
    # if a.split == 'train':  ## 如果是训练模式
    #     frames = GroupRandomHorizontalFlip()(frames)  ## 额外执行一个 随机水平翻转
    # # To tensors
    # frame_tensors = a._to_tensors(frames) * 2.0 - 1.0
    # mask_tensors = a._to_tensors(masks)
