import matplotlib.patches as patches
from matplotlib.path import Path
import os
import sys
import io
import cv2
import time
import argparse
import shutil
import random
import zipfile
from glob import glob
import math
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFilter

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('agg')


# #####################################################
# #####################################################

class ZipReader(object):  ## 读取压缩文件函数
    file_dict = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):  ## 按指定目录读取
        file_dict = ZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, video_name, image_name):  ## 从压缩包中返回指定的视频帧图片对象
        zfile = ZipReader.build_file_dict(path)
        # data = zfile.read(video_name + '/' + image_name)   linux下的目录
        data = zfile.read(image_name)
        im = Image.open(io.BytesIO(data))
        return im


# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):  ## 重写了call函数以后 这个类就变成了可调用对象
        ## 类后一个括号是实例化  重写了 call函数后就可以跟第二个括号，第二个括号的作用是调用类中的__call__函数
        ## 所以会有一种情况 类后直接跟两个括号 代表我直接实例化加调用call函数 即：
        ## A()() 代表实例化同时调用A.__call__()函数
        ## 相当于
        # b = A()
        # b()
        ## 所以第一个括号的参数是给__init__函数的   第二个括号的参数是给__call__函数的
        v = random.random()
        if v < 0.5:
            # img.transpose(Image.FLIP_LEFT_RIGHT) 对img做水平翻转
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                # print('Stack: ', img_group)
                # print('after Stack: ', np.stack(img_group, axis=2).shape)
                return np.stack(img_group, axis=2)  # np.stack将多个array在指定维度拼接成一个多了一维的array，
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            # print(img)
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # print(img)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            # torch.contiguous() 方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################

## 用随机动作创建随机形状的遮罩
def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    # 高宽随机范围是1/3~1比例的原尺寸
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)  ## 边界数
    ratio = random.randint(6, 8) / 10  ## 比率
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight - region_height), random.randint(0, imageWidth - region_width)
    velocity = get_random_velocity(max_speed=3)  ## 获取随机速度
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))  ## 将mask贴在区域上
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:  ## 随机实数大于0.5  也就是一半的可能用固定遮罩
        return masks * video_length  ## 每个视频帧都用一样的遮罩
    # return moving masks
    for _ in range(video_length - 1):  ## 另一半可能则使用移动遮罩 也就是随机移动
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):  ## 返回一个随机形状的区域
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num * 3 + 1  ## 路径点个数
    angles = np.linspace(0, 2 * np.pi, points_num)  ## 在0到2π之间均匀分布points_num个数字
    codes = np.full(points_num, Path.CURVE4)
    # plt.path是plt绘图的最高级用法，plt的所有简单形状绘图函数都是通过path实现的
    # 教程：https://blog.csdn.net/qq_27825451/article/details/82967904
    # Path.CURVE4表示路径： 2个控制点，一个终点。使用指定的2个控制点从当前位置画三次赛贝尔曲线到指定的结束位置
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    # 在cos和sin函数上分别均匀地取points_num个值拼接到一起组成（2，points_num）的数组，再逐元素乘一个(-ratio+1,ratio+1)范围的随机数
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
            (2 * ratio * np.random.random(points_num) + 1 - ratio)[:, None]
    # array[:,None] 将array每个元素变成单独一维 形变： (28) -> (28,1)
    # 路径的首尾相接形成封闭图形
    verts[-1, :] = verts[0, :]
    # vertices(简称verts) 是点的坐标，codes是点之间路径的轨迹类型（直线、曲线等）
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8)) * 255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5),
                               maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity


# ##############################################
# ##############################################

if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)
    # z = ZipReader()
    #
    # img = z.imread("../datasets/davis/JPEGImages/bear.zip","bear","00011.jpg").convert('RGB')
    # print(img)
