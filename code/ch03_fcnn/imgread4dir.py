#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> imgread4dir
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/7/3 17:02
@Desc   ：
==================================================
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

data_dir = '../ch00_dataset'

img = datasets.ImageFolder(data_dir,)  # 指明读取的文件夹和读取方式,注意指明的是到文件夹的路径,不是到图片的路径

imgLoader = torch.utils.data.DataLoader(img, batch_size=2, shuffle=False, num_workers=1)  # 指定读取配置信息

inputs, _ = next(iter(imgLoader))

inputs = torchvision.utils.make_grid(
    inputs)  # make_grid()实现图片的拼接，并去除原本Tesor中Batch_Size那一维度,因为操作之前的inputs是4维的, make_grid()返回的结果是3维的, shape为(3, h, w) 3代表通道数, w,h代表拼接后图片的宽高
inputs = inputs.numpy().transpose((1, 2, 0))  # transpose((1, 2, 0)) 是将(3, h, w) 变为 (h ,w, 3), 因为这种格式才是图像存储的标准格式
plt.imshow(inputs)  # 展示,这里会一块展示batch_size张图片,因为它们是一块被读出来的
plt.show()