#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> img_read
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/7/1 23:23
@Desc   ：
==================================================
"""
from scipy.misc import fromimage, toimage
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

path1 = '../ch00_dataset/h_4.png'   # 黑底白字
path2 = '../ch00_dataset/h_3.png'   # 白底黑子
path3 = '../ch00_dataset/hs_7.png'  # 白底彩字
path5 = '../ch00_dataset/5.png'     # 原画 黑底白字

img = Image.open(path2).convert('L')
def img_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# def img_transform(img):
img_show(img)
img_arr = np.asarray(img)
print(img_arr.size)
print(img_arr.shape)
print(np.sum(img_arr>200))
# 黑白转换
if np.sum(img_arr>200)>img_arr.size/2:
    img_arr = 255*(1-np.true_divide(img_arr, 255))
    img_arr = np.where(img_arr>150,255,img_arr)

img = Image.fromarray(img_arr)
img_show(img)
np.set_printoptions(threshold=img_arr.size)
# np.set_printoptions(threshold=np.inf)
# print(img_arr)
