#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> toos
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/7/1 12:02
@Desc   ：
==================================================
"""
from scipy.misc import fromimage, toimage
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

path = '../ch00_dataset/0_9_img/h_4.png'
path = '../ch00_dataset/0_9_img/h_7.png'

image = Image.open(path).convert('L')

img_arr = np.asarray(image)

# print(img_arr)

img_arr = np.where(img_arr == 255, 0, img_arr)
# print(img_arr)

img_arr_img = Image.fromarray(img_arr)

# plt.imshow(image, cmap='gray')
plt.imshow(img_arr_img, cmap='gray')
plt.show()

# a = np.array([[255,255,6],[255, 4, 9]])
# b = np.array([1,2,3])
#
# c = np.where(a==255, 10, np.true_divide(a, 10))

# print(c)