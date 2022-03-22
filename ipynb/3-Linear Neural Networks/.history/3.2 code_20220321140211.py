'''
Author:jhong.tao
Date: 2022-03-21 10:30:23
LastEditTime: 2022-03-21 14:02:11
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \3-Linear Neural Networks\3.2 code.py
'''
import random

import torch
from d2l import torch as d2l

def systhetic_data(w, b, num_exanmples)

# 生成线性回归的模你数据集，样本特征为2，样本量为1000，从均值为0，标准差为1的正态分布中采样
X = torch.normal(0, 1, (1000, 2), )

# 设置真实的w和b
