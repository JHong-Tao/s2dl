'''
Author:jhong.tao
Date: 2022-03-21 10:30:23
LastEditTime: 2022-03-22 09:45:23
LastEditors: Please set LastEditors
Description: 线性回归调用自动求导方法实现
FilePath: \3-Linear Neural Networks\3.2 code.py
'''

import random
import re
from importlib_metadata import requires
from numpy import arange, size
import torch

from d2l import torch as d2l


# 定义模你数据生成器
def systhetic_data(w, b, num_examples):
    """
    合成训练数据集

    Args:
        w (tensor): 表示样本的每个特征的权重，len(w)也代表了一个样本的特征个数
        b (float): 表示偏置b
        num_examples (int): 代表样本的数量
 
    Returns:
        tensor: X为样本的特征集，y为样本的标签
    """
    # 生成线性回归的模你数据集，样本特征为len(w)，样本量为num_examples，从均值为0，标准差为1的正态分布中采样
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 生成不包含误差的y
    y += torch.normal(0, 0.01, y.shape)  # 生成带有从均值为0，标准差为0.01的有观察误差的样本标签
    return X, y


# 设置真实的w和b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

num_examples = 1000  # 设置样本量num_examples=1000
X, y = systhetic_data(true_w, true_b, num_examples)  # 生成样本特征和标签。
# 打印数据集看一下效果，因为样本特征包含两个特征，这里选择第二个特征打印出来，如果都打印会是一个曲面
d2l.set_figsize((10, 6))   # 设置图片大小
d2l.plt.scatter(X[:, 1].detach().numpy(), y.detach().numpy())  # 绘制散点图
d2l.plt.show()  # 显示散点图


# 定义小批量数据生成器
def data_iter(batch_size, X, y):
    """
    构造小批量数据生成器

    Args:
        batch_size (int): 每次获取数据的批量大小
        X (tensor): 样本的特征集
        y (tensor): 样本特征对应的标签集
    
    Yields:
        tensor: batch_X为样本的小批量特征集合，batch_y为X对应的标签集合
    """
    num_X = len(X)  # 获取样本数量，len()函数当输入参数为tensor时只统计第0维度的数量，若输入是矩阵则只统计行的数量
    indices = list(range(num_X))  # 创建样本索引
    random.shuffle(indices)   # 打乱索引的顺序
    for i in range(0, num_X, batch_size):
        batch_indices_index = indices[i: min(i+batch_size, num_X)]  # 每次从打乱顺序的indices中获取一个batch_size大小的索引
        batch_indices = torch.tensor(batch_indices_index)  # 依据batch_indices_index获得一个batch_size的样本特征和标签的索引
        yield X[batch_indices], y[batch_indices]  # 根据batch_indices索引获取一个批量的样本特征和标签


# 测试获取一个小批量的数据
batch_size = 10
batch_X, batch_y = next(data_iter(batch_size, X, y))  # 获取一个小批量的数据，因为data_iter为生成器所以需要用next函数来取数据
print(batch_X[0], "\n", batch_y[0])   # 打印第一行数据

# 从均值为0，标准差为0.01的正态分布中随机初始化参数w和b,由于我们的目标就是优化这两个参数，所以需要跟踪他们的梯度
w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print("w:", w,"\nb:", b)  # 打印输出看一下随机初始化的w和b

# 定义线性回归模型
def linear(X:torch.Tensor, w:torch.tensor, b:torch.tensor)->torch.tensor:
    """
    线性回归模型: y = Xw+b
    
    Args:
        X (torch.Tensor): 样本特征集
        w (torch.tensor): 样本特征权重
        b (torch.tensor): 偏置b
    
    Returns:
        tensor: y_hat:模型预测值
    """
    return torch.matmul(X, w) + b

