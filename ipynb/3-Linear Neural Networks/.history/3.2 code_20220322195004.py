'''
Author:jhong.tao
Date: 2022-03-21 10:30:23
LastEditTime: 2022-03-22 19:49:06
LastEditors: Please set LastEditors
Description: 线性回归调用自动求导方法实现
FilePath: \3-Linear Neural Networks\3.2 code.py
'''

import random
import re
from importlib_metadata import requires
from matplotlib.pyplot import legend, title, xlabel, ylabel
import numpy as np
import torch
import visdom

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
    X = torch.normal(1, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 生成不包含误差的y
    y += torch.normal(0, 0.01, y.shape)  # 生成带有从均值为0，标准差为0.01的有观察误差的样本标签
    return X, y


# 设置真实的w和b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

num_examples = 1000  # 设置样本量num_examples=1000
X_features, y_labels = systhetic_data(true_w, true_b, num_examples)  # 生成样本特征和标签。
# 打印数据集看一下效果，因为样本特征包含两个特征，这里选择第二个特征打印出来，如果都打印会是一个曲面
# d2l.set_figsize((10, 6))   # 设置图片大小
# d2l.plt.scatter(X_features[:, 1].detach().numpy(), y_labels.detach().numpy())  # 绘制散点图
# d2l.plt.show()  # 显示散点图

# 在visdom中绘图图像
viz = visdom.Visdom(env='linreg')  # 设置visdom中的环境
# 绘制散点图
win = viz.scatter(
    X=torch.cat((X_features[:, 1].reshape(len(y_labels), 1), y_labels.reshape(len(y_labels), 1)), 1),  # 用训练集的第二个特征和标签
    opts=dict(
        markersize = 2,  # 设置点的大小
        showlegend = True,  # 显示坐标轴
        title = "样本散点图",
    )
    )  

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
batch_X, batch_y = next(data_iter(batch_size, X_features, y_labels))  # 获取一个小批量的数据，因为data_iter为生成器所以需要用next函数来取数据
print(batch_X[0], "\n", batch_y[0])   # 打印第一行数据


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


# 定义损失函数
def squared_loss(y_hat:torch.tensor, y:torch.tensor)->torch.tensor:
    """
    平方误差损失函数: loss = 1/2*(y_hat-y)**2    
    
    Args:
        y_hat (torch.tensor): 模型预测值
        y (torch.tensor): 样本真实标签
    
    Returns:
        torch.tensor: 平方损失
    """
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


# 定义梯度下降优化算法
def sgd(params:list, lr:float, batch_size:int):
    """
    梯度更新过程  theta =  theta - lr * theta.grad / batch_size
    
    Args:
        params (list): 参数列表
        lr (float): 梯度下降学习率（步长）
        batch_size (int): 批量大小
    """
    with torch.no_grad():  # 更新梯度的过程不需要跟踪梯度
        # 遍历参数w和b
        for param in params:  
            param -= lr * param.grad / batch_size  # 更新参数
            param.grad.zero_()  # pytorch默认会一直跟踪梯度，每轮梯度更新后需要置0梯度


# 模型训练
# 从均值为0，标准差为0.01的正态分布中随机初始化参数w和b,由于我们的目标就是优化这两个参数，所以需要跟踪他们的梯度
w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# print("w:", w,"\nb:", b)  # 打印输出看一下随机初始化的w和b
lr = 0.01
epochs = 100
net = linear
loss = squared_loss
loss_list = []

for epoch in range(epochs):
    for X, y in data_iter(batch_size, X_features, y_labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    
    with torch.no_grad():
        loss_epoch = loss(net(X_features, w, b), y_labels).mean()
        loss_list.append(loss_epoch.detach().numpy())

print(loss_list[0:10])

# 绘制损失函数优化过程
viz.line(
    X=np.arange(epochs),
    Y=loss_list,
    win="loss_line",
    name="loss",
    opts=dict(
        xlabel='epochs',
        ylabel='loss',
        title='loss-epochs',
        showlegend=True,
        markersize=5,
    )
)