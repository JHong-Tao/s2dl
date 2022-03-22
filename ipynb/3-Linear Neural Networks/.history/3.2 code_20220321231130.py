'''
Author:jhong.tao
Date: 2022-03-21 10:30:23
LastEditTime: 2022-03-21 23:11:30
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \3-Linear Neural Networks\3.2 code.py
'''

import random
import torch

from d2l import torch as d2l


def systhetic_data(w, b, num_examples):
    """
    _summary_
    
    
    
    Args:
        w (_type_): _description_
        b (_type_): _description_
        num_examples (_type_): _description_
    
    Returns:
        _type_: _description_
    """
    # 生成线性回归的模你数据集，样本特征为len(w)，样本量为num_examples，从均值为0，标准差为1的正态分布中采样
    X = torch.normal(0, 1, (num_examples, 2))
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

# 
