#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> 02_lr2torch_boston
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/6/6 18:33
@Desc   ：http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
==================================================
"""
import torch
import numpy
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as split

# 数据加载
data = load_boston()
x = data['data']
y = data['target'].reshape(-1, 1)

# 标准化数据
mm_scale = MinMaxScaler()
x = mm_scale.fit_transform(x)

# 切分数据集
x_train, x_test, y_train, y_test = split(x, y, test_size=0.2)
x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建网络
model = torch.nn.Sequential(
    torch.nn.Linear(13, 1)
)

# 定义优化器和损失函数
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 模型训练
if __name__ == '__main__':
    list_loss =[]
    max_epoch = 300
    for epoch in range(max_epoch):
        # 前向计算
        y_pred = model(x_train)
        # 计算损失
        loss = cost(y_pred, y_train)
        list_loss.append(loss.detach().numpy())
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

    # 打印模型参数
    list_par = list(model.named_parameters())
    print(list_par[0])
    print(list_par[1])

    # 测试
    y_hat = model(x_test)
    MSE = cost(y_test, y_hat)

    # 模型评估
    RMSE = numpy.sqrt(MSE.detach().numpy())
    print(RMSE)

    # 打印优化过程
    plt.plot(range(max_epoch), list_loss)
    plt.xlabel = 'epoch'
    plt.ylabel = 'loss'
    plt.show()
