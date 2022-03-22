# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> simplelr
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/6/6 15:21
@Desc   ：
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# 创建数据集
x = [1., 2., 3., 4., 5.]
y = [3.2, 5.2, 6.8, 9.5, 10.8]

x = np.array(x).reshape((5, 1))
y = np.array(y).reshape((5, 1))

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(x.shape)


# 定义模型
def model(w, b, x):
    return w * x + b


# 定义损失函数
def cost(w, b, x, y):
    n = len(y)
    return 1./(2.*n) * (np.square(y-w*x-b)).sum()


# 定义优化器
def optimizer(w, b, x, y, lr):
    n = len(y)
    y_hat = model(w, b, x)
    dw = (1./n) * ((y_hat-y) * x).sum()
    db = (1./n) * (y_hat-y).sum()
    w = w - lr *  dw
    b = b - lr * db
    return w, b


# 定义模型训练器
def iterater(w, b, x, y, lr, epochs):
    list_cost = []
    for epoch in range(epochs):
        w, b = optimizer(w, b, x, y, lr)
        loss = cost(w, b, x, y)
        list_cost.append(loss)
    return w, b, list_cost


# 模型训练
if __name__ == '__main__':
    # 初始化 模型参数
    w = 0.
    b = 0.
    lr = 0.01
    epochs = 100

    # 训练模型
    w, b, list_cost = iterater(w, b, x, y, lr, epochs)

    # 查看模型最终损失
    loss = cost(w, b, x, y)

    # 用模型做预测
    y_hat = model(w, b, x)

    # 打印模型
    print(w, b, loss)
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 打印模型优化过程
    plt.plot(range(epochs), list_cost)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.show()
