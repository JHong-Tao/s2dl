#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> fcnn.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/6/28 21:35
@Desc   ：
==================================================
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义网络
class MyNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# 模型学习
def iterator(D_in, H, D_out, epochs, x, y, N, device):
    loss_list = []
    model = MyNet(D_in, H, D_out)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(epochs):
        y_pred = model(x).to(device)

        loss = loss_fn(y_pred, y)
        loss_list.append(np.sqrt(loss.item() / N))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_list, model


def process_show(loss_list):
    plt.plot(range(len(loss_list)), loss_list, color='r')
    plt.xlim(0, 500)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, D_in, H, D_out = 64, 4, 3, 2
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out).to(device)
    epochs = 1000

    loss_list, model = iterator(D_in, H, D_out, epochs, x, y, N, device)

    process_show(loss_list)

    print(model.state_dict().get('linear1.weight'))
    print(loss_list[:10])
    print(loss_list[::100])

    # torch.save(model, model)
