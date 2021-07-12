#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> fcnn_minist.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/6/29 12:18
@Desc   ：
==================================================
"""
import time

import torch
from torch import nn
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def data_loader(data_dir, batch=32, workers=4, size =28):
    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        transforms.Grayscale(1),
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),  # 切割
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_set = mnist.MNIST(root=data_dir, train=True, transform=transform, download=False)
    test_set = mnist.MNIST(root=data_dir, train=False, transform=transform, download=False)

    # data_show(train_set)

    train_data = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers)
    test_set = DataLoader(test_set, batch_size=batch, shuffle=True, num_workers=workers)
    return train_data, test_set


def data_show(train_set, img_rows=1, img_cols=4):
    print('数据集大小:', train_set.data.size())
    print('数据集标签:', train_set.targets.size())
    img_size = train_set.data[0].size()
    print('图片的维度:', img_size)

    plt.figure()
    for i in range(img_cols*img_rows):
        plt.subplot(img_rows, img_cols, i+1)      # subplot(行，列，图片序号) 重点：图片序号是从1开始的，行优先
        plt.imshow(train_set.data[i].numpy(), cmap='gray')
        plt.title('%i' % train_set.targets[i])
        plt.xticks(range(0, img_size[0], 9))
        plt.yticks(range(0, img_size[0], 9))
    plt.show()
    plt.close()


class SimpleFnn(nn.Module):
    def __init__(self, x_dim, w1, w2, w3, y_dim):
        super(SimpleFnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(x_dim, w1),
            nn.BatchNorm1d(w1),
            nn.ReLU(w1)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(w1, w2),
            nn.BatchNorm1d(w2),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(w2, w3),
            nn.BatchNorm1d(w3),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(w3, y_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y_hat = self.layer4(x)

        return y_hat


def iterator_net(x_dim, w1, w2, w3, y_dim, lr, epochs, train_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    net = SimpleFnn(x_dim, w1, w2, w3, y_dim).to(device)
    # net.parameters  # 查看网络参数

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    loss_list  = []
    acc_list = []
    data_size = len(train_data)
    t1 = time.time()

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for data, y in train_data:
            data = Variable(data.view(data.size(0), -1)).to(device)
            y = Variable(y).to(device)

            y_hat = net(data).to(device)

            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            _, pred = y_hat.max(1)
            num_correct = (pred == y).sum().item()
            acc = num_correct / data.shape[0]
            train_acc += acc

        train_loss = train_loss / data_size
        train_acc = train_acc / data_size
        loss_list.append(train_loss)
        acc_list.append(train_acc)

        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch, train_loss, train_acc))

    print('训练花费时间:', time.time()-t1)
    return loss_list, acc_list, net


def train_process_show(loss_list, acc_list):
    loss_list = list(map(lambda loss: loss.cpu().numpy(), loss_list))
    epochs = len(loss_list)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('loss process')
    plt.plot(range(epochs), loss_list)
    plt.subplot(1, 2, 2)
    plt.title('accuracy process')
    plt.plot(range(epochs), acc_list)
    plt.show()
    plt.close()


def model_eval(net, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_loss = 0
    eval_acc = 0
    net.eval()
    loss_fn = nn.CrossEntropyLoss()

    data_size = len(test_data)
    for data, y in test_data:
        data = Variable(data.view(data.size(0), -1)).to(device)
        y = Variable(y).to(device)

        y_hat = net(data).to(device)

        loss = loss_fn(y_hat, y)
        eval_loss += loss

        _, pred = y_hat.max(1)
        num_correct = (pred == y).sum().item()
        acc = num_correct / data.shape[0]
        eval_acc += acc

    eval_loss = eval_loss / data_size
    eval_acc = eval_acc / data_size
    print('网络测试:\nEval Loss: {:.6f}, Eval Acc: {:.6f}'.format(eval_loss, eval_acc))


def show_img(img, save=False, path=''):
    plt.imshow(img, cmap='gray')
    plt.show()
    if save:
        plt.savefig(path)
    plt.close()


def net_predict(img, img_label, net, size=28):
    y_hat = net(img)
    _, y_hat = y_hat.max(1)

    plt.imshow(img.view(size, -1).cpu().numpy(), cmap='gray')
    plt.show()
    plt.close()

    print('y =', img_label)
    print('y_hat =', y_hat)
    return y_hat


def read_img(path, size=28):
    image = Image.open(path) .convert('L') # 转换成灰度图
    image_arr = np.asarray(image)  # .astype(np.float32)
    if np.sum(image_arr > 150) > image_arr.size / 2:
        image_arr = np.where(image_arr<150, image_arr/2, image_arr)
        image_arr = 255 -image_arr
    image = Image.fromarray(image_arr)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),  # 切割
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image)
    image = image.view(image.size(0), -1)
    return image


def init_param():
    x_dim = 28 * 28
    w1 = 384
    w2 = 194
    w3 = 96
    y_dim = 10
    lr = 1e-2
    epochs = 1
    return x_dim, w1, w2, w3, y_dim, lr, epochs


if __name__ == '__main__':
    data_dir = '../ch00_dataset'
    path = '../ch00_dataset/0_9_img/hs_7.png'  # 黑底白字

    train_data, test_data = data_loader(data_dir, 128, 4)
    data_show(train_data.dataset)
    params = init_param()

    loss, acc, net = iterator_net(*params, train_data)

    train_process_show(loss, acc)

    model_eval(net, test_data)

    data_iter = iter(train_data)
    images, labels = data_iter.next()
    img = images[0].view(images[0].size(0),-1)      # 1x784
    show_img(img.cpu().numpy().reshape(28, 28))

    y_hat = net_predict(img.cuda(), labels[0], net)

    img1 = read_img(path)

    y1_hat = net_predict(img1.cuda(), path, net)

    # print(img)
    # print(img1)











