#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
==================================================
@Project -> File   ：study-ml -> logr_cancer
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2021/6/18 6:45
@Desc   ：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
==================================================
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


# 数据获取sh
def dataloader():
    # 获取数据并添加字段名
    column_name = ['userID', '肿块厚度', '细胞大小均匀度', '细胞形状均匀度', '边缘粘', '单一上皮细胞大小', '裸核', '乏味染色体', '正常核', '有丝分裂', '2良性/4恶性']

    cancer = pd.read_csv("..\ch00_dataset/breast-cancer-wisconsin.data", names=column_name)
    cancer.head()

    # 缺失值处理
    cancer = cancer.replace(to_replace="?", value=np.nan)
    cancer = cancer.dropna()

    # 数据集划分
    # 1> 提取特征数据与目标数据
    x = cancer.iloc[:, 1:-2]
    y = cancer.iloc[:, -1]
    # 2> 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 标准化处理
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, y_train, x_test, y_test


# 模型训练和评估
def model(x_train, y_train, x_test, y_test):

    # 创建一个逻辑回归估计器
    estimator = LogisticRegression()
    # 训练模型，进行机器学习
    estimator.fit(x_train, y_train)
    # 得到模型，打印模型回归系数，即权重值
    print("logist回归系数为:\n", estimator.coef_)
    # 模型评估
    # 方法1：真实值与预测值比对
    y_predict = estimator.predict(x_test)
    print("预测值为:\n", y_predict)
    print("真实值与预测值比对:\n", y_predict==y_test)
    # 方法2：计算准确率
    print("直接计算准确率为:\n", estimator.score(x_test, y_test))

    # 接上面的肿瘤预测代码

    # 打印精确率、召回率、F1 系数以及该类占样本数
    print("精确率与召回率为:\n", classification_report(y_test,y_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    # 模型评估
    # ROC曲线与AUC值
    # 把输出的 2 4 转换为 0 或 1
    # y_test = np.where(y_test > 2, 1, 0)  # 大于2就变为1，否则变为0
    print("AUC值:\n", roc_auc_score(y_test, y_predict))

    # 预测结果生成混淆矩阵，confusion matrix
    target_names = ['begin', 'malignant']
    mat = confusion_matrix(y_test, y_predict)
    plt.title("predict-to-true confusion matrix")
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


# main
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = dataloader()
    model(x_train, x_test, y_train, y_test)
