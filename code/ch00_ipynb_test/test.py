'''
Author: your name
Date: 2022-03-17 12:36:45
LastEditTime: 2022-03-20 22:35:14
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \ch00_ipynb_test\test.py
'''
import re
import torch

x = torch.arange(4.0, requires_grad=True)
y = x * x 
y.sum().backward()
print(x.grad)

def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(next(g))