'''
Author: jhont.tao
Date: 2022-03-21 21:16:52
LastEditTime: 2022-03-22 17:06:12
Description: 
'''
import numpy as np
from matplotlib import pyplot as plt
import torch
# 绘图测试
# x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
# plt.plot(x, np.sin(x))       # Plot the sine of each x point
# plt.show()                   # Display the plot

# 根据索引获取tensor元素
# ones = torch.arange(48).reshape(4, 3, 4)
# print(ones, "\n", ones.shape)
# index = torch.tensor([0, 2])  # 索引
# print(index.shape)
# print(ones[index])  # 根据索引获取元素

# 测试requires_grid
# ones = torch.arange(4., requires_grad=True)
# print(ones)

# visdom散点图测试
import numpy as np
import visdom

viz = visdom.Visdom(env="scatter test")
 
win = viz.scatter(
        X=np.random.rand(100, 2),
        Y=np.random.rand(100, 2),
        opts=dict(
            legend=['Didnt', 'Update'],
            xtickmin=-50,
            xtickmax=50,
            xtickstep=0.5,
            ytickmin=-50,
            ytickmax=50,
            ytickstep=0.5,
            markersymbol='cross-thin-open',
        ),
    )
