'''
Author: jhont.tao
Date: 2022-03-21 21:16:52
LastEditTime: 2022-03-21 21:57:30
Description: 
'''
import numpy as np
from matplotlib import pyplot as plt
import torch
# x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
# plt.plot(x, np.sin(x))       # Plot the sine of each x point
# plt.show()                   # Display the plot
ones = torch.ones(10, 3, 4)
print(ones, "\n", ones.shape)

