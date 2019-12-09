import torch
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from torch.backends import cudnn


# 1.view 和 reshape  数据的存储/维度非常重要，必须牢记
# 1.1:view,必须要有一定的物理意义
# a = torch.rand(4,1,28,28)
# print(a.shape)
# a1 = a.view(4,28*28)
# print(a1.shape)
# print(a1)
# a2 = torch.rand(4*1*28,28)
# print(a2.shape)
# b = a.view(4,28,28,1)  #logis bug




































