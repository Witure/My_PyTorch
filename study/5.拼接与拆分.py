import torch
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from torch.backends import cudnn

# 四个经典api
# 拼接：cat stack  拆分：split:长度 chunk:数量
# 1.cat 直接进行相加合并
# a = torch.rand(4, 32, 8)
# b = torch.rand(5, 32, 8)
# x = torch.cat([a,b],dim=0).shape
# print(x)

# 2.stack：新增加一个维度，其他维度保持不变
# a1 = torch.rand(4,3,32,32)
# a2 = torch.rand(4,3,32,32)
# a3 = torch.stack([a1,a2],dim=2).shape
# print(a3)
#

# 3.split：根据长度拆分
# a = torch.rand(32,8)
# b = torch.rand(32,8)
# a1 = torch.stack([a,b],dim=0)
# print(a1.shape)
# aa,bb = a1.split([1,1],dim=0)
# print(aa.shape,bb.shape)
# x,y = a1.split(1,dim=0)
# print(x.shape,y.shape)


# 4.take：根据索引进行分割(先把tensor打平，然后再进行取值，一般而言用得少）
# src = torch.tensor([[3,4,5],[6,7,8]])
# s = torch.take(src,torch.tensor([0,2,5]))
# print(s)
#


# 5.chuunk : 通过数量进行拆分
# a = torch.rand(4,32,8)
# a1,a2 = a.chunk(2,dim=0)
# print(a1.shape,a2.shape)





































