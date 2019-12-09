import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


# 非常重要而且用途很广
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 这里参数为统计到的均值和方差
x = torch.randn(100, 16) + 0.5
layer = torch.nn.BatchNorm1d(16)

print(layer.running_mean)  # 均值
print(layer.running_var)   # 方差

out = layer(x)  # 前向传播一次，将很大范围的数据缩放到0-1范围波动

print(layer.running_mean)
print(layer.running_var)
print(vars(layer))