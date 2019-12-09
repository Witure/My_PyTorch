import torch
import torch.nn as nn
from torch.nn import functional as F


layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0) # 第一个参数是input的参数，第二个参数是kernels的数量，后面分别是kernels的大小，步长，边框,且该接口为类接口（大写）
x = torch.rand(1, 1, 28, 28)

out = layer.forward(x)  # 这个函数可以直接进行一次卷积运算
print(out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.shape)

out = layer(x)  # 这里是更为推荐的方法，不是很推荐用forward函数，因为这样会使一些pytorch自带的接口无法使用
print(out.shape)

print(layer.weight)
print(layer.weight.shape)
print(layer.bias.shape)

















