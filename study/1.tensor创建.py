import torch
import numpy as np
import pandas as pd


# numpy导入tensor
# a = np.array([2,2,3])
# b = torch.from_numpy(a)
# print(b)
# x = np.ones([2,3])
# y = torch.from_numpy(x)
# print(y)

# list导入tensor
# a = torch.tensor([2.,3.,4.])
# print(a)
# b = torch.FloatTensor([2.,2.3])
# print(b)


# 微初始化数据
# a = torch.empty(1)
# print(a)
# b = torch.FloatTensor(3,1,28,28)
# print(b)
# torch.set_default_tensor_type(torch.DoubleTensor)
# c = torch.tensor([1.,2.,3]).type()
# print(c)


# # 初始化tensor
# a = torch.rand(3,3)
# print(a)
# b = torch.randint(1,10,[3,3])
# print(b)
# # 正态分布
# c = torch.randn(3,3)
# print(c)
# 全赋值
# d = torch.full([3,3],9)
# print(d)


# a = torch.arange(10)
# print(a)
# b = torch.arange(0,10,2)
# print(b)


# a = torch.linspace(0,10,steps=22)
# print(a)
# b = torch.logspace(0,-1,steps=10)
# print(b)


# a = torch.eye(3,5)
# print(a)
# print(torch.ones(3,5))
# print(torch.zeros(5,3))


# 随机打散
# a = torch.randperm(2)
# print(a)
# print(torch.rand(2,3))
# x = torch.rand(2,3)
# y = torch.rand(2,3)
# print(x[a])
# print(y[a])


# 切片
# a = torch.rand(4,1,28,28)
# print(a.shape)
# torch.Size([4,1,28,28])
# b = a.view(4,28*28)
# print(b)
# print(b.shape)


# a = torch.rand(4,3,28,28)
# b = a[:2,1:3,2:28:2,4:28:2]
# print(b.shape)





# a = torch.rand(1,32,1,1)
# b = a.unsqueeze(0).shape
# print(b)
# c = a.squeeze(0).shape
# print(c)
#
#
# x = a.expand(4,32,14,14)
# print(x.shape)
#
# f = a.repeat(2,32,2,2).shape  # 重复次数，不推荐使用
# print(f)


# 容易导致数据污染，使用的时候要特别注意
# a = torch.randn(3,4)  # 三行四列的矩阵
# # print(a.t())
# b = torch.randn(1,2,3,4)
# # print(b)
# # repeat用来对原来的tensor进行操作
#
# print(b.shape)
# b1 = b.transpose(1,3)
# print(b1)
# b2 = b.transpose(1,3).contiguous().view(1*2,3*4).view(1,2,3,4).transpose(2,3)
# print(b2.shape)
#
# c = torch.all(torch.eq(b,b2))
# print(c)


# permute
# a = torch.rand(4,3,28,28)
# a1 = a.transpose(1,3).shape
# print("a1 = {}".format(a1) )
# b = torch.rand(4,3,28,32)
# b1 = b.transpose(1,3).shape
# print("b1 = {}".format(b1))
# b2 = b.transpose(1,3).transpose(1,2).shape
# print("b2 = {}".format(b2))
# b3 = b.permute(0,2,3,1).shape  #原维度的重新组合，比如0的位置就是原来0的维度值，同理最后面那个1就是原来1位置的维度
# print("b3 = {}".format(b3))
#




























