import torch
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd


# 1.加减乘除四则运算 add sub mul div 一般而言建议直接使用运算符
# a = torch.rand(3,4)
# b = torch.rand(4)
# x1 = a + b
# print(x1.shape)
# x2 = torch.add(a,b)
# print(x2.shape)
# print(torch.all(torch.eq(a + b, torch.add(a, b))))
# print(torch.all(torch.eq(a - b, torch.sub(a, b))))
# print(torch.all(torch.eq(a * b, torch.mul(a, b))))
# print(torch.all(torch.eq(a / b, torch.div(a, b))))
# print(torch.all(torch.eq(a + b, torch.add(a, b))))
# print(torch.all(torch.eq(a - b, torch.sub(a, b))))

# 2.矩阵运算
# a = torch.tensor([[1,2,3],[4,5,6]])
# b = torch.tensor([[1,2],[2,3],[5,6]])
# c = torch.matmul(a, b)
# d = a @ b
# print(c)
# print(d)
# x = torch.rand(4, 784)
# y = torch.rand(4, 784)
# v = torch.rand(512, 784)
# z = x @ v.transpose(0,1)  # 参数位互换的维度
# print(z.shape)
# w = torch.rand(10, 512)
# k = z @ w.t()
# print(k.shape)


# 3.多维矩阵运算
# a = torch.rand(4,3,28,64)
# b = torch.rand(4,3,64,32)
# c1 = torch.matmul(a, b)  #只取了最后的两维进行乘法运算
# print(c1.shape)


# 4.平方：pow/** 平方根：rsqrt
# a = torch.full([2,2],3)  #创建一个两行两列全为3的矩阵
# # print(a)
# a1 = a.pow(2)
# print(a1)
# print(a ** 2)
# a2 = a1.sqrt()
# print(a2)
# a3 = a1.rsqrt()
# print(a3)
# a4 = a1.pow(0.5)
# print(a4)

# 5.对数运算
# a = torch.exp(torch.ones(2,2))
# print(a)
# a1 = torch.log(a)  # 默认以e为底
# print(a1)

# 6.取整运算
# a = torch.tensor(3.14)
# a1 = a.floor()
# a2 = a.ceil()
# a3 = a.trunc()
# a4 = a.frac()
# print(a1,a2,a3,a4)
# a5 = a.round()
# print(a5)
# b = torch.tensor(3.66)
# b1 = b.round()
# print(b1)


# 6.裁剪（主要应用于梯度的裁剪）
# grad = torch.rand(2, 3) * 15
# print(grad.max())
# print(grad.median())
# print(grad)
# print(grad.clamp(5,10)) # 小于5的数全部用5代替，大于10的数全部用10代替






























