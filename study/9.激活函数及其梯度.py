import torch
from torch.nn import functional as F

# 1.什么是激活函数  需要一个值达到特定大小才能激活该函数，因此这个函数并没有导数

# 2.sigmoid（连续光滑并且大小在0-1之间，但容易出现梯度弥散导致梯度长时间得不到更新） 和 logistic
# a = torch.linspace(-100, 100, 10)
# print(a)
# a1 = torch.sigmoid(a)
# print(a1)

# 3.tanh 经常用于rnn算法
# a = torch.linspace(-1, 1, 10)
# print(a)
# a1 = torch.tanh(a)
# print(a1)

# 4.rectified linear unit （relu），非常适用于深度学习,一般而言更为优先使用这个函数
# a = torch.linspace(-1, 1, 10)
# print(a)
# a1 = torch.relu(a)  # 更为推荐的接口形式
# print(a1)
# a2 = F.relu(a)
# print(a2)

# 6.loss及其梯度
# 6.1:Mean Squared Error（MSE）均方差
# x = torch.ones(1)
# w = torch.full([1], 2)  #创建一个一维的tensor，用2来填充这个tensor
# print(w)
# mse = F.mse_loss(torch.ones(1), x*w)  #用来求预测值和真实值的均方差
# print(mse)
# w.requires_grad_()  #声明w可导
# mse = F.mse_loss(torch.ones(1), x*w)  #因为pytorch为动态图，所以必须再次声明才能进行使用
# a = torch.autograd.grad(mse, [w])    #求该函数的梯度
# print(a)
# # 方法2，继续前面的程序
# mse = F.mse_loss(torch.ones(1), x*w)
# mse.backward()
# print(w.grad)

# 7.softMax（有点难，记得回去多看看视频是怎么推导的）
# a = torch.rand(3)  #创建一个随机的tensor
# print(a)
# a.requires_grad_()  #声明 允许求导
# p = F.softmax(a, dim=0)  #进行一次softmax操作，这会让不同数之间的倍数差距变大
# print(p)
# p1 = torch.autograd.grad(p[1], [a], retain_graph=True)  # 求导，只能一步步来，不能整个p一起求导
# p2 = torch.autograd.grad(p[2], [a], retain_graph=True)
# print(p1)
# print(p2)






