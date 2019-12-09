import torch

# 1.范数 norm
# a = torch.full([8],1)
# print(a)
# print(a.norm(1))
# b = a.view(2,4)
# c = a.view(2,2,2)
# print(b)
# print(b.norm(1)) # 求解1范数，也就是各个数绝对值得求和
#
# print(a.norm(2))
# print(b.norm(2)) # 求解2范数，也就是所有元素平方和再开根号
#
# print(a)
# print(a.norm(1,dim=0)) # 第一个参数是求几范数，第二个参数是取第几维度，取第几维度，第几维度就会被消掉
# print(b)
# print(b.norm(1,dim=1))
# print(c)
# print(c.norm(1,dim=1))


# 2.统计属性 mean sum min max prod
# a = torch.arange(8).view(2,4).float()
# print(a)
# print(a.mean(),a.sum(),a.min(),a.max(),a.prod()) # 分别为平均 求和 最小 最大 累乘
# print(a.argmin(),a.argmax())  # 分别表示最大值所在索引  最小值所在索引
# print(a.argmax(dim=1))


# 3.dim 和 keepdim
# a = torch.rand(4,10)
# print(a)
# print(a.max(dim=1))   #取某个维度的最大值，同理其他操作也可以
# print(a.max(dim=1,keepdim=True))  #keepdim用来保存原结构的dim


# 4.topk 和 kthvalue
# a = torch.rand(10)
# print(a)
# a1 = a.topk(3,dim=0)  #第一个参数是要返回的最大值的个数，同时还返回值所在的索引，第二个参数是按什么维度进行返回
# print(a1)
# a2 = a.topk(3,dim=0,largest=False)  #largest表示要取最小的值
# print(a2)
#
# a3 = a.kthvalue(8,dim=0)  # 返回第8小的数（在这个例子中也就是第3大的数），和他所在的索引位置
# print(a3)

# 5.比较运算，这里pytorch版本不同可能有一些不同的地方，具体要看版本在进行学习
# a = torch.rand(8)
# print(a)
# a1 = torch.eq(a,a)
# print(a1)
# print(a>0.5)
# b = torch.ones(8)
# b1 = torch.eq(a, b)
# print(b1)

























































