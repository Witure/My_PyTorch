import torch


# 以下操作都借助了gpu 因此效率很高

# 1.where  torch.where(dondition,x,y) -> tensor
# cond = torch.tensor([[0.6769, 0.7271],[0.8884, 0.4163]])
# print(cond)
# a = torch.tensor([[0.,0.],[0.,0.]])
# print(a)
# b = torch.tensor([[1.,1.],[1.,1.]])
# print(b)
# c = torch.where(cond>0.5, a, b)  #这里cond>0.5是一个条件，如果cond>0.5成立，则在对应维度对应位置取a的值，否者取b的值
# print(c)


# 2.gather : torch.gather(input, dim, index, out=None) -> tensor （查表操作）
# prob = torch.randn(4, 10)
# idx = prob.topk(dim=1, k=3)
# print(idx)
# idx = idx[1]
# print(idx)
# label = torch.arange(10) + 100
# print(label)
# x = torch.gather(label.expand(4, 10), dim=1, index=idx.long())  #label为用来查阅的表，也就是用来衡量的原表，dim不用多说了，index里面保存的是索引，也就是原表对应的位置
# print(x)
# print(label.expand(4, 10))
# print(idx.long())









