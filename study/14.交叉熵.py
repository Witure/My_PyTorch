import torch
from torch.nn import functional as F

# entropy
# a = torch.full([4], 1/4.)
# a1 = a * torch.log2(a)
# print(a)
# print(a1)
# a2 = -(a * torch.log2(a)).sum()  # 很明显这里没有任何先验概率，因此这里的熵是最大的，因为不确定性最高
# print(a2)
#
# a = torch.tensor([0.1, 0.1, 0.1, 0.7])
# a3 = -(a * torch.log2(a)).sum()
# print(a3)
#
# a = torch.tensor([0.001, 0.001, 0.001, 0.997])
# a4 = -(a * torch.log2(a)).sum()  # 该数值反映的是不确定性，如果可以确定一个概率是最高的，那么这个值将会比较高
# print(a4)


#cross entropy
x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x@w.t()
print(logits)

pred = F.softmax(logits, dim=1)
print(pred)

pred_log = torch.log(pred)

a = F.cross_entropy(logits, torch.tensor([3]))  # 这里不可以传入pred 因为这里默认会进行一次softmax
print(a)
b = F.nll_loss(pred_log, torch.tensor([3]))
print(b)


















