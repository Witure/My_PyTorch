import torch
from torch.nn import functional as F

# 1.单层感知机求梯度
# x = torch.randn(1, 10)
# w = torch.randn(1, 10, requires_grad=True)
# print(x)
# print(w)
#
# o = torch.sigmoid(x@w.t())
# print(o.shape)
#
# loss = F.mse_loss(torch.ones(1, 1), o)
# print(loss.shape)
#
# loss.backward()
# print(w.grad)



# 2.MLP及梯度
# x = torch.randn(1, 10)
# w = torch.randn(2, 10, requires_grad=True)
#
# o = torch.sigmoid(x@w.t())
# print(o.shape)
#
# loss = F.mse_loss(torch.ones(1, 2), o)
# print(loss)
#
# loss.backward()
# print(w.grad)



# 3.链式法则
# x = torch.tensor(1.)
# w1 = torch.tensor(2., requires_grad=True)
# b1 = torch.tensor(1.)
# w2 = torch.tensor(2., requires_grad=True)
# b2 = torch.tensor(1.)
#
# y1 = w1*x + b1
# y2 = y1*w2 + b2
#
# dy2_dy1 = torch.autograd.grad(y2, [y1], retain_graph=True)[0]
# dy1_dw1 = torch.autograd.grad(y1, [w1], retain_graph=True)[0]
# dy2_dw1 = torch.autograd.grad(y2, [w1], retain_graph=True)[0]
#
# print(dy2_dw1)
# print(dy2_dy1*dy1_dw1)









































