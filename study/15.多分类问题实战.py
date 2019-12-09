import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F



# 使用交叉熵来解决多分类问题
w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)


def forword(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x


optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=1e-3)
critteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = forword(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()













