import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
import torch.nn as nn




# 全军出击：全连接层
# x = torch.rand([1, 784])
# print(x.shape)
#
# layer1 = nn.Linear(784, 200)
# layer2 = nn.Linear(200, 200)
# layer3 = nn.Linear(200, 10)
#
# x = layer1(x)
# print(x.shape)
#
# x = layer2(x)
# print(x.shape)
#
# x = layer3(x)
# print(x.shape)



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
            x = self.model(x)
            return x
net = MLP()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()


