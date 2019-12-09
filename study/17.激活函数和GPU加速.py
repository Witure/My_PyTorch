import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
import torch.nn as nn

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



device = torch.device('cuda:0')
net = MLP().to(device)
print(net)
cri = nn.CrossEntropyLoss().to(device)
print(cri)












