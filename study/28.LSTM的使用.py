import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# LSTM
# lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
# print(lstm)
# x = torch.randn(10, 3, 100)
# # print(x)
# out, (h, c) = lstm(x)
# print(out.shape, h.shape, c.shape)


# LSTMcell
x = torch.randn(10, 3, 100)
print("two layer lstm")
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])

print(h2.shape, c2.shape)




















