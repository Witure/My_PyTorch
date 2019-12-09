import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


# 循环神经网络（nlp，常用于自然语言处理），之前产生的结果会对后续网络继续产生影响也就是memory
# rnn = nn.RNN(100, 10)  # 三个参数分别为input_size， hidden_size, num_layers
# print(rnn._parameters.keys())

# 单层RNN
# rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
# print(rnn)
# x = torch.randn(10, 3, 100)
# out, h = rnn(x, torch.zeros(1, 3, 20))
# print(out.shape, h.shape)   # h是最后一层的输出，out是总的一个输出所以最前面的维度是10，因为总共是十层，三个句子，每个句子是20维数据表示


# 双层RNN
rnn = nn.RNN(100, 10, num_layers=2)
print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape)
print(rnn.weight_ih_l0.shape)

print(rnn.weight_hh_l1.shape)
print(rnn.weight_ih_l1.shape)

# 四层RNN
rnn = nn.RNN(10, 20, num_layers=4)



























