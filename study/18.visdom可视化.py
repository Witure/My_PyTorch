# 很倒霉这里配置失败了，白白忙活了一晚上
# 呸呸呸！！！我配置出来了，两千行代码眼都看瞎了
from visdom import Visdom
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
import torch.nn as nn


viz = Visdom()
a = torch.tensor([0.])

viz.line([0.], [0.], win='实战0', opts=dict(title='实战0 loss&acc.', legend=['loss', 'acc.']))
viz.line([a], [a], win='实战0', update='append')

print("我太难啦")


