import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 用来对不同维度数据进行计算并且可以省下很多内存,在低维度设计不合理的时候无法使用
a = torch.rand(4, 32, 14, 14)
# 例子另说

