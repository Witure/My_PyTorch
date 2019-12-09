import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class NumbersDataset(Dataset):
    def __init__(self, tranining=True):
        if tranining:
            self.samples = list(range(1, 1001))
        else:
            self.samples = list(range(1001, 1501))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



