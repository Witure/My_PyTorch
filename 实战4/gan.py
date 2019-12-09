import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from torch import nn, optim, autograd
import visdom
from torch.nn import functional as F


h_dim = 400
batchsz = 512
viz = visdom.Visdom()
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
            nn.ReLU(True)
        )
        

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    scale = 2
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1 / np.sqrt(2)),
        (-1. / np.sqrt(2), -1 / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while  True:
        dataset = []
        for i in range(batchsz):
            point = np.random.rand(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        yield dataset

def generate_image(D, G, xr, epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype="float32")
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    with torch.no_grad():
        points = torch.Tensor(points).cuda()
        disc_map = D(points).cpu().numpy()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)

    with torch.no_grad():
        z =torch.rand(batchsz, 2).cuda()














































