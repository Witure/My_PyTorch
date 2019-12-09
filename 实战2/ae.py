import torch
from torch import nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()

        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.ReLU()

        )

    def forward(self, x):
        """

        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, 784)
        # encoder
        x = self.encoder(x)
        # deconder
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)
        return x, None





























