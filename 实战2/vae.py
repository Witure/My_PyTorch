import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()

        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
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
        h_ = self.encoder(x)
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.rand_like(sigma)
        # deconder
        x_hat = self.decoder(x)
        # reshape
        x_hat = x_hat.view(batchsz, 1, 28, 28)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 28 * 28)

        return x_hat, None





























