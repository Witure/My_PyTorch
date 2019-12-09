import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ae import AE
import torch.optim as optim
from vae import VAE
import visdom

def main():
    mnist_train = datasets.MNIST("mnist", True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST("mnist", False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_train, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()
    print("x : ", x.shape)
    device = torch.device("cuda")
    print(device)
    model = VAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    viz = visdom.Visdom()
    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            x = x.to(device)
            x_hat, kld = model(x)
            loss = criteon(x_hat, x)
            if kld is not None:
                elbo = - loss - 1.0 * kld
                loss = - elbo
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item(), 'kld:', kld.item())

        x, _ = iter(mnist_train).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)
        viz.images(x, nrow=8, win="x", opts=dict(title="x"))
        viz.images(x_hat, nrow=8, win="x_hat", opts=dict(title="x_hat"))


if __name__ == '__main__':
    main()
