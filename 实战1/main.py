import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from lenet5 import Lenet5
from torch import nn, optim


def main():
    batchsz = 32
    cifar_train = datasets.CIFAR10("cifar", True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10("cifar", False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print("x:", x.shape, "label:",label.shape)


    device = torch.device('cuda')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimzer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    for epoch in range(1000):

        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            #[b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits : [b, 10]
            # label : [b]
            # loss : tensot scalar
            loss = criteon(logits, label)

            #backprop
            optimzer.zero_grad()
            loss.bacward()   # 这里得到的梯度会累加到原来的梯度上面，所以在上一步要有清零操作这样才能得到新的梯度而不是与旧锑度的相加
            optimzer.step()

        print(epoch, loss.item())

        # 实战0

















if __name__ == "__main__":
    main()
