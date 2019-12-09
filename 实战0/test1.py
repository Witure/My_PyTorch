import torch


# a = torch.ones(2)
# print(a)
# b = torch.randn(2,3)
# print(b)
# x = torch.rand(2,2,3)
# print(x)
# print(list(x.shape))
# y = torch.rand(4,4,4,4)
# print(y)
# print(y.numel())
points = [[1,1.9],[2,3.3],[3,4.1],[4,5.2]]
def a(w, b, points):
    loss = 0
    for i in points:
        x = i[0]
        y = i[1]
        # print(x)
        loss += (y - (w * x + b)) ** 2
    return loss / len(points)

loss = a(1,1,points)
print(loss)