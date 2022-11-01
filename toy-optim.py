import torch
import torch.optim


def f(x):
    return torch.pow(x.subtract(1), 2)


x = torch.rand(1, requires_grad=True)

optim_x = torch.optim.SGD([x], lr=1e-2)

for i in range(100):
    optim_x.zero_grad()
    val = -torch.norm(x)
    print(x, val)
    val.backward()
    optim_x.step()

