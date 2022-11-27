import torch
import torch.nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


USE_CUDA_IF_AVAILABLE = True
N = 500
ITERATIONS = 20


torch.set_printoptions(precision=4, sci_mode=False)

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def soft_tukey_depth(x, x_, z, temp):
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / temp), torch.divide
        (torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))


def draw_histogram(values, bins, title):
    fig = plt.figure()
    plt.hist(np.asarray(values), bins=bins)
    plt.title(title)
    fig.show()
    fig.savefig(f'C:/Users/Dennis Weiss/Pictures/depth_exp_sphere/temp{temp}_d{d}.png')


def sample_from_cube(dim, size):
    return torch.rand((dim), device=device).multiply(2*size).subtract(size)


def sample_from_sphere(dim, radius):
    x = sample_from_cube(dim, radius)
    if torch.linalg.norm(x) > radius:
        return sample_from_sphere(dim, radius)
    return x


for temp in [0.05, 0.1, 0.2, 0.5, 1, 2]:
    for d in range(1, 7):
        # X = torch.normal(mean=0, std=1, size=(N, d), device=device, requires_grad=False)
        X = torch.zeros((N, d), device=device)
        for i in range(N):
            X[i] = sample_from_sphere(d, 2)
        z = [torch.nn.Parameter(torch.rand(d, device=device).multiply(2).subtract(1)) for _ in range(N)]
        z_optim = torch.optim.SGD(z, lr=1e-3)

        soft_tukey_depths = []

        for i in tqdm(range(N), desc=f'temp={temp}, d={d}'):
            for j in range(ITERATIONS):
                z_optim.zero_grad()
                _soft_tukey_depth = soft_tukey_depth(X[i].reshape(1, -1), X, z[i], temp)
                _soft_tukey_depth.backward()
                z_optim.step()
            soft_tukey_depths.append(soft_tukey_depth(X[i].reshape(1, -1), X, z[i], temp).item() / N)

        draw_histogram(soft_tukey_depths, 20, f'temp={temp}, d={d}')

