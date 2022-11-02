import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset
from models.MNIST_Encoder import MNIST_Encoder
import torch.utils.data
import numpy as np


DATA_SIZE = 500
USE_CUDA_IF_AVAILABLE = True
KERNEL_BANDWIDTH = 0.05

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def get_random_matrix(m, n):
    matrix = torch.zeros((m, n), device=device)
    for i in range(m):
        for j in range(n):
            matrix[i][j] = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
    return matrix


def soft_tukey_depth(x, x_, z):
    matmul = torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1), torch.divide(
        torch.matmul(torch.subtract(x_, matmul), z),
        torch.norm(z)))))


train_data = NominalMNISTImageDataset(nominal_class=0, train=True)
train_dataloader = torch.utils.data.DataLoader(train_data)

encoder = MNIST_Encoder().to(device)
encoder.train()

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-5)

# z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
# z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
z_params = [torch.nn.Parameter(torch.rand(2, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
optimizer_z = torch.optim.SGD(z_params, lr=1e-5)


def get_mean_soft_tukey_depth(X, z_params):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))
    return mean.divide(X.size(dim=0))


def get_variance_soft_tukey_depth(X, z_params):
    mean = get_mean_soft_tukey_depth(X, z_params)
    var = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        var = var.add(torch.square(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]).subtract(mean)))
    return var.divide(X.size(dim=0) - 1)


def get_kde_norm_soft_tukey_depth(X, z_params, bandwidth):
    n = X.size(dim=0)
    Y = torch.zeros(n).to(device)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i])
    kde_norm = torch.tensor(0).to(device)
    for i in range(n):
        for j in range(n):
            diff = Y[i] - Y[j]
            kde_norm = kde_norm.add(torch.exp(-torch.square(diff).divide(torch.tensor(4 * bandwidth ** 2))))
    return kde_norm.divide(torch.square(torch.tensor(n)))


def get_kth_moment_soft_tukey_depth(X, z_params, k):
    n = X.size(dim=0)
    Y = torch.zeros(n)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / n
    return torch.pow(Y, k).mean()


def get_kth_moment_uniform_distribution(k, b):
    return torch.pow(torch.tensor(b), torch.tensor(k+1)).divide(torch.tensor(b*(k+1)))


def get_moment_loss(X, z_params, k):
    moment_loss = torch.tensor(0)
    for i in range(k):
        moment_loss = moment_loss.add(torch.square(
            get_kth_moment_soft_tukey_depth(X, z_params, i + 1).subtract(get_kth_moment_uniform_distribution(i + 1, 0.5))))
    return moment_loss


def get_inverse_sum_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    inverse_sum_loss = torch.tensor(0)
    for i in range(n):
        inverse_sum_loss = inverse_sum_loss.add(torch.divide(torch.tensor(n), soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i])))
    return torch.divide(inverse_sum_loss, torch.tensor(n))


def draw_histogram(X, z_params, data_size):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / data_size
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths[soft_tukey_depths < 0.5].detach(), bins=200)
    tukey_depth_histogram.show()


def draw_scatter_plot(X, z_params):
    X_np = X.detach().numpy()
    z_normalized = np.zeros(X.size())
    for i in range(len(z_params)):
        z_normalized[i] = z_params[i].detach() / z_params[i].detach().norm()

    X_scatter_plot = plt.figure()
    plt.scatter(
        np.append(X_np[:, 0], X_np[:, 0] + z_normalized[:, 0]),
        np.append(X_np[:, 1], X_np[:, 1] + z_normalized[:, 1]),
        c=['#0000ff' for i in range(X.size(dim=0))] + ['#ff0000' for i in range(X.size(dim=0))]
    )
    X_scatter_plot.show()


for i in range(500):
    print(i)
    n = len(train_data)
    Y = torch.zeros((n, 2)).to(device)

    for step, X in enumerate(train_dataloader):
        X = X.to(device)
        Y[step] = encoder(X)

    print('Computed encodings')

    optimizer_encoder.zero_grad()
    var = get_variance_soft_tukey_depth(Y, z_params)
    (-var).backward()
    optimizer_encoder.step()

    for j in range(n):
        optimizer_z.zero_grad()
        soft_tukey_depth(Y[j].reshape(1, -1), Y, z_params[j]).backward()
        optimizer_z.step()
        print(z_params[j])

    draw_scatter_plot(Y, z_params)


# for i in range(X.size(dim=0)):
#     print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

# print('Variance')
# print(get_variance_soft_tukey_depth(X, z_params))
#
# print('Inverse Sum')
# print(get_inverse_sum_soft_tukey_depth(X, z_params))