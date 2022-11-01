import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


DATA_SIZE = 200
KERNEL_BANDWIDTH = 0.1
# USE_CUDA_IF_AVAILABLE = True
#
# if torch.cuda.is_available():
#     print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
# else:
#     print('GPU is not available')
#
# device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
# print('The model will run with {}'.format(device))

device = 'cpu'

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


X = torch.tensor(get_random_matrix(DATA_SIZE, 2), requires_grad=True)
optimizer_X = torch.optim.SGD([X], lr=3e+2)

# z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
# z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
z_params = [torch.tensor(torch.rand(X.size(dim=1), device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)), requires_grad=True) for i in range(X.size(dim=0))]
optimizer_z = torch.optim.SGD(z_params, lr=1e-1)

print(X)
print(z_params)

for i in range(X.size(dim=0)):
    print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))


def get_mean_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    mean = torch.tensor(0)
    for i in range(n):
        mean = mean.add(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / n)
    return mean.divide(n)


def get_variance_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    mean = get_mean_soft_tukey_depth(X, z_params)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]).divide(n).subtract(mean)))
    return var.divide(n - 1)


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


def draw_histogram(X, z_params):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / n
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths[soft_tukey_depths < 1].detach(), bins=50)
    tukey_depth_histogram.show()


def draw_scatter_plot(X, z_params, show_z_vectors=False):
    X_np = X.detach().numpy()
    z_normalized = np.zeros(X.size())
    for i in range(len(z_params)):
        z_normalized[i] = z_params[i].detach() / z_params[i].detach().norm()

    X_scatter_plot = plt.figure()
    if show_z_vectors:
        plt.scatter(
            np.append(X_np[:, 0], X_np[:, 0] + z_normalized[:, 0]),
            np.append(X_np[:, 1], X_np[:, 1] + z_normalized[:, 1]),
            c=['#0000ff' for i in range(X.size(dim=0))] + ['#ff0000' for i in range(X.size(dim=0))]
        )
    else:
        plt.scatter(X_np[:, 0], X_np[:, 1])
    X_scatter_plot.show()


def get_sum_of_norm(X):
    val = torch.tensor(0)
    for i in range(X.size(dim=0)):
        val = val.add(torch.norm(X[i]))
    return val


for i in range(50):
    optimizer_X.zero_grad()
    # var = get_variance_soft_tukey_depth(X, z_params)
    # (var).backward()

    # (-torch.norm(X[0] + X[1])).backward()



    # kde_norm = get_kde_norm_soft_tukey_depth(X, z_params, KERNEL_BANDWIDTH)
    # (-kde_norm).backward()

    moment_loss = get_moment_loss(X, z_params, 6)
    moment_loss.backward()

    # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(X, z_params)
    # (-inverse_sum_loss).backward()

    optimizer_X.step()

    print(X)
    for j in range(X.size(dim=0)):
        optimizer_z.zero_grad()
        soft_tukey_depth(X[j].reshape(1, -1), X, z_params[j]).backward()
        optimizer_z.step()
        print(z_params[j])

    if i % 3 == 0:
        draw_scatter_plot(X, z_params, False)


for i in range(X.size(dim=0)):
    print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

print('Variance')
print(get_variance_soft_tukey_depth(X, z_params))

print('Inverse Sum')
print(get_inverse_sum_soft_tukey_depth(X, z_params))

print('KDE Norm')
print(get_kde_norm_soft_tukey_depth(X, z_params, KERNEL_BANDWIDTH))

print('Moment Loss')
print(get_moment_loss(X, z_params, 6))

for i in range(6):
    print(f'Moment {i+1} is {get_kth_moment_soft_tukey_depth(X, z_params, i+1)}')

draw_histogram(X, z_params)