import torch
import matplotlib.pyplot as plt


DATA_SIZE = 500
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


X = torch.nn.Parameter(get_random_matrix(DATA_SIZE, 2))
optimizer_X = torch.optim.SGD([X], lr=1e-3)

# z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
# z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
z_params = [torch.nn.Parameter(torch.rand(X.size(dim=1), device=device).multiply(torch.rand(2)).subtract(torch.tensor(1))) for i in range(X.size(dim=0))]
optimizer_z = torch.optim.SGD(z_params, lr=1e-3)

print(X)
print(z_params)

for i in range(X.size(dim=0)):
    print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))


def get_mean_soft_tukey_depth(X, z_params):
    mean = torch.tensor(0)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))
    return mean.divide(X.size(dim=0))


def get_variance_soft_tukey_depth(X, z_params):
    mean = get_mean_soft_tukey_depth(X, z_params)
    var = torch.tensor(0)
    for i in range(X.size(dim=0)):
        var = var.add(torch.square(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]).subtract(mean)))
    return var.divide(X.size(dim=0) - 1)


def get_kde_norm_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    Y = torch.zeros(n)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i])
    kde_norm = torch.tensor(0)
    for i in range(n):
        for j in range(n):
            diff = Y[i] - Y[j]
            kde_norm = kde_norm.add(-torch.square(diff))
    return kde_norm.divide(torch.square(torch.tensor(n)))


def get_kth_moment_soft_tukey_depth(X, z_params, k, data_size):
    n = X.size(dim=0)
    Y = torch.zeros(n)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / data_size
    return torch.pow(Y, k).mean()


def get_kth_moment_uniform_distribution(k, b):
    return torch.pow(torch.tensor(b), torch.tensor(k+1)).divide(torch.tensor(b*(k+1)))


def get_moment_loss(X, z_params, k, data_size):
    moment_loss = torch.tensor(0)
    for i in range(k):
        moment_loss = moment_loss.add(torch.square(
            get_kth_moment_soft_tukey_depth(X, z_params, i + 1, data_size).subtract(get_kth_moment_uniform_distribution(i + 1, 0.5))))
    return moment_loss


def draw_histogram(X, z_params, data_size):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]) / data_size
        print(soft_tukey_depths[i])
    tukey_depth_histogram = plt.figure()
    print(soft_tukey_depths[soft_tukey_depths < 0.5].detach())
    plt.hist(soft_tukey_depths[soft_tukey_depths < 0.5].detach(), bins=200)
    tukey_depth_histogram.show()


draw_histogram(X, z_params, DATA_SIZE)


for i in range(40):
    # var = get_variance_soft_tukey_depth(X, z_params)
    # var.backward()
    # kde_norm = get_kde_norm_soft_tukey_depth(X, z_params)
    # (-kde_norm).backward()

    moment_loss = get_moment_loss(X, z_params, 6, DATA_SIZE)

    moment_loss.backward()
    optimizer_X.step()

    print(X)
    for j in range(X.size(dim=0)):
        soft_tukey_depth(X[j].reshape(1, -1), X, z_params[j]).backward()
        optimizer_z.step()
        print(z_params[j])


for i in range(X.size(dim=0)):
    print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

print('Variance')
print(get_variance_soft_tukey_depth(X, z_params))

draw_histogram(X, z_params, DATA_SIZE)

X_np = X.detach().numpy()

X_scatter_plot = plt.figure()
plt.scatter(X_np[:,0], X_np[:,1])
X_scatter_plot.show()