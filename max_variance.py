import torch
import matplotlib.pyplot as plt


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


X = torch.nn.Parameter(get_random_matrix(500, 2))
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


for i in range(50):
    var = get_variance_soft_tukey_depth(X, z_params)
    var.backward()
    optimizer_X.step()

    print(X)
    for i in range(X.size(dim=0)):
        soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]).backward()
        optimizer_z.step()
        print(z_params[i])


for i in range(X.size(dim=0)):
    print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

print('Variance')
print(get_variance_soft_tukey_depth(X, z_params))

X_np = X.detach().numpy()
plt.scatter(X_np[:,0], X_np[:,1])
plt.show()