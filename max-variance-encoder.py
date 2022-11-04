import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset
from models.MNIST_Encoder_Simple import MNIST_Encoder_Simple
from models.MNIST_Encoder_DSVDD import MNIST_Encoder_DSVDD
import torch.utils.data
import numpy as np


DATA_SIZE = 500
TEST_NOMINAL_SIZE = 200
TEST_ANOMALOUS_SIZE = 200

USE_CUDA_IF_AVAILABLE = True
KERNEL_BANDWIDTH = 0.05
ENCODING_DIM = 32

torch.autograd.set_detect_anomaly(True)

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
    matmul = torch.outer(torch.ones(x_.size(dim=0), device=device), x)
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(0.1), torch.divide(
        torch.matmul(torch.subtract(x_, matmul), z),
        torch.norm(z)))))


train_data = torch.utils.data.Subset(NominalMNISTImageDataset(nominal_class=1, train=True), list(range(DATA_SIZE)))
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)

test_data_nominal = torch.utils.data.Subset(NominalMNISTImageDataset(nominal_class=1, train=False), list(range(TEST_NOMINAL_SIZE)))
test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=TEST_NOMINAL_SIZE)

test_data_anomalous = torch.utils.data.Subset(AnomalousMNISTImageDataset(nominal_class=1, train=False), list(range(TEST_ANOMALOUS_SIZE)))
test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=TEST_ANOMALOUS_SIZE)

encoder = MNIST_Encoder_DSVDD().to(device)
encoder.train()

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-2)

# z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
# z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
z_params = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
optimizer_z = torch.optim.SGD(z_params, lr=1e-1)


def get_mean_soft_tukey_depth(X, z_params):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i], X, z_params[i]))
    return mean.divide(X.size(dim=0))


def get_variance_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    mean = get_mean_soft_tukey_depth(X, z_params)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i], X, z_params[i]).divide(n).subtract(mean)))
    return var.divide(n - 1)


def get_kde_norm_soft_tukey_depth(X, z_params, bandwidth):
    n = X.size(dim=0)
    Y = torch.zeros(n).to(device)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i], X, z_params[i])
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
        Y[i] = soft_tukey_depth(X[i], X, z_params[i]) / n
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
        inverse_sum_loss = inverse_sum_loss.add(torch.divide(torch.tensor(n), soft_tukey_depth(X[i], X, z_params[i])))
    return torch.divide(inverse_sum_loss, torch.tensor(n))


def draw_histogram(X, z_params, bins=200):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i], X, z_params[i]) / n
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths[soft_tukey_depths < 0.5].detach(), bins=bins)
    tukey_depth_histogram.show()


def draw_scatter_plot(X, z_params):
    X_np = X.detach().cpu().numpy()
    z_normalized = np.zeros(X.size())
    for i in range(len(z_params)):
        z_normalized[i] = z_params[i].detach().cpu() / z_params[i].detach().cpu().norm()

    X_scatter_plot = plt.figure()
    plt.scatter(
        np.append(X_np[:, 0], X_np[:, 0] + z_normalized[:, 0]),
        np.append(X_np[:, 1], X_np[:, 1] + z_normalized[:, 1]),
        c=['#0000ff' for i in range(X.size(dim=0))] + ['#ff0000' for i in range(X.size(dim=0))]
    )
    X_scatter_plot.show()


for i in range(30):
    print(f'Epoch {i+1}')
    n = len(train_data)

    for step, X in enumerate(train_dataloader):
        X = X.to(device)
        Y = encoder(X)

        optimizer_encoder.zero_grad()

        moment_loss = get_moment_loss(Y, z_params, 4)
        moment_loss.backward()

        # var = get_variance_soft_tukey_depth(Y, z_params)
        # (-var).backward()

        # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(Y, z_params)
        # (inverse_sum_loss).backward()

        optimizer_encoder.step()

        for j in range(n):
            for k in range(5):
                optimizer_encoder.zero_grad()
                optimizer_z.zero_grad()
                _soft_tukey_depth = soft_tukey_depth(Y[j].detach(), Y.detach(), z_params[j])
                _soft_tukey_depth.backward()
                optimizer_z.step()
                # print(j, z_params[j])


        if i % 3 == 0:
            if ENCODING_DIM == 2:
                draw_scatter_plot(Y, z_params)
        if i == 29:
            draw_histogram(Y, z_params, bins=20)

            Y = encoder(X)

            for step2, X_test_nominal in enumerate(test_dataloader_nominal):
                X_test_nominal = X_test_nominal.to(device)
                Y_test_nominal = encoder(X_test_nominal)
                z_test_nominal = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(test_data_nominal))]
                optimizer_z_test_nominal = torch.optim.SGD(z_test_nominal, lr=1e-2)

                for j in range(len(test_data_nominal)):
                    for k in range(40):
                        optimizer_z_test_nominal.zero_grad()
                        _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[j].detach(), Y.detach(), z_test_nominal[j])
                        _soft_tukey_depth.backward()
                        optimizer_z_test_nominal.step()

                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y_test_nominal, z_test_nominal)
                draw_histogram(Y_test_nominal, z_test_nominal, bins=20)

            for step2, X_test_anomalous in enumerate(test_dataloader_anomalous):
                X_test_anomalous = X_test_anomalous.to(device)
                Y_test_anomalous = encoder(X_test_anomalous)
                z_test_anomalous = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(test_data_anomalous))]
                optimizer_z_test_anomalous = torch.optim.SGD(z_test_anomalous, lr=1e-2)

                for j in range(len(test_data_nominal)):
                    for k in range(40):
                        optimizer_z_test_anomalous.zero_grad()
                        _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[j].detach(), Y.detach(), z_test_anomalous[j])
                        _soft_tukey_depth.backward()
                        optimizer_z_test_anomalous.step()

                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y_test_anomalous, z_test_anomalous)
                draw_histogram(Y_test_anomalous, z_test_anomalous, bins=20)

# for i in range(X.size(dim=0)):
#     print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

# print('Variance')
# print(get_variance_soft_tukey_depth(X, z_params))
#
# print('Inverse Sum')
# print(get_inverse_sum_soft_tukey_depth(X, z_params))