import csv

import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset, NominalCIFAR10ImageDataset, \
    AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, AnomalousCIFAR10AutoencoderDataset, \
    NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset
from models.CIFAR10_AE_Encoder import CIFAR10_AE_Encoder
from models.CIFAR10_Encoder_V4 import CIFAR10_Encoder_V4
from models.CIFAR10_Encoder_V5 import CIFAR10_Encoder_V5
from models.CIFAR10_Encoder_V6 import CIFAR10_Encoder_V6
from models.MNIST_AE_Encoder import MNIST_AE_Encoder
from models.MNIST_Encoder_Simple import MNIST_Encoder_Simple
from models.MNIST_Encoder_DSVDD import MNIST_Encoder_DSVDD
from models.CIFAR10_Encoder_Simple import CIFAR10_Encoder_Simple
from models.CIFAR10_Encoder_V3 import CIFAR10_Encoder_V3
import torch.utils.data
import numpy as np
import scipy as sp


DATASET_NAME = 'MNIST_Autoencoder'
NOMINAL_DATASET = NominalMNISTAutoencoderDataset
ANOMALOUS_DATASET = AnomalousMNISTAutoencoderDataset
RESULT_NAME_DESC = 'kl_div_uniform_1000'
DATA_SIZE = 1000
TEST_NOMINAL_SIZE = 1000
TEST_ANOMALOUS_SIZE = 1000


USE_CUDA_IF_AVAILABLE = True
KERNEL_BANDWIDTH = 0.1
SOFT_TUKEY_DEPTH_TEMP = 0.1
ENCODING_DIM = 64
HISTOGRAM_BINS = 50
NUM_EPOCHS = 10
STD_ITERATIONS = 3

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
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(
        torch.matmul(torch.subtract(x_, matmul), z),
        torch.norm(z)))))


def get_mean_soft_tukey_depth(X, z_params):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i], X, z_params[i]))
    _mean = mean.divide(X.size(dim=0))
    print(_mean.item())
    return _mean


def get_variance_soft_tukey_depth_with_mean(X, z_params, mean):
    n = X.size(dim=0)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i], X, z_params[i]).divide(n).subtract(mean)))
    return var.divide(n - 1)


def get_variance_soft_tukey_depth(X, z_params):
    return get_variance_soft_tukey_depth_with_mean(X, z_params, get_mean_soft_tukey_depth(X, z_params))


def get_kl_divergence_of_kde_to_uniform_dist(X, z_params, kernel_bandwidth):
    n = X.size(dim=0)
    kl_divergence = torch.log(torch.tensor(2))
    soft_tukey_depths = []
    for i in range(n):
        soft_tukey_depths.append(soft_tukey_depth(X[i], X, z_params[i]).divide(n))
    for x in np.arange(0, 0.5, 0.005):
        val = torch.tensor(0)
        for i in range(n):
            val = val.add(torch.exp(torch.square(soft_tukey_depths[i] - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))))
        # print(val.item())
        kl_divergence = kl_divergence.subtract(torch.multiply(torch.tensor(0.01), torch.log(val.divide(n))))
    return kl_divergence


def get_kl_divergence_of_kde(X, z_params, target_dist, kernel_bandwidth):
    n = X.size(dim=0)
    kl_divergence = torch.tensor(0)
    soft_tukey_depths = []
    for i in range(n):
        soft_tukey_depths.append(soft_tukey_depth(X[i], X, z_params[i]).divide(n))
    for x in np.arange(0, 0.5, 0.01):
        _sum = torch.tensor(0)
        for i in range(n):
            _sum = _sum.add(torch.exp(torch.square(soft_tukey_depths[i] - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))))

        target_val = target_dist(x) + 1e-2
        kl_divergence = kl_divergence.subtract(torch.tensor(0.01).multiply(torch.tensor(target_val)).multiply(torch.log(_sum.divide(torch.tensor(n * target_val)))))
    return kl_divergence


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
            get_kth_moment_soft_tukey_depth(X, z_params, i + 1).subtract(get_kth_moment_uniform_distribution(i + 1, 0.5)).divide(get_kth_moment_uniform_distribution(i + 1, 0.5))))
    return moment_loss


def get_inverse_sum_soft_tukey_depth(X, z_params):
    n = X.size(dim=0)
    inverse_sum_loss = torch.tensor(0)
    for i in range(n):
        inverse_sum_loss = inverse_sum_loss.add(torch.divide(torch.tensor(n), soft_tukey_depth(X[i], X, z_params[i])))
    return torch.divide(inverse_sum_loss, torch.tensor(n))


def draw_histogram(X, X_, z_params, bins=200):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i], X_, z_params[i]).detach() / X_.size(dim=0)
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths.detach(), bins=bins)
    tukey_depth_histogram.show()

    kde_0 = sp.stats.gaussian_kde(soft_tukey_depths, bw_method=KERNEL_BANDWIDTH)
    x = np.arange(0, 0.5, 1e-5)
    y0 = kde_0(x)
    kde_fig = plt.figure()
    plt.plot(x, y0)
    kde_fig.show()


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


def weibull(_lambda, k):
    def f(x):
        return k / _lambda * ((x / _lambda) ** (k-1)) * np.exp(-((x / _lambda) ** k))
    return f


def uniform():
    def f(x):
        return 2
    return f


for NOMINAL_CLASS in range(1, 2):
    train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=True, device=device), list(range(DATA_SIZE)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)

    test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=False, device=device), list(range(TEST_NOMINAL_SIZE)))
    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=TEST_NOMINAL_SIZE, shuffle=True)

    test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=NOMINAL_CLASS, train=False, device=device), list(range(TEST_ANOMALOUS_SIZE)))
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=TEST_ANOMALOUS_SIZE, shuffle=True)

    encoder = MNIST_AE_Encoder().to(device)
    encoder.train()

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=3e-3)

    # z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
    # z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
    z_params = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
    optimizer_z = torch.optim.SGD(z_params, lr=3e-2)


    for i in range(NUM_EPOCHS):
        print(f'Epoch {i+1}')
        n = len(train_data)

        for step, X in enumerate(train_dataloader):
            X = X.to(device)
            Y = encoder(X)

            for j in range(n):
                for k in range(STD_ITERATIONS):
                    optimizer_z.zero_grad()
                    _soft_tukey_depth = soft_tukey_depth(Y[j].detach(), Y.detach(), z_params[j])
                    _soft_tukey_depth.backward()
                    optimizer_z.step()
                    # print(j, z_params[j])

            optimizer_encoder.zero_grad()

            # var = get_variance_soft_tukey_depth(Y, z_params)
            # print(f'Variance: {var.item()}')
            mean_norm = torch.linalg.norm(Y, dim=1).mean()
            print(f'Mean norm: {mean_norm.item()}')
            print(f'Mean point value: {Y.mean(dim=0).sum()}')
            # ((0 * -var).add(1e+4 * (torch.square(torch.linalg.norm(Y, dim=1).sum().subtract(DATA_SIZE)))).add(1e+3 * torch.square(Y.sum(dim=0)).sum())).backward()
            # (-var).backward()

            # moment_loss = get_moment_loss(Y, z_params, 3)
            # print(f'Moment loss: {moment_loss.item()}')
            # moment_loss.backward()

            kl_divergence = get_kl_divergence_of_kde(Y, z_params, uniform(), KERNEL_BANDWIDTH)
            print(f'KL divergence: {kl_divergence.item()}')
            # covariance_loss = torch.norm(torch.cov(torch.transpose(Y, 0, 1)) - torch.eye(ENCODING_DIM, device=device))
            # print(f'Covariance loss: {covariance_loss.item()}')
            (kl_divergence + torch.square(mean_norm.subtract(torch.tensor(1)))).backward()

            # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(Y, z_params)
            # (inverse_sum_loss).backward()

            optimizer_encoder.step()

            if i % 1 == 0:
                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y, z_params)
                draw_histogram(Y, Y, z_params, bins=HISTOGRAM_BINS)
            if i == NUM_EPOCHS - 1:
                draw_histogram(Y, Y, z_params, bins=HISTOGRAM_BINS)

                Y = encoder(X)

                for step2, X_test_nominal in enumerate(test_dataloader_nominal):
                    soft_tukey_depths = []

                    X_test_nominal = X_test_nominal.to(device)
                    Y_test_nominal = encoder(X_test_nominal)
                    z_test_nominal = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(test_data_nominal))]
                    optimizer_z_test_nominal = torch.optim.SGD(z_test_nominal, lr=3e-2)

                    for j in range(len(test_data_nominal)):
                        for k in range(10):
                            optimizer_z_test_nominal.zero_grad()
                            _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[j].detach(), Y.detach(), z_test_nominal[j])
                            _soft_tukey_depth.backward()
                            optimizer_z_test_nominal.step()
                        _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[j].detach(), Y.detach(), z_test_nominal[j])
                        print(_soft_tukey_depth.item() / len(train_data))
                        soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))

                    if ENCODING_DIM == 2:
                        draw_scatter_plot(Y_test_nominal, z_test_nominal)
                    draw_histogram(Y_test_nominal, Y, z_test_nominal, bins=HISTOGRAM_BINS)

                    writer = csv.writer(open(
                        f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Nominal_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}.csv',
                        'w'))
                    writer.writerow(soft_tukey_depths)

                for step2, X_test_anomalous in enumerate(test_dataloader_anomalous):
                    soft_tukey_depths = []

                    X_test_anomalous = X_test_anomalous.to(device)
                    Y_test_anomalous = encoder(X_test_anomalous)
                    z_test_anomalous = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(test_data_anomalous))]
                    optimizer_z_test_anomalous = torch.optim.SGD(z_test_anomalous, lr=3e-2)

                    for j in range(len(test_data_anomalous)):
                        for k in range(10):
                            optimizer_z_test_anomalous.zero_grad()
                            _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[j].detach(), Y.detach(), z_test_anomalous[j])
                            _soft_tukey_depth.backward()
                            optimizer_z_test_anomalous.step()
                        _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[j].detach(), Y.detach(), z_test_anomalous[j])
                        print(_soft_tukey_depth.item() / len(train_data))
                        soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))

                    if ENCODING_DIM == 2:
                        draw_scatter_plot(Y_test_anomalous, z_test_anomalous)
                    draw_histogram(Y_test_anomalous, Y, z_test_anomalous, bins=HISTOGRAM_BINS)

                    writer = csv.writer(open(
                        f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Anomalous_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}.csv',
                        'w'))
                    writer.writerow(soft_tukey_depths)


    torch.save(encoder.state_dict(), f'./snapshots/{DATASET_NAME}_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}')

    # for i in range(X.size(dim=0)):
    #     print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

    # print('Variance')
    # print(get_variance_soft_tukey_depth(X, z_params))
    #
    # print('Inverse Sum')
    # print(get_inverse_sum_soft_tukey_depth(X, z_params))
