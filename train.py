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


DATASET_NAME = 'CIFAR10_Autoencoder'
NOMINAL_DATASET = NominalCIFAR10AutoencoderDataset
ANOMALOUS_DATASET = AnomalousCIFAR10AutoencoderDataset
RESULT_NAME_DESC = 'max_1e-3_lambda3e-1_3epochs_4096'
DATA_SIZE = 4096
TEST_NOMINAL_SIZE = 1000
TEST_ANOMALOUS_SIZE = 1000


USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE = 256
ENCODER_LEARNING_RATE = 1e-3
HALFSPACE_OPTIMIZER_LEARNING_RATE = 3e-2
KERNEL_BANDWIDTH = 0.05
SOFT_TUKEY_DEPTH_TEMP = 0.1
ENCODING_DIM = 32
HISTOGRAM_BINS = 50
NUM_EPOCHS = 3
STD_ITERATIONS = 6
RUNS = 3

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


def soft_tukey_depth(x, x_, z, temp):
    matmul = torch.outer(torch.ones(x_.size(dim=0), device=device), x)
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / temp), torch.divide(
        torch.matmul(torch.subtract(x_, matmul), z),
        torch.norm(z)))))


def get_mean_soft_tukey_depth(X, X_, z_params, temp):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i], X_, z_params[i], temp).divide(X_.size(dim=0)))
    _mean = mean.divide(X.size(dim=0))
    return _mean


def get_variance_soft_tukey_depth_with_mean(X, z_params, mean, temp):
    n = X.size(dim=0)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i], X, z_params[i], temp).divide(n).subtract(mean)))
    return var.divide(n - 1)


def get_variance_soft_tukey_depth(X, z_params, temp):
    return get_variance_soft_tukey_depth_with_mean(X, z_params, get_mean_soft_tukey_depth(X, X, z_params, temp), temp)


def get_variance_soft_tukey_depth_batches(X, train_dataloader, encoder, z_params, device, temp):
    _mean = torch.tensor(0)
    for step, X_train in enumerate(train_dataloader):
        X_train = encoder(X_train.to(device))
        _mean = _mean.add(get_mean_soft_tukey_depth(X_train, X, z_params[(step * BATCH_SIZE):((step + 1) * BATCH_SIZE)], temp).multiply(X_train.size(dim=0)).divide(DATA_SIZE))
    _mean = _mean.detach()
    print(f'Mean: {_mean.item()}')
    var = torch.tensor(0)
    for step, X_train in enumerate(train_dataloader):
        X_train = encoder(X_train.to(device))
        for i in range(X_train.size(dim=0)):
            var = var.add(torch.square(soft_tukey_depth(X_train[i], X, z_params[step * BATCH_SIZE + i], temp).divide(X.size(dim=0)).subtract(_mean)))
    return var.divide(DATA_SIZE - 1)


def get_kl_divergence(X, z_params, kernel_bandwidth, temp):
    n = X.size(dim=0)
    kl_divergence = torch.log(torch.tensor(2))
    soft_tukey_depths = []
    for i in range(n):
        soft_tukey_depths.append(soft_tukey_depth(X[i], X, z_params[i], temp).divide(n))
    for x in np.arange(0, 0.5, 0.005):
        val = torch.tensor(0)
        for i in range(n):
            val = val.add(torch.exp(torch.square(soft_tukey_depths[i] - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))))
        # print(val.item())
        kl_divergence = kl_divergence.subtract(torch.multiply(torch.tensor(0.01), torch.log(val.divide(n))))
    return kl_divergence


def get_kde_norm_soft_tukey_depth(X, z_params, bandwidth, temp):
    n = X.size(dim=0)
    Y = torch.zeros(n).to(device)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i], X, z_params[i], temp)
    kde_norm = torch.tensor(0).to(device)
    for i in range(n):
        for j in range(n):
            diff = Y[i] - Y[j]
            kde_norm = kde_norm.add(torch.exp(-torch.square(diff).divide(torch.tensor(4 * bandwidth ** 2))))
    return kde_norm.divide(torch.square(torch.tensor(n)))


def get_kth_moment_soft_tukey_depth(X, z_params, k, temp):
    n = X.size(dim=0)
    Y = torch.zeros(n)
    for i in range(n):
        Y[i] = soft_tukey_depth(X[i], X, z_params[i], temp) / n
    return torch.pow(Y, k).mean()


def get_kth_moment_uniform_distribution(k, b):
    return torch.pow(torch.tensor(b), torch.tensor(k+1)).divide(torch.tensor(b*(k+1)))


def get_moment_loss(X, z_params, k, temp):
    moment_loss = torch.tensor(0)
    for i in range(k):
        moment_loss = moment_loss.add(torch.square(
            get_kth_moment_soft_tukey_depth(X, z_params, i + 1, temp).subtract(get_kth_moment_uniform_distribution(i + 1, 0.5)).divide(get_kth_moment_uniform_distribution(i + 1, 0.5))))
    return moment_loss


def get_inverse_sum_soft_tukey_depth(X, z_params, temp):
    n = X.size(dim=0)
    inverse_sum_loss = torch.tensor(0)
    for i in range(n):
        inverse_sum_loss = inverse_sum_loss.add(torch.divide(torch.tensor(n), soft_tukey_depth(X[i], X, z_params[i], temp)))
    return torch.divide(inverse_sum_loss, torch.tensor(n))


def draw_histogram(X, X_, z_params, temp, bins=200):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i], X_, z_params[i], temp) / X_.size(dim=0)
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths.detach(), bins=bins)
    tukey_depth_histogram.show()


def draw_histogram_tukey_depth(soft_tukey_depths, bins=200):
    tukey_depth_histogram = plt.figure()
    plt.hist(np.asarray(soft_tukey_depths), bins=bins)
    tukey_depth_histogram.show()


def get_svd_dimensionality(singular_values, alpha=0.01):
    _sum = 0
    for i in range(len(singular_values)):
        _sum += singular_values[i]
    current_sum = 0
    for i in range(len(singular_values)):
        current_sum += singular_values[i]
        if current_sum >= (1 - alpha) * _sum:
            return i+1


def draw_svd_plot(X, n, alpha=0.01):
    singular_values = torch.linalg.svdvals(X)
    svd_plot = plt.figure()
    plt.bar(np.asarray(list(range(n))), singular_values.detach().cpu().numpy()[0:n])
    svd_plot.show()
    return get_svd_dimensionality(singular_values, alpha)


def draw_scatter_plot(X, z_params):
    X_np = X.Y_mean().cpu().numpy()
    z_normalized = np.zeros(X.size())
    for i in range(len(z_params)):
        z_normalized[i] = z_params[i].Y_mean().cpu() / z_params[i].Y_mean().cpu().norm()

    X_scatter_plot = plt.figure()
    plt.scatter(
        np.append(X_np[:, 0], X_np[:, 0] + z_normalized[:, 0]),
        np.append(X_np[:, 1], X_np[:, 1] + z_normalized[:, 1]),
        c=['#0000ff' for i in range(X.size(dim=0))] + ['#ff0000' for i in range(X.size(dim=0))]
    )
    X_scatter_plot.show()


for run in range(RUNS):
    for NOMINAL_CLASS in range(0, 10):
        train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=True), list(range(DATA_SIZE)))
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

        test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_NOMINAL_SIZE)))
        test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=TEST_NOMINAL_SIZE, shuffle=True)

        test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_ANOMALOUS_SIZE)))
        test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=TEST_ANOMALOUS_SIZE, shuffle=True)

        encoder = CIFAR10_AE_Encoder().to(device)
        encoder.train()

        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LEARNING_RATE)

        # z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
        # z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
        z_params = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
        optimizer_z = torch.optim.SGD(z_params, lr=HALFSPACE_OPTIMIZER_LEARNING_RATE)


        for i in range(NUM_EPOCHS):
            print(f'Epoch {i+1}')
            n = len(train_data)

            for step, X in enumerate(train_dataloader):
                X = X.to(device)
                Y = encoder(X)

                for k in range(STD_ITERATIONS):
                    optimizer_encoder.zero_grad()
                    optimizer_z.zero_grad()
                    _soft_tukey_depth = torch.tensor(0)
                    for step2, X_train in enumerate(train_dataloader):
                        X_train = X_train.to(device)
                        Y_train = encoder(X_train)
                        for j in range(Y_train.size(dim=0)):
                            _soft_tukey_depth = _soft_tukey_depth.add(soft_tukey_depth(Y_train[j].detach(), Y.detach(), z_params[step2 * BATCH_SIZE + j], SOFT_TUKEY_DEPTH_TEMP))
                    _soft_tukey_depth.backward()
                    optimizer_z.step()
                    # print(j, z_params[j])

                optimizer_encoder.zero_grad()
                optimizer_z.zero_grad()

                # var = get_variance_soft_tukey_depth_batches(Y, train_dataloader, encoder, z_params, device, SOFT_TUKEY_DEPTH_TEMP)
                # print(f'Variance: {var.item()}')
                # print(f'Total norm: {torch.linalg.norm(Y, dim=1).sum().item()}')
                # print(f'Total point value: {Y.sum(dim=0).sum()}')
                # ((0 * -var).add(1e+4 * (torch.square(torch.linalg.norm(Y, dim=1).sum().subtract(DATA_SIZE)))).add(1e+3 * torch.square(Y.sum(dim=0)).sum())).backward()
                # (-var).backward()

                avg_soft_tukey_depth = torch.tensor(0)

                for j in range(Y.size(dim=0)):
                    avg_soft_tukey_depth = avg_soft_tukey_depth.add(
                        soft_tukey_depth(Y[j], Y, z_params[step * BATCH_SIZE + j], SOFT_TUKEY_DEPTH_TEMP).divide(Y.size(dim=0) ** 2))

                Y_mean = Y.mean(dim=0)
                Y_centered = Y.subtract(Y_mean)

                avg_latent_norm = torch.norm(Y_centered, dim=1).mean()
                norm_1_diff = torch.square(avg_latent_norm - torch.tensor(1))

                # outlyingness_loss = torch.relu(torch.norm(Y_centered, dim=1).subtract(torch.tensor(1))).mean()

                loss = -avg_soft_tukey_depth + 3e-1 * norm_1_diff
                loss.backward()
                print(f'Avg soft Tukey depth: {avg_soft_tukey_depth.item()}')
                print(f'Avg latent norm: {avg_latent_norm.item()}')
                # print(f'Outlyingness loss: {outlyingness_loss.item()}')
                print(f'Loss: {loss.item()}')

                # moment_loss = get_moment_loss(Y, z_params, 3)
                # print(f'Moment loss: {moment_loss.item()}')
                # moment_loss.backward()

                # kl_divergence = get_kl_divergence(Y, z_params, 0.1)
                # print(f'KL divergence: {kl_divergence.item()}')
                # kl_divergence.backward()

                # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(Y, z_params)
                # (inverse_sum_loss).backward()

                optimizer_encoder.step()


                if i % 1 == 0:
                    if ENCODING_DIM == 2:
                        draw_scatter_plot(Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)])
                    draw_histogram(Y, Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)], SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

        for step, X in enumerate(torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)):
            X = X.to(device)
            Y = encoder(X)

            # soft_tukey_depths = []
            # for j in range(Y.size(dim=0)):
            #     soft_tukey_depths.append(soft_tukey_depth(Y[j], Y, z_params[j]).item() / Y.size(dim=0))
            #
            # draw_histogram_tukey_depth(soft_tukey_depths, bins=HISTOGRAM_BINS)
            svd_dim = draw_svd_plot(Y, 32, 0.01)
            print(f'SVD dimensionality: {svd_dim}')


            for step2, X_test_nominal in enumerate(test_dataloader_nominal):
                soft_tukey_depths = []

                X_test_nominal = X_test_nominal.to(device)
                Y_test_nominal = encoder(X_test_nominal)
                z_test_nominal = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(test_data_nominal))]
                optimizer_z_test_nominal = torch.optim.SGD(z_test_nominal, lr=3e-2)

                for j in range(len(test_data_nominal)):
                    for k in range(10):
                        optimizer_z_test_nominal.zero_grad()
                        _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[j].detach(), Y.detach(), z_test_nominal[j], SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.backward()
                        optimizer_z_test_nominal.step()
                    _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[j].detach(), Y.detach(), z_test_nominal[j], SOFT_TUKEY_DEPTH_TEMP)
                    print(_soft_tukey_depth.item() / len(train_data))
                    soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))

                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y_test_nominal, z_test_nominal)
                draw_histogram(Y_test_nominal, Y, z_test_nominal, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                writer = csv.writer(open(
                    f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Nominal_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}_run{run}.csv',
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
                        _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[j].detach(), Y.detach(), z_test_anomalous[j], SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.backward()
                        optimizer_z_test_anomalous.step()
                    _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[j].detach(), Y.detach(), z_test_anomalous[j], SOFT_TUKEY_DEPTH_TEMP)
                    print(_soft_tukey_depth.item() / len(train_data))
                    soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))

                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y_test_anomalous, z_test_anomalous)
                draw_histogram(Y_test_anomalous, Y, z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                writer = csv.writer(open(
                    f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Anomalous_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}_run{run}.csv',
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
