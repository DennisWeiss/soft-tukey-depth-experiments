import csv

import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset, NominalCIFAR10ImageDataset, \
    AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, AnomalousCIFAR10AutoencoderDataset, \
    NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset, NominalMNISTAutoencoderCachedDataset, \
    AnomalousMNISTAutoencoderCachedDataset
from models.CIFAR10_AE_Encoder import CIFAR10_AE_Encoder
from models.CIFAR10_AE_Encoder_V2 import CIFAR10_AE_Encoder_V2
from models.CIFAR10_Encoder_V4 import CIFAR10_Encoder_V4
from models.CIFAR10_Encoder_V5 import CIFAR10_Encoder_V5
from models.CIFAR10_Encoder_V6 import CIFAR10_Encoder_V6
from models.MNIST_AE_Encoder import MNIST_AE_Encoder
from models.MNIST_AE_Encoder_V2 import MNIST_AE_Encoder_V2
from models.MNIST_Encoder import MNIST_Encoder
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
RESULT_NAME_DESC = 'kldiv_8x_30epochs'
DATA_SIZE = 2000
TEST_NOMINAL_SIZE = 1000
TEST_ANOMALOUS_SIZE = 1000


USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE = 2000
ENCODER_LEARNING_RATE = 1e-3
HALFSPACE_OPTIMIZER_LEARNING_RATE = 1e+3
KERNEL_BANDWIDTH = 0.05
SOFT_TUKEY_DEPTH_TEMP = 0.2
ENCODING_DIM = 64
HISTOGRAM_BINS = 50
NUM_EPOCHS = 30
STD_ITERATIONS = 10
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


def soft_tukey_depth_v2(X_, X, Z, temp):
    X_new = X.repeat(X_.size(dim=0), 1, 1)
    X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
    dot_products_normalized = dot_products.transpose(0, 1).divide(temp * Z.norm(dim=1))
    return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))


def get_kl_divergence(soft_tukey_depths, f, kernel_bandwidth, epsilon=0.0):
    DELTA = 0.005
    kl_divergence = torch.tensor(0)
    for x in torch.arange(0, 0.5, DELTA):
        val = torch.exp(torch.square(soft_tukey_depths - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))).mean()
        f_val = f(x)
        kl_divergence = kl_divergence.subtract(torch.multiply(torch.tensor(f_val * DELTA), torch.log(val.divide(f_val + epsilon))))
    return kl_divergence


def get_kth_moment_uniform_distribution(k, b):
    return torch.pow(torch.tensor(b), torch.tensor(k+1)).divide(torch.tensor(b*(k+1)))


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
    for NOMINAL_CLASS in range(0, 6):
        train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=True), list(range(DATA_SIZE)))
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
        train_dataloader_full_data = torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)

        test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_NOMINAL_SIZE)))
        test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=16, shuffle=True)

        test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_ANOMALOUS_SIZE)))
        test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=16, shuffle=True)

        encoder = CIFAR10_AE_Encoder().to(device)
        encoder.train()

        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LEARNING_RATE)

        # z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
        # z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
        z_params = torch.nn.Parameter(torch.rand(len(train_data), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
        optimizer_z = torch.optim.SGD([z_params], lr=HALFSPACE_OPTIMIZER_LEARNING_RATE)

        for i in range(NUM_EPOCHS):
            print(f'Epoch {i+1}')
            n = len(train_data)

            for step, X in enumerate(train_dataloader):
                X = X.to(device)
                Y = encoder(X)
                Y_detached = Y.detach()

                for k in range(STD_ITERATIONS):
                    optimizer_z.zero_grad()

                    for step2, X_train in enumerate(train_dataloader_full_data):
                        X_train = X_train.to(device)
                        Y_train = encoder(X_train)

                        _soft_tukey_depth = soft_tukey_depth_v2(Y_train.detach(), Y_detached, z_params, SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.sum().backward()
                        optimizer_z.step()
                    # print(j, z_params[j])

                optimizer_encoder.zero_grad()

                for step2, X_train in enumerate(train_dataloader_full_data):
                    X_train = X_train.to(device)
                    Y_train = encoder(X_train)

                    tds = soft_tukey_depth_v2(Y_train, Y, z_params.detach(), SOFT_TUKEY_DEPTH_TEMP)
                    var = torch.var(tds)
                    print(f'Variance: {var.item()}')
                    mean_td = tds.mean()
                    print(f'Mean: {mean_td.item()}')
                    # (-var).backward()
                    # (-mean_td).backward()
                    kl_div = get_kl_divergence(tds, lambda x: 8*x, 0.05, 1e-3)
                    print(f'KL divergence: {kl_div.item()}')
                    kl_div.backward()
                    optimizer_encoder.step()

                    draw_histogram(Y_train, Y, z_params, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                # for l in range(1):
                #     X_rand = torch.rand(BATCH_SIZE, 512, device=device).multiply(1).subtract(0.5)
                #     Y_rand = encoder(X_rand)
                #
                #     z_rand = torch.nn.Parameter(torch.rand(BATCH_SIZE, ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
                #     z_rand_optim = torch.optim.SGD([z_rand], lr=HALFSPACE_OPTIMIZER_LEARNING_RATE)
                #
                #     for k in range(2*STD_ITERATIONS):
                #         z_rand_optim.zero_grad()
                #         for step2, X_train in enumerate(train_dataloader_full_data):
                #             X_train = X_train.to(device)
                #             Y_train = encoder(X_train)
                #
                #             _soft_tukey_depth = soft_tukey_depth_v2(Y_rand.detach(), Y_train.detach(), z_rand, SOFT_TUKEY_DEPTH_TEMP)
                #             _soft_tukey_depth.sum().backward()
                #             z_rand_optim.step()
                #
                #     for step2, X_train in enumerate(train_dataloader_full_data):
                #         X_train = X_train.to(device)
                #         Y_train = encoder(X_train)
                #
                #         tds = soft_tukey_depth_v2(Y_rand, Y_train.detach(), z_rand, SOFT_TUKEY_DEPTH_TEMP)
                #         mean_td = tds.mean()
                #         print(f'Random mean: {mean_td.item()}')
                #         (mean_td).backward()
                #         optimizer_encoder.step()

                # print(f'Total norm: {torch.linalg.norm(Y, dim=1).sum().item()}')
                # print(f'Total point value: {Y.sum(dim=0).sum()}')
                # ((0 * -var).add(1e+4 * (torch.square(torch.linalg.norm(Y, dim=1).sum().subtract(DATA_SIZE)))).add(1e+3 * torch.square(Y.sum(dim=0)).sum())).backward()

                # avg_soft_tukey_depth = torch.tensor(0)
                #
                # for j in range(Y.size(dim=0)):
                #     avg_soft_tukey_depth = avg_soft_tukey_depth.add(
                #         soft_tukey_depth(Y[j], Y, z_params[step * BATCH_SIZE + j], SOFT_TUKEY_DEPTH_TEMP).divide(Y.size(dim=0) ** 2))
                #
                # Y_mean = Y.mean(dim=0)
                # Y_centered = Y.subtract(Y_mean)
                #
                # avg_latent_norm = torch.norm(Y_centered, dim=1).mean()
                # norm_1_diff = torch.square(avg_latent_norm - torch.tensor(1))
                #
                # # outlyingness_loss = torch.relu(torch.norm(Y_centered, dim=1).subtract(torch.tensor(1))).mean()
                #
                # loss = -avg_soft_tukey_depth + 3e-1 * norm_1_diff
                # loss.backward()
                # print(f'Avg soft Tukey depth: {avg_soft_tukey_depth.item()}')
                # print(f'Avg latent norm: {avg_latent_norm.item()}')
                # print(f'Outlyingness loss: {outlyingness_loss.item()}')
                # print(f'Loss: {loss.item()}')

                # moment_loss = get_moment_loss(Y, z_params, 3)
                # print(f'Moment loss: {moment_loss.item()}')
                # moment_loss.backward()

                # kl_divergence = get_kl_divergence(Y, z_params, 0.1)
                # print(f'KL divergence: {kl_divergence.item()}')
                # kl_divergence.backward()

                # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(Y, z_params)
                # (inverse_sum_loss).backward()

                if i % 1 == 0:
                    if ENCODING_DIM == 2:
                        draw_scatter_plot(Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)])

        for step, X in enumerate(train_dataloader_full_data):
            X = X.to(device)
            Y = encoder(X)

            # soft_tukey_depths = []
            # for j in range(Y.size(dim=0)):
            #     soft_tukey_depths.append(soft_tukey_depth(Y[j], Y, z_params[j]).item() / Y.size(dim=0))
            #
            # draw_histogram_tukey_depth(soft_tukey_depths, bins=HISTOGRAM_BINS)
            svd_dim = draw_svd_plot(Y, 16, 0.01)
            print(f'SVD dimensionality: {svd_dim}')


            soft_tukey_depths = []
            for step2, X_test_nominal in enumerate(test_dataloader_nominal):
                X_test_nominal = X_test_nominal.to(device)
                Y_test_nominal = encoder(X_test_nominal)
                z_test_nominal = torch.nn.Parameter(torch.rand(X_test_nominal.size(dim=0), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
                optimizer_z_test_nominal = torch.optim.SGD([z_test_nominal], lr=3e+1)

                for k in range(2*STD_ITERATIONS):
                    optimizer_z_test_nominal.zero_grad()
                    _soft_tukey_depth = soft_tukey_depth_v2(Y_test_nominal.detach(), Y.detach(), z_test_nominal, SOFT_TUKEY_DEPTH_TEMP)
                    _soft_tukey_depth.sum().backward()
                    optimizer_z_test_nominal.step()
                _soft_tukey_depth = soft_tukey_depth_v2(Y_test_nominal.detach(), Y.detach(), z_test_nominal, SOFT_TUKEY_DEPTH_TEMP)
                soft_tukey_depths += _soft_tukey_depth.tolist()

                # if ENCODING_DIM == 2:
                #     draw_scatter_plot(Y_test_nominal, z_test_nominal)
                # draw_histogram(Y_test_nominal, Y, z_test_nominal, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

            writer = csv.writer(open(
                f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Nominal_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}_run{run}.csv',
                'w'))
            writer.writerow(soft_tukey_depths)

            soft_tukey_depths = []

            for step2, X_test_anomalous in enumerate(test_dataloader_anomalous):
                X_test_anomalous = X_test_anomalous.to(device)
                Y_test_anomalous = encoder(X_test_anomalous)
                z_test_anomalous = torch.nn.Parameter(torch.rand(X_test_anomalous.size(dim=0), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
                optimizer_z_test_anomalous = torch.optim.SGD([z_test_anomalous], lr=3e+1)

                for k in range(2*STD_ITERATIONS):
                    optimizer_z_test_anomalous.zero_grad()
                    _soft_tukey_depth = soft_tukey_depth_v2(Y_test_anomalous.detach(), Y.detach(), z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP)
                    _soft_tukey_depth.sum().backward()
                    optimizer_z_test_anomalous.step()
                _soft_tukey_depth = soft_tukey_depth_v2(Y_test_anomalous.detach(), Y.detach(), z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP)
                soft_tukey_depths += _soft_tukey_depth.tolist()

                # if ENCODING_DIM == 2:
                #     draw_scatter_plot(Y_test_anomalous, z_test_anomalous)
                # draw_histogram(Y_test_anomalous, Y, z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

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
