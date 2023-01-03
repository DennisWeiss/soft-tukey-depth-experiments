import csv

import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset, NominalCIFAR10ImageDataset, \
    AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, AnomalousCIFAR10AutoencoderDataset, \
    NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset, NominalMNISTAutoencoderCachedDataset, \
    AnomalousMNISTAutoencoderCachedDataset, NominalMVTecCapsuleImageDataset, AnomalousMVTecCapsuleImageDataset
from models.AE_CIFAR10_V6 import AE_CIFAR10_V6
from models.AE_CIFAR10_V7 import AE_CIFAR10_V7
from models.AE_MNIST_V3 import AE_MNIST_V3
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
import utils

from models.MVTecCapsule_Encoder import MVTecCapsule_Encoder
from models.Wasserstein_Network import Wasserstein_Network


DATASET_NAME = 'CIFAR10_Autoencoder'
NOMINAL_DATASET = NominalCIFAR10AutoencoderDataset
ANOMALOUS_DATASET = AnomalousCIFAR10AutoencoderDataset
RESULT_NAME_DESC = 'kldiv_8x_temp0.2_dim64'
DATA_SIZE = 4000
TEST_NOMINAL_SIZE = 800
TEST_ANOMALOUS_SIZE = 800


USE_CUDA_IF_AVAILABLE = True
BATCH_SIZE = 800
ENCODER_LEARNING_RATE = 1e-3
HALFSPACE_OPTIMIZER_LEARNING_RATE = 1e+3
WASSERSTEIN_NETWORK_LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0
KERNEL_BANDWIDTH = 0.05
SOFT_TUKEY_DEPTH_TEMP = 0.2
ENCODING_DIM = 64
TARGET_DISTRIBUTION = lambda x: 8*x
HISTOGRAM_BINS = 50
NUM_EPOCHS = 30
STD_ITERATIONS = 20
STD_COMPUTATIONS = 20
WASSERSTEIN_ITERATIONS = 10
RUNS = 1


torch.set_printoptions(precision=4, sci_mode=False)

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


def get_wasserstein_loss(wasserstein_network, soft_tukey_depths):
    soft_tukey_depths_reshaped = soft_tukey_depths.detach().reshape(-1, 1)
    return wasserstein_network(soft_tukey_depths_reshaped).mean() - wasserstein_network(0.5 * torch.rand_like(soft_tukey_depths_reshaped)).mean()


def clip_weights(model, val):
    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                param.clamp_(-val, val)


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
    X_np = X.detach().cpu().numpy()
    z_normalized = z_params.detach().cpu().numpy() / torch.norm(z_params.detach(), dim=1).cpu().numpy().reshape(-1, 1)

    X_scatter_plot = plt.figure()
    plt.scatter(
        np.append(X_np[:, 0], X_np[:, 0] + z_normalized[:, 0]),
        np.append(X_np[:, 1], X_np[:, 1] + z_normalized[:, 1]),
        c=['#0000ff' for i in range(X.size(dim=0))] + ['#ff0000' for i in range(X.size(dim=0))]
    )
    X_scatter_plot.show()


for run in range(RUNS):
    for NOMINAL_CLASS in range(3, 4):
        train_data = torch.utils.data.Subset(NOMINAL_DATASET(train=True, nominal_class=NOMINAL_CLASS), list(range(DATA_SIZE)))
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
        train_dataloader_full_data = torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)

        test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(train=False, nominal_class=NOMINAL_CLASS), list(range(TEST_NOMINAL_SIZE)))
        test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=16, shuffle=True)

        test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(train=False, nominal_class=NOMINAL_CLASS), list(range(TEST_ANOMALOUS_SIZE)))
        test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=16, shuffle=True)

        encoder = CIFAR10_AE_Encoder().to(device)
        # encoder.load_weights(f'./snapshots/AE_V7_CIFAR10_{NOMINAL_CLASS}')
        # encoder.load_state_dict(torch.load(f'./snapshots/AE_V3_MNIST_{NOMINAL_CLASS}'))
        encoder.train()

        # autoencoder = AE_CIFAR10_V6().to(device)
        # autoencoder.load_state_dict(torch.load(f'./snapshots/AE_V6_CIFAR10_{NOMINAL_CLASS}'))
        #
        # encoder.load_weights_from_pretrained_autoencoder(autoencoder)

        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # wasserstein_network = Wasserstein_Network().to(device)
        # wasserstein_network.train()
        #
        # optimizer_wasserstein_network = torch.optim.Adam(wasserstein_network.parameters(), lr=WASSERSTEIN_NETWORK_LEARNING_RATE)

        # z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
        # z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
        best_z = torch.rand(len(train_data), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)).detach()

        for i in range(NUM_EPOCHS):
            print(f'Epoch {i+1}')
            n = len(train_data)

            for step, X in enumerate(train_dataloader):
                X = X.to(device)
                Y = encoder(X)
                Y_detached = Y.detach()

                for step2, X_train in enumerate(train_dataloader_full_data):
                    X_train = X_train.to(device)
                    Y_train = encoder(X_train)

                    tukey_depths = soft_tukey_depth_v2(Y_detached, Y_train.detach(), best_z[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)], SOFT_TUKEY_DEPTH_TEMP)

                    for k in range(STD_COMPUTATIONS):
                        z_params = torch.nn.Parameter(2 * torch.rand(BATCH_SIZE, ENCODING_DIM, device=device) - 1)
                        # z_params = torch.nn.Parameter(best_z)
                        optimizer_z = torch.optim.SGD([z_params], lr=HALFSPACE_OPTIMIZER_LEARNING_RATE)
                        for l in range(STD_ITERATIONS):
                            optimizer_z.zero_grad()

                            _soft_tukey_depth = soft_tukey_depth_v2(Y_detached, Y_train.detach(), z_params, SOFT_TUKEY_DEPTH_TEMP)
                            _soft_tukey_depth.sum().backward()
                            optimizer_z.step()

                        _soft_tukey_depth = soft_tukey_depth_v2(Y_detached, Y_train.detach(), z_params, SOFT_TUKEY_DEPTH_TEMP)

                        for j in range(tukey_depths.size(dim=0)):
                            if _soft_tukey_depth[j] < tukey_depths[j]:
                                tukey_depths[j] = _soft_tukey_depth[j].detach()
                                best_z[step*BATCH_SIZE+j] = z_params[j].detach()

            for step, X in enumerate(train_dataloader):
                X = X.to(device)
                Y = encoder(X)
                Y_detached = Y.detach()

                for step2, X_train in enumerate(train_dataloader_full_data):
                    X_train = X_train.to(device)
                    Y_train = encoder(X_train)

                    tds = soft_tukey_depth_v2(Y_train, Y, best_z, SOFT_TUKEY_DEPTH_TEMP)

                    # for j in range(WASSERSTEIN_ITERATIONS):
                    #     optimizer_wasserstein_network.zero_grad()
                    #     wasserstein_loss = get_wasserstein_loss(wasserstein_network, tds)
                    #     (-wasserstein_loss).backward()
                    #     optimizer_wasserstein_network.step()
                    #     clip_weights(wasserstein_network, 1)

                    optimizer_encoder.zero_grad()

                    var = torch.var(tds)
                    print(f'Variance: {var.item()}')
                    mean_td = tds.mean()
                    print(f'Mean: {mean_td.item()}')
                    # mean_td_loss = -mean_td - 1e-2 * torch.norm(Y - Y.mean(dim=0))
                    # mean_td_loss.backward()
                    # print(f'Mean TD loss: {mean_td_loss}')
                    # (-var).backward()
                    # (-mean_td).backward()
                    kl_div = get_kl_divergence(tds, TARGET_DISTRIBUTION, 0.05, 1e-3)
                    # wasserstein_loss = get_wasserstein_loss(wasserstein_network, tds)
                    print(f'KL divergence: {kl_div.item()}')
                    # print(f'Wasserstein loss: {wasserstein_loss.item()}')
                    kl_div.backward()
                    # wasserstein_loss.backward()
                    optimizer_encoder.step()

                    if i % 2 == 0:
                        draw_histogram(Y_train, Y, best_z, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                if i % 2 == 0:
                    if ENCODING_DIM == 2:
                        draw_scatter_plot(Y, best_z[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)])

            for step, X in enumerate(train_dataloader_full_data):
                X = X.to(device)
                Y = encoder(X)

                # soft_tukey_depths = []
                # for j in range(Y.size(dim=0)):
                #     soft_tukey_depths.append(soft_tukey_depth(Y[j], Y, z_params[j]).item() / Y.size(dim=0))
                #
                # draw_histogram_tukey_depth(soft_tukey_depths, bins=HISTOGRAM_BINS)
                # svd_dim = draw_svd_plot(Y, min(ENCODING_DIM, 16), 0.01)
                # print(f'SVD dimensionality: {svd_dim}')


                nominal_soft_tukey_depths = []
                for step2, X_test_nominal in enumerate(test_dataloader_nominal):
                    X_test_nominal = X_test_nominal.to(device)
                    Y_test_nominal = encoder(X_test_nominal)
                    z_test_nominal = torch.nn.Parameter(torch.rand(X_test_nominal.size(dim=0), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
                    optimizer_z_test_nominal = torch.optim.SGD([z_test_nominal], lr=3e+1)

                    for l in range(2 * STD_ITERATIONS):
                        optimizer_z_test_nominal.zero_grad()
                        _soft_tukey_depth = soft_tukey_depth_v2(Y_test_nominal.detach(), Y.detach(), z_test_nominal, SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.sum().backward()
                        optimizer_z_test_nominal.step()
                    _soft_tukey_depth = soft_tukey_depth_v2(Y_test_nominal.detach(), Y.detach(), z_test_nominal, SOFT_TUKEY_DEPTH_TEMP)
                    nominal_soft_tukey_depths += _soft_tukey_depth.tolist()

                    # if ENCODING_DIM == 2:
                    #     draw_scatter_plot(Y_test_nominal, z_test_nominal)
                    # draw_histogram(Y_test_nominal, Y, z_test_nominal, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                writer = csv.writer(open(
                    f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Nominal_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}_run{run}.csv',
                    'w'))
                writer.writerow(nominal_soft_tukey_depths)

                anomalous_soft_tukey_depths = []

                for step2, X_test_anomalous in enumerate(test_dataloader_anomalous):
                    X_test_anomalous = X_test_anomalous.to(device)
                    Y_test_anomalous = encoder(X_test_anomalous)
                    z_test_anomalous = torch.nn.Parameter(torch.rand(X_test_anomalous.size(dim=0), ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
                    optimizer_z_test_anomalous = torch.optim.SGD([z_test_anomalous], lr=3e+1)

                    for l in range(2 * STD_ITERATIONS):
                        optimizer_z_test_anomalous.zero_grad()
                        _soft_tukey_depth = soft_tukey_depth_v2(Y_test_anomalous.detach(), Y.detach(), z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.sum().backward()
                        optimizer_z_test_anomalous.step()
                    _soft_tukey_depth = soft_tukey_depth_v2(Y_test_anomalous.detach(), Y.detach(), z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP)
                    anomalous_soft_tukey_depths += _soft_tukey_depth.tolist()

                    # if ENCODING_DIM == 2:
                    #     draw_scatter_plot(Y_test_anomalous, z_test_anomalous)
                    # draw_histogram(Y_test_anomalous, Y, z_test_anomalous, SOFT_TUKEY_DEPTH_TEMP, bins=HISTOGRAM_BINS)

                writer = csv.writer(open(
                    f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Anomalous_Encoder_{RESULT_NAME_DESC}_{NOMINAL_CLASS}_run{run}.csv',
                    'w'))
                writer.writerow(anomalous_soft_tukey_depths)

                print(f'AUROC: {utils.get_auroc(nominal_soft_tukey_depths, anomalous_soft_tukey_depths)}')


        torch.save(encoder.state_dict(), f'./snapshots/{DATASET_NAME}_Encoder_V7_{RESULT_NAME_DESC}_{NOMINAL_CLASS}')

        # for i in range(X.size(dim=0)):
        #     print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

        # print('Variance')
        # print(get_variance_soft_tukey_depth(X, z_params))
        #
        # print('Inverse Sum')
        # print(get_inverse_sum_soft_tukey_depth(X, z_params))
