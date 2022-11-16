import csv

import torch
import matplotlib.pyplot as plt
from DataLoader import NominalMNISTImageDataset, AnomalousMNISTImageDataset, NominalCIFAR10ImageDataset, \
    AnomalousCIFAR10ImageDataset, NominalMVTecCapsuleDataset, AnomalousMVTecCapsuleDataset
from models.CIFAR10_Encoder_V4 import CIFAR10_Encoder_V4
from models.CIFAR10_Encoder_V5 import CIFAR10_Encoder_V5
from models.MNIST_Encoder_Simple import MNIST_Encoder_Simple
from models.MNIST_Encoder_DSVDD import MNIST_Encoder_DSVDD
from models.CIFAR10_Encoder_Simple import CIFAR10_Encoder_Simple
from models.CIFAR10_Encoder_V3 import CIFAR10_Encoder_V3
import torch.utils.data
import numpy as np

from models.MVTecCapsule_Encoder import MVTecCapsule_Encoder

DATASET_NAME = 'CIFAR10'
NOMINAL_DATASET = NominalCIFAR10ImageDataset
ANOMALOUS_DATASET = AnomalousCIFAR10ImageDataset
DATA_SIZE = 1024
TEST_NOMINAL_SIZE = 1000
TEST_ANOMALOUS_SIZE = 1000


USE_CUDA_IF_AVAILABLE = True
SAVE_MODEL = True
KERNEL_BANDWIDTH = 0.05
SOFT_TUKEY_DEPTH_TEMP = 0.1
ENCODING_DIM = 128
HISTOGRAM_BINS = 50
NUM_EPOCHS = 6
STD_ITERATIONS = 3
TEST_STD_ITERATIONS = 5
BATCH_SIZE = 256
BATCH_SIZE_STD_COMPUTATION = 16

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
    return soft_tukey_depth_v2(x, x_, z)
    # matmul = torch.outer(torch.ones(x_.size(dim=0), device=device), x)
    # return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(
    #     torch.matmul(torch.subtract(x_, matmul), z),
    #     torch.norm(z)))))


def soft_tukey_depth_v2(x, x_, z):
    return torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(torch.matmul(torch.subtract(x_, x), z), torch.norm(z)))).sum()


def get_mean_soft_tukey_depth(X, z_params):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i], X, z_params[i]))
    return mean.divide(X.size(dim=0))


def get_variance_soft_tukey_depth_with_mean(X, z_params, mean):
    n = X.size(dim=0)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i], X, z_params[i]).divide(n).subtract(mean)))
    return var.divide(n - 1)


def get_variance_soft_tukey_depth(X, z_params):
    return get_variance_soft_tukey_depth_with_mean(X, z_params, get_mean_soft_tukey_depth(X, z_params))


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


def draw_histogram(X, X_, z_params, bins=200):
    n = X.size(dim=0)
    soft_tukey_depths = torch.zeros(n)
    for i in range(n):
        soft_tukey_depths[i] = soft_tukey_depth(X[i], X_, z_params[i]) / n
    tukey_depth_histogram = plt.figure()
    plt.hist(soft_tukey_depths.detach(), bins=bins)
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


for NOMINAL_CLASS in range(5, 6):

    train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=True), list(range(DATA_SIZE)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    train_dataloader_std_computation = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE_STD_COMPUTATION)

    test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_NOMINAL_SIZE)))
    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal)

    test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=NOMINAL_CLASS, train=False), list(range(TEST_ANOMALOUS_SIZE)))
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous)

    encoder = CIFAR10_Encoder_V5().to(device)
    encoder.train()

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # z = [torch.ones(X.size(dim=1), device=device) for i in range(X.size(dim=0))]
    # z_params = [torch.nn.Parameter(z[i].divide(torch.norm(z[i]))) for i in range(len(z))]
    z_params = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
    optimizer_z = torch.optim.SGD(z_params, lr=3e-2)


    for i in range(NUM_EPOCHS):
        print(f'Epoch {i+1}')

        for step, X in enumerate(train_dataloader):
            print(f'Step {step} - {X.size()}')
            X = X.to(device)
            Y = encoder(X)

            for j in range(X.size(dim=0)):
                for k in range(STD_ITERATIONS):
                    optimizer_encoder.zero_grad()
                    optimizer_z.zero_grad()
                    _soft_tukey_depth = soft_tukey_depth(Y[j].detach(), Y.detach(), z_params[step * BATCH_SIZE + j])
                    _soft_tukey_depth.backward()
                    optimizer_z.step()
                    # print(j, z_params[j])

            optimizer_encoder.zero_grad()
            optimizer_z.zero_grad()


            var = get_variance_soft_tukey_depth(Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)])
            print(f'Variance: {var.item()}')
            print(f'Total norm: {torch.linalg.norm(Y, dim=1).sum().item()}')
            print(f'Total point value: {Y.sum(dim=0).sum()}')
            # ((0 * -var).add(1e+4 * (torch.square(torch.linalg.norm(Y, dim=1).sum().subtract(DATA_SIZE)))).add(1e+3 * torch.square(Y.sum(dim=0)).sum())).backward()
            (-var).backward()

            # moment_loss = get_moment_loss(Y, z_params, 3)
            # print(f'Moment loss: {moment_loss.item()}')
            # moment_loss.backward()

            # inverse_sum_loss = get_inverse_sum_soft_tukey_depth(Y, z_params)
            # (inverse_sum_loss).backward()

            optimizer_encoder.step()

            if i % 1 == 0:
                if ENCODING_DIM == 2:
                    draw_scatter_plot(Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)])
                draw_histogram(Y, Y, z_params[(step * BATCH_SIZE):((step + 1) * BATCH_SIZE)], bins=HISTOGRAM_BINS)
            if i == NUM_EPOCHS - 1:
                draw_histogram(Y, Y, z_params[(step * BATCH_SIZE):((step+1) * BATCH_SIZE)], bins=HISTOGRAM_BINS)

    if SAVE_MODEL:
        torch.save(encoder.state_dict(), f'./snapshots/{DATASET_NAME}_Encoder_v5_{NOMINAL_CLASS}')


    soft_tukey_depths = []

    for step, X_test_nominal in enumerate(test_dataloader_nominal):
        print(f'Step {step}')
        X_test_nominal = X_test_nominal.to(device)
        Y_test_nominal = encoder(X_test_nominal)
        z_test_nominal = torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
        optimizer_z_test_nominal = torch.optim.SGD([z_test_nominal], lr=3e-2)

        for k in range(TEST_STD_ITERATIONS):
            for step2, X in enumerate(train_dataloader_std_computation):
                X = X.to(device)
                Y = encoder(X)
                optimizer_z_test_nominal.zero_grad()
                _soft_tukey_depth = soft_tukey_depth(Y_test_nominal[0].detach(), Y.detach(), z_test_nominal)
                _soft_tukey_depth.backward()
                optimizer_z_test_nominal.step()

        _soft_tukey_depth = torch.tensor(0, device=device)
        for step2, X in enumerate(train_dataloader):
            X = X.to(device)
            Y = encoder(X)
            _soft_tukey_depth = torch.add(_soft_tukey_depth, soft_tukey_depth(Y_test_nominal[0].detach(), Y.detach(), z_test_nominal))
        soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))

    writer = csv.writer(open(
        f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Nominal_Encoder_v5_batches_{NOMINAL_CLASS}.csv',
        'w'))
    writer.writerow(soft_tukey_depths)


    soft_tukey_depths = []
    for step, X_test_anomalous in enumerate(test_dataloader_anomalous):
        print(f'Step {step}')
        X_test_anomalous = X_test_anomalous.to(device)
        Y_test_anomalous = encoder(X_test_anomalous)
        z_test_anomalous = torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
        optimizer_z_test_anomalous = torch.optim.SGD([z_test_anomalous], lr=3e-2)

        for k in range(TEST_STD_ITERATIONS):
            for step2, X in enumerate(train_dataloader_std_computation):
                X = X.to(device)
                Y = encoder(X)
                optimizer_z_test_anomalous.zero_grad()
                _soft_tukey_depth = soft_tukey_depth(Y_test_anomalous[0].detach(), Y.detach(), z_test_anomalous)
                _soft_tukey_depth.backward()
                optimizer_z_test_anomalous.step()

        _soft_tukey_depth = torch.tensor(0, device=device)
        for step2, X in enumerate(train_dataloader):
            X = X.to(device)
            Y = encoder(X)
            _soft_tukey_depth = torch.add(_soft_tukey_depth,
                                          soft_tukey_depth(Y_test_anomalous[0].detach(), Y.detach(), z_test_anomalous))
        soft_tukey_depths.append(_soft_tukey_depth.item() / len(train_data))


    writer = csv.writer(open(
        f'./results/raw/soft_tukey_depths_{DATASET_NAME}_Anomalous_Encoder_v5_batches_{NOMINAL_CLASS}.csv',
        'w'))
    writer.writerow(soft_tukey_depths)

    # for i in range(X.size(dim=0)):
    #     print(soft_tukey_depth(X[i].reshape(1, -1), X, z_params[i]))

    # print('Variance')
    # print(get_variance_soft_tukey_depth(X, z_params))
    #
    # print('Inverse Sum')
    # print(get_inverse_sum_soft_tukey_depth(X, z_params))
