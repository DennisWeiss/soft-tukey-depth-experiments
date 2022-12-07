from DataLoader import NominalCIFAR10ImageDataset, NominalMNISTImageDataset, NominalMVTecCapsuleDataset
from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_V4 import AE_CIFAR10_V4
from models.AE_MNIST_V2 import AE_MNIST_V2
from models.AE_MNIST_V3 import AE_MNIST_V3
from models.MVTec_AE import MVTec_AE
from models.RAE_CIFAR10 import RAE_CIFAR10
from models.RAE_MNIST import RAE_MNIST
from models.AE_MNIST import AE_MNIST
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
import torchvision
import torch.utils.data
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import time
import gc


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'MNIST'
# CLASS = 0
NUM_EPOCHS = 200
DATA_SIZE = 1600
SOFT_TUKEY_DEPTH_TEMP = 1
ENCODING_DIM = 64
STD_ITERATIONS = 10
BATCH_SIZE = 1600
MODEL_LEARNING_RATE = 1e-2
Z_LEARNING_RATE = 1e+4

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


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


for CLASS in range(0, 1):
    train_data = torch.utils.data.Subset(NominalMNISTImageDataset(nominal_class=CLASS, train=True), list(range(DATA_SIZE)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    train_dataloader_all_data = torch.utils.data.DataLoader(train_data, batch_size=DATA_SIZE)

    test_data = NominalMNISTImageDataset(nominal_class=CLASS, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    autoencoder = AE_MNIST_V2().to(device)
    # print(list(autoencoder.parameters()))
    print(len(train_dataloader))


    def get_loss_rec(x, x_hat):
        return ((x - x_hat).square()).sum(axis=(2, 3)).mean()


    def get_loss_rae(z):
        return (z.square()).sum(axis=1).mean()


    def get_loss_reg(model):
        return sum(parameter.square().sum() for parameter in model.parameters())


    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=MODEL_LEARNING_RATE)

    z = torch.nn.Parameter(torch.rand(DATA_SIZE, ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1)))
    optimizer_z = torch.optim.SGD([z], lr=Z_LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}')
        autoencoder.train()
        for step, X in enumerate(train_dataloader):
            start = time()
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            Z_detached = Z.detach()

            for k in range(STD_ITERATIONS):
                optimizer_z.zero_grad()
                _soft_tukey_depth = soft_tukey_depth_v2(Z_detached, Z_detached, z[(step*BATCH_SIZE):((step+1)*BATCH_SIZE)], SOFT_TUKEY_DEPTH_TEMP)
                _soft_tukey_depth.sum().backward()
                optimizer_z.step()

            if epoch == NUM_EPOCHS - 1 or epoch % 10 == 0:
                for i in range(min(16, X.size(dim=0))):
                    plt.imshow(np.transpose(X[i].cpu().numpy(), (1, 2, 0)))
                    plt.show()
                    plt.imshow(np.transpose(X_hat[i].cpu().detach().numpy(), (1, 2, 0)))
                    plt.show()

            for step2, X_train in enumerate(train_dataloader_all_data):
                optimizer.zero_grad()
                X_train = X_train.to(device)
                Z_train, X_train_hat = autoencoder(X_train)

                tds = soft_tukey_depth_v2(Z_train, Z, z, SOFT_TUKEY_DEPTH_TEMP)
                print(f'Mean TD: {tds.mean().item()}')

                var = torch.var(tds)
                print(f'Variance TD: {var.item()}')

                rec_loss = get_loss_rec(X, X_hat)
                print(f'Reconstruction loss: {rec_loss.item()}')

                total_loss = rec_loss - 10 * var + 1e-5 * get_loss_reg(autoencoder)
                print(f'Total loss: {total_loss.item()}')

                total_loss.backward()
                optimizer.step()

            print(f'Elapsed time: {(time() - start):.2f}s')

        autoencoder.eval()

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        for step, X in enumerate(train_dataloader):
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            normalizer = X.size(dim=0) / len(train_data)
            rec_loss = get_loss_rec(X, X_hat)
            total_loss = total_loss.add(
                torch.multiply(rec_loss + 1e-2 * get_loss_rae(Z) + 1e-5 * get_loss_reg(autoencoder), normalizer))
            total_rec_loss = total_rec_loss.add(torch.multiply(rec_loss, normalizer))

        print(f'Train total loss: {total_loss.item()}')
        print(f'Train reconstruction loss: {total_rec_loss.item()}')

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        for step, X in enumerate(test_dataloader):
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            normalizer = X.size(dim=0) / len(test_data)
            rec_loss = get_loss_rec(X, X_hat)
            total_loss = total_loss.add(
                torch.multiply(rec_loss + 1e-2 * get_loss_rae(Z) + 1e-5 * get_loss_reg(autoencoder), normalizer))
            total_rec_loss = total_rec_loss.add(torch.multiply(rec_loss, normalizer))

        print(f'Test total loss: {total_loss.item()}')
        print(f'Test reconstruction loss: {total_rec_loss.item()}')

        torch.save(autoencoder.state_dict(), f'./snapshots/TDAE_var_max_{DATASET_NAME}_{CLASS}')
