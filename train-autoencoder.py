from DataLoader import NominalCIFAR10ImageDataset, NominalMNISTImageDataset, NominalMVTecCapsuleDataset
from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_V4 import AE_CIFAR10_V4
from models.AE_MNIST_V2 import AE_MNIST_V2
from models.MVTec_AE import MVTec_AE
from models.RAE_CIFAR10 import RAE_CIFAR10
from models.RAE_MNIST import RAE_MNIST
from models.AE_MNIST import AE_MNIST
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
import torchvision
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10'
# CLASS = 0
NUM_EPOCHS = 300
SOFT_TUKEY_DEPTH_TEMP = 0.2
ENCODING_DIM = 512
STD_ITERATIONS = 6


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

def soft_tukey_depth(x, x_, z):
    matmul = torch.outer(torch.ones(x_.size(dim=0), device=device), x)
    return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(
        torch.matmul(torch.subtract(x_, matmul), z),
        torch.norm(z)))))


def get_mean_soft_tukey_depth(X, X_, z_params):
    mean = torch.tensor(0).to(device)
    for i in range(X.size(dim=0)):
        mean = mean.add(soft_tukey_depth(X[i], X_, z_params[i]).divide(X_.size(dim=0)))
    _mean = mean.divide(X.size(dim=0))
    return _mean


def get_variance_soft_tukey_depth_with_mean(X, z_params, mean):
    n = X.size(dim=0)
    var = torch.tensor(0)
    for i in range(n):
        var = var.add(torch.square(soft_tukey_depth(X[i], X, z_params[i]).divide(n).subtract(mean)))
    return var.divide(n - 1)


def get_variance_soft_tukey_depth(X, z_params):
    return get_variance_soft_tukey_depth_with_mean(X, z_params, get_mean_soft_tukey_depth(X, X, z_params))


for CLASS in range(0, 1):
    train_data = torch.utils.data.Subset(NominalMNISTImageDataset(nominal_class=CLASS, train=True), list(range(1024)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), pin_memory=True)

    test_data = NominalMNISTImageDataset(nominal_class=CLASS, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, pin_memory=True)

    autoencoder = AE_CIFAR10_V4().to(device)
    # print(list(autoencoder.parameters()))
    print(len(train_dataloader))

    def get_loss_rec(x, x_hat):
        return ((x - x_hat).square()).sum(axis=(2, 3)).mean()


    def get_loss_rae(z):
        return (z.square()).sum(axis=1).mean()


    def get_loss_reg(model):
        return sum(parameter.square().sum() for parameter in model.parameters())


    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-3)

    z = [torch.nn.Parameter(torch.rand(ENCODING_DIM, device=device).multiply(torch.tensor(2)).subtract(torch.tensor(1))) for i in range(len(train_data))]
    optimizer_z = torch.optim.SGD(z, lr=3e-2)

    for epoch in range(NUM_EPOCHS):
        autoencoder.train()
        for X in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch', colour='blue'):
            X = X.to(device)
            Z, X_hat = autoencoder(X)

            for i in range(Z.size(dim=0)):
                for k in range(STD_ITERATIONS):
                    optimizer_z.zero_grad()
                    _soft_tukey_depth = soft_tukey_depth(Z[i].detach(), Z.detach(), z[i])
                    _soft_tukey_depth.backward()
                    optimizer_z.step()

            if epoch == NUM_EPOCHS - 1:
                for i in range(16):
                    plt.imshow(np.transpose(X[i].cpu().numpy(), (1, 2, 0)))
                    plt.show()
                    plt.imshow(np.transpose(X_hat[i].cpu().detach().numpy(), (1, 2, 0)))
                    plt.show()

            var = get_variance_soft_tukey_depth(Z, z)
            print(f'Variance: {var.item()}')
            total_loss = get_loss_rec(X, X_hat) - 1e+3 * var + 1e-4 * get_loss_reg(autoencoder)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        autoencoder.eval()

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        for step, X in enumerate(train_dataloader):
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            rec_loss = get_loss_rec(X, X_hat)
            total_loss = total_loss.add(rec_loss + 1e-3 * get_loss_rae(Z) + 1e-4 * get_loss_reg(autoencoder))
            total_rec_loss = total_rec_loss.add(rec_loss)

        total_loss /= len(train_data)
        total_rec_loss /= len(train_data)

        print(f'Train total loss: {total_loss.item()}')
        print(f'Train reconstruction loss: {total_rec_loss.item()}')

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        for step, X in enumerate(test_dataloader):
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            rec_loss = get_loss_rec(X, X_hat)
            total_loss = total_loss.add(rec_loss + 1e-3 * get_loss_rae(Z) + 1e-4 * get_loss_reg(autoencoder))
            total_rec_loss = total_rec_loss.add(rec_loss)

        total_loss /= len(test_data)
        total_rec_loss /= len(test_data)

        print(f'Test total loss: {total_loss.item()}')
        print(f'Test reconstruction loss: {total_rec_loss.item()}')

    torch.save(autoencoder.state_dict(), f'./snapshots/TDAE_{DATASET_NAME}_{CLASS}')