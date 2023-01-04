from DataLoader import NominalCIFAR10ImageDataset, NominalMNISTImageDataset, NominalMVTecCapsuleImageDataset, \
    AnomalousMVTecCapsuleImageDataset, AnomalousCIFAR10ImageDataset, NominalFashionMNISTImageDataset, \
    AnomalousFashionMNISTImageDataset
from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_V4 import AE_CIFAR10_V4
from models.AE_CIFAR10_V5 import AE_CIFAR10_V5
from models.AE_CIFAR10_V6 import AE_CIFAR10_V6
from models.AE_CIFAR10_V7 import AE_CIFAR10_V7
from models.AE_FashionMNIST import AE_FashionMNIST
from models.AE_MNIST_V2 import AE_MNIST_V2
from models.AE_MNIST_V3 import AE_MNIST_V3
from models.MVTec_AE import MVTec_AE
from models.RAE_CIFAR10 import RAE_CIFAR10
from models.RAE_MNIST import RAE_MNIST
from models.AE_MNIST import AE_MNIST
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
import torchvision
import torch.utils.data
import torch.distributions
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.VAE_CIFAR10 import VAE_CIFAR10
from models.resnet import ResNet50



USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'FashionMNIST'
SAVE_MODEL = True
# CLASS = 0
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3



if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def get_loss_rec(x, x_hat):
    return ((x - x_hat).square()).sum(axis=(2, 3)).mean()


def get_loss_rae(z):
    return (z.square()).sum(axis=1).mean()


def get_loss_reg(model):
    return sum(parameter.square().sum() for parameter in model.parameters())


"""
Only used for variational autoencoder
"""
def get_loss_kl_div_latent(Z_mu, Z_std, Z):
    p = torch.distributions.normal.Normal(torch.zeros_like(Z_mu), torch.ones_like(Z_std))
    q = torch.distributions.normal.Normal(Z_mu, torch.exp(Z_std))
    return torch.sum(q.log_prob(Z) - p.log_prob(Z), dim=1).mean()


for CLASS in range(4, 10):
    train_data = NominalFashionMNISTImageDataset(train=True, nominal_class=CLASS)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, pin_memory=True)

    test_data = NominalFashionMNISTImageDataset(train=False, nominal_class=CLASS)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, pin_memory=True)

    test_data_anomalous = AnomalousFashionMNISTImageDataset(train=False, nominal_class=CLASS)
    test_anomalous_dataloader = torch.utils.data.DataLoader(test_data_anomalous, batch_size=BATCH_SIZE, pin_memory=True)

    autoencoder = AE_FashionMNIST().to(device)

    # print(list(autoencoder.parameters()))

    print(len(train_dataloader))

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        autoencoder.train()
        for X in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch', colour='blue'):
            X = X.to(device)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X[0].cpu().numpy(), (1, 2, 0)))
                plt.show()
            Z, X_hat = autoencoder(X)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X_hat[0].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()
                # rand_img = autoencoder.decoder(torch.rand(1, 64, device=device).multiply(4).subtract(2))
                # plt.imshow(np.transpose(rand_img[0].cpu().detach().numpy(), (1, 2, 0)))
                # plt.show()

            total_loss = get_loss_rec(X, X_hat) + 1e-4 * get_loss_rae(Z) + 1e-4 * get_loss_reg(autoencoder)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        autoencoder.eval()

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        # total_kl_div_loss = torch.tensor(0)
        for step, X in enumerate(train_dataloader):
            X = X.to(device)
            Z, X_hat = autoencoder(X)
            rec_loss = get_loss_rec(X, X_hat)
            # kl_div_loss = get_loss_kl_div_latent(Z_mu, Z_std, Z)
            total_loss = total_loss.add(torch.multiply(rec_loss + 1e-4 * get_loss_rae(Z) + 1e-4 * get_loss_reg(autoencoder), BATCH_SIZE / len(train_data)))
            total_rec_loss = total_rec_loss.add(torch.multiply(rec_loss, BATCH_SIZE / len(train_data)))
            # total_kl_div_loss = total_kl_div_loss.add(torch.multiply(kl_div_loss, BATCHS_SIZE / len(train_data)))

        print(f'Train total loss: {total_loss.item()}')
        print(f'Train reconstruction loss: {total_rec_loss.item()}')
        # print(f'Train KL divergence loss: {total_kl_div_loss.item()}')

        total_loss = torch.tensor(0)
        total_rec_loss = torch.tensor(0)
        total_kl_div_loss = torch.tensor(0)
        for step, X in enumerate(test_dataloader):
            X = X.to(device)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X[0].cpu().numpy(), (1, 2, 0)))
                plt.show()
            Z, X_hat = autoencoder(X)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X_hat[0].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()
            rec_loss = get_loss_rec(X, X_hat)
            # kl_div_loss = get_loss_kl_div_latent(Z_mu, Z_std, Z)
            total_loss = total_loss.add(torch.multiply(rec_loss + 1e-4 * get_loss_rae(Z) + 1e-4 * get_loss_reg(autoencoder), BATCH_SIZE / len(test_data)))
            total_rec_loss = total_rec_loss.add(torch.multiply(rec_loss, BATCH_SIZE / len(test_data)))
            # total_kl_div_loss = total_kl_div_loss.add(torch.multiply(kl_div_loss, BATCHS_SIZE / len(test_data)))

        for step, X in enumerate(test_anomalous_dataloader):
            X = X.to(device)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X[0].cpu().numpy(), (1, 2, 0)))
                plt.show()
            Z, X_hat = autoencoder(X)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X_hat[0].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()

        print(f'Test total loss: {total_loss.item()}')
        print(f'Test reconstruction loss: {total_rec_loss.item()}')
        # print(f'Test KL divergence loss: {total_kl_div_loss.item()}')

        if SAVE_MODEL and epoch % 10 == 0:
            torch.save(autoencoder.state_dict(), f'./snapshots/AE_{DATASET_NAME}_{CLASS}')