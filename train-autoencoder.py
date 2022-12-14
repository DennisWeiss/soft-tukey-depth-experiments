from DataLoader import NominalCIFAR10ImageDataset, NominalMNISTImageDataset, NominalMVTecCapsuleImageDataset
from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_32 import AE_CIFAR10_32
from models.AE_CIFAR10_V4 import AE_CIFAR10_V4
from models.AE_MNIST_32 import AE_MNIST_32
from models.AE_MNIST_V2 import AE_MNIST_V2
from models.AE_MNIST_V3 import AE_MNIST_V3
from models.MVTec_AE import MVTec_AE
from models.RAE_CIFAR10 import RAE_CIFAR10
from models.RAE_MNIST import RAE_MNIST
from models.AE_MNIST import AE_MNIST
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10'
# CLASS = 0
NUM_EPOCHS = 50


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

for CLASS in range(0, 10):
    train_data = NominalCIFAR10ImageDataset(nominal_class=CLASS, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, pin_memory=True)

    test_data = NominalCIFAR10ImageDataset(nominal_class=CLASS, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, pin_memory=True)

    autoencoder = AE_CIFAR10_32().to(device)
    # print(list(autoencoder.parameters()))
    print(len(train_dataloader))

    def get_loss_rec(x, x_hat):
        return ((x - x_hat).square()).mean()


    def get_loss_rae(z):
        return (z.square()).mean(axis=1).mean()


    def get_loss_reg(model):
        return sum(parameter.square().mean() for parameter in model.parameters())


    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-3)

    for epoch in range(NUM_EPOCHS):
        autoencoder.train()
        for X in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch', colour='blue'):
            X = X.to(device)
            if epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X[0].cpu().numpy(), (1, 2, 0)))
                plt.show()
            Z, X_hat = autoencoder(X)
            if epoch == NUM_EPOCHS - 1:
                plt.imshow(np.transpose(X_hat[0].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()

            total_loss = get_loss_rec(X, X_hat) + 1e-4 * get_loss_rae(Z) + 1e-5 * get_loss_reg(autoencoder)

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
            total_loss = total_loss.add(rec_loss + 1e-4 * get_loss_rae(Z) + 1e-5 * get_loss_reg(autoencoder))
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
            total_loss = total_loss.add(rec_loss + 1e-4 * get_loss_rae(Z) + 1e-5 * get_loss_reg(autoencoder))
            total_rec_loss = total_rec_loss.add(rec_loss)

        total_loss /= len(test_data)
        total_rec_loss /= len(test_data)

        print(f'Test total loss: {total_loss.item()}')
        print(f'Test reconstruction loss: {total_rec_loss.item()}')

    torch.save(autoencoder.state_dict(), f'./snapshots/AE_{DATASET_NAME}_32_{CLASS}')