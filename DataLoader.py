import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
from models.AE_MNIST import AE_MNIST
from transform import FlattenTransform
from models.RAE_CIFAR10 import RAE_CIFAR10
from models.RAE_MNIST import RAE_MNIST
from preprocessing import get_target_label_idx, global_contrast_normalization


class Cellular4GDataset(Dataset):
    def __init__(self, root, train=True):
        data = pd.read_csv(f"{root}\ML-MATT-CompetitionQT2021\ML-MATT-CompetitionQT2021_{'train' if train else 'test'}.csv", delimiter=';')
        cell_names = list(set(data['CellName'].tolist()))
        print(list(set(data['CellName'].tolist())))
        cell_name_dict = {}
        for i in range(len(cell_names)):
            cell_name_dict[cell_names[i]] = i

        def parse_time(time_str):
            parsed = time_str.split(':')
            if len(parsed) < 2:
                return 0
            return int(parsed[0]) * 60 + int(parsed[1])

        data['Time'] = data['Time'].transform(parse_time)
        cell_names_one_hot = np.eye(len(cell_names))[[cell_name_dict[cell_name] for cell_name in data['CellName'].tolist()]]
        data.drop(columns=['CellName', 'Unusual'] if train else ['CellName'], axis=1, inplace=True)

        self.data = np.concatenate((StandardScaler().fit_transform(data.to_numpy()), cell_names_one_hot), axis=1)

    def __getitem__(self, item):
        return self.data[item % self.data.shape[0]]

    def __len__(self):
        return self.data.shape[0]


class NominalCIFAR10ImageDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10ImageDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalCIFAR10AutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

        self.data_latent = torch.zeros(len(self.data), 1, 512)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_CIFAR10_V3()
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_CIFAR10_V3_{nominal_class}'))

        for step, x in enumerate(dataloader):
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10AutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

        self.data_latent = torch.zeros(len(self.data), 1, 512)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_CIFAR10_V3()
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_CIFAR10_V3_{nominal_class}'))

        for step, x in enumerate(dataloader):
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalCIFAR10Dataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10Dataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalCIFAR10GrayscaleDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10GrayscaleDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalCIFAR10PCADataset(Dataset):
    def __init__(self, nominal_class, n_components, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.data.data = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(self.data.data.reshape((self.data.data.shape[0], -1))))

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10PCADataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalMNISTDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousMNISTDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), FlattenTransform()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)



min_max_mnist = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

min_max_mnist_all = (-0.8826567065619495, 20.108062262467364)


class NominalMNISTImageDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        torchvision.transforms.Normalize([(min_max_mnist_all if nominal_class == 'all' else min_max_mnist[nominal_class])[0]],
                                                             [(min_max_mnist_all if nominal_class == 'all' else min_max_mnist[nominal_class])[1] - (min_max_mnist_all if nominal_class == 'all' else min_max_mnist[nominal_class])[0]])])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0] if nominal_class != 'all' else list(range(len(self.data)))

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousMNISTImageDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                                      torchvision.transforms.Normalize([(
                                                                                            min_max_mnist_all if nominal_class == 'all' else
                                                                                            min_max_mnist[
                                                                                                nominal_class])[0]],
                                                                                       [(
                                                                                            min_max_mnist_all if nominal_class == 'all' else
                                                                                            min_max_mnist[
                                                                                                nominal_class])[1] - (
                                                                                            min_max_mnist_all if nominal_class == 'all' else
                                                                                            min_max_mnist[
                                                                                                nominal_class])[0]])])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0] if nominal_class != 'all' else list(range(len(self.data)))

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalMNISTAutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Lambda(
                                                          lambda x: global_contrast_normalization(x, scale='l1')),
                                                      torchvision.transforms.Normalize(
                                                          [min_max_mnist[nominal_class][0]],
                                                          [min_max_mnist[nominal_class][1] -
                                                           min_max_mnist[nominal_class][0]])])
        )

        self.data_latent = torch.zeros(len(self.data), 1, 32)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_MNIST()
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_MNIST_32_{nominal_class}'))
        autoencoder.eval()

        for step, x in enumerate(dataloader):
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousMNISTAutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Lambda(
                                                          lambda x: global_contrast_normalization(x, scale='l1')),
                                                      torchvision.transforms.Normalize(
                                                          [min_max_mnist[nominal_class][0]],
                                                          [min_max_mnist[nominal_class][1] -
                                                           min_max_mnist[nominal_class][0]])])
        )

        self.data_latent = torch.zeros(len(self.data), 1, 32)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_MNIST()
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_MNIST_32_{nominal_class}'))
        autoencoder.eval()

        for step, x in enumerate(dataloader):
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0, 0], [1, 1], [-2, 2], [0, 2], [2, 2], [-2, 3], [0, 3], [2, 3]], dtype=torch.float)

    def __getitem__(self, item):
        return self.data[item % self.data.size(dim=0)]

    def __len__(self):
        return self.data.size(dim=0)