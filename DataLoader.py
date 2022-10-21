import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from transform import FlattenTransform
from models.RAE_CIFAR10 import RAE_CIFAR10


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


class NominalCIFAR10AEDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

        self.data_latent = torch.zeros(len(self.data), 1, 64)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = RAE_CIFAR10()
        autoencoder.load_state_dict(torch.load(f'./snapshots/RAE_CIFAR10_{nominal_class}'))

        for step, x in enumerate(dataloader):
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10AEDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

        self.data_latent = torch.zeros(len(self.data), 1, 64)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = RAE_CIFAR10()
        autoencoder.load_state_dict(torch.load(f'./snapshots/RAE_CIFAR10_{nominal_class}'))

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


class NominalMNISTImageDataset(Dataset):
    def __init__(self, nominal_class, train=True):
        self.data = torchvision.datasets.MNIST(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop(33)])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

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
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop(33)])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0, 0], [1, 1], [-2, 2], [0, 2], [2, 2], [-2, 3], [0, 3], [2, 3]], dtype=torch.float)

    def __getitem__(self, item):
        return self.data[item % self.data.size(dim=0)]

    def __len__(self):
        return self.data.size(dim=0)