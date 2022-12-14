import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.AE_CIFAR10 import AE_CIFAR10
from models.AE_CIFAR10_V3 import AE_CIFAR10_V3
from models.AE_CIFAR10_V4 import AE_CIFAR10_V4
from models.AE_MNIST import AE_MNIST
from models.AE_MNIST_V2 import AE_MNIST_V2
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
    def __init__(self, nominal_class, train=True, device=None):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]
        self.data_latent = torch.load(f"./datasets/CIFAR10_AE_representation/AE_CIFAR10_V3_{'train' if train else 'test'}_{nominal_class}").to(device)

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10AutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True, device=None):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]
        self.data_latent = torch.load(f"./datasets/CIFAR10_AE_representation/AE_CIFAR10_V3_{'train' if train else 'test'}_{nominal_class}").to(device)

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
    def __init__(self, nominal_class, train=True, device=None):
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

        self.data_latent = torch.load(f"./datasets/MNIST_AE_representation/AE_MNIST_32_{'train' if train else 'test'}_{nominal_class}").to(device)
        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousMNISTAutoencoderDataset(Dataset):
    def __init__(self, nominal_class, train=True, device=None):
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

        self.data_latent = torch.load(f"./datasets/MNIST_AE_representation/AE_MNIST_32_{'train' if train else 'test'}_{nominal_class}").to(device)
        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalMNISTAutoencoderAllDataset(Dataset):
    def __init__(self, nominal_class, train=True, device=None):
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

        self.data_latent = torch.zeros(len(self.data), 1, 64)

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_MNIST()
        if device is not None:
            autoencoder = autoencoder.to(device)
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_MNIST_32_all'))
        autoencoder.eval()

        for step, x in enumerate(dataloader):
            if device is not None:
                x[0] = x[0].to(device)
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousMNISTAutoencoderAllDataset(Dataset):
    def __init__(self, nominal_class, train=True, device=None):
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

        self.data_latent = torch.zeros(len(self.data), 1, 64)
        self.device = device

        dataloader = torch.utils.data.DataLoader(self.data)
        autoencoder = AE_MNIST()
        if device is not None:
            autoencoder = autoencoder.to(device)
        autoencoder.load_state_dict(torch.load(f'./snapshots/AE_MNIST_32_all'))
        autoencoder.eval()

        for step, x in enumerate(dataloader):
            if self.device is not None:
                x[0] = x[0].to(self.device)
            z, x_hat = autoencoder(x[0])
            self.data_latent[step][0] = z.detach()

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data_latent[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class NominalMVTecCapsuleImageDataset(Dataset):
    def __init__(self, train=True):
        self.path = 'datasets/mvtec/capsule'
        self.train = train
        self.samples = 219 if train else 23
        self.transform = torchvision.transforms.Resize((250, 250))

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image = torchvision.io.read_image(f"{self.path}/{'train' if self.train else 'test'}/good/{idx:03d}.png") / 255
        return self.transform(image)


class AnomalousMVTecCapsuleImageDataset(Dataset):
    def __init__(self):
        self.path = 'datasets/mvtec/capsule'
        self.samples = {
            'crack': 23,
            'faulty_imprint': 22,
            'poke': 21,
            'scratch': 23,
            'squeeze': 20
        }
        self.transform = torchvision.transforms.Resize((250, 250))

    def __len__(self):
        return sum([self.samples[type] for type in self.samples.keys()])

    def __getitem__(self, idx):
        for type in self.samples.keys():
            if idx < self.samples[type]:
                image = torchvision.io.read_image(f'{self.path}/test/{type}/{idx:03d}.png') / 255
                return self.transform(image)
            idx -= self.samples[type]


class NominalMVTecCapsuleDataset(Dataset):
    def __init__(self):
        self.path = 'datasets/mvtec/capsule'
        self.samples = {
            'crack': 23,
            'faulty_imprint': 22,
            'poke': 21,
            'scratch': 23,
            'squeeze': 20
        }
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((250, 250))])

    def __len__(self):
        return sum([self.samples[type] for type in self.samples.keys()])

    def __getitem__(self, idx):
        for type in self.samples.keys():
            if idx < self.samples[type]:
                image = torchvision.io.read_image(f'{self.path}/test/{type}/{idx:03d}.png') / 255
                return self.transform(image)
            idx -= self.samples[type]


class AnomalousMVTecCapsuleDataset(Dataset):
    def __init__(self):
        self.path = 'datasets/mvtec/capsule'
        self.samples = {
            'crack': 23,
            'faulty_imprint': 22,
            'poke': 21,
            'scratch': 23,
            'squeeze': 20
        }
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((250, 250))])

    def __len__(self):
        return sum([self.samples[type] for type in self.samples.keys()])

    def __getitem__(self, idx):
        for type in self.samples.keys():
            if idx < self.samples[type]:
                image = torchvision.io.read_image(f'{self.path}/test/{type}/{idx:03d}.png') / 255
                return self.transform(image)
            idx -= self.samples[type]


class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0, 0], [1, 1], [-2, 2], [0, 2], [2, 2], [-2, 3], [0, 3], [2, 3]], dtype=torch.float)

    def __getitem__(self, item):
        return self.data[item % self.data.size(dim=0)]

    def __len__(self):
        return self.data.size(dim=0)