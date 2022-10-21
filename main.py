import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, AnomalousCIFAR10Dataset, \
    NominalCIFAR10GrayscaleDataset, AnomalousCIFAR10GrayscaleDataset, Cellular4GDataset, ToyDataset, \
    NominalCIFAR10ImageDataset, AnomalousCIFAR10ImageDataset, NominalCIFAR10AEDataset, AnomalousCIFAR10AEDataset
from models.RAE_CIFAR10 import RAE_CIFAR10


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'MNIST'

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


autoencoder = RAE_CIFAR10().to(device)
autoencoder.load_state_dict(torch.load('./snapshots/RAE_CIFAR10_0'))



for i in range(1):
    train_data = NominalCIFAR10AEDataset(nominal_class=i, train=True)
    test_data_nominal = NominalCIFAR10AEDataset(nominal_class=i, train=False)
    test_data_anomalous = AnomalousCIFAR10AEDataset(nominal_class=i, train=False)

    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of test samples: {len(test_data_nominal)}')

    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal)
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16)

    for test_dataloader in [test_dataloader_nominal, test_dataloader_anomalous]:
        soft_tukey_depths = []

        def soft_tukey_depth(x, x_, z):
            return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1), torch.divide(torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))

        for item, x in enumerate(test_dataloader):
            print(f'Item {item}/{len(test_dataloader)}')
            x = x.to(device)
            z = torch.nn.Parameter(torch.ones(x.size(dim=1), device=device) / torch.tensor(len(train_data)))
            optimizer = torch.optim.SGD([z], lr=1e-5)

            for j in range(5):
                for item2, x2 in enumerate(train_dataloader):
                    x2 = x2.to(device)
                    _soft_tukey_depth = soft_tukey_depth(x, x2, z)
                    _soft_tukey_depth.backward(retain_graph=True)
                    optimizer.step()

            _soft_tukey_depth = torch.tensor(0.0, device=device)
            for step2, x2 in enumerate(train_dataloader):
                x2 = x2.to(device)
                _soft_tukey_depth = torch.add(_soft_tukey_depth, soft_tukey_depth(x, x2, z))

            soft_tukey_depths.append(_soft_tukey_depth.item())
            print(f'Soft tukey depth is {_soft_tukey_depth}')

        print(soft_tukey_depths)

        writer = csv.writer(open(f'./results/raw/soft_tukey_depths_{DATASET_NAME}_{test_dataloader.dataset.__class__.__name__}_AE_{i}.csv', 'w'))
        writer.writerow(soft_tukey_depths)


