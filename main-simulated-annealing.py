import random

import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, AnomalousCIFAR10Dataset, Cellular4GDataset, ToyDataset


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10'

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


for i in range(1):
    train_data = NominalMNISTDataset(nominal_class=i, train=True)
    test_data_nominal = NominalMNISTDataset(nominal_class=i, train=False)
    test_data_anomalous = AnomalousMNISTDataset(nominal_class=i, train=False)

    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of test samples: {len(test_data_nominal)}')

    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal)
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128)

    for test_dataloader in [test_dataloader_nominal, test_dataloader_anomalous]:
        soft_tukey_depths = []

        def soft_tukey_depth(x, x_, z):
            return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1), torch.divide(torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))

        for item, x in enumerate(test_dataloader):
            print(f'Item {item}/{len(test_dataloader)}')
            x = x.to(device)
            z = torch.nn.Parameter(torch.ones(x.size(dim=1), device=device) / torch.tensor(len(train_data)))
            z = z.divide(torch.norm(z))
            new_z = z

            current_tukey_depth = torch.tensor(100000000)

            for j in range(100):
                new_soft_tukey_depth = torch.tensor(0)

                for item2, x2 in enumerate(train_dataloader):
                    x2 = x2.to(device)
                    new_soft_tukey_depth = new_soft_tukey_depth.add(soft_tukey_depth(x, x2, new_z))

                if torch.exp((current_tukey_depth.subtract(new_soft_tukey_depth).divide(torch.tensor(0.01)))) >= random.random():
                    current_tukey_depth = new_soft_tukey_depth
                    z = new_z
                    print(z)
                    print(new_soft_tukey_depth.item())
                else:
                    print('no update')

                delta_vec = torch.rand(x.size(dim=1), device=device)
                delta_vec = delta_vec.divide(torch.norm(delta_vec))
                new_z = z.subtract(torch.multiply(torch.tensor(0.1), delta_vec.subtract(z.multiply(torch.dot(delta_vec, z.divide(torch.norm(z)))))))
                new_z = new_z.divide(torch.norm(new_z))


            soft_tukey_depths.append(current_tukey_depth.item())
            print(f'Soft tukey depth is {current_tukey_depth}')

        print(soft_tukey_depths)

        writer = csv.writer(open(f'./results/raw/soft_tukey_depths_{DATASET_NAME}_{test_dataloader.dataset.__class__.__name__}_{i}.csv', 'w'))
        writer.writerow(soft_tukey_depths)

