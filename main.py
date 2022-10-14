import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, Cellular4GDataset, ToyDataset


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10'

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

for i in range(10):
    train_data = NominalCIFAR10Dataset(nominal_class=0, train=True)
    test_data = NominalCIFAR10Dataset(nominal_class=i, train=False)

    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of test samples: {len(test_data)}')

    test_dataloader = torch.utils.data.DataLoader(test_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8)

    soft_tukey_depths = []


    def soft_tukey_depth(x, x_, z):
        matmul = torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)
        return torch.sum(torch.sigmoid(torch.divide(torch.matmul(torch.subtract(x2, matmul), z), torch.norm(z))))


    for item, x in enumerate(test_dataloader):
        print(f'Item {item}/{len(test_data)}')
        x = x.to(device)
        z = torch.nn.Parameter(torch.ones(x.size(dim=1), device=device) / torch.tensor(len(train_data)))
        optimizer = torch.optim.SGD([z], lr=1e-5)

        for j in range(5):
            for item2, x2 in enumerate(train_dataloader):
                x2 = x2.to(device)
                _soft_tukey_depth = soft_tukey_depth(x, x2, z)
                _soft_tukey_depth.backward()
                optimizer.step()

        _soft_tukey_depth = torch.tensor(0.0, device=device)
        for step2, x2 in enumerate(train_dataloader):
            x2 = x2.to(device)
            _soft_tukey_depth = torch.add(_soft_tukey_depth, soft_tukey_depth(x, x2, z))

        soft_tukey_depths.append(_soft_tukey_depth.item())
        print(f'Soft tukey depth is {_soft_tukey_depth}')

    print(soft_tukey_depths)

    writer = csv.writer(open(f'soft_tukey_depths_{DATASET_NAME}_{i}.csv', 'w'))
    writer.writerow(soft_tukey_depths)


