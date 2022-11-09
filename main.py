import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, AnomalousCIFAR10Dataset, \
    NominalCIFAR10GrayscaleDataset, AnomalousCIFAR10GrayscaleDataset, Cellular4GDataset, ToyDataset, \
    NominalCIFAR10ImageDataset, AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, \
    AnomalousCIFAR10AutoencoderDataset, NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset, \
    NominalMNISTAutoencoderAllDataset, AnomalousMNISTAutoencoderAllDataset
from models.RAE_CIFAR10 import RAE_CIFAR10


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'MNIST_Autoencoder'
NOMINAL_DATASET = NominalMNISTAutoencoderAllDataset
ANOMALOUS_DATASET = AnomalousMNISTAutoencoderAllDataset
N_CLASSES = 10
TUKEY_DEPTH_COMPUTATION_EPOCHS = 5
TUKEY_DEPTH_COMPUTATIONS = 1
BATCH_SIZE = 16

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


for i in range(N_CLASSES):
    train_data = NOMINAL_DATASET(nominal_class=i, train=True)
    test_data_nominal = NOMINAL_DATASET(nominal_class=i, train=False)
    test_data_anomalous = ANOMALOUS_DATASET(nominal_class=i, train=False)

    print(f'Number of training samples: {len(train_data)}')
    print(f'Number of test samples: {len(test_data_nominal)}')

    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal)
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

    for test_dataloader in [test_dataloader_nominal, test_dataloader_anomalous]:
        soft_tukey_depths = []

        def soft_tukey_depth(x, x_, z):
            return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(5), torch.divide(torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))

        for item, x in enumerate(test_dataloader):
            print(f'Item {item}/{len(test_dataloader)}')
            x = x.to(device)
            min_tukey_depth = torch.tensor(100_000_000)

            for j in range(TUKEY_DEPTH_COMPUTATIONS):
                z = torch.nn.Parameter(torch.rand(x.size(dim=1), device=device) / torch.tensor(len(train_data)))
                optimizer = torch.optim.SGD([z], lr=1e-5)

                for k in range(TUKEY_DEPTH_COMPUTATION_EPOCHS):
                    for item2, x2 in enumerate(train_dataloader):
                        x2 = x2.to(device)
                        _soft_tukey_depth = soft_tukey_depth(x, x2, z)
                        _soft_tukey_depth.backward(retain_graph=True)
                        optimizer.step()

                _soft_tukey_depth = torch.tensor(0.0, device=device)
                for step2, x2 in enumerate(train_dataloader):
                    x2 = x2.to(device)
                    _soft_tukey_depth = torch.add(_soft_tukey_depth, soft_tukey_depth(x, x2, z))

                if _soft_tukey_depth < min_tukey_depth:
                    min_tukey_depth = _soft_tukey_depth

            soft_tukey_depths.append(min_tukey_depth.item() / len(train_data))
            print(f'Soft tukey depth is {min_tukey_depth.item() / len(train_data)}')

        print(soft_tukey_depths)

        writer = csv.writer(open(f'./results/raw/soft_tukey_depths_{DATASET_NAME}_{test_dataloader.dataset.__class__.__name__}_{i}.csv', 'w'))
        writer.writerow(soft_tukey_depths)


