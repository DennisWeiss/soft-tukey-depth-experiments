import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, AnomalousCIFAR10Dataset, \
    NominalCIFAR10GrayscaleDataset, AnomalousCIFAR10GrayscaleDataset, Cellular4GDataset, ToyDataset, \
    NominalCIFAR10ImageDataset, AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, \
    AnomalousCIFAR10AutoencoderDataset, NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset, \
    NominalMNISTAutoencoderAllDataset, AnomalousMNISTAutoencoderAllDataset
from models.RAE_CIFAR10 import RAE_CIFAR10


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10_Autoencoder'
NOMINAL_DATASET = NominalCIFAR10AutoencoderDataset
ANOMALOUS_DATASET = AnomalousCIFAR10AutoencoderDataset
N_CLASSES = 10
TUKEY_DEPTH_COMPUTATION_EPOCHS = 10
TUKEY_DEPTH_COMPUTATIONS = 1
SOFT_TUKEY_DEPTH_TEMP = 1
BATCH_SIZE = 16
TRAIN_SIZE = 6000
TEST_NOMINAL_SIZE = 1000
TEST_ANOAMLOUS_SIZE = 1000


torch.manual_seed(160)

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


for i in range(0, 1):
    train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=i, train=True), list(range(TRAIN_SIZE)))
    test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=i, train=False), list(range(TEST_NOMINAL_SIZE)))
    test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=i, train=False), list(range(TEST_ANOAMLOUS_SIZE)))

    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal)
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    for test_dataloader, type in [(test_dataloader_nominal, 'Nominal'), (test_dataloader_anomalous, 'Anomalous')]:
        soft_tukey_depths = []

        def soft_tukey_depth(x, x_, z):
            return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))

        for item, x in enumerate(test_dataloader):
            print(f'Item {item}/{len(test_dataloader)}')
            x = x.to(device)
            min_tukey_depth = torch.tensor(100_000_000)

            for j in range(TUKEY_DEPTH_COMPUTATIONS):
                z = torch.nn.Parameter(torch.rand(x.size(dim=1), device=device) / torch.tensor(TRAIN_SIZE))
                optimizer = torch.optim.SGD([z], lr=1e-4)

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

            soft_tukey_depths.append(min_tukey_depth.item() / TRAIN_SIZE)
            print(f'Soft tukey depth is {min_tukey_depth.item() / TRAIN_SIZE}')

        print(soft_tukey_depths)

        writer = csv.writer(open(f'./results/raw/soft_tukey_depths_{DATASET_NAME}_{type}_TDAE_{i}.csv', 'w'))
        writer.writerow(soft_tukey_depths)


