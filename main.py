import torch.utils.data
import csv

from DataLoader import NominalMNISTDataset, AnomalousMNISTDataset, NominalCIFAR10Dataset, AnomalousCIFAR10Dataset, \
    NominalCIFAR10GrayscaleDataset, AnomalousCIFAR10GrayscaleDataset, Cellular4GDataset, ToyDataset, \
    NominalCIFAR10ImageDataset, AnomalousCIFAR10ImageDataset, NominalCIFAR10AutoencoderDataset, \
    AnomalousCIFAR10AutoencoderDataset, NominalMNISTAutoencoderDataset, AnomalousMNISTAutoencoderDataset, \
    NominalMNISTAutoencoderAllDataset, AnomalousMNISTAutoencoderAllDataset, NominalMNISTAutoencoderCachedDataset, \
    AnomalousMNISTAutoencoderCachedDataset, NominalMNISTImageDataset, AnomalousMNISTImageDataset
from models.RAE_CIFAR10 import RAE_CIFAR10


USE_CUDA_IF_AVAILABLE = True
DATASET_NAME = 'CIFAR10_Autoencoder'
NOMINAL_DATASET = NominalCIFAR10AutoencoderDataset
ANOMALOUS_DATASET = AnomalousCIFAR10AutoencoderDataset
N_CLASSES = 10
TUKEY_DEPTH_COMPUTATION_EPOCHS = 10
TUKEY_DEPTH_COMPUTATIONS = 1
SOFT_TUKEY_DEPTH_TEMP = 1
BATCH_SIZE = 64
TRAIN_SIZE = 5000
TEST_NOMINAL_SIZE = 1000
TEST_ANOAMLOUS_SIZE = 9000


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def soft_tukey_depth_v2(X_, X, Z, temp):
    X_new = X.repeat(X_.size(dim=0), 1, 1)
    X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
    dot_products_normalized = dot_products.transpose(0, 1).divide(temp * Z.norm(dim=1))
    return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))


for i in range(9, 10):
    train_data = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=i, train=True), list(range(TRAIN_SIZE)))
    test_data_nominal = torch.utils.data.Subset(NOMINAL_DATASET(nominal_class=i, train=False), list(range(TEST_NOMINAL_SIZE)))
    test_data_anomalous = torch.utils.data.Subset(ANOMALOUS_DATASET(nominal_class=i, train=False), list(range(TEST_ANOAMLOUS_SIZE)))

    test_dataloader_nominal = torch.utils.data.DataLoader(test_data_nominal, batch_size=BATCH_SIZE)
    test_dataloader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=BATCH_SIZE)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_SIZE)

    for test_dataloader, type in [(test_dataloader_nominal, 'Nominal'), (test_dataloader_anomalous, 'Anomalous')]:
        soft_tukey_depths = []

        def soft_tukey_depth(x, x_, z):
            return torch.sum(torch.sigmoid(torch.multiply(torch.tensor(1 / SOFT_TUKEY_DEPTH_TEMP), torch.divide(torch.matmul(torch.subtract(x_, torch.matmul(torch.ones((x_.size(dim=0), 1), device=device), x)), z), torch.norm(z)))))


        for item, x in enumerate(test_dataloader):
            x = x.to(device)
            x_detached = x.detach()
            min_tukey_depth = torch.ones(x.size(dim=0), device=device)

            for item2, x2 in enumerate(train_dataloader):
                x2 = x2.to(device)
                x2_detached = x2.detach()
                for j in range(TUKEY_DEPTH_COMPUTATIONS):
                    z = torch.nn.Parameter(torch.rand(x.size(dim=0), x.size(dim=1), device=device))
                    optimizer = torch.optim.SGD([z], lr=300)
                    for k in range(TUKEY_DEPTH_COMPUTATION_EPOCHS):
                        _soft_tukey_depth = soft_tukey_depth_v2(x_detached, x2_detached, z, SOFT_TUKEY_DEPTH_TEMP)
                        _soft_tukey_depth.sum().backward()
                        optimizer.step()
                        del _soft_tukey_depth


                    _soft_tukey_depth = soft_tukey_depth_v2(x_detached, x2_detached, z, SOFT_TUKEY_DEPTH_TEMP)
                    min_tukey_depth = torch.minimum(min_tukey_depth, _soft_tukey_depth)
                    del z
                    del _soft_tukey_depth
                    torch.cuda.empty_cache()


            print(f'Mean TD is {min_tukey_depth.mean().item()}')

            for j in range(x.size(dim=0)):
                print(f"Item {item * BATCH_SIZE + j + 1}/{TEST_NOMINAL_SIZE if type == 'Nominal' else TEST_ANOAMLOUS_SIZE}")
                soft_tukey_depths.append(min_tukey_depth[j].item())
                print(f'Soft tukey depth is {min_tukey_depth[j].item()}')

        print(soft_tukey_depths)

        writer = csv.writer(open(f'./results/raw/soft_tukey_depths_{DATASET_NAME}_{type}_temp1_fulldata_{i}.csv', 'w'))
        writer.writerow(soft_tukey_depths)


