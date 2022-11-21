import torch
import torchvision
import torch.utils.data
from models.AE_MNIST_V3 import AE_MNIST_V3
from preprocessing import get_target_label_idx, global_contrast_normalization


USE_CUDA_IF_AVAILABLE = True

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

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


for nominal_class in range(0, 10):
    data = torchvision.datasets.MNIST(
        'datasets',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Lambda(
                                                      lambda x: global_contrast_normalization(x, scale='l1')),
                                                  torchvision.transforms.Normalize(
                                                      [min_max_mnist[nominal_class][0]],
                                                      [min_max_mnist[nominal_class][1] -
                                                       min_max_mnist[nominal_class][0]])])
    )

    data_latent = torch.zeros(len(data), 1, 128)

    dataloader = torch.utils.data.DataLoader(data)
    autoencoder = AE_MNIST_V3().to(device)
    autoencoder.load_state_dict(torch.load(f'./snapshots/AE_MNIST_V3_{nominal_class}'))
    autoencoder.eval()

    for step, x in enumerate(dataloader):
        x[0] = x[0].to(device)
        z, x_hat = autoencoder(x[0])
        data_latent[step][0] = z.detach()

    torch.save(data_latent, f'./datasets/MNIST_AE_representation/AE_MNIST_V3_{nominal_class}')