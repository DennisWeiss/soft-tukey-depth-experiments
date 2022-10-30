from tqdm import tqdm
from models.RAE_MNIST import RAE_MNIST
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt


NOMINAL_CLASS = 0
USE_CUDA_IF_AVAILABLE = True


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

data = torchvision.datasets.MNIST(
            'datasets',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop(33)])
        )

dataloader = torch.utils.data.DataLoader(data)
autoencoder = RAE_MNIST()
autoencoder.to(device)
autoencoder.load_state_dict(torch.load(f'./snapshots/RAE_MNIST_2_{NOMINAL_CLASS}'))
autoencoder.eval()

x = []
y = []
colors = []

count = 0

for sample, _class in tqdm(dataloader, desc=f'Calculating latent space positions', unit='batch', colour='blue'):
    count += 1
    if count > 1000:
        break
    sample = sample.to(device)
    z, reconstruction = autoencoder(sample)
    x.append(z[0][0].cpu().detach().numpy())
    y.append(z[0][1].cpu().detach().numpy())
    colors.append('#0000ff' if _class == NOMINAL_CLASS else '#ff0000')

plt.scatter(x, y, c=colors, marker='X')
plt.show()