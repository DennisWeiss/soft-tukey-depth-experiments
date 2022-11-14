import torch
import matplotlib.pyplot as plt
from models.AE_MNIST import AE_MNIST
from models.MNIST_Encoder_Simple import MNIST_Encoder_Simple
from models.CIFAR10_Encoder_V4 import CIFAR10_Encoder_V4


USE_CUDA_IF_AVAILABLE = True


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


model = CIFAR10_Encoder_V4()

model.load_state_dict(torch.load(f'snapshots/CIFAR10_Encoder_temp2_2', map_location=device))


def show_filters(conv_layer):
    for i in range(conv_layer.out_channels):
        for j in range(conv_layer.in_channels):
            plt.imshow(conv_layer.weight[i, j].detach().numpy())
            plt.show()


show_filters(model.encoder[0])