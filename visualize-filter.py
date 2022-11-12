import torch
import matplotlib.pyplot as plt
from models.AE_MNIST import AE_MNIST
from models.MNIST_Encoder_Simple import MNIST_Encoder_Simple
from models.CIFAR10_Encoder_V4 import CIFAR10_Encoder_V4


model = MNIST_Encoder_Simple()

model.load_state_dict(torch.load(f'snapshots/MNIST_Encoder_9'))


def show_filters(conv_layer):
    for i in range(conv_layer.out_channels):
        for j in range(conv_layer.in_channels):
            plt.imshow(conv_layer.weight[i, j].detach().numpy())
            plt.show()


show_filters(model.conv_layer2)