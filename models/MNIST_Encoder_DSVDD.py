import torch.nn as nn
import torch.nn.functional as F


class MNIST_Encoder_DSVDD(nn.Module):
    def __init__(self):
        super(MNIST_Encoder_DSVDD, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5))
        self.max_pool_layer1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5))
        self.max_pool_layer2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(100, 32)

    def forward(self, x):
        layer1 = F.leaky_relu(self.max_pool_layer1(self.conv_layer1(x)), negative_slope=0.1)
        layer2 = F.leaky_relu(self.max_pool_layer2(self.conv_layer2(layer1)), negative_slope=0.1)
        encoding = self.encoding_layer(self.flatten_layer(layer2))
        return encoding