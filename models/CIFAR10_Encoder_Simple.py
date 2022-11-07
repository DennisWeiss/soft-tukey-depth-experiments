import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_Encoder_Simple(nn.Module):
    def __init__(self):
        super(CIFAR10_Encoder_Simple, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4, affine=False)
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(800, 128)

    def forward(self, x):
        layer1 = self.pool(F.leaky_relu(self.bn1(self.conv_layer1(x))))
        layer2 = self.pool(F.leaky_relu(self.bn2(self.conv_layer2(layer1))))
        encoding = self.encoding_layer(self.flatten_layer(layer2))
        return encoding