import torch.nn as nn
import torch.nn.functional as F


class MNIST_Encoder_Simple(nn.Module):
    def __init__(self):
        super(MNIST_Encoder_Simple, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4))
        self.bn1 = nn.BatchNorm2d(16, eps=1e-4, affine=False)
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 4))
        self.bn2 = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(8 * 4 * 4, 64)

    def forward(self, x):
        layer1 = self.pool(F.leaky_relu(self.bn1(self.conv_layer1(x))))
        layer2 = self.pool(F.leaky_relu(self.bn2(self.conv_layer2(layer1))))
        encoding = self.encoding_layer(self.flatten_layer(layer2))
        return encoding