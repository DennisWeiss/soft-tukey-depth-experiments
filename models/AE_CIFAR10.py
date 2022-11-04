import torch
import torch.nn as nn
import torch.nn.functional as F


class AE_CIFAR10(nn.Module):
    def __init__(self):
        super(AE_CIFAR10, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4, affine=False)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(16, eps=1e-4, affine=False)
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(16 * 5 * 5, 50)

        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(2, 5, 5))
        self.convT_layer1 = nn.ConvTranspose2d(in_channels=2, out_channels=16, kernel_size=(5, 5), padding=2)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-4)
        self.convT_layer2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=4)
        self.bn4 = nn.BatchNorm2d(32, eps=1e-4)
        self.convT_layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5), padding=2)

    def encoder(self, x):
        layer1 = self.pool(F.leaky_relu(self.bn1(self.conv_layer1(x))))
        layer2 = self.pool(F.leaky_relu(self.bn2(self.conv_layer2(layer1))))
        encoding = self.encoding_layer(self.flatten_layer(layer2))
        return encoding

    def decoder(self, z):
        layer1 = F.interpolate(F.leaky_relu(self.unflatten_layer(z)), scale_factor=2)
        layer2 = F.interpolate(F.leaky_relu(self.bn3(self.convT_layer1(layer1))), scale_factor=2)
        layer3 = F.interpolate(F.leaky_relu(self.bn4(self.convT_layer2(layer2))), scale_factor=2)
        output = torch.sigmoid(self.convT_layer3(layer3))
        return output

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)