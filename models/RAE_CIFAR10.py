import torch.nn as nn
import torch.nn.functional as F


class RAE_CIFAR10(nn.Module):
    def __init__(self):
        super(RAE_CIFAR10, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5))
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(1024, 128)

        self.decoding_layer = nn.Linear(128, 2 * 2 * 256)
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(256, 2, 2))
        self.bn5 = nn.BatchNorm2d(256)
        self.convT_layer1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5))
        self.bn6 = nn.BatchNorm2d(128)
        self.convT_layer2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5))
        self.bn7 = nn.BatchNorm2d(64)
        self.convT_layer3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5))
        self.bn8 = nn.BatchNorm2d(32)
        self.convT_layer4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(6, 6), stride=(2, 2))

    def encoder(self, x):
        layer1 = F.relu(self.bn1(self.conv_layer1(x)))
        layer2 = F.relu(self.bn2(self.conv_layer2(layer1)))
        layer3 = F.relu(self.bn3(self.conv_layer3(layer2)))
        layer4 = F.relu(self.bn4(self.conv_layer4(layer3)))
        encoding = F.relu(self.encoding_layer(self.flatten_layer(layer4)))
        return encoding

    def decoder(self, z):
        layer1 = F.relu(self.bn5(self.unflatten_layer(self.decoding_layer(z))))
        layer2 = F.relu(self.bn6(self.convT_layer1(layer1)))
        layer3 = F.relu(self.bn7(self.convT_layer2(layer2)))
        layer4 = F.relu(self.bn8(self.convT_layer3(layer3)))
        output = self.convT_layer4(layer4)
        return output

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)