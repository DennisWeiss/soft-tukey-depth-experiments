import torch.nn as nn
import torch.nn.functional as F


class MNIST_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Encoder, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2, 2), stride=(2, 2))
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(1024, 2)

    def forward(self, x):
        layer1 = F.relu(self.conv_layer1(x))
        layer2 = F.relu(self.conv_layer2(layer1))
        layer3 = F.relu(self.conv_layer3(layer2))
        layer4 = F.relu(self.conv_layer4(layer3))
        encoding = self.encoding_layer(self.flatten_layer(layer4))
        return encoding
