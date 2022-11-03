import torch.nn as nn
import torch.nn.functional as F


class MNIST_Encoder_Simple(nn.Module):
    def __init__(self):
        super(MNIST_Encoder_Simple, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2))
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(4608, 32)

    def forward(self, x):
        layer1 = F.relu(self.conv_layer1(x))
        layer2 = F.relu(self.conv_layer2(layer1))
        encoding = self.encoding_layer(self.flatten_layer(layer2))
        return encoding