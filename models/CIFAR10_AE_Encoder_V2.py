import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_AE_Encoder_V2(nn.Module):
    def __init__(self):
        super(CIFAR10_AE_Encoder_V2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.encoder(x)