import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_AE_Encoder(nn.Module):
    def __init__(self):
        super(CIFAR10_AE_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.encoder(x)