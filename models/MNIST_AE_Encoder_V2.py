import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_AE_Encoder_V2(nn.Module):
    def __init__(self):
        super(MNIST_AE_Encoder_V2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        return self.encoder(x)