import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_Encoder_V4(nn.Module):
    def __init__(self):
        super(CIFAR10_Encoder_V4, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]


        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(12, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.BatchNorm2d(48, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 256)
        )

    def forward(self, x):
        return self.encoder(x)