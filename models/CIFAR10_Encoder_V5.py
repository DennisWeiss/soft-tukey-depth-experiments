import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_Encoder_V5(nn.Module):
    def __init__(self):
        super(CIFAR10_Encoder_V5, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]


        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 6, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(24, 12, 6, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(12, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(12, 12, 6, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.BatchNorm2d(12, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12 * 6 * 6, 128)
        )

    def forward(self, x):
        return self.encoder(x)