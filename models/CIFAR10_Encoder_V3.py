import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_Encoder_V3(nn.Module):
    def __init__(self):
        super(CIFAR10_Encoder_V3, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 512)
        )

    def forward(self, x):
        return self.encoder(x)