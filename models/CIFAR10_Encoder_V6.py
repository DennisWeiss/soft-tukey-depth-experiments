import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AE_CIFAR10_V6 import AE_CIFAR10_V6


class CIFAR10_Encoder_V6(nn.Module):
    def __init__(self):
        super(CIFAR10_Encoder_V6, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=1, padding=1),  # [batch, 48, 4, 4]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, 3, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 32)
        )

    def forward(self, x):
        return self.encoder(x)

    def load_weights_from_pretrained_autoencoder(self, autoencoder: AE_CIFAR10_V6):
        self.encoder.load_state_dict(autoencoder.encoder.state_dict())