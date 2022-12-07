import torch
import torch.nn as nn
import torch.nn.functional as F


class AE_CIFAR10_V4(nn.Module):
    def __init__(self):
        super(AE_CIFAR10_V4, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 6, stride=2, padding=2),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(24, 36, 6, stride=2, padding=2),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(36, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(36, 48, 6, stride=2, padding=2),  # [batch, 48, 4, 4]
            nn.BatchNorm2d(48, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 48 * 4 * 4),
            nn.Unflatten(dim=1, unflattened_size=(48, 4, 4)),
			nn.ConvTranspose2d(48, 36, 6, stride=2, padding=2),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(36, 24, 6, stride=2, padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 6, stride=2, padding=2),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded