import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional


class AE_CIFAR10_V5(nn.Module):
    def __init__(self):
        super(AE_CIFAR10_V5, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.ReLU(),
			nn.Conv2d(48, 96, 3, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.BatchNorm2d(96),
			nn.Conv2d(96, 96, 3, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 512)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 96 * 4 * 4),
            nn.Unflatten(dim=1, unflattened_size=(96, 4, 4)),
			nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(96),
			nn.ConvTranspose2d(96, 48, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(48),
            nn.ConvTranspose2d(48, 24, 3, stride=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 24, 3, stride=2),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ConvTranspose2d(24, 3, 3, stride=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = torchvision.transforms.functional.crop(self.decoder(encoded), 0, 0, 32, 32)
        return encoded, decoded