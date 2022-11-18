import torch
import torch.nn as nn
import torch.nn.functional as F


class MVTec_AE(nn.Module):
    def __init__(self):
        super(MVTec_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 6, stride=3),
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(24, 12, 6, stride=3),
            nn.BatchNorm2d(12, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(12, 6, 6, stride=3),
            nn.BatchNorm2d(6, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7 * 7, 32)
        )

        self.decoder =  nn.Sequential(
            nn.Linear(32, 6 * 7 * 7),
            nn.Unflatten(dim=1, unflattened_size=(6, 7, 7)),
            nn.ConvTranspose2d(6, 12, 4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 24, 4, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded