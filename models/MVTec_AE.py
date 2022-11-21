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
            nn.Conv2d(24, 24, 6, stride=3),
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Conv2d(24, 48, 6, stride=3),
            nn.BatchNorm2d(48, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 128)
        )

        self.decoder =  nn.Sequential(
            nn.Linear(128, 48 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(48, 7, 7)),
            nn.ConvTranspose2d(48, 24, 4, stride=3),
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 24, 4, stride=3, padding=2),
            nn.BatchNorm2d(24, eps=1e-4, affine=False),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=1),
            nn.BatchNorm2d(3, eps=1e-4, affine=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded