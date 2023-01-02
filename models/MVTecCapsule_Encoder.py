import torch
import torch.nn as nn
import torch.nn.functional as F


class MVTecCapsule_Encoder(nn.Module):
    def __init__(self):
        super(MVTecCapsule_Encoder, self).__init__()

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

    def forward(self, x):
        return self.encoder(x)