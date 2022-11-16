import torch
import torch.nn as nn
import torch.nn.functional as F


class MVTecCapsule_Encoder(nn.Module):
    def __init__(self):
        super(MVTecCapsule_Encoder, self).__init__()

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

    def forward(self, x):
        return self.encoder(x)