import torch
import torch.nn as nn


class Wasserstein_Network(nn.Module):
    def __init__(self):
        super(Wasserstein_Network, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)