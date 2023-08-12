import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, needs_flattened):
        super().__init__()
        self.needs_flattened = needs_flattened
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        if self.needs_flattened:
            x = torch.flatten(x, start_dim=1)

        logits = self.model(x)
        return logits

