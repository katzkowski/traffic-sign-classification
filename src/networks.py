import torch
from torch import nn


class TrafficSignsClassifier(nn.Module):
    def __init__(self, hparams, input_size, num_classes) -> None:
        super(TrafficSignsClassifier, self).__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(input_size ** 2 * 3, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
