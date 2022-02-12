import torch
from torch import nn


class LinearLayerClassifier(nn.Module):
    def __init__(self, hparams, input_size, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(input_size ** 2 * 3, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits


class FullyConnectedClassifier(nn.Module):
    def __init__(self, hparams, input_size, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size ** 2 * 3, num_classes * 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=num_classes * 2),
            nn.Linear(num_classes * 2, num_classes),
        )

        self.init_weights()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def init_weights(self):
        for module in self.model:
            # linear layers
            if isinstance(module, nn.Linear):
                # use He initialization
                torch.nn.init.kaiming_uniform_(module.weight)

                # slightly positive bias to avoid dying ReLU
                module.bias.data.fill_(0.01)

        print(f"Weights initialized")


class ConvolutionalNN(nn.Module):
    def __init__(self, hparams, input_size, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            # 32 x 32 x 3
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            # 30 x 30 x 8
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 15 x 15 x 8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            # 13 x 13 x 16
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 6 x 6 x 16
            nn.Flatten(),
            # 576
            nn.Linear(576, num_classes * 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=num_classes * 2),
            nn.Linear(num_classes * 2, num_classes),
        )

        self.init_weights()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def init_weights(self):
        for module in self.model:
            # linear layers
            if isinstance(module, nn.Linear):
                # use He initialization
                torch.nn.init.kaiming_uniform_(module.weight)

                # slightly positive bias to avoid dying ReLU
                module.bias.data.fill_(0.01)

        print(f"Weights initialized")


class TrafficSignsClassifier(ConvolutionalNN):
    def __init__(self, hparams, input_size, num_classes) -> None:
        super().__init__(hparams, input_size, num_classes)

    def forward(self, x):
        return super().forward(x)
