import torch
from torch import nn

from residual import ResidualBlock


class LinearLayerClassifier(nn.Module):
    """Single linear layer, without non-linearities."""

    def __init__(self, hparams, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(nn.Flatten(), nn.Linear(32 ** 2 * 3, num_classes))

    def forward(self, x):
        logits = self.model(x)
        return logits


class FullyConnectedClassifier(nn.Module):
    """Fully connected neural network, using two linear layers, ReLU actiovation and batch normalization"""

    def __init__(self, hparams, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 ** 2 * 3, num_classes * 2),
            nn.BatchNorm1d(num_features=num_classes * 2),
            nn.ReLU(),
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
    """CNN with two convolutional layers for feature extraction, each one followed by a batch normalization layer and ReLU activation. Two fully connected layers for classification, with batch normalization and ReLU activation inbetween."""

    def __init__(self, hparams, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            # 32 x 32 x 3
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            # 30 x 30 x 8
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 15 x 15 x 8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            # 13 x 13 x 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 6 x 6 x 16
            nn.Flatten(),
            # 576
            nn.Linear(576, num_classes * 2),
            nn.BatchNorm1d(num_features=num_classes * 2),
            nn.ReLU(),
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


class ResidualCNN(nn.Module):
    """CNN in the style of ResNet with residual blocks using skip connections, ReLU and batch normalization. Two fully connected layers for classification, with batch normalization and ReLU activation inbetween."""

    def __init__(self, hparams, num_classes) -> None:
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            # 32 x 32 x 3
            ResidualBlock(3, 8, kernel_size=3, stride=2, padding=1),
            # 16 x 16 x 8
            ResidualBlock(8, 16, kernel_size=3, stride=2, padding=1),
            # 4 x 4 x 16
            nn.Flatten(),
            # 8 * 8 * 16 = 1024
            nn.Linear(1024, num_classes * 2),
            nn.BatchNorm1d(num_features=num_classes * 2),
            nn.ReLU(),
            nn.Linear(num_classes * 2, num_classes),
        )

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


class TrafficSignsClassifier(ResidualCNN):
    def __init__(self, hparams, num_classes) -> None:
        super().__init__(hparams, num_classes)

    def forward(self, x):
        return super().forward(x)
