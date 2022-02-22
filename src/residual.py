import torch.nn.functional as F
from torch import nn


class Identity(nn.Module):
    """Identity function for skip connection. The padding approach which doesn't introduce additional parameters is implemented. If dimensionality is increased within the residual block, the missing dimensions in the skip connection are padded with zeros."""

    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        if stride > 1 or in_channels != out_channels:
            # if dimensions reduced, pad identity to mtach dimensions
            self.padding = lambda x: F.pad(
                x[:, :, ::stride, ::stride],  # select every stride-th value
                (0, 0, 0, 0, 0, out_channels - in_channels),  # pad back dims. only
                mode="constant",
                value=0,
            )
        else:
            # return identity with same dimensions otherwise
            self.padding = lambda x: x

    def forward(self, x):
        return self.padding(x)


class ResidualBlock(nn.Module):
    """Residual block which performs two convolutions, each followed by a batch normalization layer and ReLU activation. The skip connections adds the identity of the block's input before the second ReLU is applied.

    Implementation after: https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=0
    ) -> None:
        super().__init__()

        # skip connection
        self.skip = Identity(in_channels, out_channels, stride=stride)

        # residual block, only first convolution can reduce dimensions
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # add skip connection to block ouput and apply relu
        return F.relu(self.block(x) + self.skip(x))
