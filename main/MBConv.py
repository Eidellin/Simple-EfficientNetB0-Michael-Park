import torch
import numpy as np
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

class MBConv(nn.Module):
    """
    MBConv block for EfficientNet-B0

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layers.
        expand_ratio (int): Expansion ratio for the bottleneck.
        drop_out (float): Dropout rate.
        survival_prob (float): Survival probability for stochastic depth.
        se_ratio (float): Squeeze-and-Excitation ratio.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, expand_ratio: int, drop_out: float = 0.2, survival_prob: float = 0.8, se_ratio: float = 0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        # Survival probability for stochastic depth.
        self.survival_prob = survival_prob

        # Residual Connection Possibility
        self.residual = self.stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expand_ratio

        layers = []

        # 1. Expansion Layer (1x1 Conv)
        if expand_ratio > 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])

        # 2. Depthwise Convolution (3x3 Conv)
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])

        # 3. Squeeze-and-Excitation Layer
        if self.se_ratio is not None:
            layers.append(SqueezeExcitation(input_channels=expanded_channels, squeeze_channels=int(expanded_channels * se_ratio)))

        # 4. Projection Layer (1x1 Conv)
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_out)  # Dropout layer with a rate of 0.2
        ])

        self.block = nn.Sequential(*layers)

        # self.stochastic_depth = StochasticDepth(p=self.survival_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        if self.residual:
            p = self.survival_prob
            if np.random.choice([0, 1], p=[1-p, p]) == 1:
                out = self.block(x) + x
            else:
                out = x
        else:
            out = self.block(x)

        return out

if __name__ == "__main__":
    model = MBConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, expand_ratio=6)
    print(model)