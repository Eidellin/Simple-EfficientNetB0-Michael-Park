import torch
import torch.nn as nn
try:
    from .MBConv import MBConv
except ImportError:
    from MBConv import MBConv

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 model

    Args:
        num_classes (int): Number of output classes.
        stochastic_depth_prob (float): Stochastic depth probability.
    """
    def __init__(self, num_classes: int, stochastic_depth_prob: float=0.8):
        super(EfficientNetB0, self).__init__()

        # (out_channels, expansion_factor, kernel_size, stride, repeats)
        config: list[tuple[int, int, int, int]] = [
            (16, 1, 3, 1, 1),
            (24, 6, 3, 2, 2),
            (40, 6, 5, 2, 2),
            (80, 6, 3, 2, 3),
            (112, 6, 5, 1, 3),
            (192, 6, 5, 2, 4),
            (320, 6, 3, 1, 1),
        ]

        # 1. Initial Convolution Layer (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # 2. MBConv Blocks
        blocks = []
        in_channels = 32
        block_idx = 0
        for out_channels, expand_ratio, kernel_size, stride, repeats in config:
            for i in range(repeats):
                current_stride = stride if i == 0 else 1
                blocks.append(MBConv(in_channels, out_channels, kernel_size, current_stride, expand_ratio, survival_prob=stochastic_depth_prob))
                in_channels = out_channels
                block_idx += 1

        self.main_blocks = nn.Sequential(*blocks)

        # 3. Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.stem(x)
        x = self.main_blocks(x)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    model = EfficientNetB0(num_classes=2, stochastic_depth_prob=0.2)
    print(model)