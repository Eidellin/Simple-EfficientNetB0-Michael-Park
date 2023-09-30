import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=None):
        super().__init__()
        
        expanded_channels = int(in_channels * expand_ratio)
        
        layers = []
        # Pointwise Convolution (1x1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise Convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                padding=kernel_size // 2, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU6(inplace=True))
        
        # Squeeze-and-Excitation (SE) block (optional)
        if se_ratio is not None:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Conv2d(expanded_channels, se_channels, kernel_size=1, bias=True))
            layers.append(nn.ReLU6(inplace=True))
            layers.append(nn.Conv2d(se_channels, expanded_channels, kernel_size=1, bias=True))
            layers.append(nn.Sigmoid())
        
        # Pointwise Convolution (1x1)
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the architecture of EfficientNetB0
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.mbconv1 = MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1)
        self.mbconv2 = MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2)
        self.mbconv3 = MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2)
        self.mbconv4 = MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=2)
        self.mbconv5 = MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1)
        self.mbconv6 = MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2)
        self.mbconv7 = MBConvBlock(192, 320, expand_ratio=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 2)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x