import os
from pathlib import Path

import torch
from torch import nn

os.environ.setdefault("TORCH_HOME", str(Path("checkpoints/torch_cache").resolve()))

from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet18UNet(nn.Module):
    """Small U-Net decoder on top of an ImageNet-pretrained ResNet18 encoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std

        s1 = self.stem(x_norm)
        s2 = self.enc1(self.pool(s1))
        s3 = self.enc2(s2)
        s4 = self.enc3(s3)
        bottleneck = self.enc4(s4)

        x = self.up4(bottleneck, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.final_up(x)


def build_model(pretrained: bool = True) -> nn.Module:
    return ResNet18UNet(pretrained=pretrained)
