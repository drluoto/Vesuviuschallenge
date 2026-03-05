"""Ink detection models.

Provides a U-Net based architecture using segmentation_models_pytorch,
adapted to work with multi-channel 3D surface volume input.
Also supports a lighter CNN for M2/MPS development.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class InkDetectorUNet(nn.Module):
    """U-Net ink detector using a pretrained encoder.

    Adapts a 2D segmentation model to handle multi-layer CT input
    by using a 1x1 conv to project from N z-layers to 3 channels
    (matching pretrained encoder expectations).
    """

    def __init__(
        self,
        in_channels: int = 30,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
    ):
        super().__init__()
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        return self.unet(x)


class InkDetectorLite(nn.Module):
    """Lightweight ink detector for development on constrained hardware.

    Simple encoder-decoder with depthwise separable convolutions.
    Good for quick iteration on M2 Macs with 8GB RAM.
    """

    def __init__(self, in_channels: int = 30):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
