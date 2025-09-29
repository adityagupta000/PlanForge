"""
Encoder architecture for multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block for the encoder"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MultiScaleEncoder(nn.Module):
    """
    Advanced encoder with skip connections and multi-scale feature extraction
    Based on ResNet architecture with Feature Pyramid Network (FPN)
    """

    def __init__(self, input_channels=3, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 64x64
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 32x32
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 16x16
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 8x8

        # FPN lateral connections
        self.lateral4 = nn.Conv2d(512, feature_dim, 1)
        self.lateral3 = nn.Conv2d(256, feature_dim, 1)
        self.lateral2 = nn.Conv2d(128, feature_dim, 1)
        self.lateral1 = nn.Conv2d(64, feature_dim, 1)

        # FPN output layers
        self.smooth4 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        self.smooth3 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        self.smooth2 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        self.smooth1 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(512, feature_dim), 
            nn.ReLU(), 
            nn.Linear(feature_dim, feature_dim)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up pathway
        x = self.stem(x)  # 64x64

        c1 = self.layer1(x)  # 64x64
        c2 = self.layer2(c1)  # 32x32
        c3 = self.layer3(c2)  # 16x16
        c4 = self.layer4(c3)  # 8x8

        # Global context
        global_feat = self.global_pool(c4).flatten(1)
        global_feat = self.global_fc(global_feat)

        # Top-down pathway (FPN)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2)
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2)

        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        return {
            "p1": p1,  # 64x64
            "p2": p2,  # 32x32
            "p3": p3,  # 16x16
            "p4": p4,  # 8x8
            "global": global_feat,
        }