"""
Multi-task prediction heads for the 3D Model Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Semantic segmentation head with multi-scale fusion"""

    def __init__(self, feature_dim=512, num_classes=5, dropout=0.1):
        super().__init__()

        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, num_classes, 1),
        )

    def forward(self, features):
        # Fuse multi-scale features
        p1, p2, p3, p4 = features["p1"], features["p2"], features["p3"], features["p4"]

        # Upsample all to p1 resolution
        p2_up = F.interpolate(
            p2, size=p1.shape[-2:], mode="bilinear", align_corners=False
        )
        p3_up = F.interpolate(
            p3, size=p1.shape[-2:], mode="bilinear", align_corners=False
        )
        p4_up = F.interpolate(
            p4, size=p1.shape[-2:], mode="bilinear", align_corners=False
        )

        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused = self.fusion(fused)

        # Final segmentation
        seg = self.decoder(fused)
        return F.interpolate(seg, scale_factor=4, mode="bilinear", align_corners=False)


class AttributeHead(nn.Module):
    """Attribute regression head for geometric parameters"""

    def __init__(self, feature_dim=512, num_attributes=6, dropout=0.2):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_attributes),
            nn.Sigmoid(),  # Output in [0,1] range
        )

    def forward(self, global_features):
        return self.regressor(global_features)


class SDFHead(nn.Module):
    """Signed Distance Field prediction for sharp boundaries"""

    def __init__(self, feature_dim=512, dropout=0.1):
        super().__init__()

        self.sdf_decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Tanh(),  # SDF in [-1, 1]
        )

    def forward(self, features):
        # Use highest resolution features
        p1 = features["p1"]
        sdf = self.sdf_decoder(p1)
        return F.interpolate(sdf, scale_factor=4, mode="bilinear", align_corners=False)