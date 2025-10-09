"""
Multi-task prediction heads for the Neural-Geometric 3D Model Generator
OPTIMIZED FOR 48GB GPU - Enhanced capacity and throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Semantic segmentation head with multi-scale fusion - GPU-optimized"""

    def __init__(self, feature_dim=768, num_classes=5, dropout=0.15):
        """
        Args:
            feature_dim: Increased from 512 to 768 for better representation
            num_classes: Number of segmentation classes
            dropout: Slightly higher dropout for better regularization with larger model
        """
        super().__init__()

        # Multi-scale fusion with increased capacity
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 2, 3, 1, 1),  # Wider intermediate
            nn.GroupNorm(32, feature_dim * 2),  # GroupNorm for larger batch stability
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, 1, 1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Enhanced segmentation decoder with residual connections
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
                nn.GroupNorm(32, feature_dim // 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, 1, 1),
                nn.GroupNorm(32, feature_dim // 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
                nn.GroupNorm(16, feature_dim // 4),
                nn.ReLU(inplace=True),
            ),
            nn.Conv2d(feature_dim // 4, num_classes, 1),
        ])
        
        # Skip connection projection
        self.skip_proj = nn.Conv2d(feature_dim, feature_dim // 4, 1)

    def forward(self, features):
        # Fuse multi-scale features
        p1, p2, p3, p4 = features["p1"], features["p2"], features["p3"], features["p4"]

        # Upsample all to p1 resolution with better interpolation
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
        
        # Save for skip connection
        skip = self.skip_proj(fused)

        # Decoder with progressive refinement
        x = fused
        for i, layer in enumerate(self.decoder[:-1]):
            x = layer(x)
        
        # Add skip connection before final conv
        x = x + skip
        seg = self.decoder[-1](x)
        
        # Upsample to original resolution
        seg=torch.clamp(seg, min=-10.0, max=10.0)
        return F.interpolate(seg, scale_factor=4, mode="bilinear", align_corners=False)


class AttributeHead(nn.Module):
    """Attribute regression head - Enhanced for high-capacity training"""

    def __init__(self, feature_dim=768, num_attributes=6, dropout=0.2):
        """
        Args:
            feature_dim: Increased from 512 to 768
            num_attributes: Number of geometric attributes to predict
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # Deeper, wider MLP for better attribute prediction
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),  # Expand
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim * 2, feature_dim * 2),  # Additional layer
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim * 2, feature_dim),  # Contract
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Lower dropout near output
            
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(feature_dim // 2, num_attributes),
            nn.Sigmoid(),  # Output in [0,1] range
        )

    def forward(self, global_features):
        return self.regressor(global_features)


class SDFHead(nn.Module):
    """Signed Distance Field prediction - Enhanced resolution"""

    def __init__(self, feature_dim=768, dropout=0.15):
        """
        Args:
            feature_dim: Increased from 512 to 768
            dropout: Dropout rate
        """
        super().__init__()

        # Enhanced SDF decoder with attention-like refinement
        self.sdf_decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.GroupNorm(32, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, 1, 1),
            nn.GroupNorm(32, feature_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
            nn.GroupNorm(16, feature_dim // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim // 4, feature_dim // 8, 3, 1, 1),
            nn.GroupNorm(8, feature_dim // 8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim // 8, 1, 1),
            nn.Tanh(),  # SDF in [-1, 1]
        )
        
        # Refinement module for sharp boundaries
        self.boundary_refiner = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def forward(self, features):
        # Use highest resolution features
        p1 = features["p1"]
        sdf = self.sdf_decoder(p1)
        
        # Upsample before refinement for better detail
        sdf = F.interpolate(sdf, scale_factor=4, mode="bilinear", align_corners=False)
        
        # Refine boundaries
        refined = self.boundary_refiner(sdf)
        sdf = torch.tanh(sdf + 0.1 * refined)  # Residual refinement
        
        return sdf