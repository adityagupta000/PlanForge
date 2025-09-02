"""
Main model definition combining all components
"""

import torch
import torch.nn as nn

from .encoder import MultiScaleEncoder
from .heads import SegmentationHead, AttributeHead, SDFHead
from .dvx import DifferentiableVectorization
from .extrusion import DifferentiableExtrusion


class NeuralGeometric3DGenerator(nn.Module):
    """
    Complete neural-geometric system for 2D to 3D floorplan generation
    Following the research-grade approach with DVX and multi-task supervision
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=5,
        feature_dim=512,
        num_attributes=6,
        voxel_size=64,
        max_polygons=20,
        max_points=50,
    ):
        super().__init__()

        # Encoder backbone
        self.encoder = MultiScaleEncoder(input_channels, feature_dim)

        # Multi-task prediction heads
        self.seg_head = SegmentationHead(feature_dim, num_classes)
        self.attr_head = AttributeHead(feature_dim, num_attributes)
        self.sdf_head = SDFHead(feature_dim)

        # Differentiable vectorization
        self.dvx = DifferentiableVectorization(max_polygons, max_points, feature_dim)

        # Differentiable extrusion
        self.extrusion = DifferentiableExtrusion(voxel_size)

    def forward(self, image):
        # Multi-scale feature extraction
        features = self.encoder(image)

        # Multi-task predictions
        segmentation = self.seg_head(features)
        attributes = self.attr_head(features["global"])
        sdf = self.sdf_head(features)

        # Differentiable vectorization
        dvx_output = self.dvx(features, segmentation)
        polygons = dvx_output["polygons"]
        validity = dvx_output["validity"]

        # Differentiable 3D extrusion
        voxels_pred = self.extrusion(polygons, attributes, validity)

        return {
            "segmentation": segmentation,
            "attributes": attributes,
            "sdf": sdf,
            "polygons": polygons,
            "polygon_validity": validity,
            "voxels_pred": voxels_pred,
            "features": features,
        }