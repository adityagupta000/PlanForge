"""
Differentiable Vectorization (DVX) module for polygon prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableVectorization(nn.Module):
    """
    Differentiable Vectorization (DVX) module
    Converts soft segmentation masks to polygon representations
    Using differentiable active contour fitting
    """

    def __init__(self, max_polygons=20, max_points=50, feature_dim=512):
        super().__init__()

        self.max_polygons = max_polygons
        self.max_points = max_points

        # Polygon initialization network
        self.init_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(feature_dim * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, max_polygons * max_points * 2),
            nn.Sigmoid(),
        )

        # Control point refinement network
        self.refine_net = nn.Sequential(
            nn.Linear(feature_dim + 2, 128),  # feature + xy coordinates
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # xy displacement
            nn.Tanh(),
        )

        # Polygon validity classifier
        self.validity_net = nn.Sequential(
            nn.Linear(max_points * 2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1), 
            nn.Sigmoid()
        )

    def forward(self, features, segmentation):
        batch_size = segmentation.shape[0]

        # Initialize polygon control points
        p4 = features["p4"]  # Use lowest resolution for initialization
        init_points = self.init_net(p4)
        init_points = init_points.view(
            batch_size, self.max_polygons, self.max_points, 2
        )

        # Refine control points using image features
        refined_points = self._refine_points(init_points, features, segmentation)

        # Predict polygon validity
        validity = self._predict_validity(refined_points)

        return {
            "polygons": refined_points,
            "validity": validity,
            "init_polygons": init_points,
        }

    def _refine_points(self, points, features, segmentation):
        """Refine polygon control points using local image features"""
        batch_size, num_polys, num_points, _ = points.shape

        # Sample features at control point locations
        grid = points * 2 - 1  # Convert to [-1, 1] for grid_sample

        # Use P2 features for refinement (good balance of resolution/semantics)
        p2_features = features["p2"]

        # Sample features at each control point
        sampled_features = F.grid_sample(
            p2_features, grid.view(batch_size, -1, 1, 2), align_corners=False
        )  # [B, C, num_polys*num_points, 1]

        sampled_features = sampled_features.squeeze(-1).permute(
            0, 2, 1
        )  # [B, num_polys*num_points, C]

        # Concatenate with coordinates
        coords = points.view(batch_size, -1, 2)
        input_features = torch.cat([sampled_features, coords], dim=-1)

        # Predict refinement displacements
        displacements = self.refine_net(input_features)
        displacements = displacements.view(batch_size, num_polys, num_points, 2)

        # Apply displacements with constraints
        refined = points + 0.1 * displacements  # Scale displacement
        refined = torch.clamp(refined, 0, 1)  # Keep within [0,1]

        return refined

    def _predict_validity(self, polygons):
        """Predict which polygons are valid/meaningful"""
        batch_size, num_polys, _, _ = polygons.shape

        poly_flat = polygons.view(batch_size * num_polys, -1)
        validity = self.validity_net(poly_flat)
        validity = validity.view(batch_size, num_polys)

        return validity