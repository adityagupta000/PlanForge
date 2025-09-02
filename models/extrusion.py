"""
Differentiable 3D extrusion module for converting polygons to 3D occupancy
"""

import torch
import torch.nn as nn


class DifferentiableExtrusion(nn.Module):
    """
    Differentiable 3D extrusion module
    Converts polygons + attributes to soft 3D occupancy grids
    """

    def __init__(self, voxel_size: int = 64):
        super().__init__()
        self.voxel_size = int(voxel_size)

    def forward(self, polygons: torch.Tensor, attributes: torch.Tensor, validity_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert polygons to 3D voxel occupancy.

        Args:
            polygons: [B, N, P, 2] - N polygons, P points each (normalized [0,1])
            attributes: [B, 6] - geometric parameters (normalized as in dataset)
            validity_scores: [B, N] - polygon validity scores
        Returns:
            voxels: [B, D, H, W] float tensor in [0,1]
        """
        batch_size = polygons.shape[0]
        device = polygons.device

        # Build per-sample voxels and stack to avoid in-place writes on a preallocated batched tensor
        voxels_list = []
        for b in range(batch_size):
            vox_b = self._extrude_sample(polygons[b], attributes[b], validity_scores[b])
            voxels_list.append(vox_b)

        return torch.stack(voxels_list, dim=0)  # [B, D, H, W]

    def _extrude_sample(self, polygons: torch.Tensor, attributes: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        """Extrude polygons for a single sample into a [D,H,W] voxel grid."""
        device = polygons.device
        D = self.voxel_size
        H = self.voxel_size
        W = self.voxel_size

        # Initialize voxel grid (float)
        voxels = torch.zeros((D, H, W), device=device, dtype=torch.float32)

        # Extract geometric parameters (denormalize same way dataset did)
        wall_height_m = float(attributes[0].item() * 5.0)          # back to meters
        # wall_thickness = float(attributes[1].item() * 0.5)      # not used for now

        # Convert wall height (meters) to voxel height
        height_voxels = min(int((wall_height_m / 5.0) * D), D)

        # Iterate polygons (assume polygons are [N, P, 2], normalized to [0,1])
        for poly, valid_score in zip(polygons, validity):
            # skip invalid polygons
            if valid_score.item() < 0.5:
                continue

            # If polygon has no points or all zeros, skip
            if poly.numel() == 0 or torch.all(poly == 0):
                continue

            # Scale to voxel coordinates (0..voxel_size-1)
            poly_voxel = poly * float(self.voxel_size - 1)

            # Rasterize polygon to 2D soft mask
            mask_2d = self._polygon_to_mask(poly_voxel)  # [H, W], float in [0,1]

            # Extrude mask into voxel columns (top-down along depth axis)
            # Use out-of-place update to preserve autograd history
            for z in range(0, height_voxels):
                # clone the target slice before combining to avoid in-place modification
                target_slice = voxels[z].clone()
                combined = torch.maximum(target_slice, mask_2d)
                voxels[z] = combined

        return voxels

    def _polygon_to_mask(self, polygon_voxel: torch.Tensor) -> torch.Tensor:
        """
        Convert polygon (P,2) in voxel coordinates to a flattened soft mask then reshape to [H,W].
        Uses a simple distance-to-edge soft rasterization.
        """
        device = polygon_voxel.device
        D = self.voxel_size
        H = self.voxel_size
        W = self.voxel_size

        # Prepare flattened pixel coordinates: [H*W, 2] as float
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        coords = torch.stack([x_grid.flatten().float(), y_grid.flatten().float()], dim=1)  # [H*W, 2]

        # Compute soft inside probability for each point
        inside = self._point_in_polygon_soft(coords, polygon_voxel)  # [H*W]

        # Reshape to H x W
        mask = inside.view(H, W)

        return mask

    def _point_in_polygon_soft(self, points: torch.Tensor, polygon: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Soft differentiable point-in-polygon test.
        points: [M,2]
        polygon: [P,2] (may include padded zeros)
        Returns: [M] inside probability in [0,1]
        """
        device = points.device

        # Keep only valid polygon vertices (non-zero rows)
        valid_mask = (polygon.sum(dim=1) != 0.0)
        valid_poly = polygon[valid_mask]
        if valid_poly.shape[0] < 3:
            return torch.zeros(points.shape[0], device=device, dtype=torch.float32)

        # Compute distance from each point to each polygon edge (vectorized)
        P = valid_poly.shape[0]
        M = points.shape[0]

        # Prepare p1 and p2 for each edge: [P,2]
        p1 = valid_poly
        p2 = torch.roll(valid_poly, shifts=-1, dims=0)

        # Expand to compute distances from all points to all edges
        # points: [M,2], p1: [P,2] -> point_vec: [P, M, 2]
        # We'll compute distances per edge then take min across edges.
        # For memory safety we iterate edges if P large; typical P small (<=50)

        distances = []
        for i in range(P):
            pi1 = p1[i]        # [2]
            pi2 = p2[i]        # [2]
            dist = self._point_to_line_distance(points, pi1, pi2)  # [M]
            distances.append(dist)

        distances_stack = torch.stack(distances, dim=0)  # [P, M]
        min_dist, _ = distances_stack.min(dim=0)         # [M]

        # Convert distance to inside probability: nearer to boundary => small distance
        inside_prob = torch.sigmoid(-min_dist / float(sigma))
        return inside_prob

    def _point_to_line_distance(self, points: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute shortest distance from many points to a single line segment (p1->p2).
        points: [M,2]
        p1, p2: [2]
        Returns: [M] distances
        """
        # Ensure float tensors
        points = points.float()
        p1 = p1.float()
        p2 = p2.float()

        # Line vector and length (scalar)
        line_vec = p2 - p1                       # [2]
        line_len = torch.norm(line_vec)          # scalar tensor

        if line_len.item() < 1e-6:
            # Degenerate segment -> distance to point p1
            return torch.norm(points - p1.unsqueeze(0), dim=1)

        # Unit direction
        line_unit = line_vec / line_len          # [2]

        # Vector from p1 to each point: [M,2]
        point_vec = points - p1.unsqueeze(0)

        # Projection scalars along unit line: [M] = (point_vec dot line_unit)
        proj_length = torch.matmul(point_vec, line_unit)   # [M]

        # Clamp projections to segment extents (numbers, not mixed tensor types)
        proj_length = torch.clamp(proj_length, 0.0, float(line_len.item()))  # [M]

        # Closest points coordinates on the infinite line: [M,2]
        closest = p1.unsqueeze(0) + proj_length.unsqueeze(1) * line_unit.unsqueeze(0)

        # Distances from points to closest points
        dists = torch.norm(points - closest, dim=1)  # [M]
        return dists
