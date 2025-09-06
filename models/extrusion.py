"""
Vectorized Differentiable 3D extrusion module for converting polygons to 3D occupancy
Optimized version with GPU-accelerated vectorized operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableExtrusion(nn.Module):
    """
    Vectorized Differentiable 3D extrusion module
    Converts polygons + attributes to soft 3D occupancy grids
    """

    def __init__(self, voxel_size: int = 64):
        super().__init__()
        self.voxel_size = int(voxel_size)
        # Pre-computed coordinates buffer - will be initialized on first use
        self.register_buffer("_coords", None)

    def _ensure_coords(self, device):
        """Initialize or update coordinate grid if needed"""
        if (self._coords is None or 
            self._coords.device != device or 
            self._coords.shape[0] != (self.voxel_size * self.voxel_size)):
            
            H = W = self.voxel_size
            y, x = torch.meshgrid(
                torch.arange(H, device=device), 
                torch.arange(W, device=device), 
                indexing="ij"
            )
            coords = torch.stack([x.flatten().float(), y.flatten().float()], dim=1)  # [H*W, 2]
            # Normalize to 0..1 coords to match polygon coordinates
            coords = coords / float(self.voxel_size - 1)
            self.register_buffer("_coords", coords)

    def polygon_sdf(self, polygon_xy):
        """
        Compute signed distance field for a polygon using vectorized operations.
        
        Args:
            polygon_xy: [P, 2] polygon vertices in normalized 0..1 coords
        Returns:
            sdf: [H*W] signed distances (negative inside, positive outside)
        """
        device = polygon_xy.device
        self._ensure_coords(device)
        pts = self._coords  # [M, 2], M = H*W
        P = polygon_xy.shape[0]
        
        if P < 2:
            # Degenerate case: return large positive distances (outside)
            return torch.full((pts.shape[0],), 1.0, device=device)

        # Create edges: v0->v1, v1->v2, ..., v_{P-1}->v0
        v0 = polygon_xy.unsqueeze(1)   # [P, 1, 2]
        v1 = torch.roll(polygon_xy, shifts=-1, dims=0).unsqueeze(1)  # [P, 1, 2]
        
        # Expand for broadcasting: edges x points -> [P, M, 2]
        pts_exp = pts.unsqueeze(0)     # [1, M, 2]
        
        # Compute edge vectors and point-to-edge vectors
        e = v1 - v0                   # [P, 1, 2] edge vectors
        v = pts_exp - v0              # [P, M, 2] vectors from v0 to each point
        
        # Project points onto edges
        e_norm_sq = (e**2).sum(dim=2, keepdim=True)  # [P, 1, 1]
        e_norm_sq = e_norm_sq + 1e-8  # Avoid division by zero
        
        # Projection parameter t along edge
        t = (v * e).sum(dim=2, keepdim=True) / e_norm_sq  # [P, M, 1]
        t_clamped = t.clamp(0.0, 1.0)  # Clamp to line segment
        
        # Closest point on each edge
        proj = v0 + t_clamped * e    # [P, M, 2]
        diff = pts_exp - proj        # [P, M, 2]
        dists = torch.norm(diff, dim=2)  # [P, M] distances to each edge
        
        # Find minimum distance to any edge
        min_dist_per_point, _ = dists.min(dim=0)  # [M]

        # Vectorized inside/outside test using ray casting
        x_pts = pts[:, 0].unsqueeze(0)  # [1, M]
        y_pts = pts[:, 1].unsqueeze(0)  # [1, M]
        x0 = v0[..., 0]  # [P, 1]
        y0 = v0[..., 1]  # [P, 1]
        x1 = v1[..., 0]  # [P, 1]
        y1 = v1[..., 1]  # [P, 1]

        # Check if horizontal ray crosses each edge
        y_crosses = ((y0 <= y_pts) & (y1 > y_pts)) | ((y1 <= y_pts) & (y0 > y_pts))  # [P, M]
        
        # Compute x coordinate of intersection with horizontal ray
        dy = y1 - y0
        dx = x1 - x0
        inter_x = x0 + dx * ((y_pts - y0) / (dy + 1e-8))  # [P, M]
        
        # Count crossings to the right of each point
        crossings = (inter_x > x_pts) & y_crosses  # [P, M]
        crossing_count = crossings.sum(dim=0)  # [M]
        inside = (crossing_count % 2 == 1)     # [M] boolean - odd number of crossings = inside

        # Create signed distance field: negative inside, positive outside
        sdf = min_dist_per_point.clone()
        sdf[inside] = -sdf[inside]
        return sdf  # [M]

    def forward(self, polygons: torch.Tensor, attributes: torch.Tensor, validity_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert polygons to 3D voxel occupancy using vectorized operations.

        Args:
            polygons: [B, N, P, 2] - N polygons, P points each (normalized [0,1])
            attributes: [B, 6] - geometric parameters (normalized)
            validity_scores: [B, N] - polygon validity scores
        Returns:
            voxels: [B, D, H, W] float tensor in [0,1]
        """
        device = polygons.device
        B, N, P, _ = polygons.shape
        D = H = W = self.voxel_size
        M = H * W

        # Pre-allocate output tensor
        voxels = torch.zeros((B, D, H, W), device=device)

        # Process each sample in batch (still iterate over B for memory efficiency)
        for b in range(B):
            sample_vox = torch.zeros((D, H, W), device=device)
            
            # Extract wall height for this sample
            wall_height_normalized = attributes[b, 0].item()  # Assume first attribute is wall height
            wall_height_m = wall_height_normalized * 5.0     # Denormalize to meters
            height_voxels = max(1, min(D, int((wall_height_m / 5.0) * D)))
            
            # Process each polygon (vectorized per polygon)
            for n in range(N):
                # Skip invalid polygons
                if validity_scores[b, n].item() < 0.5:
                    continue
                
                poly = polygons[b, n]  # [P, 2]
                
                # Skip empty or zero polygons
                if poly.abs().sum().item() == 0:
                    continue
                
                # Remove padding (zero vertices)
                valid_mask = (poly.sum(dim=1) != 0.0)
                if valid_mask.sum().item() < 3:  # Need at least 3 vertices
                    continue
                
                valid_poly = poly[valid_mask]  # [P_valid, 2]
                
                # Compute signed distance field for this polygon
                sdf_flat = self.polygon_sdf(valid_poly)  # [M]
                
                # Convert SDF to soft mask using sigmoid
                # Higher multiplier = sharper transition, tune as needed
                sharpness = 100.0
                mask2d = torch.sigmoid(-sdf_flat * sharpness)  # [M]
                mask2d = mask2d.view(H, W)  # [H, W]
                
                # Extrude mask to 3D using vectorized operations
                # Create mask stack for all height levels at once
                mask_3d = mask2d.unsqueeze(0).expand(height_voxels, -1, -1)  # [height_voxels, H, W]
                
                # Update voxel grid (element-wise maximum to handle overlapping polygons)
                sample_vox[:height_voxels] = torch.maximum(
                    sample_vox[:height_voxels], 
                    mask_3d
                )
            
            voxels[b] = sample_vox

        return voxels


class DifferentiableExtrusionFast(nn.Module):
    """
    Even more optimized version that batches polygon processing.
    Use this if you need maximum performance and have consistent polygon counts.
    """
    
    def __init__(self, voxel_size: int = 64):
        super().__init__()
        self.voxel_size = int(voxel_size)
        self.register_buffer("_coords", None)

    def _ensure_coords(self, device):
        """Initialize coordinate grid"""
        if (self._coords is None or 
            self._coords.device != device or 
            self._coords.shape[0] != (self.voxel_size * self.voxel_size)):
            
            H = W = self.voxel_size
            y, x = torch.meshgrid(
                torch.arange(H, device=device), 
                torch.arange(W, device=device), 
                indexing="ij"
            )
            coords = torch.stack([x.flatten().float(), y.flatten().float()], dim=1)
            coords = coords / float(self.voxel_size - 1)
            self.register_buffer("_coords", coords)

    def batch_polygon_sdf(self, polygons_batch, validity_mask):
        """
        Compute SDF for multiple polygons simultaneously.
        
        Args:
            polygons_batch: [N, P, 2] multiple polygons
            validity_mask: [N] which polygons are valid
        Returns:
            sdfs: [N, H*W] signed distance fields
        """
        device = polygons_batch.device
        self._ensure_coords(device)
        
        N, P, _ = polygons_batch.shape
        M = self._coords.shape[0]  # H*W
        
        # Initialize output
        sdfs = torch.full((N, M), 1.0, device=device)  # Default to "outside"
        
        # Process only valid polygons
        valid_indices = torch.where(validity_mask)[0]
        if len(valid_indices) == 0:
            return sdfs
            
        valid_polygons = polygons_batch[valid_indices]  # [N_valid, P, 2]
        
        # Remove zero-padded vertices for each polygon
        # This is still a limitation - different polygons may have different valid vertex counts
        # For full vectorization, you'd need to pad all polygons to the same vertex count
        
        for i, poly_idx in enumerate(valid_indices):
            poly = valid_polygons[i]  # [P, 2]
            vertex_mask = (poly.sum(dim=1) != 0.0)
            if vertex_mask.sum().item() >= 3:
                valid_poly = poly[vertex_mask]
                sdf = self.polygon_sdf(valid_poly)
                sdfs[poly_idx] = sdf
        
        return sdfs

    def polygon_sdf(self, polygon_xy):
        """Same SDF computation as the main class"""
        device = polygon_xy.device
        self._ensure_coords(device)
        pts = self._coords
        P = polygon_xy.shape[0]
        
        if P < 2:
            return torch.full((pts.shape[0],), 1.0, device=device)

        v0 = polygon_xy.unsqueeze(1)
        v1 = torch.roll(polygon_xy, shifts=-1, dims=0).unsqueeze(1)
        pts_exp = pts.unsqueeze(0)
        
        e = v1 - v0
        v = pts_exp - v0
        e_norm_sq = (e**2).sum(dim=2, keepdim=True) + 1e-8
        t = (v * e).sum(dim=2, keepdim=True) / e_norm_sq
        t_clamped = t.clamp(0.0, 1.0)
        
        proj = v0 + t_clamped * e
        diff = pts_exp - proj
        dists = torch.norm(diff, dim=2)
        min_dist_per_point, _ = dists.min(dim=0)

        x_pts = pts[:, 0].unsqueeze(0)
        y_pts = pts[:, 1].unsqueeze(0)
        x0, y0 = v0[..., 0], v0[..., 1]
        x1, y1 = v1[..., 0], v1[..., 1]

        y_crosses = ((y0 <= y_pts) & (y1 > y_pts)) | ((y1 <= y_pts) & (y0 > y_pts))
        inter_x = x0 + (x1 - x0) * ((y_pts - y0) / (y1 - y0 + 1e-8))
        crossings = (inter_x > x_pts) & y_crosses
        crossing_count = crossings.sum(dim=0)
        inside = (crossing_count % 2 == 1)

        sdf = min_dist_per_point.clone()
        sdf[inside] = -sdf[inside]
        return sdf

    def forward(self, polygons: torch.Tensor, attributes: torch.Tensor, validity_scores: torch.Tensor) -> torch.Tensor:
        """Forward pass with batch polygon processing"""
        device = polygons.device
        B, N, P, _ = polygons.shape
        D = H = W = self.voxel_size
        
        voxels = torch.zeros((B, D, H, W), device=device)
        
        for b in range(B):
            # Get validity mask for this sample
            validity_mask = validity_scores[b] >= 0.5  # [N]
            
            if not validity_mask.any():
                continue
                
            # Compute SDFs for all valid polygons in this sample
            sdfs = self.batch_polygon_sdf(polygons[b], validity_mask)  # [N, H*W]
            
            # Convert to masks
            sharpness = 100.0
            masks = torch.sigmoid(-sdfs * sharpness)  # [N, H*W]
            masks_2d = masks.view(N, H, W)  # [N, H, W]
            
            # Get wall height
            wall_height_normalized = attributes[b, 0].item()
            wall_height_m = wall_height_normalized * 5.0
            height_voxels = max(1, min(D, int((wall_height_m / 5.0) * D)))
            
            # Combine all polygon masks for this sample
            combined_mask = torch.zeros((H, W), device=device)
            for n in range(N):
                if validity_mask[n]:
                    combined_mask = torch.maximum(combined_mask, masks_2d[n])
            
            # Extrude to 3D
            mask_3d = combined_mask.unsqueeze(0).expand(height_voxels, -1, -1)
            voxels[b, :height_voxels] = mask_3d
            
        return voxels