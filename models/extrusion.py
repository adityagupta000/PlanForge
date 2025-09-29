"""
Vectorized Differentiable 3D extrusion module for converting polygons to 3D occupancy
Optimized version with GPU-accelerated vectorized operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging


# -----------------------------------------------------------------------------
# Logging and sanitization helper
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _sanitize_normalized_height(value, sample_id=None, default=0.6):
    """
    Ensure normalized height value is finite and in [0,1].
    Returns a float in [0,1].

    Args:
        value: torch scalar tensor or float
        sample_id: optional identifier for logging (string or int)
        default: fallback normalized height
    """
    try:
        if isinstance(value, torch.Tensor):
            raw = float(value.item())
        else:
            raw = float(value)
    except Exception:
        raw = float("nan")

    # Build label for logging
    sid = f"[sample={sample_id}]" if sample_id is not None else ""

    # Check finite
    if not math.isfinite(raw):
        logger.warning(f"{sid} Invalid wall height value (not finite): {raw}; using default {default}")
        raw = default

    # Clamp to [0,1]
    if raw < 0.0 or raw > 1.0:
        logger.warning(f"{sid} Wall height normalized {raw} out of [0,1]; clamping.")
        raw = max(0.0, min(1.0, raw))

    return raw


def _sanitize_tensor(tensor, default_value=0.0, name="tensor"):
    """
    Sanitize an entire tensor by replacing NaN/Inf values with default.
    
    Args:
        tensor: Input tensor
        default_value: Value to replace invalid entries with
        name: Name for logging
    
    Returns:
        Sanitized tensor
    """
    if tensor.numel() == 0:
        return tensor
    
    # Check for any invalid values
    invalid_mask = ~torch.isfinite(tensor)
    num_invalid = invalid_mask.sum().item()
    
    if num_invalid > 0:
        logger.warning(f"Found {num_invalid} invalid values in {name}, replacing with {default_value}")
        tensor = tensor.clone()
        tensor[invalid_mask] = default_value
    
    return tensor


# -----------------------------------------------------------------------------
# Main extrusion module
# -----------------------------------------------------------------------------
class DifferentiableExtrusion(nn.Module):
    """
    Vectorized Differentiable 3D extrusion module
    Converts polygons + attributes to soft 3D occupancy grids
    """

    def __init__(self, voxel_size: int = 96):
        super().__init__()
        self.voxel_size = int(voxel_size)
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
            coords = coords / float(self.voxel_size - 1)
            self.register_buffer("_coords", coords)

    def polygon_sdf(self, polygon_xy):
        """
        Compute signed distance field for a polygon using vectorized operations.
        """
        device = polygon_xy.device
        self._ensure_coords(device)
        pts = self._coords  # [M, 2]
        P = polygon_xy.shape[0]
        
        if P < 2:
            return torch.full((pts.shape[0],), 1.0, device=device)

        # Sanitize polygon coordinates
        polygon_xy = _sanitize_tensor(polygon_xy, default_value=0.0, name="polygon_xy")

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
        
        # Sanitize distances before min operation
        dists = _sanitize_tensor(dists, default_value=1.0, name="distances")
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
        
        # Final sanitization of SDF output
        sdf = _sanitize_tensor(sdf, default_value=1.0, name="sdf")
        return sdf

    def forward(self, polygons, attributes, validity_scores, sample_ids=None):
        """
        Convert polygons to 3D voxel occupancy.
        sample_ids: optional list/array of identifiers (e.g., filenames or dataset indices)
        """
        device = polygons.device
        B, N, P, _ = polygons.shape
        D = H = W = self.voxel_size
        
        # Sanitize input tensors
        polygons = _sanitize_tensor(polygons, default_value=0.0, name="input_polygons")
        attributes = _sanitize_tensor(attributes, default_value=0.6, name="input_attributes")
        validity_scores = _sanitize_tensor(validity_scores, default_value=0.0, name="input_validity_scores")
        
        voxels = torch.zeros((B, D, H, W), device=device)

        for b in range(B):
            # pick identifier if available
            sid = sample_ids[b] if sample_ids is not None else b

            # Sanitize height with logging
            wall_height_normalized = attributes[b, 0]
            sanitized_norm = _sanitize_normalized_height(
                wall_height_normalized, sample_id=sid, default=0.6
            )

            wall_height_m = sanitized_norm * 5.0
            height_frac = wall_height_m / 5.0
            height_voxels = int(round(height_frac * D))
            height_voxels = max(1, min(D, height_voxels))

            # Process each polygon for this batch
            validity_mask = validity_scores[b] >= 0.5
            if not validity_mask.any():
                continue

            combined_mask = torch.zeros((H, W), device=device)
            sharpness = 100.0

            for n in range(N):
                if not validity_mask[n]:
                    continue
                    
                polygon = polygons[b, n]  # [P, 2]
                
                # Filter out zero-padded vertices
                vertex_mask = (polygon.sum(dim=1) != 0.0)
                if vertex_mask.sum().item() < 3:
                    continue
                    
                valid_polygon = polygon[vertex_mask]
                
                # Compute SDF for this polygon
                sdf = self.polygon_sdf(valid_polygon)
                mask = torch.sigmoid(-sdf * sharpness)
                mask_2d = mask.view(H, W)
                
                # Sanitize mask before combining
                mask_2d = _sanitize_tensor(mask_2d, default_value=0.0, name=f"mask_2d_b{b}_n{n}")
                combined_mask = torch.maximum(combined_mask, mask_2d)

            # Create 3D mask by extruding to the computed height
            mask_3d = combined_mask.unsqueeze(0).expand(height_voxels, -1, -1)
            
            # Sanitize final mask before assignment
            mask_3d = _sanitize_tensor(mask_3d, default_value=0.0, name=f"final_mask_3d_b{b}")
            voxels[b, :height_voxels] = mask_3d

        # Final sanitization of output
        voxels = _sanitize_tensor(voxels, default_value=0.0, name="output_voxels")
        return voxels


# -----------------------------------------------------------------------------
# Fast extrusion module
# -----------------------------------------------------------------------------
class DifferentiableExtrusionFast(nn.Module):
    """
    Optimized version that batches polygon processing.
    """

    def __init__(self, voxel_size: int = 64):
        super().__init__()
        self.voxel_size = int(voxel_size)
        self.register_buffer("_coords", None)

    def _ensure_coords(self, device):
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
        device = polygons_batch.device
        self._ensure_coords(device)

        N, P, _ = polygons_batch.shape
        M = self._coords.shape[0]
        sdfs = torch.full((N, M), 1.0, device=device)

        valid_indices = torch.where(validity_mask)[0]
        if len(valid_indices) == 0:
            return sdfs

        valid_polygons = polygons_batch[valid_indices]
        for i, poly_idx in enumerate(valid_indices):
            poly = valid_polygons[i]
            vertex_mask = (poly.sum(dim=1) != 0.0)
            if vertex_mask.sum().item() >= 3:
                valid_poly = poly[vertex_mask]
                sdf = self.polygon_sdf(valid_poly)
                sdfs[poly_idx] = sdf

        return sdfs

    def polygon_sdf(self, polygon_xy):
        device = polygon_xy.device
        self._ensure_coords(device)
        pts = self._coords
        P = polygon_xy.shape[0]

        if P < 2:
            return torch.full((pts.shape[0],), 1.0, device=device)

        # Sanitize polygon coordinates
        polygon_xy = _sanitize_tensor(polygon_xy, default_value=0.0, name="polygon_xy_fast")

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
        
        # Sanitize distances before min operation
        dists = _sanitize_tensor(dists, default_value=1.0, name="distances_fast")
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
        
        # Final sanitization of SDF output
        sdf = _sanitize_tensor(sdf, default_value=1.0, name="sdf_fast")
        return sdf

    def forward(self, polygons: torch.Tensor, attributes: torch.Tensor, validity_scores: torch.Tensor) -> torch.Tensor:
        device = polygons.device
        B, N, P, _ = polygons.shape
        D = H = W = self.voxel_size

        # Sanitize input tensors
        polygons = _sanitize_tensor(polygons, default_value=0.0, name="input_polygons_fast")
        attributes = _sanitize_tensor(attributes, default_value=0.6, name="input_attributes_fast")
        validity_scores = _sanitize_tensor(validity_scores, default_value=0.0, name="input_validity_scores_fast")

        voxels = torch.zeros((B, D, H, W), device=device)

        for b in range(B):
            validity_mask = validity_scores[b] >= 0.5
            if not validity_mask.any():
                continue

            sdfs = self.batch_polygon_sdf(polygons[b], validity_mask)
            
            # Sanitize SDFs before sigmoid
            sdfs = _sanitize_tensor(sdfs, default_value=1.0, name=f"batch_sdfs_b{b}")
            
            sharpness = 100.0
            masks = torch.sigmoid(-sdfs * sharpness)
            masks_2d = masks.view(N, H, W)
            
            # Sanitize masks
            masks_2d = _sanitize_tensor(masks_2d, default_value=0.0, name=f"masks_2d_b{b}")

            # Sanitize height
            wall_height_normalized = attributes[b, 0]
            sanitized_norm = _sanitize_normalized_height(wall_height_normalized, sample_id=b, default=0.6)
            wall_height_m = sanitized_norm * 5.0
            height_frac = wall_height_m / 5.0
            height_voxels = int(round(height_frac * D))
            height_voxels = max(1, min(D, height_voxels))

            combined_mask = torch.zeros((H, W), device=device)
            for n in range(N):
                if validity_mask[n]:
                    combined_mask = torch.maximum(combined_mask, masks_2d[n])

            mask_3d = combined_mask.unsqueeze(0).expand(height_voxels, -1, -1)
            
            # Sanitize final mask before assignment
            mask_3d = _sanitize_tensor(mask_3d, default_value=0.0, name=f"final_mask_3d_fast_b{b}")
            voxels[b, :height_voxels] = mask_3d

        # Final sanitization of output
        voxels = _sanitize_tensor(voxels, default_value=0.0, name="output_voxels_fast")
        return voxels