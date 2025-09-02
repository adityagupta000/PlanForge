"""
Advanced loss functions for multi-task training (safe, out-of-place)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class ResearchGradeLoss(nn.Module):
    """
    Multi-task loss combining:
    - Segmentation (CE + Dice)
    - SDF regression
    - Attribute regression
    - Polygon/DVX fitting
    - 3D voxel IoU
    - Topology-aware losses
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        dice_weight: float = 1.0,
        sdf_weight: float = 0.5,
        attr_weight: float = 1.0,
        polygon_weight: float = 1.0,
        voxel_weight: float = 1.0,
        topology_weight: float = 0.5,
    ):
        super().__init__()

        self.seg_weight = float(seg_weight)
        self.dice_weight = float(dice_weight)
        self.sdf_weight = float(sdf_weight)
        self.attr_weight = float(attr_weight)
        self.polygon_weight = float(polygon_weight)
        self.voxel_weight = float(voxel_weight)
        self.topology_weight = float(topology_weight)

        # Core losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions: dict, targets: dict):
        """
        Compute multi-task loss and return (total_loss, dict_of_components)
        predictions: dict with keys like 'segmentation','sdf','attributes','polygons','voxels_pred','polygon_validity'
        targets: dict with keys 'mask','attributes','polygons_gt','voxels_gt' etc.
        """
        device = None
        # pick device if available
        if "mask" in targets and torch.is_tensor(targets["mask"]):
            device = targets["mask"].device
        elif "attributes" in targets and torch.is_tensor(targets["attributes"]):
            device = targets["attributes"].device

        losses = {}
        total_loss = torch.tensor(0.0, device=device if device is not None else None)

        # ---- 1) Segmentation ----
        if "segmentation" in predictions and "mask" in targets:
            seg_pred = predictions["segmentation"]
            seg_target = targets["mask"].long()

            ce_loss = self.ce_loss(seg_pred, seg_target)
            losses["ce"] = ce_loss
            total_loss = total_loss + self.seg_weight * ce_loss

            dice_loss = self._dice_loss(seg_pred, seg_target)
            losses["dice"] = dice_loss
            total_loss = total_loss + self.dice_weight * dice_loss

        # ---- 2) SDF loss ----
        if "sdf" in predictions and "mask" in targets:
            sdf_pred = predictions["sdf"]
            # compute SDF target (numpy/OpenCV based) - no grad needed
            sdf_target = self._mask_to_sdf(targets["mask"])
            # Ensure same device/dtype
            sdf_target = sdf_target.to(sdf_pred.device).type_as(sdf_pred)
            sdf_loss = self.mse_loss(sdf_pred, sdf_target)
            losses["sdf"] = sdf_loss
            total_loss = total_loss + self.sdf_weight * sdf_loss

        # ---- 3) Attribute regression ----
        if "attributes" in predictions and "attributes" in targets:
            pred_attr = predictions["attributes"].float()
            tgt_attr = targets["attributes"].float().to(pred_attr.device)
            attr_loss = self.l1_loss(pred_attr, tgt_attr)
            losses["attributes"] = attr_loss
            total_loss = total_loss + self.attr_weight * attr_loss

        # ---- 4) Polygon / DVX losses ----
        if "polygons" in predictions and "polygons_gt" in targets:
            poly_loss = self._polygon_loss(predictions, targets["polygons_gt"])
            losses["polygon"] = poly_loss
            total_loss = total_loss + self.polygon_weight * poly_loss

        # ---- 5) 3D voxel IoU loss ----
        if "voxels_pred" in predictions and "voxels_gt" in targets:
            pred_vox = predictions["voxels_pred"].float()
            tgt_vox = targets["voxels_gt"].float().to(pred_vox.device)
            voxel_loss = self._voxel_iou_loss(pred_vox, tgt_vox)
            losses["voxel"] = voxel_loss
            total_loss = total_loss + self.voxel_weight * voxel_loss

        # ---- 6) Topology-aware loss (segmentation dependent) ----
        if "segmentation" in predictions:
            topo_loss = self._topology_loss(predictions["segmentation"])
            losses["topology"] = topo_loss
            total_loss = total_loss + self.topology_weight * topo_loss

        losses["total"] = total_loss
        return total_loss, losses

    # -----------------------
    # Dice
    # -----------------------
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] integer labels
        """
        pred_soft = F.softmax(pred, dim=1)  # [B,C,H,W]
        B = pred_soft.shape[0]
        C = pred_soft.shape[1]

        dice_losses = []
        for c in range(C):
            pred_c = pred_soft[:, c, :, :]  # [B,H,W]
            target_c = (target == c).float().to(pred_c.device)
            # compute per-batch intersection/union
            intersection = (pred_c * target_c).view(B, -1).sum(dim=1)
            union = pred_c.view(B, -1).sum(dim=1) + target_c.view(B, -1).sum(dim=1)
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss_c = 1.0 - dice  # [B]
            dice_losses.append(dice_loss_c.mean())

        return torch.stack(dice_losses).mean()

    # -----------------------
    # SDF helper (uses OpenCV / numpy; no gradient expected)
    # -----------------------
    def _mask_to_sdf(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: [B, H, W] tensor with integer labels (0 background, >0 foreground)
        returns: [B, 1, H, W] float SDF (on same device as input)
        """
        device = mask.device if torch.is_tensor(mask) else None
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=device)

        B, H, W = mask.shape
        sdf = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for b in range(B):
            mask_np = mask[b].cpu().numpy().astype(np.uint8)

            # edges = cv2.Canny((mask_np > 0).astype(np.uint8) * 255, 50, 150)
            dist_inside = cv2.distanceTransform((mask_np > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist_outside = cv2.distanceTransform((mask_np == 0).astype(np.uint8), cv2.DIST_L2, 5)

            sdf_np = dist_inside.astype(np.float32) - dist_outside.astype(np.float32)
            # normalize a little and squash
            sdf_np = np.tanh(sdf_np / 10.0).astype(np.float32)

            sdf[b, 0] = torch.from_numpy(sdf_np)

        return sdf

    # -----------------------
    # Polygon / DVX losses
    # -----------------------
    def _polygon_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """
        predictions: should contain 'polygons' [B, N, P, 2] and 'polygon_validity' [B,N]
        targets: dict with 'polygons' [B,N,P,2] and 'valid_mask' [B,N]
        """
        pred_polys = predictions.get("polygons")  # [B,N,P,2]
        tgt_polys = targets.get("polygons")       # [B,N,P,2]
        valid_mask = targets.get("valid_mask")   # [B,N] bool or float

        if pred_polys is None or tgt_polys is None:
            return torch.tensor(0.0, device=pred_polys.device if pred_polys is not None else None)

        pred_polys = pred_polys.float()
        tgt_polys = tgt_polys.float().to(pred_polys.device)

        # L2 point-wise loss (masked)
        point_loss = self.mse_loss(pred_polys, tgt_polys)

        # validity
        pred_valid = predictions.get("polygon_validity")
        if pred_valid is None:
            validity_loss = torch.tensor(0.0, device=pred_polys.device)
        else:
            pred_valid = pred_valid.float().to(pred_polys.device)
            valid_mask_f = valid_mask.float().to(pred_polys.device)
            validity_loss = self.mse_loss(pred_valid, valid_mask_f)

        # smoothness & rectilinearity
        smoothness_loss = self._polygon_smoothness(pred_polys)
        rect_loss = self._rectilinearity_loss(pred_polys)

        return point_loss + 0.1 * validity_loss + 0.05 * smoothness_loss + 0.1 * rect_loss

    def _polygon_smoothness(self, polygons: torch.Tensor) -> torch.Tensor:
        """
        polygons: [B, N, P, 2]
        encourage low curvature (second derivative)
        """
        if polygons is None or polygons.numel() == 0:
            return torch.tensor(0.0, device=polygons.device if polygons is not None else None)

        p1 = polygons
        p2 = torch.roll(polygons, -1, dims=2)
        p3 = torch.roll(polygons, -2, dims=2)
        curvature = torch.norm(p1 - 2.0 * p2 + p3, dim=-1)  # [B,N,P]
        return curvature.mean()

    def _rectilinearity_loss(self, polygons: torch.Tensor) -> torch.Tensor:
        """
        Encourage axis-aligned structure:
        polygons: [B, N, P, 2]
        """
        if polygons is None or polygons.numel() == 0:
            return torch.tensor(0.0, device=polygons.device if polygons is not None else None)

        edges = torch.roll(polygons, -1, dims=2) - polygons  # [B,N,P,2]
        edge_norms = torch.norm(edges, dim=-1, keepdim=True)  # [B,N,P,1]
        edges_normalized = edges / (edge_norms + 1e-6)

        edge1 = edges_normalized
        edge2 = torch.roll(edges_normalized, -1, dims=2)

        cos_angles = (edge1 * edge2).sum(dim=-1)  # [B,N,P]

        # Encourage cos_angles to be near 0 (perpendicular) or near 1 (parallel)
        # We use a smooth penalty: min((cos)^2, (cos^2 - 1)^2)
        cos2 = cos_angles ** 2
        perp_penalty = cos2  # small when cos = 0
        parallel_penalty = (cos2 - 1.0) ** 2  # small when cos^2 = 1
        angle_penalty = torch.minimum(perp_penalty, parallel_penalty)
        return angle_penalty.mean()

    # -----------------------
    # 3D voxel IoU loss
    # -----------------------
    def _voxel_iou_loss(self, pred_voxels: torch.Tensor, target_voxels: torch.Tensor) -> torch.Tensor:
        """
        pred_voxels: [B,D,H,W] predicted logits (or floats)
        target_voxels: [B,D,H,W] binary floats (0/1)
        """
        pred_prob = torch.sigmoid(pred_voxels)  # [B,...]
        target = target_voxels.float().to(pred_prob.device)

        intersection = (pred_prob * target).view(pred_prob.shape[0], -1).sum(dim=1)
        union = pred_prob.view(pred_prob.shape[0], -1).sum(dim=1) + target.view(target.shape[0], -1).sum(dim=1) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()

    # -----------------------
    # Topology-aware losses
    # -----------------------
    def _topology_loss(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        """
        Encourages architectural constraints: doors/windows overlap walls, wall connectivity.
        segmentation_logits: [B,C,H,W]
        """
        seg_soft = F.softmax(segmentation_logits, dim=1)
        # safe indexing with fallback zeros if channel missing
        C = seg_soft.shape[1]
        device = seg_soft.device
        walls = seg_soft[:, 1] if C > 1 else torch.zeros_like(seg_soft[:, 0])
        doors = seg_soft[:, 2] if C > 2 else torch.zeros_like(walls)
        windows = seg_soft[:, 3] if C > 3 else torch.zeros_like(walls)

        # Overlap checks
        door_wall_overlap = doors * walls
        window_wall_overlap = windows * walls

        door_penalty = torch.maximum(doors - door_wall_overlap, torch.zeros_like(doors))
        window_penalty = torch.maximum(windows - window_wall_overlap, torch.zeros_like(windows))

        connectivity_loss = self._connectivity_loss(walls)

        return door_penalty.mean() + window_penalty.mean() + 0.1 * connectivity_loss

    def _connectivity_loss(self, wall_prob: torch.Tensor) -> torch.Tensor:
        """
        Penalize isolated wall pixels using a local average approximation (conv).
        wall_prob: [B, H, W]
        """
        if wall_prob is None or wall_prob.numel() == 0:
            return torch.tensor(0.0, device=wall_prob.device if wall_prob is not None else None)

        B = wall_prob.shape[0]
        kernel = torch.ones((1, 1, 3, 3), device=wall_prob.device, dtype=wall_prob.dtype) / 9.0
        neighbors = F.conv2d(wall_prob.unsqueeze(1), kernel, padding=1).squeeze(1)  # [B,H,W]

        isolation_penalty = wall_prob * torch.exp(-neighbors)
        return isolation_penalty.mean()
