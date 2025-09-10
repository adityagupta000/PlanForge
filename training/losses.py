"""
Advanced loss functions for multi-task training with dynamic weighting
Enhanced with cross-modal consistency, graph constraints, and GradNorm
Modified to support conditional geometric losses via run_full_geometric flag
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import networkx as nx


class DynamicLossWeighter:
    """Implements GradNorm and other dynamic weighting strategies"""
    
    def __init__(self, loss_names: List[str], alpha: float = 0.12, device: str = 'cuda'):
        self.loss_names = loss_names
        self.alpha = alpha
        self.device = device
        
        # Initialize weights
        self.weights = {name: 1.0 for name in loss_names}  # FIX: keep floats, easier logging
        self.initial_task_losses = None
        self.running_mean_losses = {name: 0.0 for name in loss_names}
        self.update_count = 0
        
    def update_weights(self, task_losses: Dict[str, torch.Tensor], 
                      shared_parameters, update_freq: int = 10):
        """Update loss weights using GradNorm algorithm"""
        if self.update_count % update_freq != 0:
            self.update_count += 1
            return self.weights
            
        # Store initial losses on first update
        if self.initial_task_losses is None:
            self.initial_task_losses = {name: loss.item() for name, loss in task_losses.items()}
            
        # Update running mean
        for name, loss in task_losses.items():
            self.running_mean_losses[name] = (0.9 * self.running_mean_losses[name] + 
                                            0.1 * loss.item())
        
        # Calculate relative decrease rates
        loss_ratios = {}
        for name in self.loss_names:
            if name in task_losses:
                current_loss = self.running_mean_losses[name]
                initial_loss = self.initial_task_losses[name]
                loss_ratios[name] = current_loss / (initial_loss + 1e-8)
        
        # Calculate average relative decrease
        if not loss_ratios:  # FIX: guard empty
            self.update_count += 1
            return self.weights
        avg_loss_ratio = sum(loss_ratios.values()) / len(loss_ratios)
        
        # Calculate gradient norms
        grad_norms = {}
        for name in self.loss_names:
            if name in task_losses:
                grads = torch.autograd.grad(
                    task_losses[name], shared_parameters, 
                    retain_graph=True, create_graph=False, allow_unused=True
                )
                grad_norm = 0.0
                for grad in grads:
                    if grad is not None:
                        grad_norm += grad.norm().item() ** 2
                if grad_norm > 0:
                    grad_norms[name] = grad_norm ** 0.5
        
        if not grad_norms:  # FIX: guard empty
            self.update_count += 1
            return self.weights
        
        avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)
        
        for name in self.loss_names:
            if name in grad_norms and name in loss_ratios:
                target_grad = avg_grad_norm * (loss_ratios[name] ** self.alpha)
                weight_update = target_grad / (grad_norms[name] + 1e-8)
                # Apply momentum-like update
                new_w = 0.9 * self.weights[name] + 0.1 * float(weight_update)
                self.weights[name] = float(np.clip(new_w, 0.1, 10.0))  # FIX: always float
        
        self.update_count += 1
        return self.weights


class GraphTopologyExtractor:
    """Extracts graph structure from segmentation for topology constraints"""
    
    @staticmethod
    def extract_room_graph(segmentation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract room connectivity graph from segmentation mask"""
        B, C, H, W = segmentation.shape
        device = segmentation.device
        
        # Get room predictions (assume classes: 0=bg, 1=wall, 2=door, 3=window, 4=room)
        room_probs = F.softmax(segmentation, dim=1)
        room_mask = room_probs[:, 4] if C > 4 else torch.zeros((B, H, W), device=device)
        wall_mask = room_probs[:, 1] if C > 1 else torch.zeros((B, H, W), device=device)
        
        # Simple connectivity: rooms connected if they share wall boundary
        adjacency_matrices = []
        room_features = []
        
        for b in range(B):
            room_b = room_mask[b].detach().cpu().numpy()
            wall_b = wall_mask[b].detach().cpu().numpy()
            
            # Find connected components (rooms)
            try:
                from scipy import ndimage
                labeled_rooms, num_rooms = ndimage.label(room_b > 0.5)
                
                # Create adjacency matrix
                adj_matrix = np.zeros((max(num_rooms, 1), max(num_rooms, 1)))
                room_centroids = []
                
                for i in range(1, num_rooms + 1):
                    room_i_mask = (labeled_rooms == i)
                    if np.sum(room_i_mask) > 0:
                        centroid = ndimage.center_of_mass(room_i_mask)
                        room_centroids.append(centroid)
                        
                        # Check connectivity to other rooms through walls
                        for j in range(i + 1, num_rooms + 1):
                            room_j_mask = (labeled_rooms == j)
                            if np.sum(room_j_mask) > 0:
                                # Check if rooms are connected via wall adjacency
                                connectivity = GraphTopologyExtractor._check_room_connectivity(
                                    room_i_mask, room_j_mask, wall_b
                                )
                                adj_matrix[i-1, j-1] = connectivity
                                adj_matrix[j-1, i-1] = connectivity
                
                # Convert to tensor
                adj_tensor = torch.from_numpy(adj_matrix).float().to(device)
                centroids_tensor = torch.from_numpy(np.array(room_centroids) if room_centroids else np.zeros((0, 2))).float().to(device)
                
            except ImportError:
                # Fallback if scipy not available
                adj_tensor = torch.zeros((1, 1), device=device)
                centroids_tensor = torch.zeros((0, 2), device=device)
            
            adjacency_matrices.append(adj_tensor)
            room_features.append(centroids_tensor)
        
        return {
            "adjacency_matrices": adjacency_matrices,
            "room_features": room_features
        }
    
    @staticmethod
    def _check_room_connectivity(room1_mask, room2_mask, wall_mask):
        """Check if two rooms are connected through walls"""
        try:
            from scipy.ndimage import binary_dilation
            
            # Dilate room masks to check wall adjacency
            dilated1 = binary_dilation(room1_mask, iterations=2)
            dilated2 = binary_dilation(room2_mask, iterations=2)
            
            # Check overlap through wall areas
            wall_overlap = (dilated1 & dilated2) & (wall_mask > 0.3)
            return float(np.sum(wall_overlap) > 0)
        except ImportError:
            # Simple distance-based fallback
            return 0.0


class ResearchGradeLoss(nn.Module):
    """
    Multi-task loss combining:
    - Traditional losses (segmentation, SDF, attributes, polygons, voxels, topology)
    - NEW: Cross-modal latent consistency
    - NEW: Graph-based topology constraints  
    - NEW: Dynamic loss weighting via GradNorm
    - NEW: Conditional geometric losses based on run_full_geometric flag
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
        latent_consistency_weight: float = 0.5,
        graph_constraint_weight: float = 0.3,
        enable_dynamic_weighting: bool = True,
        gradnorm_alpha: float = 0.12,
        device: str = 'cuda',
        weight_update_freq: int = 10,
        weight_momentum: float = 0.9,
    ):
        super().__init__()

        # Store initial weights
        self.initial_weights = {
            'seg': float(seg_weight),
            'dice': float(dice_weight), 
            'sdf': float(sdf_weight),
            'attr': float(attr_weight),
            'polygon': float(polygon_weight),
            'voxel': float(voxel_weight),
            'topology': float(topology_weight),
            'latent_consistency': float(latent_consistency_weight),
            'graph': float(graph_constraint_weight)
        }
        
        # Current weights (will be dynamically updated)
        self.weights = self.initial_weights.copy()
        
        # Core losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Dynamic weighting
        self.enable_dynamic_weighting = enable_dynamic_weighting
        if enable_dynamic_weighting:
            self.loss_weighter = DynamicLossWeighter(
                list(self.initial_weights.keys()), alpha=gradnorm_alpha, device=device,
            )
            self.loss_weighter.update_freq = weight_update_freq
            self.loss_weighter.momentum = weight_momentum
        
        self.device = device
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights (called by trainer for curriculum scheduling)"""
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = float(value)

    def forward(self, predictions: dict, targets: dict, shared_parameters=None, run_full_geometric=True):
        """
        Compute multi-task loss with conditional geometric computation and dynamic weighting.
        
        Args:
            predictions: Model predictions dict
            targets: Ground truth targets dict
            shared_parameters: Model parameters for GradNorm (optional)
            run_full_geometric: Whether geometric losses should be computed
        
        Returns:
            tuple: (total_loss, individual_losses_dict)
        """
        device = self._get_device_from_inputs(predictions, targets)
        losses = {}
        total_loss = torch.tensor(0.0, device=device)

        # ---- 1) Core losses (always computed - lightweight) ----
        if "segmentation" in predictions and "mask" in targets:
            seg_pred = predictions["segmentation"]
            seg_target = targets["mask"].long()

            ce_loss = self.ce_loss(seg_pred, seg_target)
            losses["ce"] = ce_loss
            losses["seg"] = ce_loss  # alias for dynamic weighting

            dice_loss = self._dice_loss(seg_pred, seg_target)
            losses["dice"] = dice_loss

        if "sdf" in predictions and "mask" in targets:
            sdf_pred = predictions["sdf"]
            sdf_pred = torch.clamp(sdf_pred, -1.0, 1.0)   # FIX: prevent huge values
            sdf_target = self._mask_to_sdf(targets["mask"])
            sdf_target = sdf_target.to(sdf_pred.device).type_as(sdf_pred)
            losses["sdf"] = self.mse_loss(sdf_pred, sdf_target)

        if "attributes" in predictions and "attributes" in targets:
            pred_attr = predictions["attributes"].float()
            tgt_attr = targets["attributes"].float().to(pred_attr.device)
            losses["attr"] = self.l1_loss(pred_attr, tgt_attr)

        # ---- 2) Conditional geometric losses (heavy operations) ----
        if run_full_geometric:
            # Polygon loss (only if model produced polygons)
            if ("polygons" in predictions and predictions["polygons"] is not None and
                "polygons_gt" in targets):
                losses["polygon"] = self._polygon_loss(predictions, targets["polygons_gt"])
            else:
                # Zero loss if polygons not available
                losses["polygon"] = torch.tensor(0.0, device=device)

            # Voxel loss (only if model produced voxels)
            if ("voxels_pred" in predictions and predictions["voxels_pred"] is not None and
                "voxels_gt" in targets):
                pred_vox = predictions["voxels_pred"].float()
                tgt_vox = targets["voxels_gt"].float().to(pred_vox.device)
                losses["voxel"] = self._voxel_iou_loss(pred_vox, tgt_vox)
            else:
                # Zero loss if voxels not available
                losses["voxel"] = torch.tensor(0.0, device=device)

            # Cross-modal latent consistency (only if embeddings available)
            if ("latent_2d_embedding" in predictions and "latent_3d_embedding" in predictions and
                predictions["latent_2d_embedding"] is not None and predictions["latent_3d_embedding"] is not None):
                consistency_loss = self._latent_consistency_loss(
                    predictions["latent_2d_embedding"], 
                    predictions["latent_3d_embedding"]
                )
                losses["latent_consistency"] = consistency_loss
            else:
                losses["latent_consistency"] = torch.tensor(0.0, device=device)
        else:
            # When geometric computation is skipped, use zero losses
            losses["polygon"] = torch.tensor(0.0, device=device)
            losses["voxel"] = torch.tensor(0.0, device=device)
            losses["latent_consistency"] = torch.tensor(0.0, device=device)

        # ---- 3) Independent auxiliary losses (always computed if enabled) ----
        # Traditional topology loss
        if "segmentation" in predictions:
            losses["topology"] = self._topology_loss(predictions["segmentation"])

        # Graph-based topology constraints
        if "segmentation" in predictions:
            graph_loss = self._graph_topology_loss(predictions["segmentation"])
            losses["graph"] = graph_loss

        # ---- 4) Apply weighting ----
        if self.enable_dynamic_weighting and shared_parameters is not None:
            # Only include differentiable losses for GradNorm
            task_losses = {
                name: loss for name, loss in losses.items()
                if name in self.weights and isinstance(loss, torch.Tensor) and loss.requires_grad
            }

            dynamic_weights = self.loss_weighter.update_weights(task_losses, shared_parameters)

            # Apply weights (dynamic for diff losses, static for non-diff losses)
            for name, loss in losses.items():
                if name in self.weights:
                    if name in dynamic_weights:
                        weight = dynamic_weights[name]
                    else:
                        weight = self.weights[name]
                    total_loss = total_loss + weight * loss
        else:
            # Static weights
            for name, loss in losses.items():
                if name in self.weights:
                    total_loss = total_loss + self.weights[name] * loss

        # Final NaN/Inf guard
        for k, v in list(losses.items()):
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"[Warning] {k} loss is NaN/Inf → zeroed out")
                losses[k] = torch.tensor(0.0, device=device)

        losses["total"] = total_loss
        return total_loss, losses

    def __call__(self, predictions: dict, targets: dict, shared_parameters=None, run_full_geometric=True):
        """
        Convenience method for trainer compatibility
        
        Args:
            predictions: Model predictions dict  
            targets: Ground truth targets dict
            shared_parameters: Model parameters for GradNorm (optional)
            run_full_geometric: Whether to compute geometric losses
        
        Returns:
            tuple: (total_loss, individual_losses_dict)
        """
        return self.forward(predictions, targets, shared_parameters, run_full_geometric)

    def _get_device_from_inputs(self, predictions, targets):
        """Helper to determine device from inputs"""
        for pred_dict in [predictions, targets]:
            for value in pred_dict.values():
                if torch.is_tensor(value):
                    return value.device
        return self.device

    # ---- NEW: Cross-modal latent consistency loss ----
    def _latent_consistency_loss(self, embedding_2d: torch.Tensor, embedding_3d: torch.Tensor) -> torch.Tensor:
        """
        Ensure 2D floorplan embeddings match 3D voxelized structure embeddings
        embedding_2d: [B, D] - 2D floorplan embeddings
        embedding_3d: [B, D] - 3D structure embeddings
        """
        if embedding_2d.shape != embedding_3d.shape:
            # Project to same dimension if needed
            min_dim = min(embedding_2d.shape[-1], embedding_3d.shape[-1])
            embedding_2d = embedding_2d[..., :min_dim]
            embedding_3d = embedding_3d[..., :min_dim]
        
        # Cosine similarity loss (maximize similarity)
        target = torch.ones(embedding_2d.shape[0], device=embedding_2d.device)
        cosine_loss = self.cosine_loss(embedding_2d, embedding_3d, target)
        
        # L2 consistency loss
        l2_loss = F.mse_loss(embedding_2d, embedding_3d)
        
        return 0.7 * cosine_loss + 0.3 * l2_loss

    # ---- NEW: Graph-based topology constraints ----
    def _graph_topology_loss(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        """
        Graph-based topology constraints on room connectivity
        segmentation_logits: [B, C, H, W]
        """
        try:
            # Extract graph structure
            graph_data = GraphTopologyExtractor.extract_room_graph(segmentation_logits)
            device = segmentation_logits.device
            
            total_graph_loss = torch.tensor(0.0, device=device)
            batch_size = segmentation_logits.shape[0]
            
            for b in range(batch_size):
                if b < len(graph_data["adjacency_matrices"]):
                    adj_matrix = graph_data["adjacency_matrices"][b]
                    if adj_matrix.numel() == 0:
                        continue
                        
                    # Connectivity constraint: encourage reasonable connectivity
                    # Penalize isolated rooms (degree 0) and over-connected rooms
                    degrees = adj_matrix.sum(dim=1)
                    
                    # Isolation penalty (rooms should have at least 1 connection)
                    isolation_penalty = torch.exp(-degrees).mean()
                    
                    # Over-connection penalty (rooms shouldn't connect to everything)
                    max_reasonable_connections = min(4, adj_matrix.shape[0] - 1)
                    over_connection_penalty = F.relu(degrees - max_reasonable_connections).mean()
                    
                    # Graph smoothness (connected rooms should have similar features)
                    if b < len(graph_data["room_features"]) and graph_data["room_features"][b].numel() > 0:
                        room_features = graph_data["room_features"][b]
                        if room_features.shape[0] > 1:
                            feature_distances = torch.cdist(room_features, room_features)
                            # Weight by adjacency - connected rooms should be similar
                            smoothness_loss = (adj_matrix * feature_distances).sum() / (adj_matrix.sum() + 1e-6)
                        else:
                            smoothness_loss = torch.tensor(0.0, device=device)
                    else:
                        smoothness_loss = torch.tensor(0.0, device=device)
                    
                    batch_graph_loss = (0.4 * isolation_penalty + 
                                      0.3 * over_connection_penalty + 
                                      0.3 * smoothness_loss)
                    total_graph_loss = total_graph_loss + batch_graph_loss
            
            return total_graph_loss / batch_size
            
        except Exception as e:
            # Fallback to zero loss if graph extraction fails
            return torch.tensor(0.0, device=segmentation_logits.device)

    # ---- Existing helper methods (preserved) ----
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss implementation"""
        pred_soft = F.softmax(pred, dim=1)
        B = pred_soft.shape[0]
        C = pred_soft.shape[1]

        dice_losses = []
        for c in range(C):
            pred_c = pred_soft[:, c, :, :]
            target_c = (target == c).float().to(pred_c.device)
            intersection = (pred_c * target_c).view(B, -1).sum(dim=1)
            union = pred_c.view(B, -1).sum(dim=1) + target_c.view(B, -1).sum(dim=1)
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_losses.append((1.0 - dice).mean())

        return torch.stack(dice_losses).mean()

    def _mask_to_sdf(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert mask to SDF with performance warning"""
        device = mask.device if torch.is_tensor(mask) else None
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=device)

        B, H, W = mask.shape
        sdf = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        # FIX: Add performance warning for CV2 bottleneck
        if B > 8:  # Warn for large batches
            print(f"[Performance Warning] SDF conversion with batch_size={B} uses CPU cv2 - consider GPU implementation")

        for b in range(B):
            mask_np = mask[b].detach().cpu().numpy().astype(np.uint8)  # FIX: explicit detach
            try:
                dist_inside = cv2.distanceTransform((mask_np > 0).astype(np.uint8), cv2.DIST_L2, 5)
                dist_outside = cv2.distanceTransform((mask_np == 0).astype(np.uint8), cv2.DIST_L2, 5)
                sdf_np = dist_inside.astype(np.float32) - dist_outside.astype(np.float32)
                sdf_np = np.tanh(sdf_np / 10.0).astype(np.float32)
                sdf[b, 0] = torch.from_numpy(sdf_np)
            except Exception:
                # Fallback if cv2 fails
                sdf[b, 0] = torch.zeros_like(mask[b].float())

        return sdf

    def _polygon_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Polygon/DVX loss (preserved from original)"""
        pred_polys = predictions.get("polygons")
        tgt_polys = targets.get("polygons")
        valid_mask = targets.get("valid_mask")

        if pred_polys is None or tgt_polys is None:
            return torch.tensor(0.0, device=pred_polys.device if pred_polys is not None else self.device)

        pred_polys = pred_polys.float()
        tgt_polys = tgt_polys.float().to(pred_polys.device)

        point_loss = self.mse_loss(pred_polys, tgt_polys)

        pred_valid = predictions.get("polygon_validity")
        if pred_valid is None or valid_mask is None:
            validity_loss = torch.tensor(0.0, device=pred_polys.device)
        else:
            pred_valid = pred_valid.float().to(pred_polys.device)
            valid_mask_f = valid_mask.float().to(pred_polys.device)
            validity_loss = self.mse_loss(pred_valid, valid_mask_f)

        smoothness_loss = self._polygon_smoothness(pred_polys)
        rect_loss = self._rectilinearity_loss(pred_polys)

        return point_loss + 0.1 * validity_loss + 0.05 * smoothness_loss + 0.1 * rect_loss

    def _polygon_smoothness(self, polygons: torch.Tensor) -> torch.Tensor:
        """Polygon smoothness loss (preserved)"""
        if polygons is None or polygons.numel() == 0:
            return torch.tensor(0.0, device=polygons.device if polygons is not None else self.device)

        p1 = polygons
        p2 = torch.roll(polygons, -1, dims=2)
        p3 = torch.roll(polygons, -2, dims=2)
        curvature = torch.norm(p1 - 2.0 * p2 + p3, dim=-1)
        return curvature.mean()

    def _rectilinearity_loss(self, polygons: torch.Tensor) -> torch.Tensor:
        """Encourage axis-aligned structure (preserved)"""
        if polygons is None or polygons.numel() == 0:
            return torch.tensor(0.0, device=polygons.device if polygons is not None else self.device)

        edges = torch.roll(polygons, -1, dims=2) - polygons
        edge_norms = torch.norm(edges, dim=-1, keepdim=True)
        edges_normalized = edges / (edge_norms + 1e-6)

        edge1 = edges_normalized
        edge2 = torch.roll(edges_normalized, -1, dims=2)

        cos_angles = (edge1 * edge2).sum(dim=-1)
        cos2 = cos_angles ** 2
        perp_penalty = cos2
        parallel_penalty = (cos2 - 1.0) ** 2
        angle_penalty = torch.minimum(perp_penalty, parallel_penalty)
        return angle_penalty.mean()

    def _voxel_iou_loss(self, pred_voxels: torch.Tensor, target_voxels: torch.Tensor) -> torch.Tensor:
        """3D voxel IoU loss (preserved)"""
        pred_prob = torch.sigmoid(torch.clamp(pred_voxels, -10.0, 10.0))  # FIX: safe sigmoid range
        target = target_voxels.float().to(pred_prob.device)

        intersection = (pred_prob * target).view(pred_prob.shape[0], -1).sum(dim=1)
        union = (pred_prob.view(pred_prob.shape[0], -1).sum(dim=1) + 
                target.view(target.shape[0], -1).sum(dim=1) - intersection)

        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()

    def _topology_loss(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        """Traditional topology loss (preserved)"""
        seg_soft = F.softmax(segmentation_logits, dim=1)
        C = seg_soft.shape[1]
        device = seg_soft.device
        
        walls = seg_soft[:, 1] if C > 1 else torch.zeros_like(seg_soft[:, 0])
        doors = seg_soft[:, 2] if C > 2 else torch.zeros_like(walls)
        windows = seg_soft[:, 3] if C > 3 else torch.zeros_like(walls)

        door_wall_overlap = doors * walls
        window_wall_overlap = windows * walls

        door_penalty = torch.maximum(doors - door_wall_overlap, torch.zeros_like(doors))
        window_penalty = torch.maximum(windows - window_wall_overlap, torch.zeros_like(windows))

        connectivity_loss = self._connectivity_loss(walls)

        return door_penalty.mean() + window_penalty.mean() + 0.1 * connectivity_loss

    def _connectivity_loss(self, wall_prob: torch.Tensor) -> torch.Tensor:
        """Connectivity loss for walls (preserved)"""
        if wall_prob is None or wall_prob.numel() == 0:
            return torch.tensor(0.0, device=wall_prob.device if wall_prob is not None else self.device)

        kernel = torch.ones((1, 1, 3, 3), device=wall_prob.device, dtype=wall_prob.dtype) / 9.0
        neighbors = F.conv2d(wall_prob.unsqueeze(1), kernel, padding=1).squeeze(1)

        isolation_penalty = wall_prob * torch.exp(-neighbors)
        return isolation_penalty.mean()


class LossScheduler:
    """Manages curriculum-based loss weight scheduling"""
    
    def __init__(self, config):
        self.config = config
        self.loss_schedules = config.loss_schedule
        
    def get_scheduled_weights(self, current_stage: int, current_epoch: int, 
                            stage_epoch: int, total_stage_epochs: int,
                            base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate loss weights based on curriculum schedule
        
        Args:
            current_stage: Current training stage (1, 2, 3)
            current_epoch: Global epoch count
            stage_epoch: Epoch within current stage
            total_stage_epochs: Total epochs planned for current stage
            base_weights: Base weight configuration
        """
        scheduled_weights = base_weights.copy()
        
        for loss_name, schedule_type in self.loss_schedules.items():
            if loss_name not in scheduled_weights:
                continue
                
            base_weight = scheduled_weights[loss_name]
            
            if schedule_type == "static":
                # Keep original weight
                continue
                
            elif schedule_type == "progressive":
                # Gradually increase throughout training
                if loss_name == "topology":
                    start_weight = self.config.topology_start_weight
                    end_weight = self.config.topology_end_weight
                    ramp_epochs = self.config.topology_ramp_epochs
                    progress = min(current_epoch / ramp_epochs, 1.0)
                    scheduled_weights[loss_name] = start_weight + progress * (end_weight - start_weight)
                    
            elif schedule_type == "linear_ramp":
                # Linear increase within current stage
                progress = stage_epoch / max(total_stage_epochs, 1)
                scheduled_weights[loss_name] = base_weight * progress
                
            elif schedule_type == "exponential":
                # Exponential increase
                progress = stage_epoch / max(total_stage_epochs, 1)
                scheduled_weights[loss_name] = base_weight * (progress ** 2)
                
            elif schedule_type == "early_decay":
                # Decay after Stage 1 (for SDF loss)
                if current_stage > 1:
                    scheduled_weights[loss_name] = base_weight * 0.3
                    
            elif schedule_type == "staged_ramp":
                # Ramp up during specific stage (polygon in Stage 2)
                if current_stage == 2:
                    progress = stage_epoch / max(total_stage_epochs, 1)
                    scheduled_weights[loss_name] = base_weight * progress
                elif current_stage < 2:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "late_ramp":
                # Ramp up in Stage 3 (voxel loss)
                if current_stage == 3:
                    progress = stage_epoch / max(total_stage_epochs, 1)
                    scheduled_weights[loss_name] = base_weight * progress
                elif current_stage < 3:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "mid_ramp":
                # Activate mid-training (latent consistency)
                if current_stage >= 2:
                    if current_stage == 2:
                        progress = min(stage_epoch / (total_stage_epochs * 0.5), 1.0)
                        scheduled_weights[loss_name] = base_weight * progress
                    else:  # Stage 3
                        scheduled_weights[loss_name] = base_weight
                else:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "delayed_ramp":
                # FIX: gentler ramp for graph constraints
                if current_epoch >= self.config.graph_start_epoch:
                    epochs_since_start = current_epoch - self.config.graph_start_epoch
                    ramp_duration = 50  # FIX: slower ramp (was 20)
                    progress = min(epochs_since_start / ramp_duration, 1.0)
                    scheduled_weights[loss_name] = self.config.graph_end_weight * progress
                else:
                    scheduled_weights[loss_name] = 0.0
        
        return scheduled_weights