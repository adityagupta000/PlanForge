"""
Advanced loss functions for multi-task training with dynamic weighting
Enhanced with cross-modal consistency, graph constraints, and GradNorm
Modified to support conditional geometric losses via run_full_geometric flag
FIXED: Dynamic loss component initialization for stage transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import networkx as nx


class DynamicLossWeighter:
    def __init__(self, loss_names: List[str], alpha: float = 0.12, device: str = 'cuda'):
        self.loss_names = loss_names
        self.alpha = alpha
        self.device = device
        
        # Initialize weights for all known loss components
        self.weights = {name: 1.0 for name in loss_names}
        self.initial_task_losses = {}
        # Add running normalization to prevent raw magnitude issues
        self.running_mean_losses = {name: 0.0 for name in loss_names}
        self.running_std_losses = {name: 1.0 for name in loss_names}  # NEW
        self.update_count = 0
        
        print(f"[DynamicWeighter] Initialized with loss components: {loss_names}")
        
    def update_weights(self, task_losses: Dict[str, torch.Tensor], 
                      shared_parameters, update_freq: int = 10):
        """Update loss weights using GradNorm with stability improvements"""
        if self.update_count % update_freq != 0:
            self.update_count += 1
            return self.weights
        
        # Initialize tracking for new loss components
        newly_initialized = []
        for name, loss in task_losses.items():
            if name not in self.initial_task_losses:
                loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
                # Use log-scale initialization for stability
                self.initial_task_losses[name] = max(np.log(abs(loss_val) + 1e-6), -10.0)
                
                if name not in self.weights:
                    self.weights[name] = 1.0
                if name not in self.running_mean_losses:
                    self.running_mean_losses[name] = loss_val
                if name not in self.running_std_losses:
                    self.running_std_losses[name] = max(abs(loss_val), 1e-3)
                    
                newly_initialized.append(name)
        
        # Update running statistics with EMA
        for name, loss in task_losses.items():
            loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            if name in self.running_mean_losses:
                # Exponential moving average for mean and std
                self.running_mean_losses[name] = 0.9 * self.running_mean_losses[name] + 0.1 * loss_val
                
                # Update running std using Welford's algorithm
                delta = loss_val - self.running_mean_losses[name]
                self.running_std_losses[name] = 0.9 * self.running_std_losses[name] + 0.1 * abs(delta)
                self.running_std_losses[name] = max(self.running_std_losses[name], 1e-3)
        
        # Calculate normalized relative decrease rates
        loss_ratios = {}
        for name, loss in task_losses.items():
            if name in self.initial_task_losses and self.initial_task_losses[name] > -9.0:
                # Normalize current loss by running statistics
                current_loss = self.running_mean_losses.get(name, loss.item())
                normalized_current = current_loss / (self.running_std_losses[name] + 1e-6)
                
                initial_loss = self.initial_task_losses[name]
                # Use log-space ratios for stability
                loss_ratios[name] = np.exp(min(max(normalized_current - initial_loss, -5.0), 5.0))
        
        if not loss_ratios:
            self.update_count += 1
            return self.weights
        
        # Calculate gradient norms with improved stability
        grad_norms = {}
        for name, loss in task_losses.items():
            if name in loss_ratios:
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                if not torch.isfinite(loss):
                    continue
                    
                try:
                    grads = torch.autograd.grad(
                        loss, shared_parameters, 
                        retain_graph=True, create_graph=False, allow_unused=True
                    )
                    
                    grad_norm_sq = 0.0
                    valid_grads = False
                    for grad in grads:
                        if grad is not None and torch.isfinite(grad).all():
                            # Apply gradient norm stabilization
                            clipped_grad = torch.clamp(grad, -10.0, 10.0)
                            grad_norm_sq += clipped_grad.norm().item() ** 2
                            valid_grads = True
                    
                    if valid_grads and grad_norm_sq > 0:
                        # Use log-scale gradient norms
                        grad_norms[name] = np.log(grad_norm_sq ** 0.5 + 1e-8)
                        
                except Exception as e:
                    continue
        
        if not grad_norms:
            self.update_count += 1
            return self.weights
        
        # Normalize gradient norms
        mean_grad_norm = np.mean(list(grad_norms.values()))
        
        # Update weights with improved stability
        for name in grad_norms.keys():
            if name in loss_ratios:
                # Calculate target gradient in log space
                target_grad_log = mean_grad_norm + self.alpha * np.log(loss_ratios[name] + 1e-8)
                current_grad_log = grad_norms[name]
                
                # Calculate weight update with damping
                weight_update_log = target_grad_log - current_grad_log
                weight_update = np.exp(np.clip(weight_update_log, -1.0, 1.0))  # Stronger clipping
                
                # Apply update with momentum and stronger constraints
                current_weight = self.weights.get(name, 1.0)
                new_weight = 0.8 * current_weight + 0.2 * weight_update  # More conservative
                self.weights[name] = float(np.clip(new_weight, 0.1, 2.0))  # Tighter bounds
        
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
            except Exception as e:
                # General fallback for any other issues
                print(f"Warning: Graph extraction failed: {e}")
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
    Multi-task loss with stage-aware dynamic weighting:
    - Stage 1: segmentation, dice, sdf, attributes, topology, graph
    - Stage 2: + polygon (DVX)
    - Stage 3: + voxel, latent_consistency (full geometric)
    
    FIXED: Dynamic initialization handles new loss components during stage transitions
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

        # Store initial weights for all possible loss components
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
        
        # Dynamic weighting with all possible loss names
        self.enable_dynamic_weighting = enable_dynamic_weighting
        if enable_dynamic_weighting:
            all_loss_names = list(self.initial_weights.keys())
            self.loss_weighter = DynamicLossWeighter(
                all_loss_names, alpha=gradnorm_alpha, device=device,
            )
            self.update_freq = weight_update_freq
            self.momentum = weight_momentum
            print(f"[ResearchGradeLoss] Dynamic weighting enabled for: {all_loss_names}")
        
        self.device = device
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights (called by trainer for curriculum scheduling)"""
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = float(value)

    def forward(self, predictions: dict, targets: dict, shared_parameters=None, run_full_geometric=True):
        """Compute multi-task loss with proper normalization and aggregation"""
        # Input validation and sanitization
        predictions = self._sanitize_predictions(predictions)
        targets = self._sanitize_targets(targets)
                
        device = self._get_device_from_inputs(predictions, targets)
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # ---- STAGE 1 LOSSES with proper scaling ----
        if "segmentation" in predictions and "mask" in targets:
            seg_pred = predictions["segmentation"]
            seg_target = targets["mask"].long()

            # Scale CE loss by number of pixels for consistency
            ce_loss = self.ce_loss(seg_pred, seg_target)
            losses["seg"] = ce_loss

            dice_loss = self._dice_loss(seg_pred, seg_target)
            losses["dice"] = dice_loss

        if "sdf" in predictions and "mask" in targets:
            sdf_pred = predictions["sdf"]
            sdf_pred = torch.clamp(sdf_pred, -1.0, 1.0)
            sdf_target = self._mask_to_sdf(targets["mask"])
            sdf_target = sdf_target.to(sdf_pred.device).type_as(sdf_pred)
            # Normalize SDF loss by spatial dimensions
            sdf_loss = self.mse_loss(sdf_pred, sdf_target)
            losses["sdf"] = sdf_loss

        if "attributes" in predictions and "attributes" in targets:
            pred_attr = predictions["attributes"].float()
            tgt_attr = targets["attributes"].float().to(pred_attr.device)
            # Normalize attribute loss by number of attributes
            attr_loss = self.l1_loss(pred_attr, tgt_attr) / pred_attr.shape[-1]
            losses["attr"] = attr_loss

        # Apply proper scaling to topology losses
        if "segmentation" in predictions:
            topology_loss = self._topology_loss(predictions["segmentation"])
            # Scale topology loss to reasonable magnitude
            losses["topology"] = topology_loss * 0.5
            
            graph_loss = self._graph_topology_loss(predictions["segmentation"])
            # Graph loss is already scaled in the function above
            losses["graph"] = graph_loss

        # ---- GEOMETRIC LOSSES with normalization ----
        if run_full_geometric:
            if ("polygons" in predictions and predictions["polygons"] is not None and
                "polygons_gt" in targets):
                poly_loss = self._polygon_loss(predictions, targets["polygons_gt"])
                # Normalize polygon loss by number of polygons and points
                if "polygons" in predictions and predictions["polygons"] is not None:
                    B, P, N, _ = predictions["polygons"].shape
                    poly_loss = poly_loss / (P * N)  # Normalize by polygon complexity
                losses["polygon"] = poly_loss
            else:
                losses["polygon"] = torch.tensor(0.0, device=device)

            if ("voxels_pred" in predictions and predictions["voxels_pred"] is not None and
                "voxels_gt" in targets):
                pred_vox = predictions["voxels_pred"].float()
                tgt_vox = targets["voxels_gt"].float().to(pred_vox.device)
                voxel_loss = self._voxel_iou_loss(pred_vox, tgt_vox)
                losses["voxel"] = voxel_loss
            else:
                losses["voxel"] = torch.tensor(0.0, device=device)

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
            losses["polygon"] = torch.tensor(0.0, device=device)
            losses["voxel"] = torch.tensor(0.0, device=device)
            losses["latent_consistency"] = torch.tensor(0.0, device=device)

        # ---- IMPROVED WEIGHTING AND AGGREGATION ----
        active_losses = {
            name: loss for name, loss in losses.items()
            if isinstance(loss, torch.Tensor) and loss.requires_grad and loss.item() > 1e-8
        }

        if self.enable_dynamic_weighting and shared_parameters is not None and active_losses:
            try:
                dynamic_weights = self.loss_weighter.update_weights(
                    active_losses, shared_parameters, self.update_freq
                )
                
                # Apply weights with additional stability checks
                for name, loss in losses.items():
                    if name in self.weights and isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                        weight = dynamic_weights.get(name, self.weights[name])
                        # Apply weight with gradient scaling for stability
                        weighted_loss = weight * loss
                        if torch.isfinite(weighted_loss):
                            total_loss = total_loss + weighted_loss
                            
            except Exception as e:
                print(f"[ResearchGradeLoss] Dynamic weighting failed: {e}, falling back to static weights")
                # Fallback to static weights
                for name, loss in losses.items():
                    if name in self.weights and isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                        total_loss = total_loss + self.weights[name] * loss
        else:
            # Static weights with stability
            for name, loss in losses.items():
                if name in self.weights and isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                    total_loss = total_loss + self.weights[name] * loss

        # Final loss scaling and validation
        if not torch.isfinite(total_loss):
            print("[ResearchGradeLoss] CRITICAL: Non-finite total loss detected!")
            print(f"  Active component losses:")
            for name, loss in losses.items():
                if name != "total" and isinstance(loss, torch.Tensor):
                    loss_val = loss.item() if torch.isfinite(loss) else "INF/NAN"
                    print(f"    {name}: {loss_val}")
            
            # Create a small positive fallback loss to maintain gradient flow
            total_loss = torch.tensor(0.01, device=device, requires_grad=True)
        else:
            # Clamp to reasonable range
            total_loss = torch.clamp(total_loss, 0.0, 100.0)

        losses["total"] = total_loss
        return total_loss, losses

    def __call__(self, predictions: dict, targets: dict, shared_parameters=None, run_full_geometric=True):
        """Trainer compatibility method"""
        return self.forward(predictions, targets, shared_parameters, run_full_geometric)

    def _sanitize_predictions(self, predictions: dict) -> dict:
        """Sanitize prediction tensors"""
        sanitized = {}
        for name, tensor in predictions.items():
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"WARNING: NaN/Inf in predictions[{name}] - zeroing out")
                    sanitized[name] = torch.zeros_like(tensor)
                else:
                    sanitized[name] = tensor
            else:
                sanitized[name] = tensor
        return sanitized

    def _sanitize_targets(self, targets: dict) -> dict:
        """Sanitize target tensors"""
        sanitized = {}
        for name, tensor in targets.items():
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"WARNING: NaN/Inf in targets[{name}] - zeroing out")
                    sanitized[name] = torch.zeros_like(tensor)
                else:
                    sanitized[name] = tensor
            else:
                sanitized[name] = tensor
        return sanitized

    def _get_device_from_inputs(self, predictions, targets):
        """Helper to determine device from inputs"""
        for pred_dict in [predictions, targets]:
            for value in pred_dict.values():
                if torch.is_tensor(value):
                    return value.device
        return self.device

    # ---- LOSS COMPONENT IMPLEMENTATIONS ----
    
    def _latent_consistency_loss(self, embedding_2d: torch.Tensor, embedding_3d: torch.Tensor) -> torch.Tensor:
        """Cross-modal latent consistency loss"""
        if embedding_2d.shape != embedding_3d.shape:
            min_dim = min(embedding_2d.shape[-1], embedding_3d.shape[-1])
            embedding_2d = embedding_2d[..., :min_dim]
            embedding_3d = embedding_3d[..., :min_dim]
        
        target = torch.ones(embedding_2d.shape[0], device=embedding_2d.device)
        cosine_loss = self.cosine_loss(embedding_2d, embedding_3d, target)
        l2_loss = F.mse_loss(embedding_2d, embedding_3d)
        
        return 0.7 * cosine_loss + 0.3 * l2_loss

    def _graph_topology_loss(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        """Graph-based topology constraints with proper normalization"""
        try:
            graph_data = GraphTopologyExtractor.extract_room_graph(segmentation_logits)
            device = segmentation_logits.device
            
            total_graph_loss = torch.tensor(0.0, device=device)
            batch_size = segmentation_logits.shape[0]
            valid_batches = 0
            
            for b in range(batch_size):
                if b < len(graph_data["adjacency_matrices"]):
                    adj_matrix = graph_data["adjacency_matrices"][b]
                    if adj_matrix.numel() == 0:
                        continue
                    
                    # Normalize by matrix size to prevent scale explosion
                    matrix_size = max(adj_matrix.shape[0], 1)
                    norm_factor = 1.0 / (matrix_size + 1e-6)
                    
                    degrees = adj_matrix.sum(dim=1)
                    isolation_penalty = torch.exp(-degrees).mean() * norm_factor
                    
                    max_reasonable_connections = min(4, adj_matrix.shape[0] - 1)
                    over_connection_penalty = F.relu(degrees - max_reasonable_connections).mean() * norm_factor
                    
                    if b < len(graph_data["room_features"]) and graph_data["room_features"][b].numel() > 0:
                        room_features = graph_data["room_features"][b]
                        if room_features.shape[0] > 1:
                            feature_distances = torch.cdist(room_features, room_features)
                            # Normalize distance computation
                            mean_distance = feature_distances.mean()
                            normalized_distances = feature_distances / (mean_distance + 1e-6)
                            smoothness_loss = (adj_matrix * normalized_distances).sum() / (adj_matrix.sum() + 1e-6)
                            smoothness_loss = smoothness_loss * norm_factor
                        else:
                            smoothness_loss = torch.tensor(0.0, device=device)
                    else:
                        smoothness_loss = torch.tensor(0.0, device=device)
                    
                    # Apply strong penalty scaling to keep graph loss in reasonable range
                    batch_graph_loss = (0.4 * isolation_penalty + 
                                     0.3 * over_connection_penalty + 
                                     0.3 * smoothness_loss) * 0.1  # Scale down by 10x
                    
                    total_graph_loss = total_graph_loss + batch_graph_loss
                    valid_batches += 1
            
            # Average over valid batches and apply final normalization
            if valid_batches > 0:
                return total_graph_loss / valid_batches
            else:
                return torch.tensor(0.0, device=segmentation_logits.device)
                
        except Exception as e:
            return torch.tensor(0.0, device=segmentation_logits.device)
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss implementation"""
        pred_soft = F.softmax(pred, dim=1)
        B, C = pred_soft.shape[:2]

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
        """Convert mask to SDF"""
        device = mask.device if torch.is_tensor(mask) else self.device
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=device)

        B, H, W = mask.shape
        sdf = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for b in range(B):
            mask_np = mask[b].detach().cpu().numpy().astype(np.uint8)
            try:
                dist_inside = cv2.distanceTransform((mask_np > 0).astype(np.uint8), cv2.DIST_L2, 5)
                dist_outside = cv2.distanceTransform((mask_np == 0).astype(np.uint8), cv2.DIST_L2, 5)
                sdf_np = dist_inside.astype(np.float32) - dist_outside.astype(np.float32)
                sdf_np = np.tanh(sdf_np / 10.0).astype(np.float32)
                sdf[b, 0] = torch.from_numpy(sdf_np)
            except Exception:
                sdf[b, 0] = torch.zeros_like(mask[b].float())

        return sdf

    def _polygon_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Polygon/DVX loss"""
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
        """Polygon smoothness loss"""
        if polygons is None or polygons.numel() == 0:
            return torch.tensor(0.0, device=polygons.device if polygons is not None else self.device)

        p1 = polygons
        p2 = torch.roll(polygons, -1, dims=2)
        p3 = torch.roll(polygons, -2, dims=2)
        curvature = torch.norm(p1 - 2.0 * p2 + p3, dim=-1)
        return curvature.mean()

    def _rectilinearity_loss(self, polygons: torch.Tensor) -> torch.Tensor:
        """Encourage axis-aligned structure"""
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
        """3D voxel IoU loss"""
        pred_prob = torch.sigmoid(torch.clamp(pred_voxels, -10.0, 10.0))
        target = target_voxels.float().to(pred_prob.device)

        intersection = (pred_prob * target).view(pred_prob.shape[0], -1).sum(dim=1)
        union = (pred_prob.view(pred_prob.shape[0], -1).sum(dim=1) + 
                target.view(target.shape[0], -1).sum(dim=1) - intersection)

        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()

    def _topology_loss(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        """Traditional topology loss"""
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
        """Connectivity loss for walls"""
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
        """Calculate loss weights based on curriculum schedule"""
        scheduled_weights = base_weights.copy()
        
        for loss_name, schedule_type in self.loss_schedules.items():
            if loss_name not in scheduled_weights:
                continue
                
            base_weight = scheduled_weights[loss_name]
            
            if schedule_type == "static":
                continue
                
            elif schedule_type == "progressive":
                if loss_name == "topology":
                    start_weight = self.config.topology_start_weight
                    end_weight = self.config.topology_end_weight
                    ramp_epochs = self.config.topology_ramp_epochs
                    progress = min(current_epoch / ramp_epochs, 1.0)
                    scheduled_weights[loss_name] = start_weight + progress * (end_weight - start_weight)
                    
            elif schedule_type == "linear_ramp":
                progress = stage_epoch / max(total_stage_epochs, 1)
                scheduled_weights[loss_name] = base_weight * progress
                
            elif schedule_type == "exponential":
                progress = stage_epoch / max(total_stage_epochs, 1)
                scheduled_weights[loss_name] = base_weight * (progress ** 2)
                
            elif schedule_type == "early_decay":
                if current_stage > 1:
                    scheduled_weights[loss_name] = base_weight * 0.3
                    
            elif schedule_type == "staged_ramp":
                if current_stage == 2:
                    progress = stage_epoch / max(total_stage_epochs, 1)
                    scheduled_weights[loss_name] = base_weight * progress
                elif current_stage < 2:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "late_ramp":
                if current_stage == 3:
                    progress = stage_epoch / max(total_stage_epochs, 1)
                    scheduled_weights[loss_name] = base_weight * progress
                elif current_stage < 3:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "mid_ramp":
                if current_stage >= 2:
                    if current_stage == 2:
                        progress = min(stage_epoch / (total_stage_epochs * 0.5), 1.0)
                        scheduled_weights[loss_name] = base_weight * progress
                    else:
                        scheduled_weights[loss_name] = base_weight
                else:
                    scheduled_weights[loss_name] = 0.0
                    
            elif schedule_type == "delayed_ramp":
                if current_epoch >= self.config.graph_start_epoch:
                    epochs_since_start = current_epoch - self.config.graph_start_epoch
                    ramp_duration = 50
                    progress = min(epochs_since_start / ramp_duration, 1.0)
                    scheduled_weights[loss_name] = self.config.graph_end_weight * progress
                else:
                    scheduled_weights[loss_name] = 0.0
        
        return scheduled_weights