"""
evaluation/metrics.py - Enhanced Evaluation Metrics
Comprehensive evaluation utilities for Neural-Geometric 3D Model Generator
Compatible with adaptive multi-stage training and all loss components
"""

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from models.model import NeuralGeometric3DGenerator
from dataset import AdvancedFloorPlanDataset


# ==============================================================================
# Core Metric Computation Functions
# ==============================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) for binary segmentation
    
    Args:
        pred: Binary prediction tensor
        target: Binary target tensor
    
    Returns:
        IoU score (0-1)
    """
    pred = pred.bool()
    target = target.bool()
    
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / (union + 1e-6)).item()


def compute_3d_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute 3D Intersection over Union for voxel grids
    
    Args:
        pred: Predicted voxel grid
        target: Target voxel grid
    
    Returns:
        3D IoU score (0-1)
    """
    pred_bool = pred.bool()
    target_bool = target.bool()

    intersection = (pred_bool & target_bool).float().sum()
    union = (pred_bool | target_bool).float().sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / (union + 1e-6)).item()


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Dice coefficient (F1 score)
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        Dice score (0-1)
    """
    pred_bool = pred.bool()
    target_bool = target.bool()
    
    intersection = (pred_bool & target_bool).float().sum()
    
    pred_sum = pred_bool.float().sum()
    target_sum = target_bool.float().sum()
    
    if pred_sum + target_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    return dice.item()


def compute_polygon_metrics(
    pred_polygons: torch.Tensor,
    gt_polygons: torch.Tensor,
    validity_pred: torch.Tensor,
    validity_gt: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for polygon prediction
    
    Args:
        pred_polygons: Predicted polygons [N, P, 2]
        gt_polygons: Ground truth polygons [N, P, 2]
        validity_pred: Predicted validity scores [N]
        validity_gt: Ground truth validity mask [N]
    
    Returns:
        Dictionary with chamfer_distance and validity_accuracy
    """
    # Ensure all tensors are on the same device
    device = pred_polygons.device
    pred_polygons = pred_polygons.to(device)
    gt_polygons = gt_polygons.to(device)
    validity_pred = validity_pred.to(device)
    validity_gt = validity_gt.to(device)
    
    # Filter valid polygons
    valid_pred_mask = validity_pred > 0.5
    valid_gt_mask = validity_gt.bool()
    
    valid_pred = pred_polygons[valid_pred_mask] if valid_pred_mask.any() else pred_polygons[:0]
    valid_gt = gt_polygons[valid_gt_mask] if valid_gt_mask.any() else gt_polygons[:0]

    if len(valid_pred) == 0 or len(valid_gt) == 0:
        return {
            "chamfer_distance": float('inf'),
            "validity_accuracy": 0.0,
            "num_valid_pred": len(valid_pred),
            "num_valid_gt": len(valid_gt)
        }

    # Compute Chamfer distance
    chamfer_dist = 0.0
    for pred_poly in valid_pred:
        min_dist = float('inf')
        for gt_poly in valid_gt:
            # Remove zero-padded points
            pred_points = pred_poly[pred_poly.sum(dim=-1) > 0]
            gt_points = gt_poly[gt_poly.sum(dim=-1) > 0]
            
            if len(pred_points) > 0 and len(gt_points) > 0:
                # Ensure both tensors on same device
                pred_points = pred_points.to(device)
                gt_points = gt_points.to(device)
                
                dist = torch.norm(pred_points.unsqueeze(0) - gt_points.unsqueeze(1), dim=-1)
                dist = dist.min().item()
                min_dist = min(min_dist, dist)
        
        if min_dist != float('inf'):
            chamfer_dist += min_dist

    if len(valid_pred) > 0:
        chamfer_dist /= len(valid_pred)
    else:
        chamfer_dist = float('inf')

    # Validity accuracy - move to same device for comparison
    validity_pred_bool = (validity_pred > 0.5).to(device)
    validity_gt_bool = validity_gt.bool().to(device)
    validity_acc = (validity_pred_bool == validity_gt_bool).float().mean().item()

    return {
        "chamfer_distance": chamfer_dist,
        "validity_accuracy": validity_acc,
        "num_valid_pred": len(valid_pred),
        "num_valid_gt": len(valid_gt)
    }


def compute_architectural_metrics(predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute architecture-specific metrics from predictions
    
    Args:
        predictions: Model predictions dictionary
    
    Returns:
        Dictionary with architectural metrics
    """
    metrics = {}
    
    if "segmentation" not in predictions:
        return metrics
    
    try:
        # Get segmentation prediction
        seg_pred = torch.argmax(predictions["segmentation"], dim=1)[0]
        seg_np = seg_pred.cpu().numpy().astype(np.uint8)
        
        # Room count (class 0 = background/room)
        room_mask = (seg_np == 0).astype(np.uint8)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        room_count = len([c for c in contours if cv2.contourArea(c) > 100])
        metrics["room_count"] = int(room_count)
        
        # Wall connectivity (class 1)
        wall_mask = (seg_np == 1).astype(np.uint8)
        wall_components = cv2.connectedComponents(wall_mask)[0] - 1
        metrics["wall_components"] = max(0, int(wall_components))
        
        # Door count (class 2)
        door_mask = (seg_np == 2).astype(np.uint8)
        door_contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        door_count = len([c for c in door_contours if cv2.contourArea(c) > 50])
        metrics["door_count"] = int(door_count)
        
        # Window count (class 3)
        window_mask = (seg_np == 3).astype(np.uint8)
        window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        window_count = len([c for c in window_contours if cv2.contourArea(c) > 20])
        metrics["window_count"] = int(window_count)
        
        # Polygon metrics
        if "polygons" in predictions and predictions["polygons"] is not None:
            validity = predictions["polygon_validity"][0] if predictions.get("polygon_validity") is not None else None
            if validity is not None:
                valid_polygons = (validity > 0.5).sum().item()
                metrics["valid_polygon_count"] = int(valid_polygons)
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics


# ==============================================================================
# Main Evaluator Class
# ==============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation with support for all task components
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize evaluator with pretrained model
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model = NeuralGeometric3DGenerator()

        # Load model with safe state dict loading
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        
        # Load state dict with flexible shape matching
        self.model.load_state_dict(model_state, strict=False)
        self.model.to(device)
        self.model.eval()

        print(f"✓ Loaded model from {model_path}")

    def evaluate_dataset(self, test_dataset: AdvancedFloorPlanDataset) -> Dict[str, float]:
        """
        Comprehensive evaluation on full test dataset
        
        Args:
            test_dataset: Dataset to evaluate on
        
        Returns:
            Dictionary with summary statistics
        """
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Storage for all metrics
        metrics = {
            "segmentation": {"ious": [], "class_ious": []},
            "attributes": {"maes": [], "mses": []},
            "voxels": {"ious": [], "dice_scores": []},
            "polygons": {"chamfer_distances": [], "validity_accs": []},
            "architectural": []
        }

        print("\n[INFO] Running evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move to device
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # Forward pass
                predictions = self.model(batch["image"])

                # Segmentation evaluation
                seg_metrics = self._evaluate_segmentation(
                    predictions["segmentation"], batch["mask"]
                )
                metrics["segmentation"]["ious"].append(seg_metrics["iou"])
                metrics["segmentation"]["class_ious"].append(seg_metrics["class_ious"])

                # Attributes evaluation
                attr_metrics = self._evaluate_attributes(
                    predictions["attributes"], batch["attributes"]
                )
                metrics["attributes"]["maes"].append(attr_metrics["mae"])
                metrics["attributes"]["mses"].append(attr_metrics["mse"])

                # Voxels evaluation
                if predictions.get("voxels_pred") is not None:
                    voxel_metrics = self._evaluate_voxels(
                        predictions["voxels_pred"], batch["voxels_gt"]
                    )
                    metrics["voxels"]["ious"].append(voxel_metrics["iou"])
                    metrics["voxels"]["dice_scores"].append(voxel_metrics["dice"])
                else:
                    metrics["voxels"]["ious"].append(0.0)
                    metrics["voxels"]["dice_scores"].append(0.0)

                # Polygons evaluation
                if predictions.get("polygons") is not None:
                    poly_metrics = self._evaluate_polygons(
                        predictions["polygons"],
                        predictions.get("polygon_validity"),
                        batch["polygons_gt"]
                    )
                    metrics["polygons"]["chamfer_distances"].append(poly_metrics["chamfer_distance"])
                    metrics["polygons"]["validity_accs"].append(poly_metrics["validity_accuracy"])
                else:
                    metrics["polygons"]["chamfer_distances"].append(float('inf'))
                    metrics["polygons"]["validity_accs"].append(0.0)

                # Architectural metrics
                arch_metrics = compute_architectural_metrics(predictions)
                metrics["architectural"].append(arch_metrics)

        return self._compute_summary_metrics(metrics)

    def _evaluate_segmentation(self, pred_seg: torch.Tensor, target_mask: torch.Tensor) -> Dict:
        """Evaluate segmentation performance"""
        seg_pred = torch.argmax(pred_seg, dim=1)

        # Overall IoU
        overall_iou = compute_iou(seg_pred, target_mask)

        # Per-class IoU
        num_classes = pred_seg.shape[1]
        class_ious = []

        for c in range(num_classes):
            pred_c = (seg_pred == c)
            target_c = (target_mask == c)

            if target_c.sum() > 0:
                iou_c = compute_iou(pred_c, target_c)
                class_ious.append(iou_c)

        return {
            "iou": overall_iou,
            "class_ious": class_ious
        }

    def _evaluate_attributes(self, pred_attrs: torch.Tensor, target_attrs: torch.Tensor) -> Dict:
        """Evaluate attribute prediction"""
        pred_attrs = pred_attrs.float()
        target_attrs = target_attrs.float().to(pred_attrs.device)
        
        mae = torch.mean(torch.abs(pred_attrs - target_attrs)).item()
        mse = torch.mean((pred_attrs - target_attrs) ** 2).item()

        return {"mae": mae, "mse": mse}

    def _evaluate_voxels(self, pred_voxels: torch.Tensor, target_voxels: torch.Tensor) -> Dict:
        """Evaluate 3D voxel prediction"""
        pred_binary = (torch.sigmoid(pred_voxels) > 0.5).float()
        target_float = target_voxels.float()

        # 3D IoU
        iou_3d = compute_3d_iou(pred_binary, target_float)

        # 3D Dice score
        intersection = (pred_binary * target_float).sum()
        dice = (2 * intersection) / (pred_binary.sum() + target_float.sum() + 1e-6)

        return {
            "iou": iou_3d,
            "dice": dice.item()
        }

    def _evaluate_polygons(
        self,
        pred_polygons: Optional[torch.Tensor],
        pred_validity: Optional[torch.Tensor],
        gt_polygons: Dict
    ) -> Dict:
        """Evaluate polygon prediction"""
        if pred_polygons is None or pred_validity is None:
            return {
                "chamfer_distance": float('inf'),
                "validity_accuracy": 0.0
            }
        
        # Ensure all on same device
        device = pred_polygons.device
        gt_poly_tensor = gt_polygons["polygons"][0].to(device)
        gt_valid_mask = gt_polygons["valid_mask"][0].to(device)
        
        return compute_polygon_metrics(
            pred_polygons[0],
            gt_poly_tensor,
            pred_validity[0],
            gt_valid_mask
        )

    def _compute_summary_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Compute summary statistics from collected metrics"""
        summary = {}

        # Segmentation
        summary["segmentation_mIoU"] = np.mean(metrics["segmentation"]["ious"])
        summary["segmentation_std"] = np.std(metrics["segmentation"]["ious"])

        # Attributes
        summary["attribute_MAE"] = np.mean(metrics["attributes"]["maes"])
        summary["attribute_MAE_std"] = np.std(metrics["attributes"]["maes"])
        summary["attribute_MSE"] = np.mean(metrics["attributes"]["mses"])

        # Voxels
        summary["voxel_mIoU"] = np.mean(metrics["voxels"]["ious"])
        summary["voxel_mIoU_std"] = np.std(metrics["voxels"]["ious"])
        summary["voxel_dice"] = np.mean(metrics["voxels"]["dice_scores"])

        # Polygons
        valid_chamfer = [d for d in metrics["polygons"]["chamfer_distances"] if d != float('inf')]
        if valid_chamfer:
            summary["polygon_chamfer"] = np.mean(valid_chamfer)
            summary["polygon_chamfer_std"] = np.std(valid_chamfer)
        else:
            summary["polygon_chamfer"] = float('inf')
            summary["polygon_chamfer_std"] = 0.0

        summary["polygon_validity_acc"] = np.mean(metrics["polygons"]["validity_accs"])

        return summary

    def print_evaluation_results(self, summary: Dict) -> None:
        """Print formatted evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS - COMPREHENSIVE METRICS")
        print("=" * 70)

        print(f"\nSegmentation:")
        print(f"  mIoU:     {summary['segmentation_mIoU']:.4f} ± {summary['segmentation_std']:.4f}")

        print(f"\nAttributes:")
        print(f"  MAE:      {summary['attribute_MAE']:.4f} ± {summary['attribute_MAE_std']:.4f}")
        print(f"  MSE:      {summary['attribute_MSE']:.4f}")

        print(f"\n3D Voxels:")
        print(f"  mIoU:     {summary['voxel_mIoU']:.4f} ± {summary['voxel_mIoU_std']:.4f}")
        print(f"  Dice:     {summary['voxel_dice']:.4f}")

        print(f"\nPolygons:")
        if summary['polygon_chamfer'] != float('inf'):
            print(f"  Chamfer:  {summary['polygon_chamfer']:.4f} ± {summary['polygon_chamfer_std']:.4f}")
        else:
            print(f"  Chamfer:  No valid polygons")
        print(f"  Validity: {summary['polygon_validity_acc']:.4f}")

        print("\n" + "=" * 70 + "\n")


def evaluate_model(model_path: str, data_dir: str = "./data/floorplans") -> Optional[Dict]:
    """
    Standalone evaluation function
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Path to dataset directory
    
    Returns:
        Summary metrics dictionary
    """
    # Load test dataset
    test_dataset = AdvancedFloorPlanDataset(data_dir, split="test")

    if len(test_dataset) == 0:
        print("No test samples found!")
        return None

    # Create evaluator
    evaluator = ModelEvaluator(model_path)

    # Run evaluation
    summary = evaluator.evaluate_dataset(test_dataset)

    # Print results
    evaluator.print_evaluation_results(summary)

    return summary