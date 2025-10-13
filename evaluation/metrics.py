"""
Evaluation metrics and utilities for the 3D Model Generator
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.model import NeuralGeometric3DGenerator
from dataset import AdvancedFloorPlanDataset


def compute_iou(pred, target):
    """Compute IoU for segmentation"""
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection / (union + 1e-6)).item()


def compute_3d_iou(pred, target):
    """Compute 3D IoU for voxel grids"""
    pred_bool = pred.bool()
    target_bool = target.bool()

    intersection = (pred_bool & target_bool).float().sum()
    union = (pred_bool | target_bool).float().sum()

    return (intersection / (union + 1e-6)).item()


def compute_polygon_metrics(pred_polygons, gt_polygons, validity_pred, validity_gt):
    """Compute metrics for polygon prediction"""
    # Chamfer distance between polygon sets
    valid_pred = pred_polygons[validity_pred > 0.5]
    valid_gt = gt_polygons[validity_gt]
    
    if len(valid_pred) == 0 or len(valid_gt) == 0:
        return {"chamfer_distance": float('inf'), "validity_accuracy": 0.0}
    
    # Simplified chamfer distance computation
    chamfer_dist = 0.0
    for pred_poly in valid_pred:
        min_dist = float('inf')
        for gt_poly in valid_gt:
            dist = torch.norm(pred_poly - gt_poly, dim=-1).min().item()
            min_dist = min(min_dist, dist)
        chamfer_dist += min_dist
    
    chamfer_dist /= len(valid_pred)
    
    # Validity accuracy
    validity_acc = ((validity_pred > 0.5) == validity_gt).float().mean().item()
    
    return {
        "chamfer_distance": chamfer_dist,
        "validity_accuracy": validity_acc
    }


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = NeuralGeometric3DGenerator()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")

    def evaluate_dataset(self, test_dataset):
        """Comprehensive evaluation on test dataset"""
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Metrics storage
        metrics = {
            "segmentation": {"ious": [], "class_ious": []},
            "attributes": {"maes": [], "mses": []},
            "voxels": {"ious": [], "dice_scores": []},
            "polygons": {"chamfer_distances": [], "validity_accs": []},
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}

                predictions = self.model(batch["image"])

                # Evaluate segmentation
                seg_metrics = self._evaluate_segmentation(
                    predictions["segmentation"], batch["mask"]
                )
                metrics["segmentation"]["ious"].append(seg_metrics["iou"])
                metrics["segmentation"]["class_ious"].append(seg_metrics["class_ious"])

                # Evaluate attributes
                attr_metrics = self._evaluate_attributes(
                    predictions["attributes"], batch["attributes"]
                )
                metrics["attributes"]["maes"].append(attr_metrics["mae"])
                metrics["attributes"]["mses"].append(attr_metrics["mse"])

                # Evaluate voxels
                voxel_metrics = self._evaluate_voxels(
                    predictions["voxels_pred"], batch["voxels_gt"]
                )
                metrics["voxels"]["ious"].append(voxel_metrics["iou"])
                metrics["voxels"]["dice_scores"].append(voxel_metrics["dice"])

                # Evaluate polygons
                poly_metrics = self._evaluate_polygons(
                    predictions["polygons"], 
                    predictions["polygon_validity"],
                    batch["polygons_gt"]
                )
                metrics["polygons"]["chamfer_distances"].append(poly_metrics["chamfer_distance"])
                metrics["polygons"]["validity_accs"].append(poly_metrics["validity_accuracy"])

                if (batch_idx + 1) % 10 == 0:
                    print(f"Evaluated {batch_idx + 1}/{len(test_loader)} samples")

        return self._compute_summary_metrics(metrics)

    def _evaluate_segmentation(self, pred_seg, target_mask):
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
            
            if target_c.sum() > 0:  # Only compute if class exists in ground truth
                iou_c = compute_iou(pred_c, target_c)
                class_ious.append(iou_c)
        
        return {
            "iou": overall_iou,
            "class_ious": class_ious
        }

    def _evaluate_attributes(self, pred_attrs, target_attrs):
        """Evaluate attribute prediction"""
        mae = torch.mean(torch.abs(pred_attrs - target_attrs)).item()
        mse = torch.mean((pred_attrs - target_attrs) ** 2).item()
        
        return {"mae": mae, "mse": mse}

    def _evaluate_voxels(self, pred_voxels, target_voxels):
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

    def _evaluate_polygons(self, pred_polygons, pred_validity, gt_polygons):
        """Evaluate polygon prediction"""
        return compute_polygon_metrics(
            pred_polygons[0], 
            gt_polygons["polygons"][0],
            pred_validity[0],
            gt_polygons["valid_mask"][0]
        )

    def _compute_summary_metrics(self, metrics):
        """Compute summary statistics"""
        summary = {}
        
        # Segmentation
        summary["segmentation_mIoU"] = np.mean(metrics["segmentation"]["ious"])
        summary["segmentation_std"] = np.std(metrics["segmentation"]["ious"])
        
        # Attributes
        summary["attribute_MAE"] = np.mean(metrics["attributes"]["maes"])
        summary["attribute_MAE_std"] = np.std(metrics["attributes"]["maes"])
        
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

    def print_evaluation_results(self, summary):
        """Print formatted evaluation results"""
        print("=" * 60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"Segmentation mIoU: {summary['segmentation_mIoU']:.4f} ± {summary['segmentation_std']:.4f}")
        print(f"Attribute MAE: {summary['attribute_MAE']:.4f} ± {summary['attribute_MAE_std']:.4f}")
        print(f"Voxel 3D mIoU: {summary['voxel_mIoU']:.4f} ± {summary['voxel_mIoU_std']:.4f}")
        print(f"Voxel Dice Score: {summary['voxel_dice']:.4f}")
        
        if summary['polygon_chamfer'] != float('inf'):
            print(f"Polygon Chamfer Distance: {summary['polygon_chamfer']:.4f} ± {summary['polygon_chamfer_std']:.4f}")
        else:
            print("Polygon Chamfer Distance: No valid polygons")
            
        print(f"Polygon Validity Accuracy: {summary['polygon_validity_acc']:.4f}")
        print("=" * 60)


def evaluate_model(model_path, data_dir="./data/floorplans"):
    """Standalone evaluation function"""
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