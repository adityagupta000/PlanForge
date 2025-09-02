"""
Visualization and utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path


def plot_training_history(history, save_path="training_history.png"):
    """Plot training curves for all stages"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (stage, data) in enumerate(history.items()):
        if data["train_loss"]:  # Only plot if stage was executed
            axes[idx].plot(data["train_loss"], label="Train", linewidth=2)
            axes[idx].plot(data["val_loss"], label="Validation", linewidth=2)
            axes[idx].set_title(f"{stage.upper()} Training")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Loss")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_predictions(image, predictions, targets=None, save_path=None):
    """Visualize model predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    if len(image.shape) == 4:
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image.permute(1, 2, 0).cpu().numpy()
    
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    # Predicted segmentation
    if "segmentation" in predictions:
        seg_pred = torch.argmax(predictions["segmentation"], dim=1)[0].cpu().numpy()
        axes[0, 1].imshow(seg_pred, cmap='tab10')
        axes[0, 1].set_title("Predicted Segmentation")
        axes[0, 1].axis('off')
    
    # Ground truth segmentation (if available)
    if targets and "mask" in targets:
        gt_mask = targets["mask"][0].cpu().numpy()
        axes[0, 2].imshow(gt_mask, cmap='tab10')
        axes[0, 2].set_title("Ground Truth Segmentation")
        axes[0, 2].axis('off')
    
    # SDF prediction
    if "sdf" in predictions:
        sdf_pred = predictions["sdf"][0, 0].cpu().numpy()
        im = axes[1, 0].imshow(sdf_pred, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 0].set_title("Predicted SDF")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Polygon visualization
    if "polygons" in predictions:
        poly_vis = visualize_polygons(
            predictions["polygons"][0], 
            predictions["polygon_validity"][0],
            image_size=(256, 256)
        )
        axes[1, 1].imshow(poly_vis)
        axes[1, 1].set_title("Predicted Polygons")
        axes[1, 1].axis('off')
    
    # 3D voxel slice
    if "voxels_pred" in predictions:
        voxels = torch.sigmoid(predictions["voxels_pred"][0]).cpu().numpy()
        # Show middle slice
        mid_slice = voxels[voxels.shape[0]//2]
        axes[1, 2].imshow(mid_slice, cmap='viridis')
        axes[1, 2].set_title("3D Voxels (Mid Slice)")
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def visualize_polygons(polygons, validity, image_size=(256, 256), threshold=0.5):
    """Visualize predicted polygons"""
    vis_img = np.zeros((*image_size, 3), dtype=np.uint8)
    
    for poly_idx, (polygon, valid_score) in enumerate(zip(polygons, validity)):
        if valid_score > threshold:
            # Convert to image coordinates
            points = polygon.cpu().numpy() * np.array(image_size)
            
            # Remove zero-padded points
            valid_points = points[points.sum(axis=1) > 0]
            
            if len(valid_points) >= 3:
                points_int = valid_points.astype(np.int32)
                
                # Different colors for different polygons
                color = plt.cm.tab10(poly_idx % 10)[:3]
                color = tuple(int(c * 255) for c in color)
                
                cv2.polylines(vis_img, [points_int], True, color, 2)
                
                # Add polygon index
                center = points_int.mean(axis=0).astype(int)
                cv2.putText(vis_img, str(poly_idx), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_img


def save_model_outputs(predictions, output_dir, sample_id):
    """Save all model outputs for detailed analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(exist_ok=True)
    
    # Save segmentation
    if "segmentation" in predictions:
        seg_pred = torch.argmax(predictions["segmentation"], dim=1)[0].cpu().numpy()
        cv2.imwrite(str(sample_dir / "segmentation.png"), seg_pred * 50)
    
    # Save SDF
    if "sdf" in predictions:
        sdf_pred = predictions["sdf"][0, 0].cpu().numpy()
        sdf_normalized = ((sdf_pred + 1) * 127.5).astype(np.uint8)
        cv2.imwrite(str(sample_dir / "sdf.png"), sdf_normalized)
    
    # Save attributes
    if "attributes" in predictions:
        attrs = predictions["attributes"][0].cpu().numpy()
        np.save(sample_dir / "attributes.npy", attrs)
    
    # Save polygons
    if "polygons" in predictions:
        polygons = predictions["polygons"][0].cpu().numpy()
        validity = predictions["polygon_validity"][0].cpu().numpy()
        
        np.save(sample_dir / "polygons.npy", polygons)
        np.save(sample_dir / "polygon_validity.npy", validity)
    
    # Save voxels
    if "voxels_pred" in predictions:
        voxels = torch.sigmoid(predictions["voxels_pred"][0]).cpu().numpy()
        np.save(sample_dir / "voxels.npy", voxels)


def create_comparison_grid(input_images, predictions, targets=None, num_samples=4):
    """Create a comparison grid showing inputs, predictions, and targets"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(min(num_samples, len(input_images))):
        # Input image
        img = input_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}: Input")
        axes[i, 0].axis('off')
        
        # Predicted segmentation
        seg_pred = torch.argmax(predictions["segmentation"][i], dim=0).cpu().numpy()
        axes[i, 1].imshow(seg_pred, cmap='tab10')
        axes[i, 1].set_title("Predicted Seg")
        axes[i, 1].axis('off')
        
        # Ground truth segmentation (if available)
        if targets and "mask" in targets:
            gt_mask = targets["mask"][i].cpu().numpy()
            axes[i, 2].imshow(gt_mask, cmap='tab10')
            axes[i, 2].set_title("GT Segmentation")
        else:
            axes[i, 2].text(0.5, 0.5, "No GT", ha='center', va='center', 
                           transform=axes[i, 2].transAxes)
            axes[i, 2].set_title("GT Segmentation")
        axes[i, 2].axis('off')
        
        # Polygon overlay
        poly_vis = visualize_polygons(
            predictions["polygons"][i], 
            predictions["polygon_validity"][i]
        )
        axes[i, 3].imshow(poly_vis)
        axes[i, 3].set_title("Predicted Polygons")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    return fig


def analyze_failure_cases(predictions, targets, threshold_iou=0.5):
    """Analyze failure cases for debugging"""
    failure_indices = []
    
    for i, (pred_seg, gt_mask) in enumerate(zip(predictions["segmentation"], targets["mask"])):
        seg_pred = torch.argmax(pred_seg, dim=0)
        iou = compute_iou(seg_pred, gt_mask)
        
        if iou < threshold_iou:
            failure_indices.append({
                "index": i,
                "iou": iou,
                "pred_classes": torch.unique(seg_pred).tolist(),
                "gt_classes": torch.unique(gt_mask).tolist()
            })
    
    return failure_indices


class ProgressiveVisualization:
    """Track and visualize training progress"""
    
    def __init__(self, save_dir="./training_progress"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def log_epoch_results(self, epoch, stage, predictions, targets, sample_image):
        """Log results for a specific epoch"""
        epoch_dir = self.save_dir / f"{stage}_epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Save prediction visualization
        fig = plt.figure(figsize=(12, 8))
        visualize_predictions(sample_image, predictions, targets)
        plt.savefig(epoch_dir / "predictions.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save individual outputs
        save_model_outputs(predictions, epoch_dir, "sample")
        
    def create_training_animation(self, stage, metric_name="total_loss"):
        """Create animated GIF showing training progress"""
        # This would create an animation of training progress
        # Implementation depends on having saved epoch results
        pass


def compute_architectural_metrics(predictions, image_size=(256, 256)):
    """Compute architecture-specific metrics"""
    metrics = {}
    
    if "segmentation" in predictions:
        seg_pred = torch.argmax(predictions["segmentation"], dim=1)[0]
        
        # Room count
        room_mask = (seg_pred == 0).cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        room_count = len([c for c in contours if cv2.contourArea(c) > 100])
        metrics["room_count"] = room_count
        
        # Wall connectivity
        wall_mask = (seg_pred == 1).cpu().numpy().astype(np.uint8)
        wall_components = cv2.connectedComponents(wall_mask)[0] - 1  # Subtract background
        metrics["wall_components"] = max(0, wall_components)
        
        # Door and window counts
        door_pixels = (seg_pred == 2).sum().item()
        window_pixels = (seg_pred == 3).sum().item()
        metrics["door_pixels"] = door_pixels
        metrics["window_pixels"] = window_pixels
    
    if "polygons" in predictions:
        validity = predictions["polygon_validity"][0]
        valid_polygons = (validity > 0.5).sum().item()
        metrics["valid_polygon_count"] = valid_polygons
        
        # Average polygon area
        polygons = predictions["polygons"][0]
        areas = []
        for poly_idx, (polygon, valid) in enumerate(zip(polygons, validity)):
            if valid > 0.5:
                # Compute polygon area using shoelace formula
                points = polygon.cpu().numpy() * np.array(image_size)
                valid_points = points[points.sum(axis=1) > 0]
                if len(valid_points) >= 3:
                    area = compute_polygon_area(valid_points)
                    areas.append(area)
        
        metrics["avg_polygon_area"] = np.mean(areas) if areas else 0.0
    
    return metrics


def compute_polygon_area(points):
    """Compute polygon area using shoelace formula"""
    if len(points) < 3:
        return 0.0
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Shoelace formula
    area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    return area


def create_model_summary_report(model, sample_input, save_path="model_summary.txt"):
    """Create detailed model summary report"""
    with open(save_path, "w") as f:
        f.write("Neural-Geometric 3D Model Generator - Model Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Model architecture
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 20 + "\n")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n\n")
        
        # Component breakdown
        f.write("COMPONENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        seg_params = sum(p.numel() for p in model.seg_head.parameters())
        attr_params = sum(p.numel() for p in model.attr_head.parameters())
        sdf_params = sum(p.numel() for p in model.sdf_head.parameters())
        dvx_params = sum(p.numel() for p in model.dvx.parameters())
        ext_params = sum(p.numel() for p in model.extrusion.parameters())
        
        f.write(f"Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)\n")
        f.write(f"Segmentation Head: {seg_params:,} ({seg_params/total_params*100:.1f}%)\n")
        f.write(f"Attribute Head: {attr_params:,} ({attr_params/total_params*100:.1f}%)\n")
        f.write(f"SDF Head: {sdf_params:,} ({sdf_params/total_params*100:.1f}%)\n")
        f.write(f"DVX Module: {dvx_params:,} ({dvx_params/total_params*100:.1f}%)\n")
        f.write(f"Extrusion Module: {ext_params:,} ({ext_params/total_params*100:.1f}%)\n\n")
        
        # Forward pass analysis
        f.write("FORWARD PASS ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        
        model.eval()
        with torch.no_grad():
            predictions = model(sample_input)
            
            for key, value in predictions.items():
                if torch.is_tensor(value):
                    f.write(f"{key}: {list(value.shape)} - {value.dtype}\n")
                else:
                    f.write(f"{key}: {type(value)}\n")
    
    print(f"Model summary saved to {save_path}")


def debug_gradient_flow(model, loss):
    """Debug gradient flow through the model"""
    print("Gradient Flow Analysis:")
    print("-" * 30)
    
    total_norm = 0
    component_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            
            # Group by component
            component = name.split('.')[0]
            if component not in component_norms:
                component_norms[component] = 0
            component_norms[component] += param_norm ** 2
    
    total_norm = total_norm ** 0.5
    
    print(f"Total gradient norm: {total_norm:.4f}")
    print("Component gradient norms:")
    
    for component, norm in component_norms.items():
        norm = norm ** 0.5
        print(f"  {component}: {norm:.4f} ({norm/total_norm*100:.1f}%)")


def create_3d_visualization(voxels, output_path="3d_preview.png"):
    """Create 3D visualization of voxel prediction"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Convert to binary
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.cpu().numpy()
        
        binary_voxels = voxels > 0.5
        
        # Get occupied voxel coordinates
        occupied = np.where(binary_voxels)
        
        if len(occupied[0]) == 0:
            print("No occupied voxels to visualize")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot occupied voxels
        ax.scatter(occupied[0], occupied[1], occupied[2], 
                  c=occupied[2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Voxel Occupancy')
        
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"3D visualization saved to {output_path}")
        
    except ImportError:
        print("3D visualization requires matplotlib with 3D support")