"""
evaluate.py - Enhanced Evaluation Script
Comprehensive evaluation CLI for Neural-Geometric 3D Model Generator
Compatible with adaptive multi-stage training and novel loss components

Usage examples:
    python evaluate.py --model_path checkpoints/final_model.pth --data_dir ./data/floorplans
    python evaluate.py --model_path checkpoints/final_model.pth --visualize --save_outputs --num_viz 20
    python evaluate.py --model_path checkpoints/final_model.pth --per_sample_json --limit_samples 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
from dataset import AdvancedFloorPlanDataset
from evaluation.metrics import ModelEvaluator, compute_architectural_metrics
from inference.engine import ResearchInferenceEngine
from utils.visualization import (
    visualize_predictions,
    create_comparison_grid,
    compute_architectural_metrics as viz_arch_metrics
)


def save_json(obj, path: Path):
    """Save dictionary to JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"✓ Saved JSON -> {path}")


def gather_per_sample_metrics(
    evaluator: ModelEvaluator,
    dataset: AdvancedFloorPlanDataset,
    device: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Re-run evaluation loop sample-by-sample and collect per-sample metrics.
    Uses evaluator._evaluate_* helper methods for consistency with overall evaluation.
    
    Args:
        evaluator: ModelEvaluator instance
        dataset: Dataset to evaluate on
        device: Device to use
        max_samples: Optional limit on number of samples
    
    Returns:
        List of per-sample metric dictionaries
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_sample = []

    print(f"\n[INFO] Collecting per-sample metrics...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Per-sample evaluation")):
            if max_samples is not None and idx >= max_samples:
                break

            # Move tensors to device
            batch_for_model = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch_for_model[k] = v.to(device)
                else:
                    batch_for_model[k] = v

            # Forward pass
            predictions = evaluator.model(batch_for_model["image"])

            # Evaluate each component
            seg_res = evaluator._evaluate_segmentation(
                predictions["segmentation"], 
                batch_for_model["mask"]
            )
            
            attr_res = evaluator._evaluate_attributes(
                predictions["attributes"],
                batch_for_model["attributes"].to(device)
            )
            
            # Voxel evaluation (handle None case)
            if predictions.get("voxels_pred") is not None:
                voxel_res = evaluator._evaluate_voxels(
                    predictions["voxels_pred"],
                    batch_for_model["voxels_gt"].to(device)
                )
            else:
                voxel_res = {"iou": 0.0, "dice": 0.0, "note": "voxels_pred was None"}
            
            # Polygon evaluation
            poly_res = evaluator._evaluate_polygons(
                predictions["polygons"],
                predictions.get("polygon_validity"),
                batch["polygons_gt"]
            )

            # Extract sample ID
            sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else batch["sample_id"]
            
            # Compile sample metrics
            sample_metrics = {
                "sample_id": str(sample_id),
                "segmentation": seg_res,
                "attributes": attr_res,
                "voxels": voxel_res,
                "polygons": poly_res,
            }
            
            # Add architectural metrics if available
            try:
                arch_metrics = compute_architectural_metrics(predictions)
                sample_metrics["architectural"] = arch_metrics
            except Exception as e:
                sample_metrics["architectural"] = {"error": str(e)}
            
            per_sample.append(sample_metrics)

    return per_sample


def run_visualization_and_exports(
    engine: ResearchInferenceEngine,
    dataset: AdvancedFloorPlanDataset,
    output_dir: Path,
    device: str,
    num_viz: int = 10,
    max_export: int = 5,
):
    """
    Generate visualizations and optionally run deterministic 3D exports
    
    Args:
        engine: Inference engine instance
        dataset: Dataset to visualize from
        output_dir: Directory to save outputs
        device: Device to use
        num_viz: Number of visualizations to create
        max_export: Number of 3D exports to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    viz_count = 0
    export_count = 0

    print(f"\n[INFO] Generating visualizations and exports...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Visualizations", total=min(num_viz, len(loader)))):
            if viz_count >= num_viz and export_count >= max_export:
                break

            sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else batch["sample_id"]
            sample_dir = Path(dataset.data_dir) / dataset.split / str(sample_id)

            # Prepare tensors
            image_tensor = batch["image"].to(device)
            target_mask = batch["mask"].unsqueeze(0) if batch["mask"].dim() == 2 else batch["mask"]
            target_attrs = batch["attributes"].unsqueeze(0) if batch["attributes"].dim() == 1 else batch["attributes"]

            # Model predictions
            predictions = engine.model(image_tensor)

            # Visualization
            if viz_count < num_viz:
                vis_path = output_dir / f"viz_{sample_id}.png"
                try:
                    visualize_predictions(
                        image_tensor,
                        predictions,
                        {"mask": target_mask, "attributes": target_attrs},
                        save_path=str(vis_path),
                    )
                    viz_count += 1
                except Exception as e:
                    print(f"[!] Visualization failed for {sample_id}: {e}")

            # Export deterministic 3D model
            if export_count < max_export:
                image_file = sample_dir / "image.png"
                if image_file.exists():
                    out_obj = output_dir / f"{sample_id}_predicted_model.obj"
                    try:
                        success = engine.generate_3d_model(
                            str(image_file), 
                            str(out_obj),
                            export_intermediate=True
                        )
                        if success:
                            print(f"✓ Exported 3D model: {out_obj.name}")
                        export_count += 1
                    except Exception as e:
                        print(f"[!] 3D export failed for {sample_id}: {e}")


def create_evaluation_report(
    summary: Dict,
    per_sample_metrics: Optional[List[Dict]],
    output_dir: Path,
    model_path: str,
    dataset_info: Dict
):
    """
    Create comprehensive evaluation report with statistics and insights
    
    Args:
        summary: Overall evaluation summary
        per_sample_metrics: Per-sample metrics (optional)
        output_dir: Directory to save report
        model_path: Path to evaluated model
        dataset_info: Information about dataset
    """
    report_path = output_dir / "evaluation_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("Neural-Geometric 3D Model Generator\n")
        f.write("=" * 80 + "\n\n")
        
        # Model information
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset information
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset Size: {dataset_info.get('size', 'N/A')} samples\n")
        f.write(f"Split: {dataset_info.get('split', 'test')}\n")
        f.write(f"Data Directory: {dataset_info.get('data_dir', 'N/A')}\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        
        # Segmentation
        f.write(f"Segmentation mIoU:     {summary.get('segmentation_mIoU', 0.0):.4f} ")
        f.write(f"± {summary.get('segmentation_std', 0.0):.4f}\n")
        
        # Attributes
        f.write(f"Attribute MAE:         {summary.get('attribute_MAE', 0.0):.4f} ")
        f.write(f"± {summary.get('attribute_MAE_std', 0.0):.4f}\n")
        
        # Voxels
        f.write(f"Voxel 3D mIoU:         {summary.get('voxel_mIoU', 0.0):.4f} ")
        f.write(f"± {summary.get('voxel_mIoU_std', 0.0):.4f}\n")
        f.write(f"Voxel Dice Score:      {summary.get('voxel_dice', 0.0):.4f}\n")
        
        # Polygons
        if summary.get('polygon_chamfer') != float('inf'):
            f.write(f"Polygon Chamfer Dist:  {summary.get('polygon_chamfer', 0.0):.4f} ")
            f.write(f"± {summary.get('polygon_chamfer_std', 0.0):.4f}\n")
        else:
            f.write(f"Polygon Chamfer Dist:  No valid polygons\n")
        
        f.write(f"Polygon Validity Acc:  {summary.get('polygon_validity_acc', 0.0):.4f}\n\n")
        
        # Per-sample statistics
        if per_sample_metrics:
            f.write("PER-SAMPLE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Samples Evaluated:     {len(per_sample_metrics)}\n")
            
            # Calculate statistics
            seg_ious = [s['segmentation']['iou'] for s in per_sample_metrics]
            attr_maes = [s['attributes']['mae'] for s in per_sample_metrics]
            voxel_ious = [s['voxels']['iou'] for s in per_sample_metrics if 'iou' in s['voxels']]
            
            f.write(f"\nSegmentation IoU:\n")
            f.write(f"  Min:  {min(seg_ious):.4f}\n")
            f.write(f"  Max:  {max(seg_ious):.4f}\n")
            f.write(f"  Med:  {np.median(seg_ious):.4f}\n")
            
            f.write(f"\nAttribute MAE:\n")
            f.write(f"  Min:  {min(attr_maes):.4f}\n")
            f.write(f"  Max:  {max(attr_maes):.4f}\n")
            f.write(f"  Med:  {np.median(attr_maes):.4f}\n")
            
            if voxel_ious:
                f.write(f"\nVoxel IoU:\n")
                f.write(f"  Min:  {min(voxel_ious):.4f}\n")
                f.write(f"  Max:  {max(voxel_ious):.4f}\n")
                f.write(f"  Med:  {np.median(voxel_ious):.4f}\n")
            
            # Find best and worst samples
            best_idx = seg_ious.index(max(seg_ious))
            worst_idx = seg_ious.index(min(seg_ious))
            
            f.write(f"\nBest Sample:  {per_sample_metrics[best_idx]['sample_id']} ")
            f.write(f"(IoU: {seg_ious[best_idx]:.4f})\n")
            f.write(f"Worst Sample: {per_sample_metrics[worst_idx]['sample_id']} ")
            f.write(f"(IoU: {seg_ious[worst_idx]:.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Detailed evaluation report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Neural-Geometric 3D Model Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate.py --model_path checkpoints/final_model.pth
  
  # With visualizations
  python evaluate.py --model_path checkpoints/final_model.pth --visualize --num_viz 20
  
  # Full evaluation with exports and per-sample metrics
  python evaluate.py --model_path checkpoints/final_model.pth --visualize --save_outputs --per_sample_json
  
  # Quick test with limited samples
  python evaluate.py --model_path checkpoints/final_model.pth --limit_samples 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path", "-m",
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_dir", "-d",
        default="./data/floorplans",
        help="Dataset root directory with train/val/test splits (default: ./data/floorplans)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if omitted"
    )
    
    # Evaluation options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images (predictions vs ground truth)"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Run deterministic 3D export for sample models"
    )
    parser.add_argument(
        "--per_sample_json",
        action="store_true",
        help="Save detailed per-sample metrics to JSON (may be large)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        default="./evaluation_outputs",
        help="Directory to save evaluation results (default: ./evaluation_outputs)"
    )
    parser.add_argument(
        "--num_viz",
        type=int,
        default=10,
        help="Number of visualization images to generate (default: 10)"
    )
    parser.add_argument(
        "--max_exports",
        type=int,
        default=3,
        help="Number of 3D model exports to generate (default: 3)"
    )
    
    # Performance options
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (for quick testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    
    # Reporting options
    parser.add_argument(
        "--detailed_report",
        action="store_true",
        default=True,
        help="Generate detailed text report (default: True)"
    )
    parser.add_argument(
        "--comparison_grid",
        action="store_true",
        help="Create comparison grid visualization"
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("NEURAL-GEOMETRIC 3D MODEL GENERATOR - EVALUATION")
    print("=" * 80)
    
    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n[ERROR] Model not found at: {model_path}")
        print("Please provide a valid model checkpoint path.")
        return 1

    print(f"[INFO] Model checkpoint: {model_path}")

    # Load dataset
    print(f"\n[INFO] Loading {args.split} dataset from: {args.data_dir}")
    try:
        dataset = AdvancedFloorPlanDataset(
            data_dir=args.data_dir,
            split=args.split,
            augment=False  # Never augment during evaluation
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return 1

    if len(dataset) == 0:
        print(f"\n[ERROR] No samples found in {args.split} split")
        print(f"Please check that {args.data_dir}/{args.split} contains valid samples")
        return 1

    print(f"[INFO] Found {len(dataset)} samples in {args.split} split")

    # Limit samples if requested
    if args.limit_samples is not None:
        original_count = len(dataset)
        dataset.samples = dataset.samples[:args.limit_samples]
        print(f"[INFO] Limited evaluation to {len(dataset)} samples (from {original_count})")

    # Create evaluator
    print(f"\n[INFO] Initializing model evaluator...")
    try:
        evaluator = ModelEvaluator(str(model_path), device=device)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize evaluator: {e}")
        return 1

    # Run main evaluation
    print(f"\n[INFO] Running comprehensive evaluation on {len(dataset)} samples...")
    start_time = time.time()
    
    try:
        summary = evaluator.evaluate_dataset(dataset)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    eval_time = time.time() - start_time
    print(f"\n[INFO] Evaluation completed in {eval_time:.2f}s ({eval_time/len(dataset):.2f}s per sample)")

    # Display results
    print("\n" + "=" * 80)
    evaluator.print_evaluation_results(summary)
    print("=" * 80)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    summary_path = out_dir / f"{model_path.stem}_evaluation_summary.json"
    save_json(summary, summary_path)

    # Collect per-sample metrics if requested
    per_sample_metrics = None
    if args.per_sample_json:
        print(f"\n[INFO] Collecting detailed per-sample metrics...")
        per_sample_start = time.time()
        
        try:
            per_sample_metrics = gather_per_sample_metrics(
                evaluator, 
                dataset, 
                device,
                max_samples=args.limit_samples
            )
            per_sample_path = out_dir / f"{model_path.stem}_per_sample_metrics.json"
            save_json(per_sample_metrics, per_sample_path)
            
            per_sample_time = time.time() - per_sample_start
            print(f"[INFO] Per-sample collection completed in {per_sample_time:.2f}s")
        except Exception as e:
            print(f"[WARNING] Per-sample metrics collection failed: {e}")

    # Generate visualizations and exports if requested
    if args.visualize or args.save_outputs:
        print(f"\n[INFO] Initializing inference engine for outputs...")
        try:
            engine = ResearchInferenceEngine(
                model_path=str(model_path),
                device=device
            )
            
            run_visualization_and_exports(
                engine,
                dataset,
                out_dir,
                device,
                num_viz=args.num_viz if args.visualize else 0,
                max_export=args.max_exports if args.save_outputs else 0,
            )
        except Exception as e:
            print(f"[WARNING] Visualization/export generation failed: {e}")

    # Generate detailed report
    if args.detailed_report:
        dataset_info = {
            "size": len(dataset),
            "split": args.split,
            "data_dir": args.data_dir
        }
        try:
            create_evaluation_report(
                summary,
                per_sample_metrics,
                out_dir,
                str(model_path),
                dataset_info
            )
        except Exception as e:
            print(f"[WARNING] Report generation failed: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Results saved to: {out_dir}")
    print(f"✓ Summary JSON: {summary_path.name}")
    
    if per_sample_metrics:
        print(f"✓ Per-sample metrics: {len(per_sample_metrics)} samples")
    
    if args.visualize:
        print(f"✓ Visualizations: {args.num_viz} images")
    
    if args.save_outputs:
        print(f"✓ 3D exports: {args.max_exports} models")
    
    print("\nEvaluation finished successfully!")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())