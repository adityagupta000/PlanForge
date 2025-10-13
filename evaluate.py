"""
evaluate.py
Comprehensive evaluation CLI for 3D Model Generator.

Usage examples:
  python evaluate.py --model_path checkpoints/final_model.pth --data_dir ./data/floorplans
  python evaluate.py --model_path checkpoints/final_model.pth --data_dir ./data/floorplans --visualize --save_outputs --num_viz 20
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

# Project imports (match your repo layout)
from dataset import AdvancedFloorPlanDataset
from evaluation.metrics import ModelEvaluator
from inference.engine import ResearchInferenceEngine
from utils.visualization import visualize_predictions


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[✓] Saved JSON -> {path}")


def gather_per_sample_metrics(
    evaluator: ModelEvaluator,
    dataset: AdvancedFloorPlanDataset,
    device: str,
    max_samples: int = None,
) -> List[Dict]:
    """
    Re-run evaluation loop sample-by-sample and collect per-sample metrics.
    We use evaluator._evaluate_* helper methods (present in evaluation/metrics.py)
    so metrics match the overall evaluation.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    per_sample = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_samples is not None and idx >= max_samples:
                break

            # Move tensors to device where applicable
            batch_for_model = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch_for_model[k] = v.to(device)
                else:
                    batch_for_model[k] = v  # dicts/strings stay as-is

            # Forward
            preds = evaluator.model(batch_for_model["image"])

            # segmentation
            seg_res = evaluator._evaluate_segmentation(preds["segmentation"], batch_for_model["mask"])
            # attributes
            attr_res = evaluator._evaluate_attributes(preds["attributes"], batch_for_model["attributes"].to(device))
            # voxels
            voxel_res = evaluator._evaluate_voxels(preds["voxels_pred"], batch_for_model["voxels_gt"].to(device))
            # polygons — evaluator._evaluate_polygons expects format used in metrics.py
            # batch["polygons_gt"] is a dict with "polygons" and "valid_mask"
            poly_res = evaluator._evaluate_polygons(preds["polygons"], preds.get("polygon_validity", preds.get("validity", None)), batch["polygons_gt"])

            sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else batch["sample_id"]
            sample_metrics = {
                "sample_id": str(sample_id),
                "segmentation": seg_res,
                "attributes": attr_res,
                "voxels": voxel_res,
                "polygons": poly_res,
            }
            per_sample.append(sample_metrics)

            if (idx + 1) % 10 == 0:
                print(f"[INFO] Collected per-sample metrics for {idx+1}/{len(loader)} samples")

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
    For the first `num_viz` samples, create visualizations using the model and optionally
    run deterministic 3D export to save intermediate results and a .obj.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    viz_count = 0
    export_count = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else batch["sample_id"]
            sample_dir = Path(dataset.data_dir) / "test" / str(sample_id)

            # Prepare tensors
            image_tensor = batch["image"].to(device)
            target_mask = batch["mask"].unsqueeze(0) if torch.is_tensor(batch["mask"]) else None
            target_attrs = batch["attributes"].unsqueeze(0) if torch.is_tensor(batch["attributes"]) else None

            # Model predictions using engine.model (same underlying model)
            preds = engine.model(image_tensor)

            # Visualization
            if viz_count < num_viz:
                vis_path = output_dir / f"viz_{sample_id}.png"
                try:
                    visualize_predictions(
                        image_tensor,
                        preds,
                        {"mask": target_mask, "attributes": target_attrs},
                        save_path=str(vis_path),
                    )
                    print(f"[✓] Saved visualization for sample {sample_id} -> {vis_path}")
                except Exception as e:
                    print(f"[!] Visualization failed for {sample_id}: {e}")
                viz_count += 1

            # Export deterministic 3D (uses the image file path)
            if export_count < max_export:
                image_file = sample_dir / "image.png"
                out_obj = output_dir / f"{sample_id}_predicted_model.obj"
                try:
                    success = engine.generate_3d_model(str(image_file), str(out_obj), export_intermediate=True)
                    if success:
                        print(f"[✓] Exported deterministic 3D model for {sample_id} -> {out_obj}")
                    else:
                        print(f"[!] 3D export returned False for {sample_id}")
                except Exception as e:
                    print(f"[!] 3D export failed for {sample_id}: {e}")
                export_count += 1

            if viz_count >= num_viz and export_count >= max_export:
                break


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D model")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model checkpoint (checkpoint.pth)")
    parser.add_argument("--data_dir", "-d", default="./data/floorplans", help="Dataset root with train/val/test")
    parser.add_argument("--device", default=None, help="Device to use (cuda or cpu). Auto-detect if omitted")
    parser.add_argument("--visualize", action="store_true", help="Save visual comparison images (pred vs GT)")
    parser.add_argument("--save_outputs", action="store_true", help="Run deterministic 3D export for some samples")
    parser.add_argument("--output_dir", default="./evaluation_outputs", help="Where to save reports/visuals")
    parser.add_argument("--num_viz", type=int, default=10, help="How many visualizations to produce (default 10)")
    parser.add_argument("--max_exports", type=int, default=3, help="How many deterministic 3D exports to run (default 3)")
    parser.add_argument("--per_sample_json", action="store_true", help="Save per-sample metrics JSON (may be large)")
    parser.add_argument("--limit_samples", type=int, default=None, help="If set, limit evaluation to first N samples (for quick tests)")

    args = parser.parse_args()

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        return

    # Load test dataset
    dataset = AdvancedFloorPlanDataset(data_dir=args.data_dir, split="test")
    if len(dataset) == 0:
        print("[ERROR] No test samples found (dataset may be empty or data_dir incorrect).")
        return

    # If user asked for a limited quick run, slice dataset.samples accordingly.
    if args.limit_samples is not None:
        # Create a shallow copy dataset pointing to first N samples
        dataset.samples = dataset.samples[: args.limit_samples]
        print(f"[INFO] Limiting evaluation to first {len(dataset)} samples")

    # Create evaluator and run full evaluation
    evaluator = ModelEvaluator(str(model_path), device=device)
    summary = evaluator.evaluate_dataset(dataset)
    evaluator.print_evaluation_results(summary)

    # Save summary JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / f"{model_path.stem}_summary.json")

    # Optionally collect detailed per-sample metrics
    if args.per_sample_json:
        print("[INFO] Collecting per-sample metrics (this re-runs model inference sample-by-sample)...")
        per_sample = gather_per_sample_metrics(evaluator, dataset, device, max_samples=None)
        save_json(per_sample, out_dir / f"{model_path.stem}_per_sample_metrics.json")

    # Visualization and/or exports
    if args.visualize or args.save_outputs:
        print("[INFO] Initializing inference engine for visualizations/exports...")
        engine = ResearchInferenceEngine(model_path=str(model_path), device=device)
        run_visualization_and_exports(
            engine,
            dataset,
            out_dir,
            device,
            num_viz=args.num_viz,
            max_export=args.max_exports,
        )

    print("[✓] Evaluation finished.")


if __name__ == "__main__":
    main()
