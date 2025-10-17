"""
infer.py - Enhanced Inference Script
Generate 3D models from 2D floorplans using trained Neural-Geometric 3D Generator
Compatible with adaptive multi-stage training

Usage examples:
    # Single image inference
    python infer.py --model_path checkpoints/final_model.pth --input floor1.png --output model1.obj
    
    # Batch processing directory
    python infer.py --model_path checkpoints/final_model.pth --input ./images/ --output ./models/
    
    # With intermediate exports
    python infer.py --model_path checkpoints/final_model.pth --input floor.png --output model.obj --export_intermediate
    
    # Using CPU
    python infer.py --model_path checkpoints/final_model.pth --input floor.png --output model.obj --device cpu
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch

from inference.engine import ResearchInferenceEngine


def process_single_image(
    engine: ResearchInferenceEngine,
    input_path: str,
    output_path: str,
    export_intermediate: bool = False
) -> bool:
    """
    Process a single image and generate 3D model
    
    Args:
        engine: Inference engine instance
        input_path: Path to input image
        output_path: Path to save output OBJ file
        export_intermediate: Whether to export intermediate results
    
    Returns:
        Success status
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return False
    
    if not input_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        print(f"[ERROR] Unsupported image format: {input_file.suffix}")
        return False
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Processing: {input_file.name}")
    print(f"[INFO] Output: {output_file.name}")
    
    start_time = time.time()
    
    try:
        success = engine.generate_3d_model(
            str(input_file),
            str(output_file),
            export_intermediate=export_intermediate
        )
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"✓ Successfully generated 3D model in {elapsed:.2f}s")
            print(f"  Output file: {output_file}")
            
            if export_intermediate:
                print(f"  Intermediate files:")
                print(f"    - {output_file.parent / 'predicted_mask.png'}")
                print(f"    - {output_file.parent / 'predicted_attributes.json'}")
                print(f"    - {output_file.parent / 'predicted_polygons.json'}")
                print(f"    - {output_file.parent / 'polygon_visualization.png'}")
            
            return True
        else:
            print(f"[ERROR] Failed to generate 3D model")
            return False
    
    except Exception as e:
        print(f"[ERROR] Exception during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(
    engine: ResearchInferenceEngine,
    input_dir: str,
    output_dir: str,
    export_intermediate: bool = False,
    limit: Optional[int] = None
) -> tuple:
    """
    Process all images in a directory
    Handles both flat directory structure and nested sample directories
    
    Args:
        engine: Inference engine instance
        input_dir: Directory containing input images or sample folders
        output_dir: Directory to save output models
        export_intermediate: Whether to export intermediate results
        limit: Maximum number of images to process
    
    Returns:
        Tuple of (stats dict, total_time)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return {"success": 0, "failed": 0}, 0
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files (supports both flat and nested structures)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    # First try flat structure (images directly in directory)
    flat_images = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    image_files.extend(flat_images)
    
    # Then try nested structure (images in subdirectories)
    if not image_files:
        for sample_dir in sorted(input_path.iterdir()):
            if sample_dir.is_dir():
                # Look for image.png in each sample directory
                image_file = sample_dir / "image.png"
                if image_file.exists():
                    image_files.append(image_file)
    
    if not image_files:
        print(f"[ERROR] No image files found in: {input_dir}")
        print(f"[INFO] Checked for:")
        print(f"  - Images directly in {input_dir}")
        print(f"  - image.png files in sample subdirectories")
        return {"success": 0, "failed": 0}, 0
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"\n[INFO] Found {len(image_files)} image file(s)")
    print(f"[INFO] Processing directory: {input_dir}")
    print(f"[INFO] Output directory: {output_dir}\n")
    
    stats = {"success": 0, "failed": 0}
    total_time = time.time()
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing {image_file.name}...")
        
        output_file = output_path / f"{image_file.stem}_model.obj"
        
        try:
            success = engine.generate_3d_model(
                str(image_file),
                str(output_file),
                export_intermediate=export_intermediate
            )
            
            if success:
                stats["success"] += 1
                print(f"✓ Generated: {output_file.name}")
            else:
                stats["failed"] += 1
                print(f"[!] Failed: {image_file.name}")
        
        except Exception as e:
            stats["failed"] += 1
            print(f"[!] Exception processing {image_file.name}: {e}")
    
    total_time = time.time() - total_time
    
    return stats, total_time


def validate_model_path(model_path: str) -> bool:
    """Validate that model file exists"""
    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        print(f"[ERROR] Please provide a valid model path")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D models from 2D floorplan images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python infer.py --model_path checkpoints/final_model.pth \\
    --input ./test_floor.png --output ./test_floor.obj
  
  # Batch processing
  python infer.py --model_path checkpoints/final_model.pth \\
    --input ./images/ --output ./models/
  
  # With intermediate results
  python infer.py --model_path checkpoints/final_model.pth \\
    --input ./images/ --output ./models/ --export_intermediate
  
  # Limit number of images in batch
  python infer.py --model_path checkpoints/final_model.pth \\
    --input ./images/ --output ./models/ --limit 10
  
  # Use CPU
  python infer.py --model_path checkpoints/final_model.pth \\
    --input ./images/ --output ./models/ --device cpu
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        "-m",
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input image file or directory containing images"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output OBJ file or directory for batch processing"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu). Auto-detect if omitted"
    )
    parser.add_argument(
        "--export_intermediate",
        action="store_true",
        help="Export intermediate results (masks, attributes, polygons)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process in batch mode"
    )
    parser.add_argument(
        "--batch_verbose",
        action="store_true",
        help="Show detailed progress during batch processing"
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("NEURAL-GEOMETRIC 3D MODEL GENERATOR - INFERENCE")
    print("=" * 80)
    
    # Validate model
    if not validate_model_path(args.model_path):
        return 1
    
    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Initialize inference engine
    print(f"[INFO] Loading model from: {args.model_path}")
    try:
        engine = ResearchInferenceEngine(
            model_path=args.model_path,
            device=device
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize inference engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Determine if single image or batch processing
    if input_path.is_file():
        # Single image processing
        print(f"\n[INFO] Single image mode")
        
        success = process_single_image(
            engine,
            str(input_path),
            str(output_path),
            export_intermediate=args.export_intermediate
        )
        
        if success:
            print("\n" + "=" * 80)
            print("INFERENCE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("INFERENCE FAILED")
            print("=" * 80)
            return 1
    
    elif input_path.is_dir():
        # Batch processing
        print(f"\n[INFO] Batch mode - Processing directory")
        
        try:
            stats, total_time = process_directory(
                engine,
                str(input_path),
                str(output_path),
                export_intermediate=args.export_intermediate,
                limit=args.limit
            )
            
            # Print summary
            print("\n" + "=" * 80)
            print("BATCH PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total time: {total_time:.2f}s")
            print(f"Successful: {stats['success']}")
            print(f"Failed: {stats['failed']}")
            
            if stats['success'] > 0:
                avg_time = total_time / stats['success']
                print(f"Average time per model: {avg_time:.2f}s")
            
            print(f"Output directory: {output_path}")
            print("=" * 80 + "\n")
            
            return 0 if stats['failed'] == 0 else 1
        
        except Exception as e:
            print(f"\n[ERROR] Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        print(f"\n[ERROR] Input path is neither a file nor directory: {args.input}")
        return 1


if __name__ == "__main__":
    exit(main())