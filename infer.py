"""
Main inference script for generating 3D models from 2D floorplans
"""

import argparse
from pathlib import Path

from inference.engine import ResearchInferenceEngine
from config import DEFAULT_INFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Generate 3D models from 2D floorplans")
    parser.add_argument("--model_path", type=str, default="final_model.pth",
                       help="Path to trained model")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image path or directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path or directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Inference device")
    parser.add_argument("--export_intermediate", action="store_true",
                       help="Export intermediate results")
    parser.add_argument("--polygon_threshold", type=float, default=0.5,
                       help="Threshold for polygon validity")
    
    args = parser.parse_args()

    # Initialize inference engine
    print(f"Initializing inference engine...")
    engine = ResearchInferenceEngine(
        model_path=args.model_path,
        device=args.device
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        
        if not output_path.suffix:
            output_path = output_path / f"{input_path.stem}_model.obj"
        
        success = engine.generate_3d_model(
            str(input_path),
            str(output_path),
            export_intermediate=args.export_intermediate
        )
        
        if success:
            print(f"✓ Successfully generated: {output_path}")
        else:
            print(f"✗ Failed to generate model for: {input_path}")

    elif input_path.is_dir():
        # Batch processing
        print(f"Processing directory: {input_path}")
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print("No image files found in input directory!")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Create output directory
        output_path.mkdir(exist_ok=True)
        
        # Process batch
        results = engine.process_batch(image_files, output_path)
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nBatch processing completed:")
        print(f"✓ Successful: {successful}/{len(results)}")
        print(f"✗ Failed: {len(results) - successful}/{len(results)}")
        
        # List failed cases
        failed_cases = [r for r in results if not r["success"]]
        if failed_cases:
            print("\nFailed cases:")
            for case in failed_cases:
                error_msg = case.get("error", "Unknown error")
                print(f"  - {Path(case['input']).name}: {error_msg}")

    else:
        print(f"Error: Input path {input_path} does not exist!")


if __name__ == "__main__":
    main()