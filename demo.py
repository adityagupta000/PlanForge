"""
Demo script for the Neural-Geometric 3D Model Generator
"""

import torch
import cv2
import numpy as np
from pathlib import Path

from models.model import NeuralGeometric3DGenerator
from dataset import create_synthetic_data_sample
from utils.visualization import visualize_predictions, create_model_summary_report


def demo_pipeline():
    """Demonstrate the complete pipeline with synthetic data"""
    print("Neural-Geometric 3D Model Generator Demo")
    print("=" * 50)

    # Create output directory
    demo_dir = Path("./demo_outputs")
    demo_dir.mkdir(exist_ok=True)

    # Create synthetic sample
    print("Creating synthetic data sample...")
    image, mask, attributes, voxels, polygons = create_synthetic_data_sample()

    # Save synthetic data
    cv2.imwrite(str(demo_dir / "demo_input.png"), image)
    cv2.imwrite(str(demo_dir / "demo_mask.png"), mask * 50)

    # Create model (random weights for demo)
    print("Initializing model...")
    model = NeuralGeometric3DGenerator()
    model.eval()

    # Convert to tensors
    image_tensor = torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0)

    # Create model summary
    create_model_summary_report(model, image_tensor, str(demo_dir / "model_summary.txt"))

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        predictions = model(image_tensor)

    print("Forward pass completed")
    print(f"Segmentation shape: {predictions['segmentation'].shape}")
    print(f"Attributes shape: {predictions['attributes'].shape}")
    print(f"Polygons shape: {predictions['polygons'].shape}")
    print(f"Voxels shape: {predictions['voxels_pred'].shape}")

    # Extract and save results
    seg_pred = torch.argmax(predictions["segmentation"], dim=1).squeeze().numpy()
    attr_pred = predictions["attributes"].squeeze().numpy()

    cv2.imwrite(str(demo_dir / "demo_seg_pred.png"), seg_pred * 50)

    print(f"Predicted attributes: {attr_pred}")

    # Create visualization
    print("Creating visualizations...")
    
    # Create targets for visualization
    targets = {
        "mask": torch.from_numpy(mask).unsqueeze(0),
        "attributes": torch.from_numpy(np.array([
            attributes["wall_height"] / 5.0,
            attributes["wall_thickness"] / 0.5,
            attributes["window_base_height"] / 3.0,
            attributes["window_height"] / 2.0,
            attributes["door_height"] / 5.0,
            attributes["pixel_scale"] / 0.02,
        ])).float().unsqueeze(0)
    }
    
    visualize_predictions(
        image_tensor, 
        predictions, 
        targets, 
        save_path=str(demo_dir / "demo_predictions.png")
    )

    print(f"Demo completed successfully! Results saved to {demo_dir}")


def demo_with_pretrained(model_path, input_image_path=None):
    """Demo with a pretrained model"""
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found!")
        return

    print(f"Running demo with pretrained model: {model_path}")

    # Load model
    model = NeuralGeometric3DGenerator()
    checkpoint = torch.load(model_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()

    # Use provided image or create synthetic
    if input_image_path and Path(input_image_path).exists():
        image = cv2.imread(input_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image_tensor = torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0).cuda()
        print(f"Using input image: {input_image_path}")
    else:
        print("Using synthetic data...")
        image, _, _, _, _ = create_synthetic_data_sample()
        image_tensor = torch.from_numpy(image / 255.0).float().permute(2, 0, 1).unsqueeze(0).cuda()

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Visualize results
    demo_dir = Path("./demo_outputs")
    demo_dir.mkdir(exist_ok=True)
    
    visualize_predictions(
        image_tensor, 
        predictions, 
        save_path=str(demo_dir / "pretrained_demo.png")
    )

    print(f"Pretrained demo completed! Results saved to {demo_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo Neural-Geometric 3D Model Generator")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model (optional)")
    parser.add_argument("--input_image", type=str, default=None,
                       help="Input image path (optional)")
    
    args = parser.parse_args()

    if args.model_path:
        demo_with_pretrained(args.model_path, args.input_image)
    else:
        demo_pipeline()