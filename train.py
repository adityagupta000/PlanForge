"""
Main training script for the Neural-Geometric 3D Model Generator
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import AdvancedFloorPlanDataset
from models.model import NeuralGeometric3DGenerator
from training.trainer import MultiStageTrainer
from utils.visualization import plot_training_history
from config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train Neural-Geometric 3D Model Generator")
    parser.add_argument("--data_dir", type=str, default="./data/floorplans", 
                       help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--device", type=str, default=None, help="Training device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "all"], default="all",
                       help="Training stage to run")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                       help="Output directory for checkpoints")
    
    args = parser.parse_args()

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create datasets
    print("Loading datasets...")
    train_dataset = AdvancedFloorPlanDataset(
        args.data_dir, split="train", augment=True
    )
    val_dataset = AdvancedFloorPlanDataset(
        args.data_dir, split="val", augment=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("Initializing model...")
    model = NeuralGeometric3DGenerator()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create trainer
    trainer = MultiStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training stages
    if args.stage == "all":
        print("Starting complete multi-stage training...")
        history = trainer.train_all_stages()
    elif args.stage == "1":
        print("Training Stage 1: Segmentation + Attributes")
        trainer.train_stage1()
        history = trainer.history
    elif args.stage == "2":
        print("Training Stage 2: DVX Polygon Fitting")
        trainer.train_stage2()
        history = trainer.history
    elif args.stage == "3":
        print("Training Stage 3: End-to-End Fine-tuning")
        trainer.train_stage3()
        history = trainer.history

    # Save final model
    final_model_path = output_dir / "final_model.pth"
    trainer._save_checkpoint(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")

    # Plot training history
    plot_save_path = output_dir / "training_history.png"
    plot_training_history(history, save_path=str(plot_save_path))

    print("Training completed successfully!")


if __name__ == "__main__":
    main()