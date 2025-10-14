"""
Enhanced training script for the 3D Model Generator
Implements novel training strategies: dynamic curriculum, adaptive weighting, cross-modal consistency
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from dataset import AdvancedFloorPlanDataset
from models.model import NeuralGeometric3DGenerator
from training.trainer import AdaptiveMultiStageTrainer, MultiStageTrainer
from utils.visualization import plot_training_history, plot_curriculum_analysis
from config import (
    DEFAULT_DATA_CONFIG, 
    DEFAULT_MODEL_CONFIG, 
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_LOSS_CONFIG,
    TrainingConfig,
    CurriculumConfig
)


def create_enhanced_config(args):
    """Create enhanced training configuration with novel strategies"""
    config = TrainingConfig()
    
    # Basic settings
    config.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dynamic curriculum settings
    if args.dynamic_curriculum:
        config.curriculum = CurriculumConfig()
        config.curriculum.use_dynamic_curriculum = True
        config.curriculum.stage_switch_patience = args.patience
        config.curriculum.min_improvement_threshold = args.min_improvement
        
        # Adjust epoch limits for dynamic training
        config.max_stage1_epochs = args.max_stage1_epochs
        config.max_stage2_epochs = args.max_stage2_epochs  
        config.max_stage3_epochs = args.max_stage3_epochs
        
        print("Dynamic curriculum learning enabled")
        print(f"  Stage switch patience: {config.curriculum.stage_switch_patience}")
        print(f"  Min improvement threshold: {config.curriculum.min_improvement_threshold}")
    else:
        # Disable dynamic curriculum for traditional training
        config.curriculum.use_dynamic_curriculum = False
        print("Using traditional fixed-epoch training")
    
    # GradNorm dynamic weighting
    if args.gradnorm:
        config.curriculum.use_gradnorm = True
        config.curriculum.gradnorm_alpha = args.gradnorm_alpha
        config.curriculum.gradnorm_update_freq = args.gradnorm_freq
        print(f"GradNorm dynamic weighting enabled (alpha={args.gradnorm_alpha})")
    
    # Topology-aware scheduling  
    if args.topology_schedule != "static":
        config.curriculum.topology_schedule = args.topology_schedule
        config.curriculum.topology_start_weight = args.topology_start
        config.curriculum.topology_end_weight = args.topology_end
        print(f"Topology-aware scheduling: {args.topology_schedule}")
        print(f"  Weights: {args.topology_start} -> {args.topology_end}")
    
    return config


def create_enhanced_model(args):
    """Create enhanced model with auxiliary heads"""
    model = NeuralGeometric3DGenerator(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        num_attributes=args.num_attributes,
        voxel_size=args.voxel_size,
        max_polygons=args.max_polygons,
        max_points=args.max_points,
        use_latent_consistency=args.latent_consistency,
        use_graph_constraints=args.graph_constraints,
        latent_embedding_dim=args.embedding_dim
    )
    
    print(f"Enhanced model created:")
    print(f"  Feature dim: {args.feature_dim}")
    print(f"  Latent consistency: {args.latent_consistency}")
    print(f"  Graph constraints: {args.graph_constraints}")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def visualize_training_results(history, output_dir):
    """Create comprehensive training visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Traditional loss curves
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))
    
    # Novel curriculum analysis plots
    if "stage_transitions" in history and history["stage_transitions"]:
        plot_curriculum_analysis(history, save_path=str(output_dir / "curriculum_analysis.png"))
    
    # Dynamic weight evolution
    if "dynamic_weights" in history and history["dynamic_weights"]:
        plt.figure(figsize=(12, 8))
        
        # Extract weight evolution data
        epochs = [entry["epoch"] for entry in history["dynamic_weights"]]
        weight_names = list(history["dynamic_weights"][0]["weights"].keys())
        
        for weight_name in weight_names:
            weights = [entry["weights"].get(weight_name, 0) for entry in history["dynamic_weights"]]
            if any(w > 0.001 for w in weights):  # Only plot significant weights
                plt.plot(epochs, weights, label=weight_name, linewidth=2)
        
        plt.xlabel("Global Epoch")
        plt.ylabel("Loss Weight")
        plt.title("Dynamic Loss Weight Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "weight_evolution.png", dpi=300)
        plt.close()
    
    # Component loss breakdown
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    stage_names = ["stage1", "stage2", "stage3"]
    
    for idx, stage_name in enumerate(stage_names):
        if stage_name in history and "component_losses" in history[stage_name]:
            component_data = history[stage_name]["component_losses"]
            if component_data:
                # Get component names from first entry
                component_names = list(component_data[0].keys())
                
                for comp_name in component_names:
                    if comp_name in ['seg', 'dice', 'polygon', 'voxel', 'topology', 
                                   'latent_consistency', 'graph']:
                        values = [entry.get(comp_name, 0) for entry in component_data]
                        if any(v > 0.001 for v in values):  # Only plot significant losses
                            axes[idx].plot(values, label=comp_name, linewidth=2)
                
                axes[idx].set_title(f"{stage_name.upper()} Component Losses")
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel("Loss Value")
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "component_losses.png", dpi=300)
    plt.close()
    
    print(f"Training visualizations saved to {output_dir}")


def save_training_summary(history, config, output_dir):
    """Save comprehensive training summary"""
    output_dir = Path(output_dir)
    
    summary = {
        "training_config": {
            "dynamic_curriculum": config.curriculum.use_dynamic_curriculum,
            "gradnorm_enabled": config.curriculum.use_gradnorm,
            "topology_schedule": config.curriculum.topology_schedule,
            "max_epochs": [config.max_stage1_epochs, config.max_stage2_epochs, config.max_stage3_epochs]
        },
        "training_results": {},
        "novel_strategies_summary": {}
    }
    
    # Training results
    for stage_name, data in history.items():
        if isinstance(data, dict) and "val_loss" in data and data["val_loss"]:
            summary["training_results"][stage_name] = {
                "final_val_loss": data["val_loss"][-1],
                "best_val_loss": min(data["val_loss"]),
                "epochs_trained": len(data["val_loss"])
            }
    
    # Novel strategies summary
    if "stage_transitions" in history:
        summary["novel_strategies_summary"]["adaptive_transitions"] = len(history["stage_transitions"])
        
    if "dynamic_weights" in history:
        summary["novel_strategies_summary"]["weight_updates"] = len(history["dynamic_weights"])
        
    if "curriculum_events" in history:
        summary["novel_strategies_summary"]["curriculum_events"] = len(history["curriculum_events"])
    
    # Save as JSON
    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Training summary saved to {output_dir / 'training_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="3D Model Training")
    
    # Basic arguments
    parser.add_argument("--data_dir", type=str, default="./data/floorplans", 
                       help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--device", type=str, default=None, help="Training device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                       help="Output directory for checkpoints")
    
    # Training mode selection
    parser.add_argument("--training_mode", type=str, choices=["traditional", "adaptive"], 
                       default="adaptive", help="Training mode (traditional fixed epochs vs adaptive)")
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "all"], default="all",
                       help="Training stage to run (only for traditional mode)")
    
    # Novel training strategies
    parser.add_argument("--dynamic-curriculum", action="store_true", default=True,
                       help="Enable adaptive stage transitioning")
    parser.add_argument("--patience", type=int, default=5,
                       help="Epochs without improvement before stage transition")
    parser.add_argument("--min-improvement", type=float, default=0.001,
                       help="Minimum relative improvement threshold")
    
    parser.add_argument("--gradnorm", action="store_true", default=True,
                       help="Enable GradNorm dynamic loss weighting")
    parser.add_argument("--gradnorm-alpha", type=float, default=0.08,
                       help="GradNorm restoring force parameter")
    parser.add_argument("--gradnorm-freq", type=int, default=5,
                       help="GradNorm update frequency (batches)")
    
    parser.add_argument("--topology-schedule", type=str, 
                       choices=["static", "progressive", "linear_ramp"], 
                       default="progressive", help="Topology loss scheduling strategy")
    parser.add_argument("--topology-start", type=float, default=0.05,
                       help="Starting weight for topology loss")
    parser.add_argument("--topology-end", type=float, default=0.5,
                       help="Ending weight for topology loss")
    
    # Model enhancements
    parser.add_argument("--latent-consistency", action="store_true", default=True,
                       help="Enable cross-modal latent consistency loss")
    parser.add_argument("--graph-constraints", action="store_true", default=True,
                       help="Enable graph-based topology constraints")
    parser.add_argument("--embedding-dim", type=int, default=256,
                       help="Latent embedding dimension")
    
    # Model architecture
    parser.add_argument("--input_channels", type=int, default=3, help="Input image channels")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--feature_dim", type=int, default=512, help="Feature dimension")
    parser.add_argument("--num_attributes", type=int, default=6, help="Number of attribute predictions")
    parser.add_argument("--voxel_size", type=int, default=64, help="3D voxel grid size")
    parser.add_argument("--max_polygons", type=int, default=20, help="Maximum number of polygons")
    parser.add_argument("--max_points", type=int, default=50, help="Maximum points per polygon")
    
    # Dynamic epoch limits
    parser.add_argument("--max-stage1-epochs", type=int, default=50, help="Max epochs for Stage 1")
    parser.add_argument("--max-stage2-epochs", type=int, default=55, help="Max epochs for Stage 2") 
    parser.add_argument("--max-stage3-epochs", type=int, default=70, help="Max epochs for Stage 3")
    
    parser.add_argument("--persistent_workers",action="store_true",default=False,help="Keep DataLoader workers alive between epochs (requires num_workers > 0).")

    parser.add_argument("--prefetch_factor",type=int,default=2,help="Number of batches preloaded by each worker.")

    
    args = parser.parse_args()

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    import torch.backends.cudnn as cudnn
    if device == "cuda":
        cudnn.benchmark = True

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create enhanced configuration
    config = create_enhanced_config(args)
    
    print("\n" + "="*80)
    print(" 3D MODEL TRAINING")
    print("="*80)
    print("Novel Training Strategies Enabled:")
    if config.curriculum.use_dynamic_curriculum:
        print("✓ Adaptive Stage Transitioning (Dynamic Curriculum)")
    if config.curriculum.use_gradnorm:
        print("✓ Multi-objective Optimization with GradNorm")
    if config.curriculum.topology_schedule != "static":
        print("✓ Topology-aware Loss Scheduling")
    if args.latent_consistency:
        print("✓ Cross-modal Latent Consistency Learning")
    if args.graph_constraints:
        print("✓ Graph-based Topology Constraints")
    print("="*80)

    # Create datasets
    print("\nLoading datasets...")
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
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    # Create enhanced model
    print("\nInitializing enhanced model...")
    model = create_enhanced_model(args)

    # Create appropriate trainer
    if args.training_mode == "adaptive":
        print("\nUsing Adaptive Multi-Stage Trainer with Novel Strategies")
        trainer = AdaptiveMultiStageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config
        )
    else:
        print("\nUsing Traditional Multi-Stage Trainer")
        trainer = MultiStageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config
        )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    if args.training_mode == "adaptive" or args.stage == "all":
        print("\nStarting adaptive multi-stage training with novel strategies...")
        history = trainer.train_all_stages()
    else:
        # Traditional single-stage training
        stage_num = int(args.stage)
        print(f"Training Stage {stage_num} only...")
        if stage_num == 1:
            trainer.train_stage1()
        elif stage_num == 2:
            trainer.train_stage2()  
        elif stage_num == 3:
            trainer.train_stage3()
        history = trainer.history

    # Save final model
    final_model_path = output_dir / "final_enhanced_model.pth"
    if hasattr(trainer, '_save_checkpoint'):
        trainer._save_checkpoint(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")

    # Create comprehensive visualizations
    print("\nGenerating training analysis...")
    visualize_training_results(history, output_dir)
    
    # Save training summary
    save_training_summary(history, config, output_dir)

    print(f"\n Training completed successfully!")
    print(f"Results saved to: {output_dir}")
    # print("\nNovel contributions implemented:")
    # print("- Dynamic curriculum learning with adaptive stage transitions")
    # print("- Multi-objective optimization with gradient-based reweighting") 
    # print("- Topology-aware progressive constraint injection")
    # print("- Cross-modal latent consistency learning")
    # print("- Graph-based architectural constraint learning")


if __name__ == "__main__":
    main()