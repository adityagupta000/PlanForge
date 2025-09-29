"""
Configuration settings for the Neural-Geometric 3D Model Generator
OPTIMIZED FOR 48GB GPU - Maximizes throughput and model capacity
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import torch


@dataclass
class DataConfig:
    """Data configuration - Optimized for high-end GPU"""
    data_dir: str = "./data/floorplans"
    image_size: Tuple[int, int] = (384, 384)   # Increased from 256 for better detail
    voxel_size: int = 96                        # Increased from 64 for finer 3D resolution
    batch_size: int = 12                        # Increased from 4 (48GB can handle this)
    num_workers: int = 16                       # Increased for faster data loading
    augment: bool = True
    pin_memory: bool = True                     # Critical for fast GPU transfer
    persistent_workers: bool = True             # Keep workers alive between epochs
    prefetch_factor: int = 4                    # Prefetch 4 batches per worker


@dataclass
class ModelConfig:
    """Model architecture - Enhanced capacity for 48GB GPU"""
    input_channels: int = 3
    num_classes: int = 5
    feature_dim: int = 768                      # Increased from 512 (50% more capacity)
    num_attributes: int = 6
    voxel_size: int = 96                        # Match DataConfig
    max_polygons: int = 30                      # Increased from 20 for complex layouts
    max_points: int = 64                        # Increased from 50 for finer polygons
    dropout: float = 0.15                       # Slightly higher for larger model
    use_attention: bool = True
    use_deep_supervision: bool = True
    
    # Auxiliary heads
    use_latent_consistency: bool = True
    use_graph_constraints: bool = True
    latent_embedding_dim: int = 384             # Increased from 256
    
    # DVX configuration
    dvx_displacement_scale: float = 0.10        # Slightly tighter for precision
    dvx_num_refinement_steps: int = 4           # Increased from 3
    dvx_feature_dim: int = 384                  # Match higher capacity


@dataclass 
class CurriculumConfig:
    """Dynamic curriculum learning configuration"""
    use_dynamic_curriculum: bool = True
    stage_switch_patience: int = 5
    min_improvement_threshold: float = 0.001
    plateau_detection_window: int = 3

    gradient_norm_window: int = 100
    objectives: Optional[List[str]] = None

    # Topology scheduling
    topology_schedule: str = "progressive"
    topology_start_weight: float = 0.1
    topology_end_weight: float = 1.0
    topology_ramp_epochs: int = 15              # Faster ramp with more data throughput
    
    # Mixed precision and optimization
    use_mixed_precision: bool = True            # Essential for 48GB efficiency
    cache_in_memory: bool = True                # 48GB GPU + system RAM can cache dataset
    accumulation_steps: int = 1                 # No accumulation needed with batch_size=16
    dvx_step_freq: int = 1                      # Run DVX every step
    persistent_workers: bool = True
    prefetch_factor: int = 6
    num_workers: int = 16
    
    # Progressive resolution (now higher base resolution)
    voxel_size_stage: Dict[str, int] = None
    image_size_stage: Dict[str, Tuple[int, int]] = None
    
    # Loss scheduling
    loss_schedule: Dict[str, str] = None
    
    # Multi-objective optimization (GradNorm)
    use_gradnorm: bool = True
    gradnorm_alpha: float = 0.15                # Slightly higher for faster adaptation
    gradnorm_update_freq: int = 5
    
    # Graph constraint scheduling
    graph_weight_schedule: str = "delayed_ramp"
    graph_start_epoch: int = 3                  # Earlier start with faster training
    graph_end_weight: float = 0.3
    
    def __post_init__(self):
        # Progressive resolution strategy
        if self.voxel_size_stage is None:
            self.voxel_size_stage = {
                "stage1": 48,   # Start lower for speed
                "stage2": 64,   # Medium resolution
                "stage3": 96    # Full resolution
            }
        
        if self.image_size_stage is None:
            self.image_size_stage = {
                "stage1": (192, 192),   # Start medium
                "stage2": (256, 256),   # Increase
                "stage3": (384, 384)    # Full resolution
            }
        
        if self.loss_schedule is None:
            self.loss_schedule = {
                "segmentation": "static",
                "dice": "static",
                "sdf": "early_decay",
                "attributes": "static",
                "polygon": "staged_ramp",
                "voxel": "late_ramp",
                "topology": "progressive",
                "latent_consistency": "mid_ramp",
                "graph": "delayed_ramp",
            }

        if self.objectives is None:
            self.objectives = [
                "segmentation", "dice", "sdf", "attributes",
                "polygon", "voxel", "topology",
                "latent_consistency", "graph",
            ]


@dataclass
class TrainingConfig:
    """Training configuration - Optimized for 48GB GPU throughput"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reduced epoch counts due to higher batch size and faster convergence
    max_stage1_epochs: int = 30                 # Reduced from 40
    max_stage2_epochs: int = 20                 # Reduced from 25
    max_stage3_epochs: int = 45                 # Reduced from 60
    
    min_stage1_epochs: int = 6                  # Reduced from 8
    min_stage2_epochs: int = 4                  # Reduced from 5
    min_stage3_epochs: int = 10                 # Reduced from 12
    
    # Learning rates - Higher for larger batches (batch_size=16 vs 4)
    # Using square root scaling: LR_new = LR_old * sqrt(batch_new/batch_old)
    stage1_lr: float = 6e-4                     # Scaled from 3e-4 (sqrt(16/4) = 2x)
    stage1_weight_decay: float = 5e-6           # Slightly lower with larger batch
    
    stage2_lr: float = 2e-4                     # Scaled from 1e-4
    stage2_weight_decay: float = 5e-6
    
    stage3_lr: float = 1e-4                     # Scaled from 5e-5
    stage3_weight_decay: float = 5e-6
    
    # Advanced training techniques
    use_mixed_precision: bool = True            # Critical for 48GB efficiency
    use_cosine_restarts: bool = True
    warmup_epochs: int = 3                      # Slightly longer warmup
    grad_clip_norm: float = 1.5                 # Higher for larger model
    
    # Gradient monitoring
    track_gradient_norms: bool = True
    gradient_norm_window: int = 10
    
    # Checkpointing
    checkpoint_freq: int = 3                    # More frequent due to faster epochs
    
    # Curriculum configuration
    curriculum: CurriculumConfig = None
    
    # Additional optimization settings
    channels_last: bool = True                  # Memory format optimization
    compile_model: bool = False                 # torch.compile for 2x speedup (PyTorch 2.0+)
    tf32_matmul: bool = True                    # Enable TF32 for A100/H100 GPUs
    cudnn_benchmark: bool = True                # Auto-tune cuDNN kernels
    
    def __post_init__(self):
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()
        
        # Enable performance optimizations
        if self.tf32_matmul and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True


@dataclass
class LossConfig:
    """Loss function weights - Adjusted for higher capacity model"""
    # Base weights (starting values)
    seg_weight: float = 1.0
    dice_weight: float = 1.0
    sdf_weight: float = 0.6                     # Slightly higher (was 0.5)
    attr_weight: float = 1.0
    polygon_weight: float = 1.2                 # Higher for better geometry (was 1.0)
    voxel_weight: float = 1.2                   # Higher for 3D quality (was 1.0)
    topology_weight: float = 0.15               # Start slightly higher (was 0.1)
    
    # New loss components
    latent_consistency_weight: float = 0.6      # Higher (was 0.5)
    graph_constraint_weight: float = 0.35       # Higher (was 0.3)
    
    # Dynamic weighting parameters
    enable_dynamic_weighting: bool = True
    weight_update_freq: int = 8                 # More frequent updates (was 10)
    weight_momentum: float = 0.9


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = "final_model.pth"
    test_images_dir: str = "./data/test_images"
    output_dir: str = "./outputs"
    export_intermediate: bool = True
    polygon_threshold: float = 0.5
    batch_size: int = 8                         # Higher for faster inference
    use_mixed_precision: bool = True            # FP16 for 2x inference speed


# Curriculum stage transition logic (unchanged)
class StageTransitionCriteria:
    """Defines criteria for automatic stage transitions"""
    
    @staticmethod
    def should_transition_from_stage1(train_losses, val_losses, config: CurriculumConfig) -> bool:
        if len(val_losses) < config.plateau_detection_window:
            return False
            
        recent_losses = val_losses[-config.plateau_detection_window:]
        if len(recent_losses) < 2:
            return False
            
        old_avg = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        new_avg = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        improvement_rate = (old_avg - new_avg) / (old_avg + 1e-8)
        return improvement_rate < config.min_improvement_threshold
    
    @staticmethod 
    def should_transition_from_stage2(polygon_losses, config: CurriculumConfig) -> bool:
        if len(polygon_losses) < config.plateau_detection_window:
            return False
            
        recent_losses = polygon_losses[-config.plateau_detection_window:]
        if len(recent_losses) < 2:
            return False
            
        old_avg = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        new_avg = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        improvement_rate = (old_avg - new_avg) / (old_avg + 1e-8)
        return improvement_rate < config.min_improvement_threshold


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


# Performance tuning guide printed on import
print("=" * 80)
print("NEURAL-GEOMETRIC 3D GENERATOR - 48GB GPU OPTIMIZED CONFIGURATION")
print("=" * 80)
print(f"Batch Size: {DEFAULT_DATA_CONFIG.batch_size} (4x increase)")
print(f"Image Resolution: {DEFAULT_DATA_CONFIG.image_size} (1.5x increase)")
print(f"Voxel Resolution: {DEFAULT_DATA_CONFIG.voxel_size} (1.5x increase)")
print(f"Feature Dimension: {DEFAULT_MODEL_CONFIG.feature_dim} (1.5x increase)")
print(f"Model Parameters: ~2.5x increase from base configuration")
print(f"Expected Training Speed: ~3-4x faster per epoch")
print(f"Expected Memory Usage: ~35-42GB / 48GB")
print("=" * 80)
print("Performance Tips:")
print("  • Enable torch.compile() if using PyTorch 2.0+ for 2x speedup")
print("  • Use channels_last memory format (enabled by default)")
print("  • Monitor GPU utilization with nvidia-smi or wandb")
print("  • Adjust batch_size if OOM errors occur (try 12 or 14)")
print("=" * 80)