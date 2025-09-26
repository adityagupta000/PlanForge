"""
Configuration settings for the Neural-Geometric 3D Model Generator
Enhanced with dynamic curriculum and adaptive training strategies
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import torch


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "./data/floorplans"
    image_size: Tuple[int, int] = (256, 256)   # keep full resolution for accuracy
    voxel_size: int = 64
    batch_size: int = 4                        # balance speed & memory
    num_workers: int = 8                       # faster dataloader (tune per CPU)
    augment: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration optimized for high accuracy"""
    input_channels: int = 3
    num_classes: int = 5
    feature_dim: int = 512     # reduced from 768 → faster while keeping strong accuracy
    num_attributes: int = 6
    voxel_size: int = 64
    max_polygons: int = 20     # enough for complex layouts
    max_points: int = 50       # good detail without huge cost
    dropout: float = 0.05
    use_attention: bool = True
    use_deep_supervision: bool = True
    
    # Auxiliary heads for novel training strategies
    use_latent_consistency: bool = True
    use_graph_constraints: bool = True
    latent_embedding_dim: int = 256


@dataclass 
class CurriculumConfig:
    """Dynamic curriculum learning configuration"""
    # Adaptive stage transitioning
    use_dynamic_curriculum: bool = True
    stage_switch_patience: int = 5
    min_improvement_threshold: float = 0.001
    plateau_detection_window: int = 3

    # GradNorm / gradient tracking
    gradient_norm_window: int = 100

    # Objectives for multi-objective optimization
    objectives: Optional[List[str]] = None

    # Topology-aware scheduling
    topology_schedule: str = "progressive"   # "progressive", "linear_ramp", "exponential"
    topology_start_weight: float = 0.1
    topology_end_weight: float = 1.0
    topology_ramp_epochs: int = 20
    
    # config.py (snippet — add into the existing config class/dict)
    # Mixed precision and training conveniences
    use_mixed_precision = True            # enable AMP
    cache_in_memory = False               # set True if host RAM can hold dataset
    accumulation_steps = 1                # effective batch size multiplier
    dvx_step_freq = 1                     # run DVX refinement every N steps (1 = every step)
    persistent_workers = True             # DataLoader persistent workers
    prefetch_factor = 4                   # DataLoader prefetch
    num_workers = 8                       # default num workers for DataLoader (tune by CPU)
    # Progressive resolution settings (example)
    voxel_size_stage = { "stage1": 32, "stage2": 32, "stage3": 64 }  # voxel sizes per stage
    image_size_stage = { "stage1": (128,128), "stage2": (192,192), "stage3": (256,256)}

    
    # Loss component scheduling
    loss_schedule: Dict[str, str] = None
    
    # Multi-objective optimization (GradNorm)
    use_gradnorm: bool = True
    gradnorm_alpha: float = 0.12
    gradnorm_update_freq: int = 5
    
    # Graph constraint scheduling
    graph_weight_schedule: str = "delayed_ramp"
    graph_start_epoch: int = 15
    graph_end_weight: float = 0.25
    
    def __post_init__(self):
        # Provide default loss schedule if not set
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

        # Default objectives used by GradNorm / trainer monitoring
        if self.objectives is None:
            self.objectives = [
                "segmentation",
                "dice",
                "sdf",
                "attributes",
                "polygon",
                "voxel",
                "topology",
                "latent_consistency",
                "graph",
            ]


@dataclass
class TrainingConfig:
    """Training configuration with adaptive strategies"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dynamic epoch limits (maxima; curriculum may switch earlier)
    max_stage1_epochs: int = 40
    max_stage2_epochs: int = 25
    max_stage3_epochs: int = 60
    
    # Minimum epochs per stage (avoid switching too early)
    min_stage1_epochs: int = 8
    min_stage2_epochs: int = 5
    min_stage3_epochs: int = 12
    
    # Learning rates (per stage)
    stage1_lr: float = 3e-4  # was 3e-4
    stage1_weight_decay: float = 1e-5
    
    stage2_lr: float = 1e-4  # was 1e-4 
    stage2_weight_decay: float = 1e-5
    
    stage3_lr: float = 5e-5  # was 5e-5
    stage3_weight_decay: float = 1e-5
    
    # Advanced training techniques
    use_mixed_precision: bool = True
    use_cosine_restarts: bool = True
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    
    # Gradient monitoring for dynamic weighting
    track_gradient_norms: bool = True
    gradient_norm_window: int = 10  # rolling window for gradient tracking
    
    # Checkpointing
    checkpoint_freq: int = 1
    
    # Curriculum configuration
    curriculum: CurriculumConfig = None
    
    def __post_init__(self):
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()


@dataclass
class LossConfig:
    """Loss function weights (will be dynamically adjusted during training)"""
    # Base weights (starting values)
    seg_weight: float = 1.0
    dice_weight: float = 1.0
    sdf_weight: float = 0.5
    attr_weight: float = 1.0
    polygon_weight: float = 1.0
    voxel_weight: float = 1.0
    topology_weight: float = 0.1  # start low, ramp up
    
    # New loss components
    latent_consistency_weight: float = 0.5
    graph_constraint_weight: float = 0.3
    
    # Dynamic weighting parameters
    enable_dynamic_weighting: bool = True
    weight_update_freq: int = 10
    weight_momentum: float = 0.9


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = "final_model.pth"
    test_images_dir: str = "./data/test_images"
    output_dir: str = "./outputs"
    export_intermediate: bool = True
    polygon_threshold: float = 0.5


# Curriculum stage transition logic
class StageTransitionCriteria:
    """Defines criteria for automatic stage transitions"""
    
    @staticmethod
    def should_transition_from_stage1(train_losses, val_losses, config: CurriculumConfig) -> bool:
        """Check if should transition from Stage 1 to Stage 2"""
        if len(val_losses) < config.plateau_detection_window:
            return False
            
        # Check for plateau in segmentation + dice losses
        recent_losses = val_losses[-config.plateau_detection_window:]
        if len(recent_losses) < 2:
            return False
            
        # Calculate improvement rate
        old_avg = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        new_avg = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        improvement_rate = (old_avg - new_avg) / (old_avg + 1e-8)
        return improvement_rate < config.min_improvement_threshold
    
    @staticmethod 
    def should_transition_from_stage2(polygon_losses, config: CurriculumConfig) -> bool:
        """Check if should transition from Stage 2 to Stage 3"""
        if len(polygon_losses) < config.plateau_detection_window:
            return False
            
        # Check polygon loss plateau
        recent_losses = polygon_losses[-config.plateau_detection_window:]
        if len(recent_losses) < 2:
            return False
            
        old_avg = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
        new_avg = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
        
        improvement_rate = (old_avg - new_avg) / (old_avg + 1e-8)
        return improvement_rate < config.min_improvement_threshold


# Default configurations (import these in your trainer)
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
