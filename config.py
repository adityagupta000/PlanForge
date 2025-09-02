"""
Configuration settings for the Neural-Geometric 3D Model Generator
Enhanced with dynamic curriculum and adaptive training strategies
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "./data/floorplans"
    image_size: Tuple[int, int] = (256, 256)
    voxel_size: int = 64
    batch_size: int = 8
    num_workers: int = 4
    augment: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration optimized for high accuracy"""
    input_channels: int = 3
    num_classes: int = 5
    feature_dim: int = 768  # Increased for better representation
    num_attributes: int = 6
    voxel_size: int = 64
    max_polygons: int = 30  # More polygons for complex layouts
    max_points: int = 100   # More points per polygon for precision
    dropout: float = 0.05   # Reduced dropout for better performance
    use_attention: bool = True  # Add attention mechanisms
    use_deep_supervision: bool = True  # Multi-scale supervision
    
    # New: Auxiliary heads for novel training strategies
    use_latent_consistency: bool = True  # Cross-modal consistency
    use_graph_constraints: bool = True   # Graph-based topology
    latent_embedding_dim: int = 256      # Consistency embedding size


@dataclass 
class CurriculumConfig:
    """Dynamic curriculum learning configuration"""
    # Adaptive stage transitioning
    use_dynamic_curriculum: bool = True
    stage_switch_patience: int = 5  # Epochs without improvement before switching
    min_improvement_threshold: float = 0.001  # Minimum relative improvement
    plateau_detection_window: int = 3  # Rolling window for plateau detection
    
    # Topology-aware scheduling
    topology_schedule: str = "progressive"  # "progressive", "linear_ramp", "exponential"
    topology_start_weight: float = 0.1
    topology_end_weight: float = 1.0
    topology_ramp_epochs: int = 20
    
    # Loss component scheduling
    loss_schedule: Dict[str, str] = None  # Will be set in __post_init__
    
    # Multi-objective optimization
    use_gradnorm: bool = True
    gradnorm_alpha: float = 0.12  # GradNorm restoring force
    gradnorm_update_freq: int = 5  # Update loss weights every N batches
    
    # Graph constraint scheduling
    graph_weight_schedule: str = "delayed_ramp"  # Start after polygon stabilizes
    graph_start_epoch: int = 15
    graph_end_weight: float = 0.5
    
    def __post_init__(self):
        if self.loss_schedule is None:
            self.loss_schedule = {
                "segmentation": "static",      # Keep constant
                "dice": "static",             
                "sdf": "early_decay",         # Decay after Stage 1
                "attributes": "static",
                "polygon": "staged_ramp",     # Ramp up in Stage 2
                "voxel": "late_ramp",        # Ramp up in Stage 3
                "topology": "progressive",    # Progressive increase
                "latent_consistency": "mid_ramp",  # Activate mid-training
                "graph": "delayed_ramp"       # Activate after polygons stable
            }


@dataclass
class TrainingConfig:
    """Training configuration with adaptive strategies"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dynamic epoch limits (maximums, actual may be less with early switching)
    max_stage1_epochs: int = 80    # Increased maximum
    max_stage2_epochs: int = 50
    max_stage3_epochs: int = 150
    
    # Minimum epochs per stage (prevent premature switching)
    min_stage1_epochs: int = 10
    min_stage2_epochs: int = 5 
    min_stage3_epochs: int = 15
    
    # Learning rates (will be dynamically adjusted)
    stage1_lr: float = 2e-4
    stage1_weight_decay: float = 1e-5
    
    stage2_lr: float = 1e-4
    stage2_weight_decay: float = 1e-5
    
    stage3_lr: float = 5e-5
    stage3_weight_decay: float = 1e-5
    
    # Advanced training techniques
    use_mixed_precision: bool = True
    use_cosine_restarts: bool = True
    warmup_epochs: int = 5
    grad_clip_norm: float = 0.5
    
    # Gradient monitoring for dynamic weighting
    track_gradient_norms: bool = True
    gradient_norm_window: int = 10  # Rolling window for gradient tracking
    
    checkpoint_freq: int = 5
    
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
    topology_weight: float = 0.1  # Start low, ramp up
    
    # New loss components
    latent_consistency_weight: float = 0.5
    graph_constraint_weight: float = 0.3
    
    # Dynamic weighting parameters
    enable_dynamic_weighting: bool = True
    weight_update_freq: int = 10  # Update weights every N batches
    weight_momentum: float = 0.9   # Momentum for weight updates


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


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()