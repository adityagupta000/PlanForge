"""
Configuration settings for the Neural-Geometric 3D Model Generator
"""

from dataclasses import dataclass
from typing import Tuple
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


@dataclass
class TrainingConfig:
    """Training configuration optimized for high performance"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Increased training for better convergence
    stage1_epochs: int = 50   # More 2D supervision
    stage1_lr: float = 2e-4   # Higher initial LR
    stage1_weight_decay: float = 1e-5  # Less regularization
    
    stage2_epochs: int = 30   # More DVX training
    stage2_lr: float = 1e-4   # Higher DVX LR
    stage2_weight_decay: float = 1e-5
    
    stage3_epochs: int = 100  # Much longer end-to-end training
    stage3_lr: float = 5e-5   # Careful fine-tuning
    stage3_weight_decay: float = 1e-5
    
    # Advanced training techniques
    use_mixed_precision: bool = True
    use_cosine_restarts: bool = True
    warmup_epochs: int = 5
    grad_clip_norm: float = 0.5  # Tighter gradient clipping
    
    # Data augmentation
    use_advanced_augmentation: bool = True
    augmentation_strength: float = 0.3
    
    checkpoint_freq: int = 5


@dataclass
class LossConfig:
    """Loss function weights"""
    seg_weight: float = 1.0
    dice_weight: float = 1.0
    sdf_weight: float = 0.5
    attr_weight: float = 1.0
    polygon_weight: float = 1.0
    voxel_weight: float = 1.0
    topology_weight: float = 0.5


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = "final_model.pth"
    test_images_dir: str = "./data/test_images"
    output_dir: str = "./outputs"
    export_intermediate: bool = True
    polygon_threshold: float = 0.5


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()