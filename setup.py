"""
Setup script for the 3D Model Generator project
"""

from pathlib import Path
import os

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define directory structure
    directories = [
        "models",
        "training", 
        "inference",
        "evaluation",
        "utils",
        "data/floorplans/train",
        "data/floorplans/val", 
        "data/floorplans/test",
        "data/test_images",
        "checkpoints",
        "outputs",
        "demo_outputs",
        "evaluation_results",
        "logs"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "models/__init__.py",
        "training/__init__.py", 
        "inference/__init__.py",
        "evaluation/__init__.py",
        "utils/__init__.py"
    ]
    
    init_content = {
        "models/__init__.py": '''"""
Model components for 3D Model Generator
"""

from .encoder import MultiScaleEncoder, ResidualBlock
from .heads import SegmentationHead, AttributeHead, SDFHead
from .dvx import DifferentiableVectorization
from .extrusion import DifferentiableExtrusion
from .model import NeuralGeometric3DGenerator

__all__ = [
    'MultiScaleEncoder',
    'ResidualBlock', 
    'SegmentationHead',
    'AttributeHead',
    'SDFHead',
    'DifferentiableVectorization',
    'DifferentiableExtrusion',
    'NeuralGeometric3DGenerator'
]''',
        
        "training/__init__.py": '''"""
Training components for 3D Model Generator
"""

from .losses import ResearchGradeLoss
from .trainer import MultiStageTrainer

__all__ = [
    'ResearchGradeLoss',
    'MultiStageTrainer'
]''',
        
        "inference/__init__.py": '''"""
Inference components for 3D Model Generator
"""

from .engine import ResearchInferenceEngine

__all__ = [
    'ResearchInferenceEngine'
]''',
        
        "evaluation/__init__.py": '''"""
Evaluation components for 3D Model Generator
"""

from .metrics import ModelEvaluator, evaluate_model, compute_iou, compute_3d_iou

__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'compute_iou',
    'compute_3d_iou'
]''',
        
        "utils/__init__.py": '''"""
Utility functions for 3D Model Generator
"""

from .visualization import (
    plot_training_history,
    visualize_predictions,
    visualize_polygons,
    save_model_outputs,
    create_comparison_grid,
    create_3d_visualization
)

__all__ = [
    'plot_training_history',
    'visualize_predictions', 
    'visualize_polygons',
    'save_model_outputs',
    'create_comparison_grid',
    'create_3d_visualization'
]'''
    }
    
    for file_path, content in init_content.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created: {file_path}")


def create_sample_config():
    """Create a sample configuration file for easy customization"""
    
    sample_config = '''"""
Sample configuration for quick customization
Copy this to config_custom.py and modify as needed
"""

from config import *

# Custom configuration example
CUSTOM_DATA_CONFIG = DataConfig(
    data_dir="./my_data/floorplans",
    batch_size=16,  # Larger batch if you have more GPU memory
    num_workers=8,  # More workers if you have more CPU cores
)

CUSTOM_TRAINING_CONFIG = TrainingConfig(
    stage1_epochs=30,  # More epochs for better 2D learning
    stage2_epochs=20,  # More DVX training
    stage3_epochs=50,  # Longer end-to-end training
    stage1_lr=2e-4,    # Higher learning rate
)

CUSTOM_MODEL_CONFIG = ModelConfig(
    feature_dim=768,   # Larger model
    voxel_size=128,    # Higher resolution 3D
    max_polygons=30,   # More polygons
)
'''
    
    with open("config_custom_example.py", "w") as f:
        f.write(sample_config)
    print("Created: config_custom_example.py")


def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Data
data/
datasets/
*.npz
*.obj
*.off
*.ply

# Outputs
outputs/
results/
checkpoints/
logs/
demo_outputs/
evaluation_results/
training_progress/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# Images and videos  
*.png
*.jpg
*.jpeg
*.gif
*.mp4
*.avi

# Except sample images
!sample_images/
!docs/images/
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Created: .gitignore")


def main():
    """Main setup function"""
    print("Setting up 3D Model Generator project...")
    print("=" * 60)
    
    # Create directory structure
    create_project_structure()
    print()
    
    # Create sample config
    create_sample_config()
    print()
    
    # Create gitignore
    create_gitignore()
    print()
    
    print("Project setup completed!")
    print("=" * 60)
    print("IMPORTANT ACCURACY EXPECTATIONS:")
    print("- 90%+ accuracy in 2D-to-3D generation is extremely challenging")
    print("- Actual accuracy depends heavily on:")
    print("  * Dataset quality and size (need 10K+ samples)")
    print("  * Ground truth accuracy")
    print("  * Problem complexity (simple vs complex floorplans)")
    print("  * Evaluation metrics used")
    print("- Realistic expectations:")
    print("  * Segmentation: 75-85% mIoU with good data")
    print("  * 3D reconstruction: 60-75% IoU for architectural scenes")
    print("  * Polygon fitting: 70-80% accuracy")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare high-quality dataset (critical for accuracy)")
    print("3. Run demo: python demo.py")
    print("4. Start training: python train.py")


if __name__ == "__main__":
    main()