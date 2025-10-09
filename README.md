# Neural-Geometric 3D Model Generator

A research-grade system for converting 2D floorplan images to 3D architectural models using neural networks and differentiable geometric operations.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Demo](#demo)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Loss Functions](#loss-functions)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Formats](#output-formats)
- [Research Features](#research-features)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Project Structure

```
neural-geometric-3d/
├── config.py                    # Configuration settings
├── dataset.py                   # Dataset classes and data loading
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── train.py                     # Main training script
├── evaluate.py                  # Model evaluation script                    
│
├── models/                      # Model architecture components
│   ├── __init__.py
│   ├── encoder.py               # Multi-scale encoder backbone
│   ├── heads.py                 # Prediction heads (segmentation, attributes, SDF)
│   ├── dvx.py                   # Differentiable Vectorization module
│   ├── extrusion.py             # Differentiable 3D extrusion
│   └── model.py                 # Main model definition
│
├── training/                    # Training system
│   ├── __init__.py
│   ├── losses.py                # Advanced loss functions
│   └── trainer.py               # Multi-stage training system
│
├── inference/                   # Inference system
│   ├── __init__.py
│   └── engine.py                # Research-grade inference engine
│
├── evaluation/                  # Evaluation utilities
│   ├── __init__.py
│   └── metrics.py               # Evaluation metrics and utilities
│
└── utils/                       # Utility functions
    ├── __init__.py
    └── visualization.py          # Visualization and utility functions
```

## Features

### Multi-Task Learning Architecture
- **Multi-scale encoder**: ResNet backbone with Feature Pyramid Network (FPN)
- **Segmentation head**: Semantic segmentation with multi-scale fusion
- **Attribute head**: Regression for geometric parameters
- **SDF head**: Signed Distance Field prediction for sharp boundaries

### Differentiable Vectorization (DVX)
- Converts soft segmentation masks to polygon representations
- Differentiable active contour fitting
- Polygon validity prediction

### Differentiable 3D Extrusion
- Converts polygons + attributes to 3D occupancy grids
- Differentiable geometric operations
- Soft 3D reconstruction

### Multi-Stage Training
- **Stage 1**: Segmentation + Attributes (2D supervision)
- **Stage 2**: DVX training (polygon fitting)
- **Stage 3**: End-to-end fine-tuning (all losses)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd neural-geometric-3d

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The dataset should be organized as follows:

```
data/floorplans/
├── train/
│   ├── sample_001/
│   │   ├── image.png           # Input floorplan image
│   │   ├── mask.png            # Segmentation ground truth
│   │   ├── params.json         # Geometric parameters
│   │   ├── model.obj           # 3D mesh ground truth
│   │   ├── voxel_GT.npz        # Voxelized occupancy
│   │   └── polygon.json        # Polygon ground truth
│   └── ...
├── val/
└── test/
```

## Usage

### Training

```bash
# Train complete pipeline (all stages)
python train.py --data_dir ./data/floorplans --batch_size 8

# Train specific stage
python train.py --stage 1  # Only stage 1 (2D supervision)
python train.py --stage 2  # Only stage 2 (DVX training)
python train.py --stage 3  # Only stage 3 (end-to-end)

# Resume from checkpoint
python train.py --resume checkpoint.pth
```

### Inference

```bash
# Single image inference
python infer.py --model_path final_model.pth --input image.png --output model.obj

# Batch processing
python infer.py --model_path final_model.pth --input ./test_images/ --output ./results/

# Export intermediate results
python infer.py --model_path final_model.pth --input image.png --output model.obj --export_intermediate
```

### Evaluation

```bash
# Comprehensive evaluation
python evaluate.py --model_path final_model.pth --data_dir ./data/floorplans

# With visualizations
python evaluate.py --model_path final_model.pth --visualize --save_outputs
```

### Demo

```bash
# Run demo with synthetic data
python demo.py

# Demo with pretrained model
python demo.py --model_path final_model.pth --input_image sample.png
```

## Configuration

All configuration settings are centralized in `config.py`:

- **DataConfig**: Dataset and preprocessing settings
- **ModelConfig**: Model architecture parameters
- **TrainingConfig**: Training hyperparameters and schedules
- **LossConfig**: Loss function weights
- **InferenceConfig**: Inference settings

## Model Architecture

### Multi-Scale Encoder
- ResNet-based backbone with FPN
- Multi-resolution feature extraction
- Global context modeling

### Prediction Heads
- **Segmentation**: Multi-class semantic segmentation
- **Attributes**: Geometric parameter regression
- **SDF**: Signed distance field prediction

### Differentiable Vectorization (DVX)
- Polygon initialization network
- Control point refinement
- Validity prediction

### 3D Extrusion
- Differentiable polygon-to-voxel conversion
- Geometric parameter integration
- Soft 3D reconstruction

## Training Pipeline

### Stage 1: 2D Supervision (20 epochs)
- Train encoder + segmentation/attribute/SDF heads
- Freeze DVX and extrusion modules
- Focus on 2D representation learning

### Stage 2: DVX Training (15 epochs)
- Freeze encoder and 2D heads
- Train DVX module for polygon fitting
- Learn vectorization from segmentation

### Stage 3: End-to-End (30 epochs)
- Unfreeze all modules
- Joint optimization with all losses
- Fine-tune complete pipeline

## Loss Functions

- **Segmentation**: Cross-entropy + Dice loss
- **SDF**: MSE loss with distance transform
- **Attributes**: L1 regression loss
- **Polygons**: Point matching + validity + smoothness + rectilinearity
- **Voxels**: 3D IoU loss
- **Topology**: Architectural constraint enforcement

## Evaluation Metrics

- **Segmentation**: mIoU, per-class IoU, Dice score
- **Attributes**: MAE, MSE for geometric parameters
- **Polygons**: Chamfer distance, validity accuracy
- **3D Reconstruction**: 3D IoU, voxel accuracy
- **Architecture-specific**: Room count, wall connectivity

## Output Formats

- **3D Mesh**: OBJ format with vertices and faces
- **Intermediate Results**: Segmentation masks, polygons, attributes
- **Visualizations**: Training curves, prediction comparisons
- **Evaluation Reports**: Comprehensive metrics and analysis

## Research Features

- **Multi-representation learning**: Combined 2D and 3D supervision
- **Differentiable geometry**: End-to-end trainable geometric operations
- **Architectural constraints**: Topology-aware loss functions
- **Progressive training**: Multi-stage curriculum learning

## Citation

If you use this code in your research, please cite:

```bibtex
@article{neural_geometric_3d,
  title={Neural-Geometric 3D Model Generator: End-to-End Learning for 2D to 3D Floorplan Generation},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

- **CUDA out of memory**: Reduce batch size in config or use smaller model
- **No training data found**: Check data directory structure matches expected format
- **Slow training**: Ensure CUDA is properly installed and configured

### Performance Tips

- Use mixed precision training for faster training
- Increase batch size if GPU memory allows
- Use multiple GPUs with DataParallel for larger datasets
- Precompute SDF ground truth for faster training

## Contact

For questions or issues, please open a GitHub issue or contact [adityagupta.d7@gmail.com].
