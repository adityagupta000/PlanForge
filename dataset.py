"""
Dataset classes for the Neural-Geometric 3D Model Generator
"""

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from config import DEFAULT_DATA_CONFIG


class AdvancedFloorPlanDataset(Dataset):
    """
    Research-grade dataset with complete ground truth:
    - Floorplan image + segmentation mask
    - Attribute dictionary (geometric parameters)
    - Ground-truth mesh + voxelized occupancy
    - Polygon outlines for vectorization supervision
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        image_size: Tuple[int, int] = None,
        voxel_size: int = None,
        augment: bool = None,
        config=None,
    ):
        # Use config if provided, otherwise defaults from config.py
        if config is None:
            config = DEFAULT_DATA_CONFIG

        self.data_dir = Path(data_dir or config.data_dir)
        self.split = split
        self.image_size = image_size or config.image_size
        self.voxel_size = voxel_size or config.voxel_size
        self.augment = (
            augment if augment is not None else config.augment
        ) and split == "train"

        # Collect all samples that contain every required file
        self.samples = self._find_complete_samples()
        print(f"Found {len(self.samples)} complete samples for {split}")

    # ----------------------------------------------------------------------
    def _find_complete_samples(self):
        """Locate samples that contain all the expected files."""
        samples = []
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            print(f"Warning: Split directory {split_dir} does not exist")
            return samples

        for sample_dir in split_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            required_files = {
                "image": sample_dir / "image.png",
                "mask": sample_dir / "mask.png",
                "params": sample_dir / "params.json",
                "mesh": sample_dir / "model.obj",
                "voxel": sample_dir / "voxel_GT.npz",
                "polygon": sample_dir / "polygon.json",
            }

            if all(f.exists() for f in required_files.values()):
                samples.append(required_files)

        return samples

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = cv2.imread(str(sample["image"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size) / 255.0

        mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        mask[mask == 5] = 0

        # Load attributes
        with open(sample["params"], "r") as f:
            attributes = json.load(f)

        # Load voxel ground truth
        voxel_data = np.load(sample["voxel"])
        voxels_gt = voxel_data["voxels"]

        # Load polygon ground truth
        with open(sample["polygon"], "r") as f:
            polygons_gt = json.load(f)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()
        voxels_tensor = torch.from_numpy(voxels_gt.astype(np.float32))

        attr_tensor = self._process_attributes(attributes)
        polygon_tensor = self._process_polygons(polygons_gt)

        if self.augment:
            image_tensor, mask_tensor = self._augment(image_tensor, mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "attributes": attr_tensor,
            "voxels_gt": voxels_tensor,
            "polygons_gt": polygon_tensor,
            "sample_id": sample["image"].parent.name,
        }

    # ----------------------------------------------------------------------
    def _process_attributes(self, attributes):
        """Convert attribute dictionary to a normalized tensor."""
        # Normalize common architectural parameters into [0,1]
        attr_list = [
            attributes.get("wall_height", 2.6) / 5.0,
            attributes.get("wall_thickness", 0.15) / 0.5,
            attributes.get("window_base_height", 0.7) / 3.0,
            attributes.get("window_height", 0.95) / 2.0,
            attributes.get("door_height", 2.6) / 5.0,
            attributes.get("pixel_scale", 0.01) / 0.02,
        ]
        return torch.tensor(attr_list, dtype=torch.float32)

    # ----------------------------------------------------------------------
    def _process_polygons(self, polygons_gt):
        """Convert polygon ground truth into a fixed tensor representation.
        Handles both formats:
        1. Nested dict: { "walls": [...], "doors": [...], ... }
        2. Flat list:   [ {"type": "wall", "points": [...]}, ... ]
        """
        max_polygons = 20   # number of polygons per sample
        max_points = 50     # max points per polygon

        processed = torch.zeros(max_polygons, max_points, 2)
        valid_mask = torch.zeros(max_polygons, dtype=torch.bool)

        poly_idx = 0

        # --- Case 1: dict format ---
        if isinstance(polygons_gt, dict):
            for class_name, polygon_list in polygons_gt.items():
                if not isinstance(polygon_list, list):
                    continue
                for polygon in polygon_list:
                    if poly_idx >= max_polygons:
                        break
                    if "points" not in polygon:
                        continue

                    points = np.array(polygon["points"])
                    if len(points) > max_points:
                        # Subsample evenly if too many points
                        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
                        points = points[indices]

                    # Normalize to [0,1] relative to image size
                    points = points / np.array(self.image_size)
                    processed[poly_idx, : len(points)] = torch.from_numpy(points).float()
                    valid_mask[poly_idx] = True
                    poly_idx += 1

        # --- Case 2: list format ---
        elif isinstance(polygons_gt, list):
            for polygon in polygons_gt:
                if poly_idx >= max_polygons:
                    break
                if "points" not in polygon:
                    continue

                points = np.array(polygon["points"])
                if len(points) > max_points:
                    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
                    points = points[indices]

                points = points / np.array(self.image_size)
                processed[poly_idx, : len(points)] = torch.from_numpy(points).float()
                valid_mask[poly_idx] = True
                poly_idx += 1

        return {"polygons": processed, "valid_mask": valid_mask}

    # ----------------------------------------------------------------------
    def _augment(self, image, mask):
        """Simple data augmentation with rotations and flips."""
        # Random rotation (multiples of 90° only)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        return image, mask


# ======================================================================
# Synthetic sample generator for testing without dataset
# ======================================================================
def create_synthetic_data_sample():
    """Generate a synthetic floorplan with attributes, voxels, and polygons."""
    image = np.ones((256, 256, 3), dtype=np.uint8) * 255
    mask = np.zeros((256, 256), dtype=np.uint8)

    # Simple square room
    room_points = np.array([[50, 50], [200, 50], [200, 200], [50, 200]])
    cv2.fillPoly(mask, [room_points], 1)  # Room = class 1
    cv2.polylines(image, [room_points], True, (0, 0, 0), 3)

    # Add door
    cv2.rectangle(mask, (90, 50), (110, 70), 2)  # Door = class 2
    cv2.rectangle(image, (90, 50), (110, 70), (255, 0, 0), -1)

    # Attributes
    attributes = {
        "wall_height": 2.6,
        "wall_thickness": 0.15,
        "window_base_height": 0.7,
        "window_height": 0.95,
        "door_height": 2.6,
        "pixel_scale": 0.02,
    }

    # Simple voxel GT
    voxels = np.zeros((64, 64, 64), dtype=bool)
    voxels[:20, 10:50, 10:50] = True

    # Polygon GT
    polygons = {"walls": [{"points": room_points.tolist()}]}

    return image, mask, attributes, voxels, polygons
