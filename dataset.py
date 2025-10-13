"""
Dataset classes for the Neural-Geometric 3D Model Generator
Enhanced with in-memory caching for faster training
FIXED: UnboundLocalError in __getitem__ method
"""

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time

from config import DEFAULT_DATA_CONFIG


class AdvancedFloorPlanDataset(Dataset):
    """
    Research-grade dataset with complete ground truth:
    - Floorplan image + segmentation mask
    - Attribute dictionary (geometric parameters)
    - Ground-truth mesh + voxelized occupancy
    - Polygon outlines for vectorization supervision

    Enhanced with optional in-memory caching for performance
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

        # NEW: In-memory caching for performance
        self.cache_in_memory = getattr(config, "cache_in_memory", False)
        self._cache = None

        if self.cache_in_memory and len(self.samples) > 0:
            print(
                f"[DATA] Preloading {len(self.samples)} samples into RAM (cache_in_memory=True)."
            )
            print(
                "[DATA] This may take significant memory but will speed up training..."
            )

            # Estimate memory usage
            estimated_mb = self._estimate_memory_usage()
            print(f"[DATA] Estimated memory usage: {estimated_mb:.1f} MB")

            start_time = time.time()
            self._preload_cache()
            load_time = time.time() - start_time
            print(f"[DATA] Cache preloading completed in {load_time:.2f}s")

    def _estimate_memory_usage(self):
        """Estimate memory usage for caching"""
        if not self.samples:
            return 0.0

        H, W = self.image_size
        n_samples = len(self.samples)

        # Rough estimates in bytes
        image_bytes = H * W * 3  # RGB uint8
        mask_bytes = H * W  # grayscale uint8
        voxel_bytes = self.voxel_size**3 * 4  # float32
        json_bytes = 1024  # rough estimate for params + polygons

        if self.voxel_size >= 128:
            voxel_gb = (voxel_bytes * n_samples) / (1024**3)
            print(
                f"[WARNING] Large voxel grid ({self.voxel_size}^3) may lead to high memory usage: {voxel_gb:.2f} GB just for voxels"
            )
            print(
                "[Warning] Consider reducing voxel_size or disabling cache_in_memory."
            )

        total_per_sample = image_bytes + mask_bytes + voxel_bytes + json_bytes
        total_mb = (total_per_sample * n_samples) / (1024 * 1024)

        return total_mb

    def _preload_cache(self):
        """Preload all samples into memory"""
        self._cache = []

        for i, sample in enumerate(self.samples):
            if i % 100 == 0:
                print(f"[DATA] Loading sample {i+1}/{len(self.samples)}")

            try:
                # Load image
                img = cv2.imread(str(sample["image"]))
                if img is None:
                    print(f"Warning: Could not load image {sample['image']}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size)  # (W, H) format for cv2.resize

                # Load mask
                mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not load mask {sample['mask']}")
                    continue
                mask = cv2.resize(
                    mask, self.image_size, interpolation=cv2.INTER_NEAREST
                )

                # Load voxel data
                try:
                    voxel_data = np.load(sample["voxel"])
                    vox = voxel_data["voxels"]  # Keep as numpy array
                except Exception as e:
                    print(f"Warning: Could not load voxel data {sample['voxel']}: {e}")
                    # Create dummy voxel data
                    vox = np.zeros(
                        (self.voxel_size, self.voxel_size, self.voxel_size),
                        dtype=np.float32,
                    )

                # Load parameters
                try:
                    with open(sample["params"], "r") as f:
                        params = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load params {sample['params']}: {e}")
                    params = self._get_default_attributes()

                # Load polygons
                try:
                    with open(sample["polygon"], "r") as f:
                        polygons = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load polygons {sample['polygon']}: {e}")
                    polygons = {"walls": []}

                self._cache.append(
                    {
                        "image": img,
                        "mask": mask,
                        "vox": vox,
                        "params": params,
                        "polygons": polygons,
                        "sample_id": sample["image"].parent.name,
                    }
                )

            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue

    def _get_default_attributes(self):
        """Return default attributes for missing param files"""
        return {
            "wall_height": 1.0,  # Changed to match dataset
            "wall_thickness": 0.15,  # Unchanged
            "window_base_height": 0.269,  # Changed to match dataset
            "window_height": 0.365,  # Changed to match dataset
            "door_height": 1.0,  # Changed to match dataset
            "pixel_scale": 0.01,
        }

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
        return len(self._cache) if self._cache is not None else len(self.samples)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        # Use cached data if available
        if self._cache is not None:
            cached_sample = self._cache[idx]
            image = cached_sample["image"]
            mask = cached_sample["mask"]
            vox = cached_sample["vox"]
            attributes = cached_sample["params"]
            polygons_gt = cached_sample["polygons"]
            sample_id = cached_sample["sample_id"]
        else:
            # Fallback: load from disk on-the-fly
            sample = self.samples[idx]

            # Load image and mask
            image = cv2.imread(str(sample["image"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)

            mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            # Load attributes
            with open(sample["params"], "r") as f:
                attributes = json.load(f)

            # Load voxel ground truth
            voxel_data = np.load(sample["voxel"])
            vox = voxel_data["voxels"]

            # Load polygon ground truth
            with open(sample["polygon"], "r") as f:
                polygons_gt = json.load(f)

            sample_id = sample["image"].parent.name

        # Normalize image to [0,1]
        image = image.astype(np.float32) / 255.0

        # Clean mask (remove class 5 if present)
        # CRITICAL: Validate and clean mask - ONLY CLASSES 0-4 ALLOWED
        # Class 5 (cube) should have been filtered during dataset generation
        valid_classes = set(range(5))  # {0, 1, 2, 3, 4}
        mask_classes = set(np.unique(mask).tolist())
        invalid_classes = mask_classes - valid_classes

        if invalid_classes:
            print(
                f"ERROR: Sample {idx} ({sample_id}) contains INVALID classes {invalid_classes}"
            )
            print(
                f"  Expected only classes 0-4 (background, wall, door, window, floor)"
            )
            print(
                f"  This indicates dataset generation failed to filter class 5 (cube)"
            )
            print(f"  Forcing cleanup: remapping {invalid_classes} -> 0 (background)")

            # Force remap invalid classes to background
            for invalid_class in invalid_classes:
                mask[mask == invalid_class] = 0

        # Additional validation: warn if mask is empty
        non_bg_pixels = np.sum(mask > 0)
        if non_bg_pixels == 0:
            print(
                f"WARNING: Sample {idx} ({sample_id}) has EMPTY mask (all background)"
            )

        # Convert to tensors FIRST (before validation that uses attr_tensor)
        # FIX: This was moved here from later in the function
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()
        voxels_tensor = torch.from_numpy(vox.astype(np.float32))

        attr_tensor = self._process_attributes(attributes)
        polygon_tensor = self._process_polygons(polygons_gt)

        # VALIDATION: Verify parameter ranges match dataset generation expectations
        # FIX: Now attr_tensor is defined before we use it
        attr_array = attr_tensor.numpy()

        validation_failed = False

        # Check wall_height (normalized): should be ~0.52 (range 0.4-0.7)
        if not (0.3 <= attr_array[0] <= 0.7):
            print(
                f"WARNING: Sample {idx} wall_height normalized = {attr_array[0]:.3f} (expected ~0.5)"
            )
            validation_failed = True

        # Check wall_thickness (normalized): should be ~0.3 (range 0.2-0.5)
        if not (0.2 <= attr_array[1] <= 0.5):
            print(
                f"WARNING: Sample {idx} wall_thickness normalized = {attr_array[1]:.3f} (expected ~0.3)"
            )
            validation_failed = True

        # Check window_base_height (normalized): should be ~0.233 (range 0.15-0.35)
        if not (0.4 <= attr_array[2] <= 0.7):
            print(
                f"WARNING: Sample {idx} window_base_height normalized = {attr_array[2]:.3f} (expected ~0.538)"
            )
            validation_failed = True

        # Check window_height (normalized): should be ~0.475 (range 0.35-0.65)
        if not (0.6 <= attr_array[3] <= 0.9):
            print(
                f"WARNING: Sample {idx} window_height normalized = {attr_array[3]:.3f} (expected ~0.73)"
            )
            validation_failed = True

        if validation_failed:
            print(
                f"  Sample {idx} ({sample_id}): Architectural parameters outside expected ranges"
            )
            print(
                f"  This indicates dataset was generated with incorrect config.py values"
            )
            print(f"  Raw attributes: {attributes}")

        # Apply augmentation if enabled
        if self.augment:
            image_tensor, mask_tensor = self._augment(image_tensor, mask_tensor)

        # Add validation before returning
        self._validate_sample_data(
            idx, image_tensor, mask_tensor, attr_tensor, voxels_tensor, polygon_tensor
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "attributes": attr_tensor,
            "voxels_gt": voxels_tensor,
            "polygons_gt": polygon_tensor,
            "sample_id": sample_id,
        }

    # ----------------------------------------------------------------------
    def _validate_sample_data(self, idx, image, mask, attributes, voxels, polygons):
        """Validate sample data for NaN/Inf values"""
        tensors_to_check = [
            ("image", image),
            ("mask", mask),
            ("attributes", attributes),
            ("voxels", voxels),
            ("polygons", polygons["polygons"]),
        ]

        corrupted_data = False

        for name, tensor in tensors_to_check:
            if torch.isnan(tensor).any():
                print(f"ERROR: {name} contains NaN values at sample {idx}")
                corrupted_data = True
            if torch.isinf(tensor).any():
                print(f"ERROR: {name} contains Inf values at sample {idx}")
                corrupted_data = True

        if corrupted_data:
            print(
                f"WARNING: Corrupted data detected in sample {idx}, replacing with safe fallback values"
            )

            # Replace corrupted tensors with safe fallback values
            for name, tensor in tensors_to_check:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    if name == "image":
                        # Replace with zeros (black image)
                        image.data = torch.zeros_like(image)
                    elif name == "mask":
                        # Replace with zeros (background class)
                        mask.data = torch.zeros_like(mask).long()
                    elif name == "attributes":
                        # Replace with reasonable default values (0.5 normalized)
                        attributes.data = torch.ones_like(attributes) * 0.5
                    elif name == "voxels":
                        # Replace with empty voxel grid
                        voxels.data = torch.zeros_like(voxels)
                    elif name == "polygons":
                        # Replace polygons with zeros
                        polygons["polygons"].data = torch.zeros_like(
                            polygons["polygons"]
                        )

    # ----------------------------------------------------------------------
    def _process_attributes(self, attributes):
        """Convert attribute dictionary to a normalized tensor."""
        # Normalize common architectural parameters into [0,1]
        attr_list = [
            attributes.get("wall_height", 1.0)
            / 2.0,  # Changed: 1.0m wall, normalize by 2.0
            attributes.get("wall_thickness", 0.15) / 0.5,  # Unchanged
            attributes.get("window_base_height", 0.269)
            / 0.5,  # Changed: 0.269m, normalize by 0.5
            attributes.get("window_height", 0.365)
            / 0.5,  # Changed: 0.365m, normalize by 0.5
            attributes.get("door_height", 1.0)
            / 2.0,  # Changed: 1.0m door, normalize by 2.0
            attributes.get("pixel_scale", 0.01) / 0.02,  # Unchanged
        ]

        # Ensure no NaN/Inf values in attribute processing
        safe_attr_list = []
        for val in attr_list:
            if np.isnan(val) or np.isinf(val):
                safe_attr_list.append(0.5)  # Default normalized value
            else:
                safe_attr_list.append(max(0.0, min(1.0, val)))  # Clamp to [0,1]

        return torch.tensor(safe_attr_list, dtype=torch.float32)

    # ----------------------------------------------------------------------
    def _process_polygons(self, polygons_gt):
        """Convert polygon ground truth into a fixed tensor representation.
        Handles both formats:
        1. Nested dict: { "walls": [...], "doors": [...], ... }
        2. Flat list:   [ {"type": "wall", "points": [...]}, ... ]
        """
        max_polygons = 30  # number of polygons per sample
        max_points = 100  # max points per polygon

        processed = torch.zeros(max_polygons, max_points, 2)
        valid_mask = torch.zeros(max_polygons, dtype=torch.bool)

        poly_idx = 0

        try:
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
                            indices = np.linspace(
                                0, len(points) - 1, max_points, dtype=int
                            )
                            points = points[indices]

                        # Check for NaN/Inf in points
                        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                            print(
                                f"Warning: Invalid polygon points detected, skipping polygon"
                            )
                            continue

                        # Normalize to [0,1] relative to image size
                        points = points / np.array(self.image_size)
                        # Clamp to valid range
                        points = np.clip(points, 0.0, 1.0)

                        processed[poly_idx, : len(points)] = torch.from_numpy(
                            points
                        ).float()
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

                    # Check for NaN/Inf in points
                    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                        print(
                            f"Warning: Invalid polygon points detected, skipping polygon"
                        )
                        continue

                    points = points / np.array(self.image_size)
                    points = np.clip(points, 0.0, 1.0)

                    processed[poly_idx, : len(points)] = torch.from_numpy(
                        points
                    ).float()
                    valid_mask[poly_idx] = True
                    poly_idx += 1

        except Exception as e:
            print(f"Warning: Error processing polygons: {e}")
            # Return safe empty polygon data
            processed = torch.zeros(max_polygons, max_points, 2)
            valid_mask = torch.zeros(max_polygons, dtype=torch.bool)

        return {"polygons": processed, "valid_mask": valid_mask}

    # ----------------------------------------------------------------------
    def _augment(self, image, mask):
        """Enhanced data augmentation with rotations, flips, and intensity changes."""
        # Random rotation (multiples of 90° only for architectural data)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        # Slight brightness/contrast adjustment with safety checks
        if torch.rand(1) < 0.3:
            brightness = torch.rand(1) * 0.2 - 0.1  # ±0.1
            contrast = torch.rand(1) * 0.2 + 0.9  # 0.9-1.1
            image = torch.clamp(image * contrast + brightness, 0, 1)

            # Additional safety check for augmented image
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(
                    "Warning: Augmentation produced invalid values, reverting to original"
                )
                # Revert to safe values
                image = torch.clamp(image, 0, 1)
                image = torch.where(
                    torch.isnan(image) | torch.isinf(image),
                    torch.zeros_like(image),
                    image,
                )

        return image, mask

    # ----------------------------------------------------------------------
    def get_cache_info(self):
        """Return information about caching status"""
        return {
            "cache_enabled": self.cache_in_memory,
            "cache_loaded": self._cache is not None,
            "cached_samples": len(self._cache) if self._cache else 0,
            "total_samples": len(self.samples),
            "estimated_memory_mb": (
                self._estimate_memory_usage() if self.cache_in_memory else 0
            ),
        }

    def disable_cache(self):
        """Disable caching and free memory"""
        if self._cache is not None:
            print(
                f"[DATA] Disabling cache and freeing memory for {len(self._cache)} samples"
            )
            self._cache = None
            self.cache_in_memory = False

    def enable_cache(self):
        """Enable caching if not already enabled"""
        if not self.cache_in_memory and self.samples:
            self.cache_in_memory = True
            print("[DATA] Enabling cache...")
            self._preload_cache()


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
        "wall_height": 1.0,
        "wall_thickness": 0.15,
        "window_base_height": 0.269,
        "window_height": 0.365,
        "door_height": 1.0,
        "pixel_scale": 0.01,
    }

    # Simple voxel GT
    voxels = np.zeros((64, 64, 64), dtype=bool)
    voxels[:20, 10:50, 10:50] = True

    # Polygon GT
    polygons = {"walls": [{"points": room_points.tolist()}]}

    return image, mask, attributes, voxels, polygons


class SyntheticFloorPlanDataset(Dataset):
    """
    Synthetic dataset for testing and development when real data is not available
    """

    def __init__(self, num_samples=1000, image_size=(256, 256), voxel_size=64):
        self.num_samples = num_samples
        self.image_size = image_size
        self.voxel_size = voxel_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate deterministic synthetic data based on index
        np.random.seed(idx)
        torch.manual_seed(idx)

        image, mask, attributes, voxels, polygons_gt = create_synthetic_data_sample()

        # Convert to tensors
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        mask_tensor = torch.from_numpy(mask).long()
        voxels_tensor = torch.from_numpy(voxels.astype(np.float32))

        # Process attributes and polygons using same methods as main dataset
        dataset = AdvancedFloorPlanDataset.__new__(AdvancedFloorPlanDataset)
        dataset.image_size = self.image_size

        attr_tensor = dataset._process_attributes(attributes)
        polygon_tensor = dataset._process_polygons(polygons_gt)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "attributes": attr_tensor,
            "voxels_gt": voxels_tensor,
            "polygons_gt": polygon_tensor,
            "sample_id": f"synthetic_{idx:06d}",
        }   