import cv2
import numpy as np
from pathlib import Path
import json

# Base dataset path
data_root = Path("./data/floorplans")

def is_valid_mask(mask_file):
    m = cv2.imread(str(mask_file), 0)
    return m is not None and np.sum(m) > 0

def is_valid_voxel(voxel_file):
    try:
        data = np.load(str(voxel_file))
        key = "voxels"  # Use the correct key in your npz files
        if key not in data.files:
            print(f"⚠️ Key '{key}' not found in {voxel_file}, available keys: {data.files}")
            return False
        v = data[key]
        return not np.isnan(v).any()
    except Exception as e:
        print(f"⚠️ Error reading {voxel_file}: {e}")
        return False

def is_valid_polygon(polygon_file):
    try:
        with open(polygon_file, "r") as f:
            json.load(f)
        return True
    except Exception as e:
        print(f"⚠️ Invalid polygon {polygon_file}: {e}")
        return False

def check_split(split="train"):
    split_dir = data_root / split
    mask_files = list(split_dir.rglob("mask.png"))
    voxel_files = list(split_dir.rglob("voxel_GT.npz"))
    polygon_files = list(split_dir.rglob("polygon.json"))

    print(f"\n🔍 Checking split: {split}")

    # Check mask files
    total_masks = len(mask_files)
    bad_masks = 0
    for f in mask_files:
        if not is_valid_mask(f):
            bad_masks += 1
            print(f"⚠️ Invalid mask: {f}")

    # Check voxel files
    total_voxels = len(voxel_files)
    bad_voxels = 0
    for f in voxel_files:
        if not is_valid_voxel(f):
            bad_voxels += 1
            print(f"⚠️ Invalid voxel: {f}")

    # Check polygon files
    total_polygons = len(polygon_files)
    bad_polygons = 0
    for f in polygon_files:
        if not is_valid_polygon(f):
            bad_polygons += 1

    # Summary
    print(f"\n✅ Summary for split: {split}")
    print(f"Total mask files checked: {total_masks}, Invalid: {bad_masks}")
    print(f"Total voxel files checked: {total_voxels}, Invalid: {bad_voxels}")
    print(f"Total polygon files checked: {total_polygons}, Invalid: {bad_polygons}")

def main():
    for split in ["train", "val", "test"]:
        check_split(split)

if __name__ == "__main__":
    main()
