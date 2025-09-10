import json
import numpy as np
from pathlib import Path

# Adjust this to your dataset path
data_root = Path("./data/floorplans")  

# Expected attributes with their default values
expected_keys = {
    "wall_height": 2.6,
    "wall_thickness": 0.15,
    "window_base_height": 0.7,
    "window_height": 0.95,
    "door_height": 2.6,
    "pixel_scale": 0.01
}

def is_valid_number(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return np.isfinite(value)
    return False

def check_params_file(params_file):
    invalid_entries = []
    try:
        with open(params_file, "r") as f:
            params = json.load(f)
    except Exception as e:
        invalid_entries.append(f"Could not load JSON: {e}")
        return invalid_entries

    for key in expected_keys.keys():
        val = params.get(key)
        if val is None:
            invalid_entries.append(f"missing '{key}'")
        elif not is_valid_number(val):
            invalid_entries.append(f"{key}={val} (invalid)")
    return invalid_entries

def check_split(split="train"):
    split_dir = data_root / split
    total_files = 0
    good_files = 0
    bad_files = 0

    print(f"\nChecking split: {split}")
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return

    # Recursively find all params.json files
    for params_file in split_dir.rglob("params.json"):
        total_files += 1
        invalid_entries = check_params_file(params_file)

        if invalid_entries:
            print(f"[BAD] {params_file}")
            for entry in invalid_entries:
                print(f"    - {entry}")
            bad_files += 1
        else:
            good_files += 1

    print(f"\nSummary for split: {split}")
    print(f"Total files checked: {total_files}")
    print(f"Good files: {good_files}")
    print(f"Bad files: {bad_files}")

def main():
    for split in ["train", "val", "test"]:
        check_split(split)

if __name__ == "__main__":
    main()
