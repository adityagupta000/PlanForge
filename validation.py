"""
Dataset Validation Script - STRICT CLASS 0-4 VALIDATION
Run this BEFORE training to verify dataset only contains classes 0-4
Usage: python validation.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import sys

# CRITICAL: Only these classes are valid (class 5 cube must be excluded)
VALID_CLASSES = {0, 1, 2, 3, 4}
INVALID_CLASSES = {5}  # Class 5 (cube) should never appear

def validate_sample(sample_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a single sample directory
    Returns: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required files exist
    required_files = ["image.png", "mask.png", "params.json", "polygon.json", "voxel_GT.npz", "model.obj"]
    for fname in required_files:
        if not (sample_dir / fname).exists():
            errors.append(f"Missing file: {fname}")
    
    if errors:
        return False, errors, warnings
    
    # ========================================
    # CRITICAL: Validate mask classes (STRICT)
    # ========================================
    try:
        mask = np.array(Image.open(sample_dir / "mask.png"))
        unique_classes = set(np.unique(mask).tolist())
        
        # Check for class 5 (cube) - THIS IS A CRITICAL ERROR
        if 5 in unique_classes:
            errors.append(f"CRITICAL: Class 5 (cube) found in mask! This must be filtered during dataset generation.")
            errors.append(f"  Found classes: {sorted(unique_classes)}")
            errors.append(f"  Class 5 pixel count: {np.sum(mask == 5)}")
        
        # Check for any invalid classes (> 4)
        invalid = unique_classes - VALID_CLASSES
        if invalid:
            errors.append(f"CRITICAL: Invalid classes found: {invalid} (only 0-4 allowed)")
            errors.append(f"  All classes in mask: {sorted(unique_classes)}")
        
        # Check if mask is completely empty
        if unique_classes == {0}:
            warnings.append("Mask contains only background (class 0) - no architectural elements")
        
        # Validate each class has reasonable pixel count
        total_pixels = mask.size
        for cls in range(5):
            if cls in unique_classes:
                pixel_count = np.sum(mask == cls)
                percentage = (pixel_count / total_pixels) * 100
                if cls > 0 and pixel_count < 100:  # Non-background classes
                    warnings.append(f"Class {cls} has very few pixels ({pixel_count}, {percentage:.2f}%)")
        
    except Exception as e:
        errors.append(f"Failed to load/validate mask: {e}")
    
    # ========================================
    # Validate params.json
    # ========================================
    try:
        with open(sample_dir / "params.json") as f:
            params = json.load(f)
        
        # Check class mapping
        if "class_mapping" in params:
            class_map = params["class_mapping"]
            expected_map = {
                "background": 0,
                "wall": 1,
                "door": 2,
                "window": 3,
                "floor": 4
            }
            if class_map != expected_map:
                errors.append(f"class_mapping mismatch: {class_map} != {expected_map}")
        
        # Check for excluded_classes field
        if "excluded_classes" in params:
            excluded = params["excluded_classes"]
            if 5 not in excluded:
                warnings.append(f"Class 5 not in excluded_classes list: {excluded}")
        
        # Validate architectural parameters
        expected_params = {
            "wall_height": (2.0, 3.5, 2.6),      # (min, max, expected)
            "wall_thickness": (0.10, 0.30, 0.15),
            "window_base_height": (0.3, 1.5, 0.7),
            "window_height": (0.5, 1.5, 0.95),
            "door_height": (2.0, 3.5, 2.6),
            "pixel_scale": (0.005, 0.02, 0.01)
        }
        
        for param_name, (min_val, max_val, expected_val) in expected_params.items():
            if param_name in params:
                value = params[param_name]
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"{param_name}={value} outside valid range [{min_val}, {max_val}] "
                        f"(expected ~{expected_val})"
                    )
                elif abs(value - expected_val) > expected_val * 0.3:  # 30% tolerance
                    warnings.append(
                        f"{param_name}={value} differs significantly from expected {expected_val}"
                    )
            else:
                errors.append(f"Missing parameter: {param_name}")
                
    except Exception as e:
        errors.append(f"Failed to load params.json: {e}")
    
    # ========================================
    # Validate voxel dimensions
    # ========================================
    try:
        voxel_data = np.load(sample_dir / "voxel_GT.npz")
        voxels = voxel_data["voxels"]
        
        if voxels.shape != (64, 64, 64):
            errors.append(f"Voxel shape {voxels.shape} != (64, 64, 64)")
        
        # Check if voxels are completely empty
        if voxels.sum() == 0:
            warnings.append("Voxel grid is completely empty")
        
        # Check voxel height distribution (should be ~33 voxels high for walls with 2.6m height)
        occupied_z = np.any(voxels > 0, axis=(1, 2))
        max_occupied_z = np.where(occupied_z)[0].max() if occupied_z.any() else 0
        
        expected_height_voxels = int(round((2.6 / 5.0) * 64))  # ~33
        if abs(max_occupied_z - expected_height_voxels) > 10:
            warnings.append(
                f"Voxel height ({max_occupied_z}) differs from expected (~{expected_height_voxels})"
            )
        
    except Exception as e:
        errors.append(f"Failed to load voxels: {e}")
    
    # ========================================
    # Validate polygon format
    # ========================================
    try:
        with open(sample_dir / "polygon.json") as f:
            polygons = json.load(f)
        
        expected_keys = ["walls", "doors", "windows", "floors"]
        for key in expected_keys:
            if key not in polygons:
                errors.append(f"Missing polygon key: {key}")
        
        # Check if polygons are empty
        total_polys = sum(len(polygons.get(k, [])) for k in expected_keys)
        if total_polys == 0:
            warnings.append("No polygons found in any class")
        
    except Exception as e:
        errors.append(f"Failed to load polygons: {e}")
    
    return len(errors) == 0, errors, warnings


def validate_dataset(data_dir: Path) -> Dict:
    """Validate entire dataset"""
    results = {"train": [], "val": [], "test": []}
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            continue
        
        sample_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"\nValidating {split} split: {len(sample_dirs)} samples")
        
        for sample_dir in sample_dirs:
            valid, errors, warnings = validate_sample(sample_dir)
            results[split].append({
                "sample": sample_dir.name,
                "valid": valid,
                "errors": errors,
                "warnings": warnings
            })
    
    return results


def print_validation_report(results: Dict):
    """Print detailed validation report"""
    print("\n" + "="*80)
    print("DATASET VALIDATION REPORT - CLASS 0-4 STRICT VALIDATION")
    print("="*80)
    
    total_valid = 0
    total_samples = 0
    critical_issues = []
    
    for split, samples in results.items():
        if not samples:
            continue
            
        valid_count = sum(1 for s in samples if s["valid"])
        total_count = len(samples)
        total_valid += valid_count
        total_samples += total_count
        
        print(f"\n{split.upper()} SPLIT:")
        print(f"  Valid: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
        
        # Collect samples with class 5
        class_5_samples = [
            s for s in samples 
            if any("Class 5" in err or "class 5" in err for err in s["errors"])
        ]
        if class_5_samples:
            print(f"  ❌ CRITICAL: {len(class_5_samples)} samples contain CLASS 5 (cube)!")
            critical_issues.extend(class_5_samples)
            for s in class_5_samples[:3]:
                print(f"    - {s['sample']}")
            if len(class_5_samples) > 3:
                print(f"    ... and {len(class_5_samples)-3} more")
        
        # Show other errors
        other_errors = [s for s in samples if not s["valid"] and s not in class_5_samples]
        if other_errors:
            print(f"  Other errors: {len(other_errors)} samples")
            for err_sample in other_errors[:3]:
                print(f"    - {err_sample['sample']}:")
                for error in err_sample['errors'][:2]:
                    print(f"        {error}")
            if len(other_errors) > 3:
                print(f"    ... and {len(other_errors)-3} more")
        
        # Show warnings summary
        total_warnings = sum(len(s["warnings"]) for s in samples)
        if total_warnings > 0:
            print(f"  ⚠️  Warnings: {total_warnings} total across all samples")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {total_valid}/{total_samples} valid samples ({100*total_valid/total_samples:.1f}%)")
    print(f"{'='*80}\n")
    
    if critical_issues:
        print("❌ CRITICAL FAILURE: CLASS 5 (CUBE) DETECTED IN DATASET")
        print(f"   {len(critical_issues)} samples contain class 5")
        print("   Dataset generation DID NOT properly filter class 5")
        print("   ACTION REQUIRED:")
        print("   1. Check Dataset_code/config.py has: exclude_classes: tuple = (5,)")
        print("   2. Check Dataset_code/processor.py has filter_class_5() method")
        print("   3. Regenerate entire dataset with corrected code")
        print("   4. DO NOT proceed to training until this is fixed")
        return False
    
    if total_valid == total_samples:
        print("✅ ALL SAMPLES VALID - Dataset ready for training!")
        print("   ✓ Only classes 0-4 present")
        print("   ✓ Class 5 (cube) successfully filtered")
        print("   ✓ Architectural parameters in valid ranges")
        return True
    else:
        print("❌ VALIDATION FAILED - Fix errors before training!")
        return False


if __name__ == "__main__":
    # Check if data directory exists
    data_dir = Path("data/floorplans")
    if not data_dir.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        print("Please run the dataset generation pipeline first.")
        sys.exit(1)
    
    print("="*80)
    print("DATASET VALIDATION - CLASS 5 (CUBE) EXCLUSION CHECK")
    print("="*80)
    print(f"Dataset directory: {data_dir}")
    print(f"Valid classes: {sorted(VALID_CLASSES)}")
    print(f"Invalid classes (must be excluded): {sorted(INVALID_CLASSES)}")
    print()
    
    # Run validation
    results = validate_dataset(data_dir)
    
    # Print report
    success = print_validation_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)