from PIL import Image
import os
import shutil

data_dir = r"data/floorplans"

def safe_clean_image(path):
    """Safely clean image by only removing ICC profiles, preserving all pixel data"""
    try:
        # Create backup first (optional safety measure)
        backup_path = path + ".backup"
        
        with Image.open(path) as img:
            # Check if image is already clean
            if 'icc_profile' not in img.info:
                print(f"Already clean: {path}")
                return True
            
            # Create backup
            shutil.copy2(path, backup_path)
            
            # Method 1: Just strip ICC profile while preserving everything else
            img_data = img.copy()
            
            # Remove only the problematic ICC profile
            if 'icc_profile' in img_data.info:
                del img_data.info['icc_profile']
            
            # Save with same format and quality, just without ICC profile
            img_data.save(path, format="PNG", optimize=False)  # No optimization to preserve exact pixels
            
            # Remove backup if successful
            os.remove(backup_path)
            
        print(f"Cleaned ICC profile from: {path}")
        return True
        
    except Exception as e:
        # Restore backup if it exists
        backup_path = path + ".backup"
        if os.path.exists(backup_path):
            shutil.move(backup_path, path)
            print(f"Restored backup for: {path}")
        
        print(f"Failed {path}: {e}")
        return False

def verify_image_integrity(path):
    """Verify image can still be loaded properly after cleaning"""
    try:
        with Image.open(path) as img:
            # Try to access pixel data to ensure image is valid
            _ = img.size
            _ = img.mode
            # Try to load a small sample of pixel data
            _ = img.getpixel((0, 0))
        return True
    except Exception as e:
        print(f"WARNING: Image integrity check failed for {path}: {e}")
        return False

# Process only image.png files
processed_files = []
cleaned_count = 0
failed_count = 0
already_clean = 0

print("Starting safe ICC profile removal for dataset...")
print("This preserves all pixel data and only removes problematic metadata.")

for root, _, files in os.walk(data_dir):
    for f in files:
        if f == "image.png":  # Only process image.png files
            path = os.path.join(root, f)
            processed_files.append(path)
            
            # Check if already clean
            try:
                with Image.open(path) as img:
                    if 'icc_profile' not in img.info:
                        already_clean += 1
                        continue
            except:
                pass
            
            if safe_clean_image(path):
                # Verify integrity after cleaning
                if verify_image_integrity(path):
                    cleaned_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1

print(f"\n" + "="*50)
print(f"DATASET CLEANING SUMMARY")
print(f"="*50)
print(f"Total image.png files found: {len(processed_files)}")
print(f"Already clean (no ICC profile): {already_clean}")
print(f"Successfully cleaned: {cleaned_count}")
print(f"Failed to clean: {failed_count}")
print(f"Total files processed: {already_clean + cleaned_count + failed_count}")

if failed_count > 0:
    print(f"\nWARNING: {failed_count} files couldn't be cleaned.")
    print(f"Check these files manually - they may be corrupted.")

print(f"\nDataset should now be ready for training without libpng warnings!")

# Optional: Test load a few random images to verify dataset integrity
print(f"\nTesting random samples for integrity...")
import random
test_files = random.sample(processed_files, min(5, len(processed_files)))
for test_path in test_files:
    if verify_image_integrity(test_path):
        print(f"✓ {test_path}")
    else:
        print(f"✗ {test_path} - POTENTIAL ISSUE")