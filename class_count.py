import cv2, numpy as np, glob
from collections import Counter

all_classes = set()
class_counts = Counter()

# Go through all mask images
for mask_file in glob.glob("./data/floorplans/train/*/mask.png"):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[u] += c
    all_classes.update(unique)

# Total pixels
total_pixels = sum(class_counts.values())

print("Classes found in dataset:", sorted(all_classes))
print("\nPixel distribution per class:")
for cls in sorted(class_counts.keys()):
    percentage = (class_counts[cls] / total_pixels) * 100
    print(f"Class {cls}: {class_counts[cls]} pixels ({percentage:.2f}%)")
