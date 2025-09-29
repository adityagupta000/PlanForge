# check_config.py
from models.encoder import MultiScaleEncoder
from models.dvx import DifferentiableVectorization
from models.extrusion import DifferentiableExtrusion
from models.heads import SegmentationHead
from config import DEFAULT_MODEL_CONFIG

print("Configuration Check:")
print("=" * 60)

# Test encoder
enc = MultiScaleEncoder(3, DEFAULT_MODEL_CONFIG.feature_dim)
print(f"Encoder feature_dim: {enc.feature_dim if hasattr(enc, 'feature_dim') else 'check __init__'}")

# Test DVX
dvx = DifferentiableVectorization()
print(f"DVX max_polygons: {dvx.max_polygons} (expect 30)")
print(f"DVX max_points: {dvx.max_points} (expect 64)")
print(f"DVX feature_dim: {dvx.feature_dim} (expect 384)")

# Test extrusion
ext = DifferentiableExtrusion()
print(f"Extrusion voxel_size: {ext.voxel_size} (expect 96)")

# Test heads
seg_head = SegmentationHead(DEFAULT_MODEL_CONFIG.feature_dim)
print(f"SegHead accepts feature_dim: {DEFAULT_MODEL_CONFIG.feature_dim} (expect 768)")

print("=" * 60)