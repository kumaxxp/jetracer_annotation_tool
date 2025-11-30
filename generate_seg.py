#!/usr/bin/env python3
"""Generate segmentation images for all images in a directory."""

from pathlib import Path
from core.segmentation import SegmentationImage
from tqdm import tqdm

# Image directory
image_dir = Path("/home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358")

# Find all images (excluding segmentation files)
all_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
image_files = sorted([
    img for img in all_images
    if not img.name.endswith('_seg.png')
])

print(f"Found {len(image_files)} images")
print("Generating segmentation images...")

for image_path in tqdm(image_files):
    # Check if segmentation already exists
    seg_path = image_path.parent / f"{image_path.stem}_seg.png"

    if seg_path.exists():
        print(f"  ✓ Skipping {image_path.name} (segmentation exists)")
        continue

    # Generate segmentation
    print(f"  Generating segmentation for {image_path.name}...")
    seg_image = SegmentationImage(
        image_path=str(image_path),
        seg_path=None  # Force generation
    )
    seg_image.load()
    print(f"  ✓ Generated {seg_path.name}")

print("\nDone!")
