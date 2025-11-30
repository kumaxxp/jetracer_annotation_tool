#!/usr/bin/env python3
"""Generate ADE20K segmentation images using OneFormer."""

from pathlib import Path
from core.ade20k_segmentation import ADE20KSegmenter

# Image directory
image_dir = Path("/home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358")

# Find all images (excluding segmentation files)
all_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
image_files = sorted([
    img for img in all_images
    if not img.name.endswith('_seg.png')
])

print(f"Found {len(image_files)} images")
print("Loading ADE20K OneFormer model...")

# Create segmenter
segmenter = ADE20KSegmenter()

# Generate segmentation and save to the same directory
print(f"\nGenerating segmentation images...")
output_paths = segmenter.segment_batch(
    image_paths=image_files,
    output_dir=image_dir,
    verbose=True
)

print(f"\n✓ Generated {len(output_paths)} segmentation images in {image_dir}")
