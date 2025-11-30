"""Training data export utilities."""

from pathlib import Path
from typing import List, Tuple
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from data.ade20k_labels import ADE20K_LABELS


def generate_multiclass_mask(
    seg_image: np.ndarray,
    road_mapping: dict,
    vehicle_mask: np.ndarray = None
) -> np.ndarray:
    """
    Generate 3-class mask from ADE20K segmentation and vehicle mask.

    Args:
        seg_image: ADE20K segmentation (H, W) - each pixel is class ID (0-150, 255)
        road_mapping: {label_name: is_road} dictionary
        vehicle_mask: Vehicle mask (H, W) - 255 where vehicle is present, 0 otherwise

    Returns:
        Multiclass mask (H, W) with values:
            0: Other (non-ROAD, non-MYCAR)
            1: ROAD (drivable area)
            2: MYCAR (vehicle body)
    """
    mask = np.zeros_like(seg_image, dtype=np.uint8)

    # Create reverse mapping: class_id -> label_name
    id_to_name = {class_id: label_name for class_id, label_name in ADE20K_LABELS.items()}

    # Mark ROAD pixels (class 1)
    for class_id in np.unique(seg_image):
        # Skip mycar if present in segmentation
        if class_id == 255:
            continue

        label_name = id_to_name.get(int(class_id))
        if label_name and road_mapping.get(label_name, False):
            mask[seg_image == class_id] = 1

    # Mark MYCAR pixels (class 2) using vehicle mask
    if vehicle_mask is not None:
        mycar_pixels = vehicle_mask > 127
        mask[mycar_pixels] = 2
    # Also check for mycar in segmentation (ID=255)
    elif 255 in seg_image:
        mask[seg_image == 255] = 2

    return mask


def export_training_data(
    image_files: List[Path],
    road_mapping: dict,
    output_dir: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    vehicle_mask_path: Path = None
) -> Tuple[int, int]:
    """
    Export training dataset with train/val split.

    Args:
        image_files: List of image file paths
        road_mapping: {label_name: is_road} dictionary
        output_dir: Output directory for training data
        train_ratio: Ratio of training data (default: 0.8)
        random_seed: Random seed for reproducibility
        vehicle_mask_path: Path to vehicle mask PNG (optional)

    Returns:
        (num_train, num_val) tuple
    """
    # Load vehicle mask if provided
    vehicle_mask = None
    if vehicle_mask_path and vehicle_mask_path.exists():
        vehicle_mask = np.array(Image.open(vehicle_mask_path))
        print(f"✓ Loaded vehicle mask from {vehicle_mask_path}")

    # Create output directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(random_seed)
    indices = list(range(len(image_files)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = set(indices[:split_idx])
    val_indices = set(indices[split_idx:])

    num_train = 0
    num_val = 0
    errors = []

    # Process each image
    for idx, image_path in enumerate(tqdm(image_files, desc="Exporting")):
        try:
            # Find corresponding segmentation file
            seg_path = image_path.parent / f"{image_path.stem}_seg.png"
            if not seg_path.exists():
                errors.append(f"Segmentation not found: {seg_path.name}")
                continue

            # Load segmentation
            seg_img = Image.open(seg_path)
            seg_array = np.array(seg_img)

            # Generate 3-class mask (0: Other, 1: ROAD, 2: MYCAR)
            multiclass_mask = generate_multiclass_mask(seg_array, road_mapping, vehicle_mask)

            # Determine train or val
            is_train = idx in train_indices
            img_dir = train_img_dir if is_train else val_img_dir
            lbl_dir = train_lbl_dir if is_train else val_lbl_dir

            # Copy original image
            shutil.copy(image_path, img_dir / image_path.name)

            # Copy segmentation image if exists
            if seg_path.exists():
                shutil.copy(seg_path, img_dir / f"{image_path.stem}_seg.png")

            # Save multiclass mask (values: 0, 1, 2)
            mask_img = Image.fromarray(multiclass_mask)
            mask_img.save(lbl_dir / f"{image_path.stem}.png")

            if is_train:
                num_train += 1
            else:
                num_val += 1

        except Exception as e:
            errors.append(f"{image_path.name}: {str(e)}")

    # Report errors
    if errors:
        print("\nWarnings/Errors during export:")
        for err in errors:
            print(f"  - {err}")

    return num_train, num_val
