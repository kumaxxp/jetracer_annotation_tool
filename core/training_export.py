"""Training data export utilities."""

from pathlib import Path
from typing import List, Tuple
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from data.ade20k_labels import ADE20K_LABELS


def generate_binary_mask(seg_image: np.ndarray, road_mapping: dict) -> np.ndarray:
    """
    Generate binary ROAD mask from ADE20K segmentation and mapping.

    Args:
        seg_image: ADE20K segmentation (H, W) - each pixel is class ID (0-150)
        road_mapping: {label_name: is_road} dictionary

    Returns:
        Binary mask (H, W) - 0: non-ROAD, 255: ROAD
    """
    mask = np.zeros_like(seg_image, dtype=np.uint8)

    # Create reverse mapping: class_id -> label_name
    id_to_name = {class_id: label_name for class_id, label_name in ADE20K_LABELS.items()}

    # Mark ROAD pixels
    for class_id in np.unique(seg_image):
        label_name = id_to_name.get(int(class_id))
        if label_name and road_mapping.get(label_name, False):
            mask[seg_image == class_id] = 255

    return mask


def export_training_data(
    image_files: List[Path],
    road_mapping: dict,
    output_dir: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[int, int]:
    """
    Export training dataset with train/val split.

    Args:
        image_files: List of image file paths
        road_mapping: {label_name: is_road} dictionary
        output_dir: Output directory for training data
        train_ratio: Ratio of training data (default: 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        (num_train, num_val) tuple
    """
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

            # Generate binary mask
            binary_mask = generate_binary_mask(seg_array, road_mapping)

            # Determine train or val
            is_train = idx in train_indices
            img_dir = train_img_dir if is_train else val_img_dir
            lbl_dir = train_lbl_dir if is_train else val_lbl_dir

            # Copy original image
            shutil.copy(image_path, img_dir / image_path.name)

            # Save binary mask
            mask_img = Image.fromarray(binary_mask)
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
