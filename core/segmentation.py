"""Segmentation image handling and demo data generation."""

import numpy as np
from PIL import Image
from pathlib import Path
from data.ade20k_labels import get_label_color


class SegmentationImage:
    """Represents a segmentation image with label data."""

    def __init__(self, image_path: str, seg_path: str = None):
        """
        Initialize segmentation image.

        Args:
            image_path: Path to original image
            seg_path: Path to segmentation label image (optional, will generate demo if None)
        """
        self.image_path = Path(image_path)
        self.seg_path = Path(seg_path) if seg_path else None
        self.image = None
        self.seg_labels = None  # Numpy array of label IDs
        self.overlay = None  # Colored overlay image

    def load(self):
        """Load image and segmentation data."""
        if self.image_path.exists():
            self.image = Image.open(self.image_path).convert("RGB")
        else:
            # Generate demo image
            self.image = self._generate_demo_image()

        if self.seg_path and self.seg_path.exists():
            seg_img = Image.open(self.seg_path)
            self.seg_labels = np.array(seg_img)
        else:
            # Generate demo segmentation
            self.seg_labels = self._generate_demo_segmentation()

        self._create_overlay()

    def _generate_demo_image(self, width=640, height=480):
        """Generate a demo base image."""
        # Create a simple gradient background
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(height // 3):
            color = int(135 + (y / (height // 3)) * 100)
            img_array[y, :] = [color, color, 255]

        # Road/ground
        for y in range(height // 3, height):
            color = int(100 - ((y - height // 3) / (2 * height // 3)) * 30)
            img_array[y, :] = [color, color, color]

        return Image.fromarray(img_array)

    def _generate_demo_segmentation(self):
        """Generate demo segmentation labels."""
        if self.image is None:
            raise ValueError("Image must be loaded first")

        width, height = self.image.size
        seg = np.zeros((height, width), dtype=np.uint8)

        # Sky (top third)
        seg[0:height//3, :] = 3  # sky

        # Road (bottom half)
        seg[height//2:, width//4:3*width//4] = 7  # road

        # Sidewalks
        seg[height//2:, :width//4] = 12  # sidewalk left
        seg[height//2:, 3*width//4:] = 12  # sidewalk right

        # Grass on sides
        seg[height//3:height//2, :width//5] = 10  # grass left
        seg[height//3:height//2, 4*width//5:] = 10  # grass right

        # Some trees
        tree_positions = [(width//6, height//3), (5*width//6, height//3)]
        for x, y in tree_positions:
            # Simple circular tree
            for dy in range(-20, 40):
                for dx in range(-15, 15):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if dx*dx + (dy+10)*(dy+10) < 225:  # Circle
                            seg[ny, nx] = 5  # tree

        # Building in background
        seg[height//3:height//2, 2*width//5:3*width//5] = 2  # building

        # Path
        seg[2*height//5:height//2, 2*width//5:3*width//5] = 53  # path

        return seg

    def _create_overlay(self):
        """Create colored overlay from segmentation labels."""
        if self.seg_labels is None or self.image is None:
            raise ValueError("Image and segmentation must be loaded first")

        height, width = self.seg_labels.shape
        overlay_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Color each pixel according to its label
        for label_id in np.unique(self.seg_labels):
            mask = self.seg_labels == label_id
            color = get_label_color(int(label_id))
            overlay_array[mask] = color

        self.overlay = Image.fromarray(overlay_array)

    def get_blended_image(self, alpha=0.5):
        """
        Get blended image with segmentation overlay.

        Args:
            alpha: Overlay transparency (0=invisible, 1=opaque)

        Returns:
            PIL Image with overlay blended
        """
        if self.image is None or self.overlay is None:
            raise ValueError("Image and overlay must be loaded first")

        return Image.blend(self.image, self.overlay, alpha)

    def get_label_at_position(self, x: int, y: int) -> int:
        """
        Get label ID at pixel position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Label ID at position
        """
        if self.seg_labels is None:
            raise ValueError("Segmentation must be loaded first")

        height, width = self.seg_labels.shape
        if 0 <= y < height and 0 <= x < width:
            return int(self.seg_labels[y, x])
        return 0  # unlabeled

    def apply_road_overlay(self, road_labels: set, stripe_alpha=0.3):
        """
        Apply stripe pattern overlay to ROAD regions.

        Args:
            road_labels: Set of label IDs marked as ROAD
            stripe_alpha: Transparency of stripe pattern

        Returns:
            PIL Image with ROAD stripes
        """
        if self.overlay is None:
            raise ValueError("Overlay must be created first")

        # Create a copy of the overlay
        result = self.overlay.copy()
        result_array = np.array(result)

        # Create stripe pattern
        height, width = self.seg_labels.shape
        stripe_pattern = np.zeros((height, width), dtype=bool)

        # Diagonal stripes
        for i in range(height):
            for j in range(width):
                if (i + j) % 20 < 10:  # Stripe width of 10 pixels
                    stripe_pattern[i, j] = True

        # Apply stripes to ROAD regions
        for label_id in road_labels:
            mask = self.seg_labels == label_id
            road_stripe_mask = mask & stripe_pattern
            # Make stripes yellow
            result_array[road_stripe_mask] = [255, 255, 0]

        return Image.fromarray(result_array)


def generate_demo_images(output_dir: Path, num_images: int = 3):
    """
    Generate demo images for testing.

    Args:
        output_dir: Directory to save demo images
        num_images: Number of demo images to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create segmentation image
        seg_img = SegmentationImage(
            image_path=output_dir / f"demo_{i}.jpg",
            seg_path=None
        )
        seg_img.load()

        # Save base image
        seg_img.image.save(output_dir / f"demo_{i}.jpg")

        # Save segmentation
        seg_img.overlay.save(output_dir / f"demo_{i}_seg.png")
