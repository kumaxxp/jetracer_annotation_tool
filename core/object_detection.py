"""Object detection module using YOLOv8 for obstacle detection."""

import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed. Object detection disabled.")


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center x, y
    area: int  # bbox area in pixels
    danger_level: str  # 'danger', 'caution', 'safe'


class ObjectDetector:
    """YOLOv8-based object detector for obstacle detection."""

    # Classes that are relevant for obstacle detection
    OBSTACLE_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        15: 'cat',
        16: 'dog',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        60: 'dining table',
        62: 'tv',
        63: 'laptop',
        67: 'cell phone',
    }

    # High priority obstacles (should always avoid)
    HIGH_PRIORITY = {0, 15, 16}  # person, cat, dog

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        use_cuda: bool = True,
        vehicle_mask_path: str = "output/vehicle_mask.png"
    ):
        """
        Initialize object detector.

        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            use_cuda: Use GPU acceleration if available
            vehicle_mask_path: Path to vehicle mask (white = own vehicle)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.use_cuda = use_cuda
        self.model: Optional[YOLO] = None
        self.device = 'cuda:0' if use_cuda else 'cpu'

        # Load vehicle mask for excluding own vehicle
        self.vehicle_mask: Optional[np.ndarray] = None
        self._load_vehicle_mask(vehicle_mask_path)

    def _load_vehicle_mask(self, mask_path: str):
        """Load vehicle mask to exclude own vehicle from detections."""
        mask_file = Path(mask_path)
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Threshold to binary (white = vehicle)
                _, self.vehicle_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                print(f"✓ Vehicle mask loaded: {mask_file}")
            else:
                print(f"⚠ Could not read vehicle mask: {mask_file}")
        else:
            print(f"⚠ Vehicle mask not found: {mask_file}")

    def load_model(self) -> bool:
        """Load YOLO model."""
        if not HAS_YOLO:
            print("ultralytics not available")
            return False

        try:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(str(self.model_path))

            # Warm-up inference
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)

            print(f"✓ YOLO model loaded (device: {self.device})")
            return True

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False

    def _is_in_vehicle_mask(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> bool:
        """
        Check if detection bbox is mostly inside vehicle mask.

        Args:
            bbox: (x1, y1, x2, y2)
            image_shape: (height, width)

        Returns:
            True if detection should be excluded (in vehicle mask)
        """
        if self.vehicle_mask is None:
            return False

        x1, y1, x2, y2 = bbox
        h, w = image_shape

        # Resize mask to match image if needed
        mask = self.vehicle_mask
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Get mask region for bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # If more than 50% of bbox is in vehicle mask, exclude it
        vehicle_ratio = np.sum(roi > 127) / roi.size
        return vehicle_ratio > 0.5

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run object detection on image.

        Args:
            image: BGR image from cv2

        Returns:
            Tuple of (list of Detection objects, inference time in ms)
        """
        if self.model is None:
            return [], 0.0

        start_time = time.perf_counter()

        # Run inference
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        inference_time = (time.perf_counter() - start_time) * 1000

        # Parse results
        detections = []
        h, w = image.shape[:2]

        for box in results.boxes:
            class_id = int(box.cls[0])

            # Skip if not an obstacle class
            if class_id not in self.OBSTACLE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Skip if in vehicle mask (own vehicle)
            if self._is_in_vehicle_mask((x1, y1, x2, y2), (h, w)):
                continue

            # Calculate center and area
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            # Determine danger level based on position and size
            danger_level = self._calculate_danger_level(
                cy, y2, h, area, w * h, class_id
            )

            detections.append(Detection(
                class_id=class_id,
                class_name=self.OBSTACLE_CLASSES[class_id],
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area=area,
                danger_level=danger_level
            ))

        # Sort by danger level (danger first)
        danger_order = {'danger': 0, 'caution': 1, 'safe': 2}
        detections.sort(key=lambda d: (danger_order[d.danger_level], -d.area))

        return detections, inference_time

    def detect_obstacles_from_segmentation(
        self,
        pred_mask: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Detect obstacles (walls, barriers) from segmentation mask.

        ROAD class = 1, anything else in bottom half = potential obstacle

        Args:
            pred_mask: Segmentation mask (0=Other, 1=ROAD, 2=MYCAR)
            image_shape: (height, width)

        Returns:
            List of wall/obstacle detections
        """
        h, w = image_shape
        detections = []

        # Resize mask if needed
        if pred_mask.shape[0] != h or pred_mask.shape[1] != w:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Get vehicle mask resized
        vehicle_excluded = np.zeros((h, w), dtype=np.uint8)
        if self.vehicle_mask is not None:
            mask = self.vehicle_mask
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            vehicle_excluded = mask > 127

        # Find obstacles: Not ROAD (class != 1) and not MYCAR (class != 2) and not in vehicle mask
        # Focus on bottom 60% of image (closer objects)
        bottom_start = int(h * 0.4)

        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_region = (pred_mask != 1) & (pred_mask != 2) & (~vehicle_excluded)
        obstacle_mask[obstacle_region] = 255

        # Only look at bottom portion
        obstacle_mask[:bottom_start, :] = 0

        # Find contours
        contours, _ = cv2.findContours(
            obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Minimum area threshold (ignore tiny regions)
        min_area = (w * h) * 0.01  # At least 1% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            # Calculate center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determine danger level based on position
            rel_bottom = y2 / h
            rel_size = area / (w * h)

            if rel_bottom > 0.85 or rel_size > 0.15:
                danger_level = 'danger'
            elif rel_bottom > 0.65 or rel_size > 0.08:
                danger_level = 'caution'
            else:
                danger_level = 'safe'

            detections.append(Detection(
                class_id=-1,  # Special ID for wall/obstacle
                class_name='wall/obstacle',
                confidence=1.0,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area=int(area),
                danger_level=danger_level
            ))

        return detections

    def _calculate_danger_level(
        self,
        center_y: int,
        bottom_y: int,
        image_height: int,
        bbox_area: int,
        image_area: int,
        class_id: int
    ) -> str:
        """
        Calculate danger level based on position and size.

        Objects in lower part of image = closer = more dangerous
        Larger objects = closer = more dangerous
        """
        # Relative position (0 = top, 1 = bottom)
        rel_bottom = bottom_y / image_height

        # Relative size
        rel_size = bbox_area / image_area

        # High priority classes have lower thresholds
        is_high_priority = class_id in self.HIGH_PRIORITY

        # Danger: object in bottom 40% OR very large (>15%)
        # For high priority: bottom 50% OR >10%
        if is_high_priority:
            if rel_bottom > 0.5 or rel_size > 0.10:
                return 'danger'
            elif rel_bottom > 0.35 or rel_size > 0.05:
                return 'caution'
        else:
            if rel_bottom > 0.6 or rel_size > 0.15:
                return 'danger'
            elif rel_bottom > 0.4 or rel_size > 0.08:
                return 'caution'

        return 'safe'

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Draw detection boxes on image with danger zone coloring.

        Args:
            image: BGR image
            detections: List of Detection objects
            alpha: Transparency for overlay

        Returns:
            Image with drawn detections
        """
        result = image.copy()
        overlay = image.copy()

        # Color scheme (BGR)
        colors = {
            'danger': (0, 0, 255),    # Red
            'caution': (0, 200, 255),  # Yellow/Orange
            'safe': (0, 255, 0)        # Green
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors[det.danger_level]

            # Draw filled rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Draw border
            thickness = 3 if det.danger_level == 'danger' else 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Label
            label = f"{det.class_name}"
            if det.confidence < 1.0:  # YOLO detection has confidence
                label += f" {det.confidence:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Label background
            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0] + 4, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                result,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Blend overlay
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        return result

    def get_danger_summary(self, detections: List[Detection]) -> dict:
        """
        Get summary of danger levels.

        Returns:
            Dict with counts and most dangerous object
        """
        summary = {
            'total': len(detections),
            'danger_count': 0,
            'caution_count': 0,
            'safe_count': 0,
            'most_dangerous': None,
            'warning': '',
            'has_wall': False
        }

        for det in detections:
            if det.class_name == 'wall/obstacle':
                summary['has_wall'] = True

            if det.danger_level == 'danger':
                summary['danger_count'] += 1
                if summary['most_dangerous'] is None:
                    summary['most_dangerous'] = det
            elif det.danger_level == 'caution':
                summary['caution_count'] += 1
            else:
                summary['safe_count'] += 1

        # Generate warning message
        if summary['danger_count'] > 0:
            obj = summary['most_dangerous']
            summary['warning'] = f"⚠️ DANGER: {obj.class_name}!"
        elif summary['caution_count'] > 0:
            summary['warning'] = f"⚡ Caution: {summary['caution_count']} obstacle(s)"

        return summary
