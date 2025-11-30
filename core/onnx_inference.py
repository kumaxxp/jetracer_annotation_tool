"""ONNX model inference using OpenCV DNN."""

from pathlib import Path
from typing import Tuple
import time
import cv2
import numpy as np


class ONNXSegmenter:
    """ONNX segmentation model inference with OpenCV DNN."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 240),
        use_cuda: bool = True
    ):
        """
        Args:
            model_path: Path to ONNX model
            input_size: (width, height) for model input
            use_cuda: Whether to use CUDA backend
        """
        self.input_width, self.input_height = input_size
        self.use_cuda = use_cuda

        # Load model
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"Loading ONNX model from {path}...")
        self.net = cv2.dnn.readNetFromONNX(str(path))

        # Configure backend
        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                print("✓ Using CUDA backend (FP16)")
            except Exception as e:
                print(f"CUDA not available: {e}")
                print("Using CPU backend")
                self.use_cuda = False
        else:
            print("Using CPU backend")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Blob tensor (1, 3, H, W)
        """
        if image is None or image.ndim != 3:
            raise ValueError("image must be a BGR numpy array with shape (H, W, 3)")

        # Resize
        resized = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR
        )

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std

        # Convert to blob (1, 3, H, W)
        blob = cv2.dnn.blobFromImage(
            normalized,
            scalefactor=1.0,
            size=(self.input_width, self.input_height),
            swapRB=False  # Already converted to RGB
        )

        return blob

    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            (mask, inference_time) where:
                mask: Binary mask (H, W) - 0 or 255, resized to original size
                inference_time: Inference time in milliseconds
        """
        original_h, original_w = image.shape[:2]

        # Preprocess
        blob = self.preprocess(image)

        # Inference
        self.net.setInput(blob)
        start = time.perf_counter()
        output = self.net.forward()
        inference_time = (time.perf_counter() - start) * 1000  # ms

        # Postprocess
        mask = self.postprocess(output)

        # Resize to original size
        mask = cv2.resize(
            mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return mask, inference_time

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to binary mask.

        Args:
            output: Model output (1, 1, H, W) or (1, H, W) - logits

        Returns:
            Binary mask (H, W) - 0 or 255
        """
        # Remove batch dimension
        if output.ndim == 4:
            logits = output[0, 0]  # (H, W)
        elif output.ndim == 3:
            logits = output[0]  # (H, W)
        else:
            logits = output

        # Apply sigmoid and threshold
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        mask = (probs > 0.5).astype(np.uint8) * 255

        return mask
