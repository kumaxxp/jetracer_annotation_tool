"""PyTorch model inference (alternative to ONNX)."""

from pathlib import Path
from typing import Tuple
import time
import torch
import cv2
import numpy as np
from torchvision import transforms


class PyTorchSegmenter:
    """PyTorch segmentation model inference."""

    def __init__(
        self,
        model_path: str,
        model_class,
        input_size: Tuple[int, int] = (320, 240),
        use_cuda: bool = True
    ):
        """
        Args:
            model_path: Path to PyTorch model (.pth)
            model_class: Model class to instantiate
            input_size: (width, height) for model input
            use_cuda: Whether to use CUDA
        """
        self.input_width, self.input_height = input_size
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

        # Load model
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"Loading PyTorch model from {path}...")
        self.model = model_class
        self.model.load_state_dict(torch.load(str(path), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Using PyTorch on {self.device}")

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tensor (1, 3, H, W)
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

        # Normalize and convert to tensor
        tensor = self.transform(rgb).unsqueeze(0)  # Add batch dimension

        return tensor

    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            (mask, inference_time) where:
                mask: Multiclass mask (H, W) - 0: Other, 1: ROAD, 2: MYCAR, resized to original size
                inference_time: Inference time in milliseconds
        """
        original_h, original_w = image.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)

        # Inference
        with torch.no_grad():
            start = time.perf_counter()
            output = self.model(input_tensor)
            inference_time = (time.perf_counter() - start) * 1000  # ms

        # Postprocess
        mask = self.postprocess(output.cpu().numpy())

        # Resize to original size
        mask = cv2.resize(
            mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return mask, inference_time

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to multiclass mask.

        Args:
            output: Model output (1, 3, H, W) - logits for 3 classes

        Returns:
            Multiclass mask (H, W) - 0: Other, 1: ROAD, 2: MYCAR
        """
        # Remove batch dimension: (1, 3, H, W) -> (3, H, W)
        logits = output[0]

        # Argmax over class dimension: (3, H, W) -> (H, W)
        mask = np.argmax(logits, axis=0).astype(np.uint8)

        return mask
