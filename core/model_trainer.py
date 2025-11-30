"""Model training utilities using segmentation_models_pytorch."""

from pathlib import Path
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm


class ROADDataset(Dataset):
    """Dataset for ROAD segmentation."""

    def __init__(self, image_dir: Path, label_dir: Path, transform=None):
        """
        Args:
            image_dir: Directory containing input images
            label_dir: Directory containing label masks
            transform: Albumentations transform
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted(list(image_dir.glob("*.jpg")))
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label
        label_path = self.label_dir / f"{img_path.stem}.png"
        label = np.array(Image.open(label_path).convert('L'))

        # Label is already multiclass (0, 1, 2)
        # 0: Other, 1: ROAD, 2: MYCAR
        label = label.astype(np.int64)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            # Ensure label is Long (int64) for CrossEntropyLoss
            # After ToTensorV2, label is torch.Tensor, not numpy
            if isinstance(label, np.ndarray):
                label = label.astype(np.int64)
            elif torch.is_tensor(label):
                label = label.long()

        return image, label  # No channel dimension for CrossEntropyLoss


def get_transforms(input_size: Tuple[int, int] = (320, 240), is_train: bool = True):
    """
    Get data augmentation transforms.

    Args:
        input_size: (width, height) for resize
        is_train: Whether for training (with augmentation) or validation

    Returns:
        Albumentations compose transform
    """
    if is_train:
        return A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class ROADTrainer:
    """Trainer for ROAD segmentation model."""

    def __init__(
        self,
        model_name: str = "DeepLabV3Plus",
        encoder_name: str = "mobilenet_v2",
        input_size: Tuple[int, int] = (320, 240),
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: Model architecture (DeepLabV3Plus, Unet, etc.)
            encoder_name: Encoder backbone (mobilenet_v2, resnet18, etc.)
            input_size: (width, height) for model input
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        model_class = getattr(smp, model_name)
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=3,  # 3-class segmentation: Other, ROAD, MYCAR
            activation=None  # We'll use CrossEntropyLoss
        )
        self.model.to(self.device)

        # Loss and optimizer (will be set in train())
        self.criterion = None
        self.optimizer = None

    def train(
        self,
        train_dir: Path,
        val_dir: Path,
        output_dir: Path,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        progress_callback: Optional[Callable[[int, float, float], None]] = None
    ):
        """
        Train the model.

        Args:
            train_dir: Training data directory (must contain images/ and labels/)
            val_dir: Validation data directory (must contain images/ and labels/)
            output_dir: Directory to save models and logs
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            progress_callback: Optional callback(epoch, train_loss, val_loss)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets
        train_dataset = ROADDataset(
            image_dir=train_dir / "images",
            label_dir=train_dir / "labels",
            transform=get_transforms(self.input_size, is_train=True)
        )
        val_dataset = ROADDataset(
            image_dir=val_dir / "images",
            label_dir=val_dir / "labels",
            transform=get_transforms(self.input_size, is_train=False)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss = self._validate_epoch(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    output_dir / "best_model.pth"
                )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    output_dir / f"model_epoch_{epoch+1}.pth"
                )

            # Callback
            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss)

        # Save final model
        torch.save(self.model.state_dict(), output_dir / "final_model.pth")

    def _train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        with tqdm(dataloader, desc="Training") as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def export_onnx(self, onnx_path: Path):
        """Export model to ONNX format compatible with OpenCV DNN."""
        self.model.eval()
        self.model.to('cpu')  # Move to CPU for export

        dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0])

        # Use legacy exporter for better OpenCV compatibility
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=13,  # OpenCV supports up to 13-14
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch'},
                    'output': {0: 'batch'}
                }
            )

        print(f"Model exported to {onnx_path}")

        # Move back to device
        self.model.to(self.device)
