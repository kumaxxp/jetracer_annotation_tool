"""NiceGUI-based training and testing interface."""

import io
import base64
import threading
from pathlib import Path
from typing import Optional
from nicegui import ui
from PIL import Image
import numpy as np
import cv2

import segmentation_models_pytorch as smp

from core.model_trainer import ROADTrainer
from core.onnx_inference import ONNXSegmenter
from core.pytorch_inference import PyTorchSegmenter
from data.ade20k_labels import ADE20K_COLORS


class TrainingTestingUI:
    """Training and testing interface."""

    def __init__(self, default_training_data_dir: str = "output/training_data"):
        """
        Initialize training/testing UI.

        Args:
            default_training_data_dir: Default training data directory
        """
        self.default_training_data_dir = Path(default_training_data_dir)

        # Training state
        self.trainer: Optional[ROADTrainer] = None
        self.is_training = False

        # Testing state
        self.model_path: Optional[Path] = None
        self.test_images: list = []
        self.current_test_index = 0

        # UI components (created in create_ui)
        self.train_log = None
        self.train_button = None
        self.test_images_select = None
        self.ade20k_display = None
        self.ground_truth_display = None
        self.prediction_display = None
        self.inference_time_label = None

    def create_ui(self):
        """Create the NiceGUI interface."""
        ui.page_title("JetRacer Model Training & Testing")

        with ui.header().classes('items-center justify-between'):
            ui.label('JetRacer Model Training & Testing').classes('text-h4')

        with ui.tabs().classes('w-full') as tabs:
            train_tab = ui.tab('Training')
            test_tab = ui.tab('Testing')

        with ui.tab_panels(tabs, value=train_tab).classes('w-full'):
            # Training panel
            with ui.tab_panel(train_tab):
                self._create_training_panel()

            # Testing panel
            with ui.tab_panel(test_tab):
                self._create_testing_panel()

    def _create_training_panel(self):
        """Create training tab content."""
        ui.label('Model Training').classes('text-h5 mb-4')

        with ui.card().classes('w-full'):
            ui.label('Training Configuration').classes('text-h6 mb-2')

            # Training data directory
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Training Data:').classes('w-32')
                self.train_data_input = ui.input(
                    value=str(self.default_training_data_dir),
                    placeholder='/path/to/training_data'
                ).classes('flex-1')

            # Model parameters
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Epochs:').classes('w-32')
                self.epochs_input = ui.number(value=50, min=1, max=1000).classes('w-32')

            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Batch Size:').classes('w-32')
                self.batch_size_input = ui.number(value=8, min=1, max=64).classes('w-32')

            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Learning Rate:').classes('w-32')
                self.lr_input = ui.number(value=0.0001, format='%.6f').classes('w-32')

            # Start button
            with ui.row().classes('w-full gap-2 mt-4'):
                self.train_button = ui.button(
                    'Start Training',
                    on_click=self._start_training,
                    icon='play_arrow',
                    color='green'
                )
                ui.button(
                    'Stop Training',
                    on_click=self._stop_training,
                    icon='stop',
                    color='red'
                ).props('disable' if not self.is_training else '')

        # Training log
        with ui.card().classes('w-full mt-4'):
            ui.label('Training Log').classes('text-h6 mb-2')
            self.train_log = ui.log().classes('w-full h-96')

    def _create_testing_panel(self):
        """Create testing tab content."""
        ui.label('Model Testing').classes('text-h5 mb-4')

        with ui.card().classes('w-full'):
            ui.label('Model Selection').classes('text-h6 mb-2')

            with ui.row().classes('w-full items-center gap-2'):
                ui.label('ONNX Model:').classes('w-32')
                self.model_path_input = ui.input(
                    placeholder='/path/to/model.onnx'
                ).classes('flex-1')
                ui.button('Load Model', on_click=self._load_model, icon='upload')

            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Test Images:').classes('w-32')
                self.test_dir_input = ui.input(
                    placeholder='/path/to/test/images'
                ).classes('flex-1')
                ui.button('Load Images', on_click=self._load_test_images, icon='folder')

        # Image selection
        with ui.card().classes('w-full mt-4'):
            ui.label('Image Selection').classes('text-h6 mb-2')
            with ui.row().classes('w-full items-center gap-2'):
                ui.button('Previous', on_click=self._previous_test_image, icon='arrow_back')
                self.test_image_label = ui.label('No images loaded').classes('flex-1 text-center')
                ui.button('Next', on_click=self._next_test_image, icon='arrow_forward')
                ui.button('Run Inference', on_click=self._run_inference, icon='play_arrow', color='green')

        # 3-column comparison
        with ui.row().classes('w-full gap-2 p-2 mt-4'):
            # Left: ADE20K Segmentation
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('ADE20K Segmentation (OneFormer)').classes('text-h6 mb-2')
                    self.ade20k_display = ui.interactive_image().classes('w-full')

            # Middle: Ground Truth
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Ground Truth (Annotation)').classes('text-h6 mb-2')
                    self.ground_truth_display = ui.interactive_image().classes('w-full')

            # Right: Prediction
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Prediction (Trained Model)').classes('text-h6 mb-2')
                    self.prediction_display = ui.interactive_image().classes('w-full')

        # Performance metrics
        with ui.card().classes('w-full mt-4'):
            ui.label('Performance Metrics').classes('text-h6 mb-2')
            self.inference_time_label = ui.label('Inference time: -').classes('text-bold')

    def _start_training(self):
        """Start model training in background thread."""
        if self.is_training:
            ui.notify('Training already in progress', type='warning')
            return

        # Validate inputs
        train_data_dir = Path(self.train_data_input.value)
        if not train_data_dir.exists():
            ui.notify(f'Training data directory not found: {train_data_dir}', type='negative')
            return

        # Create trainer
        self.trainer = ROADTrainer(
            model_name="DeepLabV3Plus",
            encoder_name="mobilenet_v2",
            input_size=(320, 240)
        )

        self.train_log.clear()
        self.train_log.push(f"Initializing training...")
        self.train_log.push(f"Model: DeepLabV3Plus (MobileNetV2)")
        self.train_log.push(f"Input size: 320x240")
        self.train_log.push(f"Epochs: {int(self.epochs_input.value)}")
        self.train_log.push(f"Batch size: {int(self.batch_size_input.value)}")
        self.train_log.push(f"Learning rate: {self.lr_input.value}")
        self.train_log.push("-" * 60)

        # Start training in background
        self.is_training = True

        def train_thread():
            try:
                output_dir = Path("output/models")
                self.trainer.train(
                    train_dir=train_data_dir / "train",
                    val_dir=train_data_dir / "val",
                    output_dir=output_dir,
                    epochs=int(self.epochs_input.value),
                    batch_size=int(self.batch_size_input.value),
                    learning_rate=self.lr_input.value,
                    progress_callback=self._training_progress_callback
                )

                # Export to ONNX
                self.train_log.push("\nExporting to ONNX...")
                onnx_path = output_dir / "road_segmentation.onnx"
                self.trainer.export_onnx(onnx_path)
                self.train_log.push(f"✓ Model exported to {onnx_path}")

                ui.notify('Training complete!', type='positive')

            except Exception as e:
                self.train_log.push(f"\n✗ Training failed: {e}")
                import traceback
                traceback.print_exc()
                ui.notify(f'Training error: {e}', type='negative')

            finally:
                self.is_training = False

        threading.Thread(target=train_thread, daemon=True).start()

    def _stop_training(self):
        """Stop training (not implemented - requires interrupt mechanism)."""
        ui.notify('Stop training not implemented yet', type='warning')

    def _training_progress_callback(self, epoch: int, train_loss: float, val_loss: float):
        """Training progress callback."""
        self.train_log.push(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    def _load_model(self):
        """Load ONNX model for testing."""
        model_path = Path(self.model_path_input.value)
        if not model_path.exists():
            ui.notify(f'Model not found: {model_path}', type='negative')
            return

        try:
            self.model_path = model_path
            ui.notify(f'Model loaded: {model_path.name}', type='positive')
        except Exception as e:
            ui.notify(f'Error loading model: {e}', type='negative')

    def _load_test_images(self):
        """Load test images."""
        test_dir = Path(self.test_dir_input.value)
        if not test_dir.exists():
            ui.notify(f'Directory not found: {test_dir}', type='negative')
            return

        # Load images
        self.test_images = sorted(list(test_dir.glob("*.jpg")))
        if not self.test_images:
            ui.notify('No images found', type='warning')
            return

        self.current_test_index = 0
        self._update_test_image_label()
        ui.notify(f'Loaded {len(self.test_images)} images', type='positive')

    def _update_test_image_label(self):
        """Update test image label."""
        if self.test_images:
            img_name = self.test_images[self.current_test_index].name
            self.test_image_label.set_text(
                f'Image {self.current_test_index + 1} / {len(self.test_images)} - {img_name}'
            )
        else:
            self.test_image_label.set_text('No images loaded')

    def _previous_test_image(self):
        """Navigate to previous test image."""
        if self.test_images and self.current_test_index > 0:
            self.current_test_index -= 1
            self._update_test_image_label()

    def _next_test_image(self):
        """Navigate to next test image."""
        if self.test_images and self.current_test_index < len(self.test_images) - 1:
            self.current_test_index += 1
            self._update_test_image_label()

    def _run_inference(self):
        """Run inference and display 3-way comparison."""
        if not self.test_images:
            ui.notify('No test images loaded', type='warning')
            return

        if not self.model_path:
            ui.notify('No model loaded', type='warning')
            return

        try:
            # Get current image
            img_path = self.test_images[self.current_test_index]

            # 1. Load ADE20K segmentation
            seg_path = img_path.parent / f"{img_path.stem}_seg.png"
            if seg_path.exists():
                ade20k_seg = Image.open(seg_path)
                ade20k_colored = self._colorize_segmentation(np.array(ade20k_seg))
                self._update_display(self.ade20k_display, ade20k_colored)
            else:
                ui.notify('ADE20K segmentation not found', type='warning')

            # 2. Load Ground Truth
            gt_path = img_path.parent.parent / "labels" / f"{img_path.stem}.png"
            if gt_path.exists():
                gt_mask = Image.open(gt_path)
                gt_colored = self._colorize_binary_mask(np.array(gt_mask))
                self._update_display(self.ground_truth_display, gt_colored)
            else:
                # Try training_data directory structure
                alt_gt_path = Path("output/training_data/val/labels") / f"{img_path.stem}.png"
                if alt_gt_path.exists():
                    gt_mask = Image.open(alt_gt_path)
                    gt_colored = self._colorize_binary_mask(np.array(gt_mask))
                    self._update_display(self.ground_truth_display, gt_colored)

            # 3. Run model inference
            # Check if model is .pth (PyTorch) or .onnx
            if self.model_path.suffix == '.pth':
                # Use PyTorch inference
                model_class = smp.DeepLabV3Plus(
                    encoder_name="mobilenet_v2",
                    encoder_weights=None,  # Don't use pretrained
                    in_channels=3,
                    classes=1
                )
                segmenter = PyTorchSegmenter(
                    str(self.model_path),
                    model_class,
                    input_size=(320, 240),
                    use_cuda=True
                )
            else:
                # Use ONNX inference
                segmenter = ONNXSegmenter(str(self.model_path), input_size=(320, 240))

            image = cv2.imread(str(img_path))
            pred_mask, inference_time = segmenter.inference(image)

            pred_colored = self._colorize_binary_mask(pred_mask)
            self._update_display(self.prediction_display, pred_colored)

            # Update metrics
            fps = 1000.0 / inference_time if inference_time > 0 else 0
            status = "✅" if inference_time < 40 else "⚠️" if inference_time < 66 else "❌"
            self.inference_time_label.set_text(
                f'{status} Inference time: {inference_time:.1f}ms ({fps:.1f} FPS)'
            )

        except Exception as e:
            ui.notify(f'Inference error: {e}', type='negative')
            import traceback
            traceback.print_exc()

    def _colorize_segmentation(self, seg_array: np.ndarray) -> Image.Image:
        """Colorize ADE20K segmentation using color palette."""
        h, w = seg_array.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in np.unique(seg_array):
            if class_id < len(ADE20K_COLORS):
                mask = seg_array == class_id
                colored[mask] = ADE20K_COLORS[class_id]

        return Image.fromarray(colored)

    def _colorize_binary_mask(self, mask: np.ndarray) -> Image.Image:
        """Colorize binary mask (ROAD = green, non-ROAD = black)."""
        h, w = mask.shape if mask.ndim == 2 else mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # ROAD pixels = green
        road_pixels = mask > 127
        colored[road_pixels] = [0, 255, 0]  # Green

        return Image.fromarray(colored)

    def _update_display(self, display_widget, img: Image.Image):
        """Update image display widget."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        display_widget.set_source(f'data:image/png;base64,{img_str}')
