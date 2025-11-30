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
from core.segmentation import SegmentationImage
from core.mapping import ROADMapping
from core.depth_estimation import DepthEstimator
from data.ade20k_labels import ADE20K_COLORS, ADE20K_LABELS


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

        # Load ROAD mapping for display
        mapping_file = Path("output/road_mapping.json")
        self.road_mapping = ROADMapping(str(mapping_file)) if mapping_file.exists() else None

        # Depth estimator
        self.depth_estimator: Optional[DepthEstimator] = None
        self._init_depth_estimator()

        # UI components (created in create_ui)
        self.train_log = None
        self.train_button = None
        self.test_images_select = None
        self.ade20k_display = None
        self.overlay_display = None
        self.depth_display = None
        self.inference_time_label = None
        self.depth_info_label = None

    def _init_depth_estimator(self):
        """Initialize Depth Anything V2 estimator."""
        # Try to find depth model
        depth_model_paths = [
            Path("/home/jetson/auto_recorder/configs/models/depth_anything_v2_vits_dynamic.onnx"),  # Dynamic (ONNX Runtime)
            Path("output/models/depth_anything_v2_static.onnx"),  # Static shape (OpenCV compatible)
            Path("output/models/depth_anything_v2_small.onnx"),
            Path("models/depth_anything_v2_small.onnx"),
        ]

        for model_path in depth_model_paths:
            if model_path.exists():
                try:
                    self.depth_estimator = DepthEstimator(
                        model_path=str(model_path),
                        input_size=(518, 518),  # Depth Anything V2 optimal size
                        use_cuda=True
                    )
                    if self.depth_estimator.load_model():
                        print(f"✓ Depth model loaded: {model_path}")
                        return
                except Exception as e:
                    print(f"Error loading depth model: {e}")

        print("⚠ Depth model not found. Depth estimation disabled.")
        self.depth_estimator = None

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

            # Middle: Original + Prediction Overlay
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Original + Prediction Overlay').classes('text-h6 mb-2')
                    self.overlay_display = ui.interactive_image().classes('w-full')

            # Right: Depth Estimation
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Depth Estimation (Depth Anything V2)').classes('text-h6 mb-2')
                    self.depth_display = ui.interactive_image().classes('w-full')
                    self.depth_info_label = ui.label('').classes('text-caption mt-2')

        # Performance metrics
        with ui.card().classes('w-full mt-4'):
            ui.label('Performance Metrics').classes('text-h6 mb-2')
            self.inference_time_label = ui.label('Segmentation: - | Depth: -').classes('text-bold')

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
                self.train_log.push("\n✓ Training complete!")

            except Exception as e:
                self.train_log.push(f"\n✗ Training failed: {e}")
                import traceback
                traceback.print_exc()

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

            # 1. Load ADE20K segmentation with ROAD overlay (same as annotation tool)
            seg_path = img_path.parent / f"{img_path.stem}_seg.png"
            if seg_path.exists() and self.road_mapping:
                # Use SegmentationImage class to apply ROAD overlay
                seg_image = SegmentationImage(
                    image_path=str(img_path),
                    seg_path=str(seg_path)
                )
                seg_image.load()

                # Get ROAD label IDs
                road_label_ids = set()
                for label_name in self.road_mapping.get_road_labels():
                    for lid, lname in ADE20K_LABELS.items():
                        if lname == label_name:
                            road_label_ids.add(lid)
                            break

                # Apply ROAD overlay (same as annotation tool)
                if road_label_ids:
                    display_img = seg_image.apply_road_overlay(road_label_ids)
                else:
                    display_img = seg_image.get_blended_image(alpha=0.6)

                self._update_display(self.ade20k_display, display_img)
            elif seg_path.exists():
                # Fallback: simple colorization if no ROAD mapping
                ade20k_seg = Image.open(seg_path)
                ade20k_colored = self._colorize_segmentation(np.array(ade20k_seg))
                self._update_display(self.ade20k_display, ade20k_colored)
            else:
                ui.notify('ADE20K segmentation not found', type='warning')

            # 2. Run model inference
            # Check if model is .pth (PyTorch) or .onnx
            if self.model_path.suffix == '.pth':
                # Use PyTorch inference
                model_class = smp.DeepLabV3Plus(
                    encoder_name="mobilenet_v2",
                    encoder_weights=None,  # Don't use pretrained
                    in_channels=3,
                    classes=3  # 3-class segmentation: Other, ROAD, MYCAR
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
            pred_mask, seg_inference_time = segmenter.inference(image)

            # Create and display original + prediction overlay (middle)
            overlay_img = self._create_prediction_overlay(image, pred_mask)
            self._update_display(self.overlay_display, overlay_img)

            # 3. Run depth estimation (right)
            depth_inference_time = 0.0
            if self.depth_estimator is not None:
                try:
                    depth_map, depth_inference_time = self.depth_estimator.inference(image)

                    # Create depth visualization with plasma colormap
                    depth_colored = self._colorize_depth_map(depth_map)
                    self._update_display(self.depth_display, depth_colored)

                    # Update depth info
                    depth_min = depth_map.min()
                    depth_max = depth_map.max()
                    depth_mean = depth_map.mean()
                    self.depth_info_label.set_text(
                        f'Depth: min={depth_min:.3f}, max={depth_max:.3f}, mean={depth_mean:.3f}'
                    )
                except Exception as e:
                    print(f"Depth estimation error: {e}")
                    self.depth_info_label.set_text(f'Depth error: {e}')
            else:
                self.depth_info_label.set_text('Depth model not loaded')

            # Update metrics
            seg_fps = 1000.0 / seg_inference_time if seg_inference_time > 0 else 0
            depth_fps = 1000.0 / depth_inference_time if depth_inference_time > 0 else 0
            seg_status = "✅" if seg_inference_time < 40 else "⚠️" if seg_inference_time < 66 else "❌"
            depth_status = "✅" if depth_inference_time < 100 else "⚠️" if depth_inference_time < 200 else "❌"
            self.inference_time_label.set_text(
                f'{seg_status} Seg: {seg_inference_time:.1f}ms ({seg_fps:.1f}fps) | '
                f'{depth_status} Depth: {depth_inference_time:.1f}ms ({depth_fps:.1f}fps)'
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
        """Colorize multiclass mask (0: Other=black, 1: ROAD=green, 2: MYCAR=orange)."""
        h, w = mask.shape if mask.ndim == 2 else mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Class 0: Other - black (already 0)
        # Class 1: ROAD - green
        colored[mask == 1] = [0, 255, 0]  # Green (RGB)
        # Class 2: MYCAR - orange
        colored[mask == 2] = [255, 128, 0]  # Orange (RGB)

        return Image.fromarray(colored)

    def _create_prediction_overlay(self, bgr_image: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Create overlay of prediction mask on original image.

        Args:
            bgr_image: Original BGR image from cv2.imread
            pred_mask: Multiclass mask (0: Other, 1: ROAD, 2: MYCAR)
            alpha: Transparency (0.0-1.0)

        Returns:
            PIL Image with overlay
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Create colored overlay
        overlay = np.zeros_like(rgb_image)
        # Class 1: ROAD - green
        overlay[pred_mask == 1] = [0, 255, 0]  # Green
        # Class 2: MYCAR - orange
        overlay[pred_mask == 2] = [255, 128, 0]  # Orange

        # Blend original + overlay
        blended = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)

        return Image.fromarray(blended)

    def _update_display(self, display_widget, img: Image.Image):
        """Update image display widget."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        display_widget.set_source(f'data:image/png;base64,{img_str}')

    def _colorize_depth_map(self, depth_map: np.ndarray) -> Image.Image:
        """
        Colorize depth map using plasma colormap.

        Args:
            depth_map: Normalized depth map (0-1, higher = closer)

        Returns:
            PIL Image with colorized depth
        """
        # Ensure depth_map is 2D
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]

        # Convert to 0-255 range
        depth_uint8 = (depth_map * 255).astype(np.uint8)

        # Apply plasma colormap (COLORMAP_PLASMA: purple->yellow, closer=brighter)
        # Invert so that closer objects appear brighter/warmer
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

        # Convert BGR to RGB for PIL
        depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return Image.fromarray(depth_rgb)
