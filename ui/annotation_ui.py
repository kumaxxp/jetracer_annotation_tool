"""NiceGUI-based annotation interface."""

import io
import base64
from pathlib import Path
from typing import Optional, List
from nicegui import ui, events
from PIL import Image
import numpy as np

from core.segmentation import SegmentationImage
from core.mapping import ROADMapping
from core.training_export import export_training_data
from core.vehicle_mask_generator import VehicleMaskGenerator
from data.ade20k_labels import get_label_name
import cv2


class AnnotationUI:
    """Main annotation interface."""

    def __init__(self, image_dir: str, output_dir: str = "output"):
        """
        Initialize annotation UI.

        Args:
            image_dir: Directory containing images
            output_dir: Directory for output files
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load images (exclude segmentation files)
        all_images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        # Filter out segmentation files (*_seg.png)
        self.image_files = sorted([
            img for img in all_images
            if not img.name.endswith('_seg.png')
        ])
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

        self.current_index = 0
        self.current_seg_image: Optional[SegmentationImage] = None

        # Mapping
        mapping_file = self.output_dir / "road_mapping.json"
        self.mapping = ROADMapping(str(mapping_file))

        # UI state
        self.selected_label: Optional[str] = None
        self.selected_label_id: Optional[int] = None

        # Vehicle mask
        self.vehicle_mask: Optional[np.ndarray] = None
        self.vehicle_mask_path = self.output_dir / "vehicle_mask.png"

        # UI components (will be created in create_ui)
        self.image_display = None
        self.label_info_card = None
        self.road_labels_list = None
        self.page_label = None

    def create_ui(self):
        """Create the NiceGUI interface."""
        ui.page_title("JetRacer ROAD Annotation Tool")

        with ui.header().classes('items-center justify-between'):
            ui.label('JetRacer ROAD Annotation Tool').classes('text-h4')
            ui.label(f'Images: {len(self.image_files)}').classes('text-subtitle1')

        with ui.row().classes('w-full gap-2 p-2 items-start'):
            # Left panel: Segmentation Image (clickable)
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Segmentation Image (Click to select label)').classes('text-h6 mb-2')
                    self.image_display = ui.interactive_image().classes('w-full')
                    self.image_display.on('click', self._handle_image_click)

            # Middle panel: Original + Overlay
            with ui.column().classes('flex-1'):
                with ui.card():
                    ui.label('Original + Segmentation Overlay').classes('text-h6 mb-2')
                    self.overlay_display = ui.interactive_image().classes('w-full')

            # Right panel: Controls and info
            with ui.column().classes('flex-1 gap-2'):
                # Selected label info
                with ui.card():
                    ui.label('Selected Label').classes('text-h6 mb-2')
                    self.label_info_card = ui.column()
                    with self.label_info_card:
                        ui.label('Click on image to select').classes('text-grey')

                # ROAD labels list
                with ui.card().classes('max-h-96 overflow-auto'):
                    ui.label('ROAD Labels').classes('text-h6 mb-2')
                    self.road_labels_list = ui.column()

        # Bottom navigation
        with ui.footer().classes('bg-primary text-white p-4'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.button('Previous', on_click=self._previous_image, icon='arrow_back')

                self.page_label = ui.label()

                ui.button('Next', on_click=self._next_image, icon='arrow_forward')

                ui.button('Save Mapping', on_click=self._save_mapping,
                         icon='save', color='green')

                ui.button('Export Training Data', on_click=self._export_training_data,
                         icon='upload', color='orange')

                ui.button('Generate Vehicle Mask', on_click=self._generate_vehicle_mask,
                         icon='directions_car', color='purple')

        # Load first image
        self._load_current_image()
        self._update_road_labels_display()

    def _load_current_image(self):
        """Load and display the current image."""
        image_path = self.image_files[self.current_index]

        # Try to find corresponding segmentation file
        seg_path = None
        potential_seg_path = image_path.parent / f"{image_path.stem}_seg.png"
        if potential_seg_path.exists():
            seg_path = str(potential_seg_path)

        # Load segmentation image
        self.current_seg_image = SegmentationImage(
            image_path=str(image_path),
            seg_path=seg_path  # Use real segmentation if available
        )
        self.current_seg_image.load()

        # Get road label IDs
        road_label_ids = set()
        for label_name in self.mapping.get_road_labels():
            # Find label ID from name
            from data.ade20k_labels import ADE20K_LABELS
            for lid, lname in ADE20K_LABELS.items():
                if lname == label_name:
                    road_label_ids.add(lid)
                    break

        # Create display image with ROAD overlay
        if road_label_ids:
            display_img = self.current_seg_image.apply_road_overlay(road_label_ids)
        else:
            display_img = self.current_seg_image.get_blended_image(alpha=0.6)

        # Create overlay image (original + segmentation blend)
        overlay_img = self.current_seg_image.get_blended_image(alpha=0.5)

        # Convert to base64 for display
        self._update_image_display(display_img)
        self._update_overlay_display(overlay_img)

        # Update page label
        self.page_label.set_text(
            f'Image {self.current_index + 1} / {len(self.image_files)} - {image_path.name}'
        )

    def _update_image_display(self, img: Image.Image):
        """Update the image display with a PIL Image."""
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Update interactive image
        self.image_display.set_source(f'data:image/png;base64,{img_str}')

    def _update_overlay_display(self, img: Image.Image):
        """Update the overlay display with a PIL Image."""
        try:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Update overlay image
            if hasattr(self, 'overlay_display') and self.overlay_display:
                self.overlay_display.set_source(f'data:image/png;base64,{img_str}')
        except Exception as e:
            print(f"Error updating overlay display: {e}")

    def _handle_image_click(self, e):
        """Handle click on image to select label."""
        if not self.current_seg_image:
            return

        # Get click coordinates from event args
        # NiceGUI interactive_image sends coordinates in e.args
        if not hasattr(e, 'args') or not e.args:
            return

        img_width, img_height = self.current_seg_image.image.size

        # e.args contains the click event data
        # Extract relative coordinates (0-1 range)
        if 'image_x' in e.args and 'image_y' in e.args:
            x = int(e.args['image_x'] * img_width)
            y = int(e.args['image_y'] * img_height)
        else:
            # Fallback: use pixel coordinates if available
            x = int(e.args.get('offsetX', 0))
            y = int(e.args.get('offsetY', 0))

        # Get label at position
        label_id = self.current_seg_image.get_label_at_position(x, y)
        label_name = get_label_name(label_id)

        # Update selected label
        self.selected_label = label_name
        self.selected_label_id = label_id

        # Update label info display
        self._update_label_info()

    def _update_label_info(self):
        """Update the selected label info card."""
        self.label_info_card.clear()

        with self.label_info_card:
            if self.selected_label:
                ui.label(f'Label: {self.selected_label}').classes('text-bold')
                ui.label(f'ID: {self.selected_label_id}').classes('text-caption')

                is_road = self.mapping.is_road(self.selected_label)
                status = 'ROAD' if is_road else 'NOT ROAD'
                color = 'green' if is_road else 'grey'

                ui.label(f'Status: {status}').classes(f'text-bold text-{color}')

                # Toggle button
                ui.button(
                    'Toggle ROAD' if not is_road else 'Remove from ROAD',
                    on_click=self._toggle_current_label,
                    color='primary' if not is_road else 'red'
                ).classes('mt-2')
            else:
                ui.label('Click on image to select').classes('text-grey')

    def _toggle_current_label(self):
        """Toggle ROAD status of currently selected label."""
        if not self.selected_label:
            ui.notify('No label selected', type='warning')
            return

        # Toggle in mapping
        new_state = self.mapping.toggle_road(self.selected_label)

        # Show notification
        status = 'ROAD' if new_state else 'NOT ROAD'
        ui.notify(f'{self.selected_label} is now {status}', type='positive')

        # Refresh displays
        self._update_label_info()
        self._update_road_labels_display()
        self._load_current_image()  # Refresh to show new overlay

    def _update_road_labels_display(self):
        """Update the ROAD labels list."""
        self.road_labels_list.clear()

        with self.road_labels_list:
            road_labels = sorted(self.mapping.get_road_labels())

            if road_labels:
                for label_name in road_labels:
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('check_circle', color='green')
                        ui.label(label_name)
            else:
                ui.label('No ROAD labels yet').classes('text-grey')

            # Stats
            stats = self.mapping.get_stats()
            ui.separator()
            ui.label(f"Total: {stats['road_labels']} ROAD labels").classes('text-caption mt-2')

    def _previous_image(self):
        """Navigate to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()
            self.selected_label = None
            self.selected_label_id = None
            self._update_label_info()

    def _next_image(self):
        """Navigate to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_current_image()
            self.selected_label = None
            self.selected_label_id = None
            self._update_label_info()

    def _save_mapping(self):
        """Save the current mapping to file."""
        try:
            self.mapping.save()
            ui.notify('Mapping saved successfully!', type='positive')
        except Exception as e:
            ui.notify(f'Error saving mapping: {e}', type='negative')

    def _export_training_data(self):
        """Export training dataset with train/val split."""
        try:
            # Check if any ROAD labels are defined
            road_labels = self.mapping.get_road_labels()
            if not road_labels:
                ui.notify('No ROAD labels defined. Please annotate first.', type='warning')
                return

            # Save mapping first
            self.mapping.save()

            # Export dataset
            output_dir = self.output_dir / "training_data"
            ui.notify('Exporting training data...', type='info')

            num_train, num_val = export_training_data(
                image_files=self.image_files,
                road_mapping=self.mapping.mapping,
                output_dir=output_dir,
                train_ratio=0.8,
                random_seed=42,
                vehicle_mask_path=self.vehicle_mask_path if self.vehicle_mask_path.exists() else None
            )

            ui.notify(
                f'✓ Export complete! Train: {num_train}, Val: {num_val}',
                type='positive'
            )

        except Exception as e:
            ui.notify(f'Error exporting: {e}', type='negative')
            import traceback
            traceback.print_exc()

    def _generate_vehicle_mask(self):
        """Generate vehicle mask from all images."""
        try:
            ui.notify('Generating vehicle mask...', type='info')

            # Generate mask using proven parameters
            generator = VehicleMaskGenerator(
                difference_threshold=20.0,
                static_ratio_threshold=0.8,
                bottom_region_ratio=0.5,
                morphology_kernel_size=15
            )

            mask, stats = generator.generate_mask(
                image_paths=self.image_files,
                num_samples=min(10, len(self.image_files))
            )

            # Save mask
            cv2.imwrite(str(self.vehicle_mask_path), mask)
            self.vehicle_mask = mask

            # Create and save visualization
            sample_img = cv2.imread(str(self.image_files[0]))
            vis = generator.create_visualization(sample_img, mask)
            vis_path = self.output_dir / "vehicle_mask_visualization.jpg"
            cv2.imwrite(str(vis_path), vis)

            # Show statistics
            ratio = stats['vehicle_ratio_total'] * 100
            ui.notify(
                f'✓ Vehicle mask generated! Coverage: {ratio:.1f}%',
                type='positive'
            )

            # Apply mask to all images
            self._apply_vehicle_mask_to_all()

        except Exception as e:
            ui.notify(f'Error generating vehicle mask: {e}', type='negative')
            import traceback
            traceback.print_exc()

    def _apply_vehicle_mask_to_all(self):
        """Apply vehicle mask to all segmentation images."""
        if self.vehicle_mask is None:
            # Try to load existing mask
            if self.vehicle_mask_path.exists():
                self.vehicle_mask = cv2.imread(str(self.vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                ui.notify('No vehicle mask found. Generate one first.', type='warning')
                return

        # Vehicle mask is saved and will be used during training data export
        # We do NOT modify segmentation images to preserve original OneFormer results
        ui.notify(
            '✓ Vehicle mask ready. It will be applied during data export.',
            type='positive'
        )

        # Reload current image
        self._load_current_image()
