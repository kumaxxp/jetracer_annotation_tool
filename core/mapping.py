"""Label to ROAD mapping management."""

import json
from pathlib import Path
from typing import Dict, Set
from data.ade20k_labels import get_label_name, get_all_label_names


class ROADMapping:
    """Manages mapping between ADE20K labels and ROAD attribute."""

    def __init__(self, mapping_file: str = None):
        """
        Initialize ROAD mapping.

        Args:
            mapping_file: Path to JSON mapping file (optional)
        """
        self.mapping_file = Path(mapping_file) if mapping_file else None
        self.mapping: Dict[str, bool] = {}  # label_name -> is_road
        self.road_labels: Set[str] = set()

        if self.mapping_file and self.mapping_file.exists():
            self.load()
        else:
            # Initialize with default likely ROAD labels
            self._init_default_mapping()

    def _init_default_mapping(self):
        """Initialize with default ROAD labels."""
        # Common labels that are likely to be ROAD (drivable surfaces)
        default_road_labels = {
            "road", "floor", "path", "sidewalk", "runway",
            "dirt track", "field", "sand"
        }

        # Initialize all known labels as non-ROAD
        for label_name in get_all_label_names():
            self.mapping[label_name] = label_name in default_road_labels

        # Update road_labels set
        self.road_labels = {label for label, is_road in self.mapping.items() if is_road}

    def toggle_road(self, label_name: str) -> bool:
        """
        Toggle ROAD attribute for a label.

        Args:
            label_name: Name of the label

        Returns:
            New ROAD state (True if now ROAD, False otherwise)
        """
        if label_name not in self.mapping:
            self.mapping[label_name] = False

        # Toggle
        self.mapping[label_name] = not self.mapping[label_name]

        # Update road_labels set
        if self.mapping[label_name]:
            self.road_labels.add(label_name)
        else:
            self.road_labels.discard(label_name)

        return self.mapping[label_name]

    def set_road(self, label_name: str, is_road: bool):
        """
        Set ROAD attribute for a label.

        Args:
            label_name: Name of the label
            is_road: Whether this label is ROAD
        """
        self.mapping[label_name] = is_road

        if is_road:
            self.road_labels.add(label_name)
        else:
            self.road_labels.discard(label_name)

    def is_road(self, label_name: str) -> bool:
        """
        Check if a label is marked as ROAD.

        Args:
            label_name: Name of the label

        Returns:
            True if label is ROAD, False otherwise
        """
        return self.mapping.get(label_name, False)

    def get_road_labels(self) -> Set[str]:
        """
        Get set of all ROAD label names.

        Returns:
            Set of label names marked as ROAD
        """
        return self.road_labels.copy()

    def save(self, filepath: str = None):
        """
        Save mapping to JSON file.

        Args:
            filepath: Path to save file (uses self.mapping_file if None)
        """
        save_path = Path(filepath) if filepath else self.mapping_file
        if not save_path:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "road_labels": sorted(list(self.road_labels)),
            "mapping": self.mapping
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Mapping saved to {save_path}")

    def load(self, filepath: str = None):
        """
        Load mapping from JSON file.

        Args:
            filepath: Path to load file (uses self.mapping_file if None)
        """
        load_path = Path(filepath) if filepath else self.mapping_file
        if not load_path or not load_path.exists():
            raise ValueError(f"Mapping file not found: {load_path}")

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.mapping = data.get("mapping", {})
        self.road_labels = set(data.get("road_labels", []))

        print(f"Mapping loaded from {load_path}")

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the mapping.

        Returns:
            Dictionary with stats (total_labels, road_labels, non_road_labels)
        """
        return {
            "total_labels": len(self.mapping),
            "road_labels": len(self.road_labels),
            "non_road_labels": len(self.mapping) - len(self.road_labels)
        }
