"""Data package for ADE20K labels and related data."""

from .ade20k_labels import (
    ADE20K_LABELS,
    ADE20K_COLORS,
    get_label_name,
    get_label_color,
    get_all_label_names,
)

__all__ = [
    "ADE20K_LABELS",
    "ADE20K_COLORS",
    "get_label_name",
    "get_label_color",
    "get_all_label_names",
]
