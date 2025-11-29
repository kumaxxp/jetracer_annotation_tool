#!/usr/bin/env python3
"""
Segment images using ADE20K OneFormer model.

This script processes images and generates ADE20K segmentation masks.
"""

import argparse
import logging
from pathlib import Path

from core.ade20k_segmentation import segment_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for segmentation script."""
    parser = argparse.ArgumentParser(
        description='Segment images using ADE20K OneFormer model'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save segmentation masks (default: <input-dir>_segmented)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='shi-labs/oneformer_ade20k_swin_tiny',
        help='HuggingFace model name (default: shi-labs/oneformer_ade20k_swin_tiny)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='Glob pattern for image files (default: *.jpg)'
    )

    args = parser.parse_args()

    # Determine output directory
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_segmented"

    logger.info("="*60)
    logger.info("ADE20K Image Segmentation")
    logger.info("="*60)
    logger.info(f"Input dir:  {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model:      {args.model}")
    logger.info(f"Pattern:    {args.pattern}")
    logger.info("="*60)

    # Check input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    # Segment images
    try:
        output_paths = segment_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=args.model,
            pattern=args.pattern
        )

        logger.info("\n" + "="*60)
        logger.info(f"✓ Segmentation complete!")
        logger.info(f"  Processed: {len(output_paths)} images")
        logger.info(f"  Output: {output_dir}")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ in {"__main__", "__mp_main__"}:
    exit(main())
