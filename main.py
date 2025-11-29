#!/usr/bin/env python3
"""
JetRacer ROAD Annotation Tool

ADE20K segmentation to ROAD mapping tool for autonomous driving.
"""

import argparse
from pathlib import Path
from nicegui import ui

from core.segmentation import generate_demo_images
from ui.annotation_ui import AnnotationUI


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='JetRacer ROAD Annotation Tool - Map ADE20K labels to ROAD'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='demo_images',
        help='Directory containing images (default: demo_images)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for output files (default: output)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    parser.add_argument(
        '--generate-demo',
        action='store_true',
        help='Generate demo images if image directory does not exist'
    )
    parser.add_argument(
        '--num-demo-images',
        type=int,
        default=3,
        help='Number of demo images to generate (default: 3)'
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # Generate demo images if needed
    if not image_dir.exists() or not list(image_dir.glob("*.jpg")):
        if args.generate_demo or args.image_dir == 'demo_images':
            print(f"Generating {args.num_demo_images} demo images in {image_dir}...")
            generate_demo_images(image_dir, args.num_demo_images)
            print("Demo images generated!")
        else:
            print(f"Error: Image directory {image_dir} does not exist or is empty.")
            print("Use --generate-demo to create demo images.")
            return

    # Create annotation UI
    print(f"Loading images from {image_dir}...")
    annotation_ui = AnnotationUI(
        image_dir=str(image_dir),
        output_dir=str(output_dir)
    )

    # Create the UI
    annotation_ui.create_ui()

    # Print access instructions
    print("\n" + "="*60)
    print("JetRacer ROAD Annotation Tool is starting...")
    print(f"Access the tool at: http://{args.host}:{args.port}")
    print("\nIf running on Jetson, access from another device using:")
    print(f"  http://<JETSON_IP>:{args.port}")
    print("\nTo find Jetson IP, run: hostname -I")
    print("="*60 + "\n")

    # Run the UI
    ui.run(
        host=args.host,
        port=args.port,
        title='JetRacer ROAD Annotation Tool',
        favicon='🤖'
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
