#!/usr/bin/env python3
"""Main entry point for training/testing UI."""

import argparse
from nicegui import ui

from ui.training_ui import TrainingTestingUI


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='JetRacer Model Training & Testing Tool'
    )
    parser.add_argument(
        '--training-data',
        type=str,
        default='output/training_data',
        help='Default training data directory'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8082,
        help='Port to run the web server'
    )

    args = parser.parse_args()

    # Create UI
    app_ui = TrainingTestingUI(default_training_data_dir=args.training_data)
    app_ui.create_ui()

    # Run server
    ui.run(
        port=args.port,
        title='JetRacer Training & Testing',
        reload=False
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
