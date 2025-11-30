#!/bin/bash
# Start training/testing UI

# Activate virtual environment
source venv/bin/activate

# Run training UI
python main_training.py "$@"
