#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib tinker

echo "Setup complete. To activate the environment, run: source venv/bin/activate"
