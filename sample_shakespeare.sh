#!/bin/bash
#SBATCH --job-name=myjob_array
#SBATCH --output=results/output_%A_%a.txt
#SBATCH --error=results/error_%A_%a.txt
#SBATCH -c 10
#SBATCH -t 6:00:00
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
set -e

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup_env.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Prepare the Shakespeare dataset if needed
if [ ! -f "out-shakespeare-char/ckpt.pt" ]; then
    echo "No model trained!"
    exit 1
else
    echo "Found trained model."
fi

echo "Starting inferencing..."

python sample.py --out_dir=out-shakespeare-char
