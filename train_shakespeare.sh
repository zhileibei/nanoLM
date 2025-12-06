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
if [ ! -f "data/shakespeare/train.bin" ]; then
    echo "Preparing Shakespeare dataset..."
    python data/shakespeare/prepare.py
else
    echo "Shakespeare dataset already prepared."
fi

echo "Starting training..."

# Hyperparameters explanation:
# --n_layer: Number of transformer layers. Deeper models can capture more complex patterns but are harder to train.
# --n_head: Number of attention heads. More heads allow the model to focus on different parts of the input.
# --n_embd: Embedding dimension. Size of the vector representation for each token.
# --block_size: Context length. How many previous characters the model can see.
# --batch_size: Number of samples per training step.
# --dropout: Dropout rate for regularization. Helps prevent overfitting.
# --learning_rate: Step size for the optimizer.
# --max_iters: Total number of training iterations.
# --out_dir: Directory to save checkpoints and logs.
# --compile: Whether to use PyTorch 2.0 compilation (True/False).

# Training command with explicit hyperparameters
# You can modify these values directly here or pass them as arguments to this script if you modify it further.
# python train.py config/train_shakespeare.py \
#     --n_layer=6 \
#     --n_head=6 \
#     --n_embd=384 \
#     --block_size=256 \
#     --batch_size=64 \
#     --dropout=0.2 \
#     --learning_rate=1e-3 \
#     --max_iters=5000 \
#     --out_dir=out-shakespeare \
#     --compile=False
# python train.py config/train_gpt2_debug.py
python train.py config/train_diffusion_debug.py