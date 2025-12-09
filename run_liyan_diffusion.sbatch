#!/bin/bash
#SBATCH --job-name=dl-project-liyan-diffusion
#SBATCH --output=results/output_%A_%a.txt
#SBATCH --error=results/error_%A_%a.txt
#SBATCH -c 10
#SBATCH -t 6:00:00
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
set -e

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup_env.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# -----------------------------
# Weights & Biases API key
# -----------------------------
export WANDB_API_KEY="794c522fe284c988c6504096077f83c779dc3af0"
export WANDB_START_METHOD="thread"

# Tinker API key
export TINKER_API_KEY=tml-qsuOJx8ByMWeObWa9bTEGE5ivgL0IPW9OCZd4kE2ShbYTzylSqvm6sGOAfJXtKKtBAAAA


# Run training for all permutation configurations
for config in config/train_diffusion_random_order_{1,2,4,8,16}.py; do
    echo "Running training with config: $config"
    python train.py "$config"
done

# Run benchmarking and perplexity curves for each output directory
for num_perm in 1 2 4 8 16; do
    INPUT_DIR="out-diffusion-random-order-${num_perm}perm-10ksteps"

    echo "Processing results for $INPUT_DIR"

    # Process all JSON files in one run (loads model only once)
    python bench_perplexity.py --input_file $INPUT_DIR/*.json

    python perplexity_curve.py --out_dir="$INPUT_DIR/perplexity"
done
