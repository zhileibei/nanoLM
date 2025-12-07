#!/bin/bash
#SBATCH --job-name=dl-project-liyan
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

# -----------------------------
# Weights & Biases API key
# -----------------------------
export WANDB_API_KEY="794c522fe284c988c6504096077f83c779dc3af0"
export WANDB_START_METHOD="thread"

# Tinker API key
export TINKER_API_KEY=tml-qsuOJx8ByMWeObWa9bTEGE5ivgL0IPW9OCZd4kE2ShbYTzylSqvm6sGOAfJXtKKtBAAAA

python train.py config/train_shakespeare.py

INPUT_DIR="out-shakespeare-100ksteps"
OUT_DIR="$INPUT_DIR/perplexity"

mkdir -p "$OUT_DIR"

for input_file in "$INPUT_DIR"/*.json; do
    base=$(basename "$input_file")
    output_file="$OUT_DIR/$base"

    python bench_perplexity.py \
        --input_file "$input_file" \
        --output_file "$output_file"
done


python perplexity_curve.py --out_dir=$OUT_DIR