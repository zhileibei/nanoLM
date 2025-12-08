#!/usr/bin/env bash
set -euo pipefail

# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare"
# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare-diffusion"
# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare-diffusion-longrun"
# INPUT_DIR="/home/beizl42/projects/nanoLM/diffusion-10ksteps-notime"
INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare-10ksteps"
OUT_DIR="$INPUT_DIR/perplexity-tinker"

export TINKER_API_KEY=tml-qsuOJx8ByMWeObWa9bTEGE5ivgL0IPW9OCZd4kE2ShbYTzylSqvm6sGOAfJXtKKtBAAAA

mkdir -p "$OUT_DIR"

for input_file in "$INPUT_DIR"/*.json; do
    base=$(basename "$input_file")
    output_file="$OUT_DIR/$base"

    python bench_perplexity_tinker.py \
        --input_file "$input_file" \
        --output_file "$output_file"
done
