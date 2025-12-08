#!/usr/bin/env bash
set -euo pipefail

# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare"
# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare-diffusion"
# INPUT_DIR="/home/beizl42/projects/nanoLM/out-shakespeare-diffusion-longrun"
INPUT_DIR="/home/beizl42/projects/nanoLM/diffusion-25ksteps-notime"
OUT_DIR="$INPUT_DIR/perplexity"


mkdir -p "$OUT_DIR"

for input_file in "$INPUT_DIR"/*.json; do
    base=$(basename "$input_file")
    output_file="$OUT_DIR/$base"

    python bench_perplexity.py \
        --input_file "$input_file" \
        --output_file "$output_file"
done
