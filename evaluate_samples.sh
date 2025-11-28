#!/bin/bash
# Example workflow: Generate samples and evaluate perplexity
set -e

# Default values
OUT_DIR="out-shakespeare-char"
NUM_SAMPLES=10
MAX_TOKENS=500
SAMPLES_FILE="samples.json"
PERPLEXITY_FILE="perplexity_results.json"
MODEL_NAME="Qwen/Qwen2.5-3B"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--out_dir DIR] [--num_samples N] [--max_tokens N] [--model_name MODEL]"
            exit 1
            ;;
    esac
done

echo "====================================="
echo "Sample Generation & Perplexity Evaluation"
echo "====================================="
echo "Output directory: $OUT_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Max tokens per sample: $MAX_TOKENS"
echo "Perplexity model: $MODEL_NAME"
echo "====================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
    echo "Error: No checkpoint found at $OUT_DIR/ckpt.pt"
    echo "Please train a model first or specify a different output directory."
    exit 1
fi

# Step 1: Generate samples
echo "Step 1: Generating samples..."
python sample.py \
    --out_dir="$OUT_DIR" \
    --num_samples="$NUM_SAMPLES" \
    --max_new_tokens="$MAX_TOKENS" \
    > "$SAMPLES_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Samples generated and saved to $SAMPLES_FILE"
else
    echo "✗ Error generating samples"
    exit 1
fi

echo ""

# Step 2: Compute perplexity
echo "Step 2: Computing perplexity scores..."
python bench_perplexity.py \
    --input_file="$SAMPLES_FILE" \
    --output_file="$PERPLEXITY_FILE" \
    --model_name="$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Perplexity evaluation complete!"
    echo "✓ Results saved to $PERPLEXITY_FILE"
else
    echo "✗ Error computing perplexity"
    exit 1
fi

echo ""
echo "====================================="
echo "Evaluation complete!"
echo "Samples: $SAMPLES_FILE"
echo "Results: $PERPLEXITY_FILE"
echo "====================================="
