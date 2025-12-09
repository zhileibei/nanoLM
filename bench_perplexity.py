"""
Benchmark perplexity of generated samples using an external LLM.
Reads samples from sample.py output (JSON) and computes perplexity using a reference model.
"""
import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# default config values
input_file = 'samples.json'  # input JSON file with samples from sample.py
output_file = 'perplexity_results.json'  # output file for perplexity results
model_name = 'Qwen/Qwen2.5-3B'  # reference model for perplexity computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device to run on
batch_size = 1  # batch size for processing (set to 1 for varied length sequences)
max_length = 2048  # maximum sequence length
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Compute perplexity of generated samples')
    parser.add_argument('--input_file', type=str, nargs='+', default=[input_file],
                        help='Input JSON file(s) containing samples (can specify multiple)')
    parser.add_argument('--output_file', type=str, default=output_file,
                        help='Output JSON file for perplexity results (ignored when processing multiple files)')
    parser.add_argument('--model_name', type=str, default=model_name,
                        help='HuggingFace model name for perplexity computation')
    parser.add_argument('--device', type=str, default=device,
                        help='Device to run on (cuda, cpu, mps)')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=max_length,
                        help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default=dtype,
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Data type for model')
    return parser.parse_args()

def load_samples(input_file):
    """Load samples from JSON file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Input JSON must be a list of text samples")

    print(f"Loaded {len(samples)} samples from {input_file}")
    return samples

def compute_perplexity(text, model, tokenizer, device, max_length):
    """
    Compute perplexity of a text sample.
    Perplexity = exp(average negative log-likelihood)
    """
    # Tokenize input
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)

    # Handle empty or very short sequences
    if input_ids.size(1) < 2:
        return float('inf')

    # Compute loss (negative log-likelihood)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    # Perplexity = exp(loss)
    perplexity = np.exp(loss)

    return perplexity

def main():
    args = parse_args()

    print(f"Loading reference model: {args.model_name}")
    print(f"Device: {args.device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # Set pad token if not already set (required for models like Qwen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Set dtype
    torch_dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    torch_dtype = torch_dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != 'cpu' else None,
        trust_remote_code=True
    )

    if args.device == 'cpu':
        model = model.to('cpu')

    model.eval()

    print(f"Model loaded successfully on {args.device}")

    # Process each input file
    input_files = args.input_file if isinstance(args.input_file, list) else [args.input_file]

    for file_idx, input_file_path in enumerate(input_files):
        print(f"\n{'='*60}")
        print(f"Processing file {file_idx+1}/{len(input_files)}: {input_file_path}")
        print(f"{'='*60}")

        # Load samples
        samples = load_samples(input_file_path)

        # Compute perplexity for each sample
        perplexities = []
        print("\nComputing perplexity for each sample...")

        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                ppl = compute_perplexity(sample, model, tokenizer, args.device, args.max_length)
                perplexities.append(ppl)
                tqdm.write(f"Sample {i+1}: perplexity = {ppl:.2f}")
            except Exception as e:
                print(f"\nError processing sample {i+1}: {e}")
                perplexities.append(float('inf'))

        # Compute statistics
        valid_perplexities = [p for p in perplexities if not np.isinf(p)]

        if len(valid_perplexities) == 0:
            print("\nWarning: No valid perplexity scores computed!")
            avg_perplexity = float('inf')
            median_perplexity = float('inf')
            min_perplexity = float('inf')
            max_perplexity = float('inf')
        else:
            avg_perplexity = np.mean(valid_perplexities)
            median_perplexity = np.median(valid_perplexities)
            min_perplexity = np.min(valid_perplexities)
            max_perplexity = np.max(valid_perplexities)

        # Prepare results
        results = {
            'input_file': input_file_path,
            'model_name': args.model_name,
            'num_samples': len(samples),
            'perplexities': perplexities,
            'statistics': {
                'mean': float(avg_perplexity),
                'median': float(median_perplexity),
                'min': float(min_perplexity),
                'max': float(max_perplexity),
                'num_valid': len(valid_perplexities),
                'num_invalid': len(perplexities) - len(valid_perplexities)
            }
        }

        # Determine output file name
        if len(input_files) == 1:
            output_path = args.output_file
        else:
            # Auto-generate output filename from input filename
            input_dir = os.path.dirname(input_file_path)
            input_basename = os.path.basename(input_file_path)
            base_name = os.path.splitext(input_basename)[0]

            # Create perplexity subdirectory
            perplexity_dir = os.path.join(input_dir, 'perplexity')
            os.makedirs(perplexity_dir, exist_ok=True)

            output_path = os.path.join(perplexity_dir, input_basename)

        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print("Perplexity Results:")
        print(f"{'='*60}")
        print(f"Reference Model: {args.model_name}")
        print(f"Number of samples: {len(samples)}")
        print(f"Valid samples: {len(valid_perplexities)}")
        print(f"Average perplexity: {avg_perplexity:.2f}")
        print(f"Median perplexity: {median_perplexity:.2f}")
        print(f"Min perplexity: {min_perplexity:.2f}")
        print(f"Max perplexity: {max_perplexity:.2f}")
        print(f"\nResults saved to: {output_path}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()