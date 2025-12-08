"""
Benchmark perplexity of generated samples using Tinker API.
Reads samples from sample.py output (JSON) and computes perplexity using Tinker's reference model.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import tinker
from tinker import types

# -----------------------------------------------------------------------------
# default config values
input_file = 'samples.json'  # input JSON file with samples from sample.py
output_file = 'perplexity_results.json'  # output file for perplexity results
api_key = None  # Tinker API key (can be set via environment variable)
model_name = 'Qwen/Qwen3-8B-Base'  # reference model for perplexity computation
batch_size = 1  # batch size for processing
max_length = 2048  # maximum sequence length
# -----------------------------------------------------------------------------

# Supported models by Tinker:
# - deepseek-ai/DeepSeek-V3.1
# - deepseek-ai/DeepSeek-V3.1-Base
# - meta-llama/Llama-3.1-70B
# - meta-llama/Llama-3.1-8B
# - meta-llama/Llama-3.1-8B-Instruct
# - meta-llama/Llama-3.2-1B
# - meta-llama/Llama-3.2-3B
# - meta-llama/Llama-3.3-70B-Instruct
# - Qwen/Qwen3-235B-A22B-Instruct-2507
# - Qwen/Qwen3-30B-A3B
# - Qwen/Qwen3-30B-A3B-Base
# - Qwen/Qwen3-30B-A3B-Instruct-2507
# - Qwen/Qwen3-32B
# - Qwen/Qwen3-4B-Instruct-2507
# - Qwen/Qwen3-8B
# - Qwen/Qwen3-8B-Base
# - openai/gpt-oss-120b
# - openai/gpt-oss-20b

def parse_args():
    parser = argparse.ArgumentParser(description='Compute perplexity of generated samples using Tinker API')
    parser.add_argument('--input_file', type=str, default=input_file,
                        help='Input JSON file containing samples')
    parser.add_argument('--output_file', type=str, default=output_file,
                        help='Output JSON file for perplexity results')
    parser.add_argument('--model_name', type=str, default=model_name,
                        help='Model name for perplexity computation')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=max_length,
                        help='Maximum sequence length')
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

def compute_perplexity(text, sampling_client, tokenizer):
    """
    Compute perplexity of a text sample using Tinker API.
    Perplexity = exp(average negative log-likelihood)

    Args:
        text: Input text to evaluate
        sampling_client: Tinker API sampling client

    Returns:
        perplexity: Perplexity score (float)
    """
    # Handle empty or very short sequences
    if len(text.strip()) < 2:
        return float('inf')

    prompt = types.ModelInput.from_ints(tokenizer.encode(text))
    logprobs = sampling_client.compute_logprobs(prompt).result()

    # logprobs format: [None, -9.54505, -1.64629, -8.81116, -3.50217, -8.25927, ...]
    # First element is None (no prediction for first token), rest are negative log-likelihoods

    # Filter out None values and extract valid negative log-likelihoods
    valid_logprobs = [lp for lp in logprobs if lp is not None]

    # Handle empty or invalid logprobs
    if not valid_logprobs or len(valid_logprobs) == 0:
        return float('inf')

    # Compute average negative log-likelihood
    avg_neg_log_likelihood = - np.mean(valid_logprobs)

    # Perplexity = exp(average negative log-likelihood)
    perplexity = np.exp(avg_neg_log_likelihood)

    return perplexity

def main():
    args = parse_args()

    print(f"Initializing Tinker API client")
    print(f"Model: {args.model_name}")

    # Initialize Tinker client
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = service_client.create_sampling_client(
        base_model=model_name
    )

    print(f"Tinker client initialized successfully")

    # Load samples
    samples = load_samples(args.input_file)

    # Compute perplexity for each sample
    perplexities = []
    print("\nComputing perplexity for each sample...")

    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        try:
            ppl = compute_perplexity(sample, sampling_client, tokenizer)
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

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
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
    print(f"\nResults saved to: {args.output_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
