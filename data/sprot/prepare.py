import os
import numpy as np
from collections import Counter

# Standard amino acids + special tokens
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Special tokens
SPECIAL_TOKENS = [
    '<PAD>',  # Padding token
    '<UNK>',  # Unknown amino acid
    '<CLS>',  # Classification/start token
    '<SEP>',  # Separator/end token
]

# Create vocabulary
vocab = SPECIAL_TOKENS + AMINO_ACIDS
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")

# Load FASTA file
input_file_path = '/home/beizl42/orcd/pool/datasets/uniprotkb/uniprot_sprot.fasta'

def parse_fasta(file_path):
    """Parse FASTA file and return list of sequences"""
    sequences = []
    current_seq = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Header line - save previous sequence if exists
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                # Sequence line
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences

print("Loading sequences...")
sequences = parse_fasta(input_file_path)
print(f"Loaded {len(sequences):,} protein sequences")

# Calculate statistics
seq_lengths = [len(seq) for seq in sequences]
print(f"Average sequence length: {np.mean(seq_lengths):.1f}")
print(f"Median sequence length: {np.median(seq_lengths):.1f}")
print(f"Max sequence length: {max(seq_lengths):,}")
print(f"Min sequence length: {min(seq_lengths):,}")

# Check for non-standard amino acids
all_chars = set(''.join(sequences))
non_standard = all_chars - set(AMINO_ACIDS)
if non_standard:
    print(f"\nNon-standard amino acids found: {non_standard}")

# Encode sequences with <CLS> and <SEP> tokens
def encode_sequence(seq):
    """Encode a protein sequence with special tokens"""
    tokens = [char_to_idx['<CLS>']]
    for char in seq.upper():
        if char in char_to_idx:
            tokens.append(char_to_idx[char])
        else:
            tokens.append(char_to_idx['<UNK>'])
    tokens.append(char_to_idx['<SEP>'])
    return tokens

print("\nEncoding sequences...")
all_encoded = []
for seq in sequences:
    all_encoded.extend(encode_sequence(seq))

print(f"Total tokens: {len(all_encoded):,}")

# Split into train/val (90/10)
n = len(all_encoded)
train_ids = all_encoded[:int(n*0.9)]
val_ids = all_encoded[int(n*0.9):]

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
output_dir = os.path.dirname(input_file_path)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_path = os.path.join(output_dir, 'train.bin')
val_path = os.path.join(output_dir, 'val.bin')
vocab_path = os.path.join(output_dir, 'vocab.txt')

train_ids.tofile(train_path)
val_ids.tofile(val_path)

# Save vocabulary
with open(vocab_path, 'w') as f:
    for idx, char in idx_to_char.items():
        f.write(f"{idx}\t{char}\n")

# Save meta.pkl for compatibility with training code
import pickle
meta = {
    'vocab_size': len(vocab),
    'itos': idx_to_char,
    'stoi': char_to_idx,
}
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"\nSaved to:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {vocab_path}")
print(f"  {meta_path}")

# Token distribution
token_counts = Counter(all_encoded)
print("\nTop 10 most common tokens:")
for token_idx, count in token_counts.most_common(10):
    print(f"  {idx_to_char[token_idx]}: {count:,} ({100*count/len(all_encoded):.2f}%)")