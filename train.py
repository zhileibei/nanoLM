"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
import pickle
import tiktoken
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, Transformer, DiffusionConfig, encode_text, decode_tokens

# set random seed
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
sample_batch_size = 1
block_size = 1024
data_permuted = False # if True, use get_batch_random_order instead of get_batch
num_permutations = 16 # Number of permutations for permuted get_batch. Only useful when data_permuted=True.
# model type
model_type = 'gpt2' # 'gpt2' or 'diffusion'
# diffusion-specific parameters (only used when model_type='diffusion')
diffusion_steps = 128
context_len = 16
sample_interval = 250 # how often to generate samples during diffusion training
sample_iter_list = [2**i-1 for i in range(9)] # non-linear sample interval at the beginning
confidence_threshold = 0.95 # for diffusion sampling
time_conditioned = True if model_type == 'diffusion' else False
predict_timestep = False
nonmask_only = False # diffusion attention mask
mask_token_id = 50257
arlike_mask = False   # if True, use AR-like left-to-right masking schedule during training
ar_sampling = False   # if True, use AR left-to-right decoding during sampling
diff_casual = False  # if True, use causal + nonmask_only combined mask for diffusion
causal = False        # default causal flag for diffusion attention (can be overridden)
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# device = 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# MaskedDiffusionSchedule for discrete diffusion (only used when model_type='diffusion')
# -----------------------------------------------------------------------------
class MaskedDiffusionSchedule:
    """
    Masked diffusion schedule for discrete diffusion.

    Two modes:
    - arlike_mask == False:
        Original behavior. At each timestep t, randomly mask tokens with probability p_t.
    - arlike_mask == True:
        AR-like behavior. At each timestep t, mask a right-side suffix so that the
        left prefix is "known context" and the right suffix is "future" (all [MASK]).
    """
    def __init__(self, num_timesteps, mask_token_id, context_len=0, arlike_mask=False):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        # self.context_len = context_len
        self.arlike_mask = arlike_mask

        # Linear schedule: probability of masking increases linearly.
        # This is only used in the random masking mode.
        self.mask_probs = torch.linspace(1.0 / num_timesteps, 1.0, num_timesteps)

    def add_masks(self, x_0, t):
        """
        Add masks to tokens x_0 at timestep t.

        Args:
            x_0: Clean tokens, shape (B, T)
            t:   Timestep indices, shape (B,)

        Returns:
            x_t: Masked tokens at timestep t, shape (B, T)
            mask: Boolean tensor, True at positions that are masked, shape (B, T)
        """
        B, T = x_0.shape
        device = x_0.device

        # -----------------------------
        # Mode 1: original random masking
        # -----------------------------
        if not self.arlike_mask:
            # Get masking probability for each sample according to its timestep.
            mask_prob = self.mask_probs[t.cpu()].to(device)  # (B,)

            # Sample random mask positions independently for each token.
            mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)  # (B, T)

            # Ensure at least one masked position per sample to avoid NaNs in the loss.
            no_mask_rows = (mask.sum(dim=1) == 0)  # (B,)
            if no_mask_rows.any():
                rand_idx = torch.randint(0, T, (no_mask_rows.sum(),), device=device)
                mask[no_mask_rows, :] = False
                mask[no_mask_rows, rand_idx] = True

            # If you want to never mask the first context_len tokens, uncomment:
            # if self.context_len > 0:
            #     mask[:, : self.context_len] = False

            x_t = torch.where(mask, self.mask_token_id, x_0)
            return x_t, mask

        # -----------------------------
        # Mode 2: AR-like left-to-right masking
        # -----------------------------
        # Here t controls how long the unknown "future" suffix is.
        # Larger t -> longer masked suffix -> less visible context.
        t_float = t.float()
        if self.num_timesteps > 1:
            frac_future = t_float / (self.num_timesteps - 1)  # in [0, 1]
        else:
            frac_future = torch.zeros_like(t_float, device=device)

        # future_len[b] is how many tokens on the right are considered "future" for sample b.
        future_len = (frac_future * T).long().clamp(0, T)  # (B,)

        mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        for b in range(B):
            f = int(future_len[b].item())  # length of masked suffix on the right
            visible_len = T - f            # number of visible tokens on the left

            # Always keep at least context_len tokens visible at the start if desired.
            # visible_len = max(visible_len, self.context_len)

            if visible_len < T:
                # Mask the right-side suffix [visible_len, T).
                mask[b, visible_len:] = True

        # Again ensure at least one mask per sample.
        no_mask_rows = (mask.sum(dim=1) == 0)
        if no_mask_rows.any():
            rand_idx = torch.randint(0, T, (no_mask_rows.sum(),), device=device)
            mask[no_mask_rows, :] = False
            mask[no_mask_rows, rand_idx] = True

        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if model_type == 'gpt2':
        return x, y
    elif model_type == 'diffusion':
        return x, x
    return x, y

def load_get_batch_random_order():
    # parameters: num_permutations
    permutation_list = []
    # permutation_list.append([i for i in range(block_size+1)])
    for index in range(num_permutations):
        base_list = torch.arange(block_size, dtype=torch.float32)
        base_list = base_list + torch.randn(block_size) * index
        permutation = torch.argsort(base_list)
        permutation = torch.cat((torch.tensor([0]), permutation+1))
        permutation_list.append(permutation)
    permutation_tensor = torch.stack(permutation_list)  # Shape: (num_permutations, block_size+1)

    def get_batch_random_order(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        segments = torch.stack([torch.from_numpy((data[i:i+block_size+1]).astype(np.int64)) for i in ix])
        # permutation_indices = torch.stack([permutation_list[torch.randint(num_permutations, (1,))] for _ in range(block_size)])
        perm_idx = torch.randint(0, num_permutations, (batch_size,))
        permutation_indices = permutation_tensor[perm_idx]
        segments_permuted = torch.gather(segments, dim=1, index=permutation_indices)
        x = segments_permuted[:, :-1]
        y = segments_permuted[:, 1:]
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        if model_type == 'gpt2':
            return x, y
        elif model_type == 'diffusion':
            return x, x
        return x, y
    
    return get_batch_random_order

# Select get_batch implementation based on data_permuted flag
if data_permuted:
    get_batch = load_get_batch_random_order()

# def get_random_context(dataset_tokens, context_len, batch_size=1):
def get_random_context(context_len, batch_size=1):
    """Get random context tokens from dataset (for sampling)"""
    dataset_tokens = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    max_start = len(dataset_tokens) - context_len
    start_indices = torch.randint(0, max_start, (batch_size,))
    context_tokens = torch.stack(
        [torch.from_numpy(dataset_tokens[start:start+context_len].astype(np.int64)) for start in start_indices]
    )
    return context_tokens.to(device)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    MASK_TOKEN_ID = mask_token_id
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode([i for i in l if i != MASK_TOKEN_ID])

# model init
model_args = dict(n_layer=n_layer, 
                  n_head=n_head, 
                  n_embd=n_embd, 
                  sequence_len=block_size,
                  bias=bias, 
                  vocab_size=None, 
                  dropout=dropout,
                  context_len=context_len,
                  model_type=model_type) # start with model_args from command line
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50258

if init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = Transformer.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'sequence_len', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    model_args['model_type'] = 'gpt2'
elif init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    if model_type == 'diffusion':
        # Diffusion model
        # model_args['vocab_size'] = 128
        model_args['diffusion_steps'] = diffusion_steps
        model_args['time_conditioned'] = time_conditioned
        model_args['predict_timestep'] = predict_timestep
        model_args['nonmask_only'] = nonmask_only
        model_args['mask_token_id'] = mask_token_id
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    keys_to_match = ['n_layer', 'n_head', 'n_embd', 'sequence_len', 'bias', 'vocab_size']
    if 'model_type' in checkpoint_model_args:
        model_type = checkpoint_model_args['model_type']
        if model_type == 'diffusion':
            keys_to_match += ['diffusion_steps']
    for k in keys_to_match:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
# create the model
if model_type == 'diffusion':
    diffusion_config = DiffusionConfig(
        sequence_len=model_args['sequence_len'],
        vocab_size=model_args['vocab_size'],
        mask_token_id=model_args['mask_token_id'],
        causal=causal, ###Change back
        nonmask_only=model_args['nonmask_only'],
        time_conditioned=model_args['time_conditioned'],
        predict_timestep=model_args['predict_timestep'],
        diffusion_steps=model_args.get('diffusion_steps', diffusion_steps),
        context_len=model_args.get('context_len', context_len),
        n_layer=model_args['n_layer'],
        n_head=model_args['n_head'],
        n_kv_head=model_args['n_head'],
        n_embd=model_args['n_embd'],
        dropout=dropout,
        bias=model_args['bias'],
        diff_casual=diff_casual
    )
    model = Transformer(diffusion_config)
else:
    gptconf = GPTConfig(
        sequence_len=model_args['sequence_len'],
        vocab_size=model_args['vocab_size'],
        causal=True,
        time_conditioned=False,
        predict_timestep=False,
        nonmask_only=False,
        n_layer=model_args['n_layer'],
        n_head=model_args['n_head'],
        n_kv_head=model_args['n_head'],
        n_embd=model_args['n_embd'],
        dropout=dropout,
        bias=model_args['bias'],
        diff_casual=diff_casual
    )
    model = Transformer(gptconf)


# Initialize weights and move to device
if init_from == 'resume':
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    model.init_weights()
    iter_num = 0
    best_val_loss = 1e9
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------------

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if model_type == 'gpt2':
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1, reduction='mean')
                elif model_type == 'diffusion':
                    # Sample random timesteps
                    t = torch.randint(0, mask_schedule.num_timesteps, (batch_size,), device=device)
                    X_t, mask = mask_schedule.add_masks(X, t)
                    # mask = X_t == mask_schedule.mask_token_id # (batch_size, seq_len)
                    if predict_timestep:
                        logits, timestep_logits = model(X_t, t=t)
                    else:
                        logits = model(X_t, t=t)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
                    loss = (loss.view(batch_size, -1) * mask).sum() / mask.sum()
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def evaluate_timestep_prediction():
    """Evaluate timestep prediction accuracy on train and val sets"""
    model.eval()
    out = {}
    
    for split in ['train', 'val']:
        accuracies = torch.zeros(eval_iters)
        mae_errors = torch.zeros(eval_iters)  # Mean Absolute Error
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                # Sample random timesteps
                t = torch.randint(0, mask_schedule.num_timesteps, (batch_size,), device=device)
                X_t, mask = mask_schedule.add_masks(X, t)
                
                # Get predictions
                logits, timestep_logits = model(X_t, t=t)
                
                # Timestep prediction accuracy
                timestep_pred = timestep_logits.argmax(dim=-1)
                accuracy = (timestep_pred == t).float().mean()
                accuracies[k] = accuracy.item()
                
                # Mean absolute error in timestep prediction
                mae = (timestep_pred - t).abs().float().mean()
                mae_errors[k] = mae.item()
        
        out[f'{split}_timestep_acc'] = accuracies.mean()
        out[f'{split}_timestep_mae'] = mae_errors.mean()
    
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Setup for diffusion training (if model_type == 'diffusion')
# -----------------------------------------------------------------------------
if model_type == 'diffusion':
    print("Setting up for diffusion training...")
    # Create masked diffusion schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=model.config.diffusion_steps,
        mask_token_id=model.config.mask_token_id,
        # context_len=model.config.context_len,
        arlike_mask=arlike_mask,
    )
    print("Diffusion setup complete!")

# training loop
print(f"Starting {model_type} training loop...")
# X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
# local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
# running_mfu = -1.0  # commented out MFU tracking
def should_sample(iter_num):
    return (iter_num < 256 and iter_num in sample_iter_list) or (iter_num >=256 and iter_num % sample_interval == 0)
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Sample generation (diffusion-specific)
    if iter_num != 0 and should_sample(iter_num) and master_process:
        model.eval()
        with torch.no_grad():
            print(f"\n--- Sample at iter {iter_num} ---")
            # Get random context if context_len > 0
            context_tokens = None
            if raw_model.config.context_len > 0:
                context_tokens = get_random_context(
                    raw_model.config.context_len, batch_size=sample_batch_size
                )
            print(f"\n--- Context ---")
            for i in range(sample_batch_size):
                context = decode(context_tokens[i].tolist())
                print(context)
            if model_type == 'diffusion':
                # Choose decoding method based on ar_sampling flag
                decode_method = "ar" if ar_sampling else "confidence"
                samples = raw_model.sample(
                    batch_size=sample_batch_size,
                    seq_len=raw_model.config.sequence_len,
                    num_steps=None,
                    temperature=1.0,
                    device=raw_model.get_device(),
                    context_tokens=context_tokens,
                    method=decode_method,
                    confidence_threshold=confidence_threshold,
                )
            elif model_type == 'gpt2':
                samples = raw_model.generate(
                    ids=context_tokens,
                    max_tokens=raw_model.config.sequence_len - raw_model.config.context_len,
                    temperature=1.0,
                    device=raw_model.get_device(),
                )
            # Decode samples to text
            # text = decode_tokens(samples[0]) # ASCII
            seqs = []
            for i in range(sample_batch_size):
                text = decode(samples[i].tolist())
                print(f"\n--- Generated ---")
                print(text)
                print("--- End sample ---\n")
                seqs.append(text)
            # TODO evaluate samples with LLM perplexity
            with open(os.path.join(out_dir, f'sample_step{iter_num}.json'), 'w') as f:
                json.dump(seqs, f, indent=4)
        model.train()
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_dict = {
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
        }
        if predict_timestep:
            timestep_metrics = evaluate_timestep_prediction()
            log_dict.update({
                "train/timestep_acc": timestep_metrics['train_timestep_acc'],
                "train/timestep_mae": timestep_metrics['train_timestep_mae'],
                "val/timestep_acc": timestep_metrics['val_timestep_acc'],
                "val/timestep_mae": timestep_metrics['val_timestep_mae'],
            })
            print(f"  train timestep acc: {timestep_metrics['train_timestep_acc']:.4f}, mae: {timestep_metrics['train_timestep_mae']:.2f}")
            print(f"  val timestep acc: {timestep_metrics['val_timestep_acc']:.4f}, mae: {timestep_metrics['val_timestep_mae']:.2f}")
        if wandb_log:
            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        X, Y = get_batch('train')
        with ctx:
            if model_type == 'gpt2':
                logits = model(X) # (batch_size, seq_len, vocab_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1, reduction='mean')
            elif model_type == 'diffusion':
                # Sample random timesteps
                t = torch.randint(0, mask_schedule.num_timesteps, (batch_size,), device=device)
                X_t, mask = mask_schedule.add_masks(X, t)
                # mask = X_t == mask_schedule.mask_token_id # (batch_size, seq_len)
                if predict_timestep:
                    logits, timestep_logits = model(X_t, t=t)
                    # Token prediction loss (only on masked positions)
                    token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
                    token_loss = (token_loss.view(batch_size, -1) * mask).sum() / mask.sum()
                    
                    # Timestep prediction loss
                    timestep_loss = F.cross_entropy(timestep_logits, t)
                    
                    # Combined loss
                    timestep_loss_weight = 0.1
                    loss = token_loss + timestep_loss_weight * timestep_loss
                else:
                    logits = model(X_t, t=t)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
                    loss = (loss.view(batch_size, -1) * mask).sum() / mask.sum()
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    # local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
