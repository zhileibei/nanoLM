# train a diffusion model on shakespeare dataset - cluster version
# Cluster-optimized configuration

# Model type
model_type = 'diffusion'  # 'gpt2' or 'diffusion'

out_dir = 'out-shakespeare-diffusion'
eval_interval = 250  # keep frequent because we'll overfit
sample_interval = 250  # generate samples frequently to see progress
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-diffusion'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby Diffusion model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Diffusion-specific parameters
diffusion_steps = 128
context_len = 16  # Number of prefix tokens that are never masked
confidence_threshold = 0.95  # For confidence-based sampling

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# Cluster-specific settings
device = 'cuda'  # Use GPU
dtype = 'bfloat16'  # Use bfloat16 for H100/H200
compile = True  # Enable torch.compile for