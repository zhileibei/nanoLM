# train an auto-regressive model on shakespeare dataset, but with data loader modified.

# Model type
model_type = 'gpt2'  # 'gpt2' or 'diffusion'

# Data loading
data_permuted = True  # Use random order data loader (get_batch_random_order)
num_permutations = 16

out_dir = 'out-shakespeare-random-order-5ksteps'
eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 200
sample_interval = 100  # generate samples frequently to see progress
sample_batch_size = 100
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt-random-order'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters
context_len = 16  # Number of prefix tokens that are never masked

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 500  # not super necessary potentially

# Cluster-specific settings
# device = 'cuda'  # Use GPU
# dtype = 'bfloat16'  # Use bfloat16 for H100/H200
# compile = True  # Enable torch.compile for
