"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    causal: bool = True
    time_conditioned: bool = False
    nonmask_only: bool = False # custom attention mask
    n_layer: int = 6
    n_head: int = 6 # number of query heads (GQA)
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 384 
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    context_len: int = 16  # Number of prefix tokens that are never masked

@dataclass
class DiffusionConfig(GPTConfig):
    vocab_size: int = 128  # Full ASCII (0-127), where 0 is reserved for mask
    mask_token_id: int = 0  # NUL character used as [MASK] token
    nonmask_only: bool = False # custom attention mask
    causal: bool = False  # non-causal attention for diffusion
    time_conditioned: bool = True  # time-conditioning for diffusion
    diffusion_steps: int = 128
    context_len: int = 16  # Number of prefix tokens that are never masked

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        # output projection
        self.layer_idx = layer_idx
        self.causal = config.causal
        self.nonmask_only = getattr(config, 'nonmask_only', False)
        self.mask_token_id = getattr(config, 'mask_token_id', None)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # # regularization
        # self.dropout = config.dropout
        # self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)
        # # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cos_sin, kv_cache, input_ids=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)
        
        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # if self.flash:
        #     # efficient attention using Flash Attention CUDA kernels
        #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # else:
        #     # manual implementation of attention
        #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #     att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #     att = F.softmax(att, dim=-1)
        #     att = self.attn_dropout(att)
        #     y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Attention: queries attend to key/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (QGA): duplicate key/value heads to match query heads
        
        # Create custom mask for nonmask_only mode
        custom_mask = None
        if self.nonmask_only:
            assert input_ids is not None and self.mask_token_id is not None
            # Create mask where True = attend, False = don't attend
            # Shape: (B, Tk) -> need to broadcast to (B, n_head, Tq, Tk)
            nonmask_positions = (input_ids != self.mask_token_id)  # (B, Tk)
            custom_mask = nonmask_positions.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Tk)
            custom_mask = custom_mask.expand(B, self.n_head, Tq, Tk)  # (B, n_head, Tq, Tk)
        if not self.causal:
            # Non-causal attention (e.g. for diffusion models)
            attn_mask = custom_mask if custom_mask is not None else None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=enable_gqa)
        elif kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cahce, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        # x = self.gelu(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        # x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config, layer_idx)
        # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, input_ids=None):
        # x = x + self.attn(self.ln_1(x))
        x = x + self.attn(norm(x), cos_sin, kv_cache, input_ids=input_ids)
        # x = x + self.mlp(self.ln_2(x))
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # assert config.vocab_size is not None
        # assert config.block_size is not None
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.time_conditioned:
            self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)
        # wpe = nn.Embedding(config.block_size, config.n_embd),
        # drop = nn.Dropout(config.dropout),
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        # ln_f = LayerNorm(config.n_embd, bias=config.bias),
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # # with weight tying when using torch.compile() some warnings get generated:
        # # "UserWarning: functional_call was passed multiple values for tied weights.
        # # This behavior is deprecated and will be an error in future versions"
        # # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        
    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # Zero out c_proj weights in all blocks
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        # if self.transformer.wte.weight.device.type == "cuda":
        #     self.transformer.wte.to(dtype=torch.bfloat16)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            # device = self.transformer.wte.weight.device
            device = self.token_emb.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq) # (T, head_dim // 2)
        cos, sin = freqs.cos(), freqs.sin()
        # cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin
    
    def get_device(self):
        # return self.transformer.wte.weight.device
        return self.token_emb.weight.device
    
    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        # nparams_embedding = self.transformer.wte.weight.numel()
        nparams_embedding = self.token_emb.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    def forward(self, idx, t=None, kv_cache=None):
        B, T = idx.size()
        
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        # assert self.cos.dtype == torch.bfloat16, f"Rotary embeddings mus be in bfloat16"
        
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.token_emb(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.time_conditioned:
            # time embedding and add to token embeddings
            assert t is not None, "time step t must be provided for time-conditioned model"
            t_emb = self.time_emb(t) # (B, n_embd)
            x = x + t_emb[:, None, :] # broadcast add to (B, T, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        x = norm(x)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache, input_ids=idx)
        # x = self.transformer.ln_f(x)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # soft cap to stabilize training
        logits = logits.float() # use tf32/fp32 for logits
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    @torch.inference_mode()
    def generate(self, ids, max_tokens, temperature=1.0, device=None, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        # assert isinstance(tokens, list)
        device = self.get_device() if device is None else device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            # append sampled index to the running sequence and continue
            ids = torch.cat((ids, next_ids), dim=1)
            # token = next_ids.item()
            # yield token

        return ids

    @torch.inference_mode()
    def sample_topk(
        self,
        batch_size,
        seq_len,
        k,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
        seed=42,
    ):
        """
        Generate samples using top-K parallel decoding (LLaDA baseline).
        At each step, decode exactly K tokens with highest confidence.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            k: Number of tokens to decode per step
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len  # Maximum possible steps
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens.to(device)

        # Track which positions are still masked
        masked_positions = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        if context_tokens is not None:
            masked_positions[:, :context_len] = False

        # Decode step by step
        for step in range(num_steps - 1, -1, -1):
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use step as proxy for timestep)
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Get confidence scores (max probability for each position)
            probs = F.softmax(logits / temperature, dim=-1)
            if temperature > 0:
                # Sample tokens stochastically
                B, T, V = probs.shape
                predicted_tokens = torch.multinomial(
                    probs.view(B * T, V), 
                    num_samples=1
                ).view(B, T)  # (B, T)

                # Get confidence (probability) of sampled tokens
                confidences = probs.gather(dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)
            else:
                confidences, predicted_tokens = torch.max(probs, dim=-1)  # (B, T)

            # Mask out already-decoded positions
            confidences = confidences.masked_fill(~masked_positions, -float("inf"))

            # Select top-K positions per batch
            k_actual = min(k, masked_positions.sum(dim=1).max().item())
            _, topk_indices = torch.topk(confidences, k=k_actual, dim=1)  # (B, K)

            # Update the top-K positions
            for b in range(batch_size):
                for idx in topk_indices[b]:
                    if masked_positions[b, idx]:
                        x[b, idx] = predicted_tokens[b, idx]
                        masked_positions[b, idx] = False

        return x

    @torch.inference_mode()
    def sample_confidence(
        self,
        batch_size,
        seq_len,
        confidence_threshold=0.95,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
        seed=42,
    ):
        """
        Generate samples using confidence-aware parallel decoding (Fast-dLLM).
        At each step, decode all tokens whose confidence exceeds a threshold.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            confidence_threshold: Threshold τ for token acceptance
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len  # Maximum possible steps
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens.to(device)

        # Track which positions are still masked
        masked_positions = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        if context_tokens is not None:
            masked_positions[:, :context_len] = False

        # Decode step by step
        for step in range(num_steps - 1, -1, -1):
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use step as proxy for timestep)
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Get confidence scores (max probability for each position)
            probs = F.softmax(logits / temperature, dim=-1)
            if temperature > 0:
                # Sample tokens stochastically
                B, T, V = probs.shape
                predicted_tokens = torch.multinomial(
                    probs.view(B * T, V), 
                    num_samples=1
                ).view(B, T)  # (B, T)

                # Get confidence (probability) of sampled tokens
                confidences = probs.gather(dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)
            else:
                confidences, predicted_tokens = torch.max(probs, dim=-1)  # (B, T)

            # Select positions above threshold (only among masked positions)
            if step < num_steps - 1:
                above_threshold = (confidences >= confidence_threshold) & masked_positions
            else:
                above_threshold = masked_positions

            # Ensure at least one token is decoded per batch if any remain masked
            for b in range(batch_size):
                if masked_positions[b].any() and not above_threshold[b].any():
                    # Decode the highest confidence masked token
                    masked_confidences = confidences[b].clone()
                    masked_confidences[~masked_positions[b]] = -float("inf")
                    best_idx = torch.argmax(masked_confidences)
                    above_threshold[b, best_idx] = True

            # Update positions above threshold
            x = torch.where(above_threshold, predicted_tokens, x)
            masked_positions = masked_positions & ~above_threshold

        return x

    @torch.inference_mode()
    def sample(
        self,
        batch_size,
        seq_len,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
        method="confidence",
        k=None,
        confidence_threshold=0.95,
    ):
        """
        Generate samples using parallel decoding methods.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
            method: Decoding method - 'topk' or 'confidence'
            k: Number of tokens per step (for 'topk' method)
            confidence_threshold: Confidence threshold τ (for 'confidence' method)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if method == "topk":
            if k is None:
                k = max(1, seq_len // 10)  # Default: decode 10% per step
            return self.sample_topk(
                batch_size, seq_len, k, num_steps, temperature, device, context_tokens
            )
        elif method == "confidence":
            return self.sample_confidence(
                batch_size,
                seq_len,
                confidence_threshold,
                num_steps,
                temperature,
                device,
                context_tokens,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")


def encode_text(text):
    """Convert text to vocab indices using direct ASCII mapping"""
    tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)
    return tokens


def decode_tokens(tokens):
    """Convert vocab indices to text using direct ASCII mapping"""
    text = "".join([chr(int(t)) for t in tokens])
    return text