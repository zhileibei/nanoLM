import matplotlib.pyplot as plt
import numpy as np
import wandb

def visualize_layer_attention(attentions, tokens=None, mask=None, num_layers_to_show=None):
    """
    Visualize attention weights across multiple layers.
    
    Args:
        attentions: (B, L, H, T, T) - batch, layers, heads, seq_len, seq_len
        tokens: Optional token IDs for labeling
        mask: Optional (B, T) boolean mask where True = masked
        num_layers_to_show: How many layers to visualize (None = all)
    """
    batch_size, num_layers, num_heads, seq_len, _ = attentions.shape
    
    # Take first sample from batch
    attentions = attentions[0]  # (L, H, T, T)
    
    if mask is not None:
        mask = mask[0]  # (T,)
    
    if num_layers_to_show is None:
        num_layers_to_show = num_layers
    
    # Average across heads for each layer
    layer_attns = attentions.mean(dim=1).cpu().numpy()  # (L, T, T)
    
    # Calculate layers to show (evenly spaced)
    if num_layers_to_show < num_layers:
        layer_indices = np.linspace(0, num_layers-1, num_layers_to_show, dtype=int)
    else:
        layer_indices = range(num_layers)
    
    # Create subplot grid
    n_cols = min(4, len(layer_indices))
    n_rows = (len(layer_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, layer_idx in enumerate(layer_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        attn_map = layer_attns[layer_idx]
        
        # Plot heatmap
        im = ax.imshow(attn_map, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add mask boundaries if provided
        if mask is not None:
            masked_indices = np.where(mask.cpu().numpy())[0]
            unmasked_indices = np.where(~mask.cpu().numpy())[0]
            
            if len(unmasked_indices) > 0 and len(masked_indices) > 0:
                boundary = len(unmasked_indices) - 0.5
                ax.axhline(y=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12)
        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
    
    # Remove extra subplots
    for idx in range(len(layer_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    return fig

def create_sorted_attention_heatmap(attn_matrix, mask, split_name):
    """
    Create a sorted attention heatmap where masked and unmasked tokens are grouped.
    
    Args:
        attn_matrix: (seq_len, seq_len) attention weights
        mask: (seq_len,) boolean array where True = masked
        split_name: 'train' or 'val'
    
    Returns:
        wandb.Image object for logging
    """
    seq_len = len(mask)
    
    # Get indices of masked and unmasked tokens
    masked_indices = np.where(mask)[0]
    unmasked_indices = np.where(~mask)[0]
    
    # Create sorted index: unmasked first, then masked
    sorted_indices = np.concatenate([unmasked_indices, masked_indices])
    
    # Reorder attention matrix
    sorted_attn = attn_matrix[sorted_indices][:, sorted_indices]
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot heatmap
    im = ax.imshow(sorted_attn, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add boundary line between masked and unmasked
    num_unmasked = len(unmasked_indices)
    if num_unmasked > 0 and num_unmasked < seq_len:
        ax.axhline(y=num_unmasked - 0.5, color='red', linestyle='--', linewidth=2, label='Mask boundary')
        ax.axvline(x=num_unmasked - 0.5, color='red', linestyle='--', linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Key Position (Unmasked | Masked)', fontsize=11)
    ax.set_ylabel('Query Position (Unmasked | Masked)', fontsize=11)
    ax.set_title(f'Attention Heatmap - {split_name.capitalize()} Split\n'
                 f'Unmasked: {num_unmasked}, Masked: {len(masked_indices)}', 
                 fontsize=12, pad=10)
    
    # Add text annotations for quadrants
    if num_unmasked > 0 and len(masked_indices) > 0:
        # Unmasked → Unmasked (top-left)
        ax.text(num_unmasked/2, num_unmasked/2, 'U→U', 
                ha='center', va='center', color='white', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Unmasked → Masked (top-right)
        ax.text(num_unmasked + len(masked_indices)/2, num_unmasked/2, 'U→M',
                ha='center', va='center', color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Masked → Unmasked (bottom-left)
        ax.text(num_unmasked/2, num_unmasked + len(masked_indices)/2, 'M→U',
                ha='center', va='center', color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Masked → Masked (bottom-right)
        ax.text(num_unmasked + len(masked_indices)/2, 
                num_unmasked + len(masked_indices)/2, 'M→M',
                ha='center', va='center', color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    
    # Convert to wandb Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    
    return wandb_image