import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module as described in the original Transformer paper.
    """
    def __init__(self, emb_dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        # Ensure embedding dimension is divisible by the number of heads
        assert emb_dim % num_heads == 0, f"Embedding dimension ({emb_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = emb_dim // num_heads
        
        # Combined query, key, and value projections in a single linear layer for efficiency
        self.qkv_projection = nn.Linear(emb_dim, emb_dim * 3, bias=True)
        self.out_projection = nn.Linear(emb_dim, emb_dim, bias=True)
        
        self.dropout = dropout