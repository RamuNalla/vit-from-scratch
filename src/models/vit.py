import torch
import torch.nn as nn
from src.models.embeddings import ViTEmbeddings
from src.models.block import TransformerEncoderBlock

class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) Architecture from scratch.
    """
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        emb_dim: int = 64,
        depth: int = 6,              # Number of transformer blocks
        num_heads: int = 8,
        expansion_factor: int = 4,
        num_classes: int = 100,      # CIFAR-100 default
        dropout: float = 0.1
    ):
        super().__init__()

                # 1. Embeddings (Patches + [CLS] Token + Positional Encoding)
        self.embeddings = ViTEmbeddings(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            image_size=image_size,
            dropout=dropout
        )