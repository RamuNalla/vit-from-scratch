import torch
import pytest
from src.models.embeddings import PatchEmbedding, ViTEmbeddings
from src.models.attention import MultiHeadAttention
from src.models.block import TransformerEncoderBlock
from src.models.vit import VisionTransformer


def test_patch_embedding_shape():
    batch_size = 4
    in_channels = 3
    image_size = 32  # Standard CIFAR-100 dimension
    patch_size = 4
    emb_dim = 64
    
    
    dummy_img = torch.randn(batch_size, in_channels, image_size, image_size)
    
    patch_layer = PatchEmbedding(
        in_channels=in_channels, 
        patch_size=patch_size, 
        emb_dim=emb_dim, 
        image_size=image_size
    )
    
    output = patch_layer(dummy_img)
    
    # Expected number of patches: (32/4) * (32/4) = 8 * 8 = 64
    expected_num_patches = (image_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches, emb_dim)
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_vit_embeddings_shape():
    batch_size = 4
    in_channels = 3
    image_size = 32
    patch_size = 4
    emb_dim = 64
    
    dummy_img = torch.randn(batch_size, in_channels, image_size, image_size)
    
    vit_embed = ViTEmbeddings(
        in_channels=in_channels,
        patch_size=patch_size,
        emb_dim=emb_dim,
        image_size=image_size
    )
    
    output = vit_embed(dummy_img)
    
    # Expected sequence length: Num_Patches (64) + 1 ([CLS] token) = 65
    expected_num_patches = (image_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches + 1, emb_dim)
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

    def test_mha_shape():
        batch_size = 4
        num_patches_plus_cls = 65
        emb_dim = 64
        
        dummy_input = torch.randn(batch_size, num_patches_plus_cls, emb_dim)
        mha = MultiHeadAttention(emb_dim=emb_dim, num_heads=8)
        
        output = mha(dummy_input)
        assert output.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {output.shape}"