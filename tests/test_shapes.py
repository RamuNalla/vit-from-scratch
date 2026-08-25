import torch
import pytest
from src.models.embeddings import PatchEmbedding, ViTEmbeddings

def test_patch_embedding_shape():
    batch_size = 4
    in_channels = 3
    image_size = 32  # Standard CIFAR-100 dimension
    patch_size = 4
    emb_dim = 64
    
    # Create dummy image tensor (e.g., CIFAR batch)
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