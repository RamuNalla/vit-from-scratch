

# ViT-From-Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular, and fully tested implementation of the **Vision Transformer (ViT)** architecture built entirely from scratch in PyTorch—omitting high-level framework abstractions to expose the raw underlying mechanics of patch extraction, multi-head self-attention, and transformer encoder blocks.

---

## Architecture

Unlike Convolutional Neural Networks (CNNs) that process images via local pixel neighborhoods (inductive bias), the Vision Transformer treats an image identically to a sentence of text:
1. **Patchification (`nn.Conv2d`):** Splits a $32 \times 32$ image into non-overlapping $4 \times 4$ patches and flattens them into spatial tokens.
2. **Tokenization & Positional Embedding:** Prepends a learnable `[CLS]` (classification) token and adds 1D learnable positional encodings to retain spatial awareness.
3. **Transformer Encoder Blocks:** Passes the sequence through a stack of Pre-Norm blocks featuring Multi-Head Self-Attention (MHSA) and Multi-Layer Perceptrons (MLP) with GELU activations.
4. **Classification Head:** Maps the final `[CLS]` token representation to target classes via a linear layer.


```
[Input Image: 3.32.32]
      │
      ▼ (Conv2d Patchify)
[Patches: 64 x 64] ──> + [CLS Token & Positional Encodings (65 x 64)]
      │
      ▼
[Transformer Encoder Blocks x Depth] (Pre-Norm + MHSA + MLP)
      │
      ▼
[Extract [CLS] Token Representation]
      │
      ▼
[Linear Classifier Head] ──> [CIFAR-100 Logits]
```
