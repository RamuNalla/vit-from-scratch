

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

```text
vit-from-scratch/
│
├── configs/
│   └── train_cifar100.yaml       # Hyperparameters (lr, batch size, epochs, etc.)
│
├── data/                         # CIFAR-100 dataset storage
│
├── docs/
│   └── assets/                   # Performance curves and training telemetry
│       └── training_performance.png
│
├── notebooks/
│   └── verification.ipynb        # Shape verification and sanity checks
│
├── src/
│   ├── data/
│   │   └── dataset.py            # Dataloader with aggressive data augmentations
│   ├── models/
│   │   ├── attention.py          # Multi-Head Self-Attention (MHSA)
│   │   ├── block.py              # Pre-Norm Transformer Encoder Block
│   │   ├── embeddings.py         # Patchification + [CLS] + PosEmbed
│   │   ├── mlp.py                # MLP / Feed-Forward Network
│   │   └── vit.py                # Full model assembly
│   ├── engine/
│   │   └── trainer.py            # Training and evaluation loops
│   └── utils/
│       └── logger.py             # Matplotlib metrics logger
│
├── tests/
│   └── test_shapes.py            # Automated Pytest tensor dimension assertions
│
├── requirements.txt
└── README.md
```

## Installation & Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/RamuNalla/vit-from-scratch.git
   cd vit-from-scratch
   ```

2. **Set up the environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install torch torchvision tqdm matplotlib pyyaml pytest
   ```

3. **Configure hyperparameters**

   Edit [configs/train_cifar100.yaml](configs/train_cifar100.yaml) to adjust model architecture, batch size, learning rate, epochs, etc. The CIFAR-100 dataset is downloaded automatically into `data/` on first run.

4. **Train the model**

   ```bash
   python -m src.train --config configs/train_cifar100.yaml
   ```

   Training progress (loss/accuracy per epoch) is printed to the console, the best checkpoint is saved to `best_vit_cifar100.pth`, and the full history is written to `docs/assets/training_history.json`.

5. **Run the test suite**

   ```bash
   pytest tests/
   ```

---

## Training Performance & Analysis

Training Vision Transformers from scratch on smaller datasets like CIFAR-100 presents unique challenges due to the absence of convolutional inductive biases. Without massive pre-training datasets (like ImageNet-21k), ViTs easily overfit and require careful tuning of data augmentations, weight decay, and label smoothing.

![Training Performance](docs/assets/training_performance.png)

Generated performance telemetry capturing Cross-Entropy Loss and Accuracy over epochs, for both the training and validation sets.

Key Engineering Highlights
Modular Design: Completely decoupled components (embeddings, attention, mlp, block, vit) mirroring production-grade deep learning standards.
Optimized Attention: Leverages PyTorch's native scaled dot-product routines for compute-efficient multi-head processing.
Test-Driven Rigor: Comprehensive unit-testing suite (pytest) verifying layer-by-layer tensor constraints.