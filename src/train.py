import argparse
import yaml
import torch
import torch.nn as nn
import json
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.vit import VisionTransformer
from src.data.dataset import get_cifar100_loaders
from src.engine.trainer import train_one_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer from scratch on CIFAR-100")
    parser.add_argument("--config", type=str, default="configs/train_cifar100.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup device
    device_str = config["training"]["device"]
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str in ("cuda", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loaders
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )

    # Instantiate Model
    print("Initializing Vision Transformer model...")
    model = VisionTransformer(
        in_channels=config["model"]["in_channels"],
        image_size=config["model"]["image_size"],
        patch_size=config["model"]["patch_size"],
        emb_dim=config["model"]["emb_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        expansion_factor=config["model"]["expansion_factor"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=config["training"]["weight_decay"]
    )
    
    epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training for {epochs} epochs...")
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vit_cifar100.pth")
            print(f"--> Saved new best model checkpoint with Val Acc: {val_acc:.2f}%")

    # Save training history to json for plotting
    os.makedirs("docs/assets", exist_ok=True)
    with open("docs/assets/training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"\nTraining completed! Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()