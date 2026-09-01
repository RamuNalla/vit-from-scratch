import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping to stabilize ViT training from scratch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
