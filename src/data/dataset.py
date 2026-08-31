import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 2):
    """
    Prepares training and validation DataLoaders for CIFAR-100 with heavy augmentations
    necessary for training Vision Transformers from scratch.
    """
    # Standard CIFAR-100 channel mean and std values
    cifar100_mean = (0.5071, 0.4865, 0.4409)
    cifar100_std = (0.2009, 0.1984, 0.2023)

    # Training transforms with aggressive spatial augmentations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # Validation transforms (no augmentation, only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # Download and load training dataset
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Download and load test dataset
    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader