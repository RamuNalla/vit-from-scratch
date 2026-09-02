import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.vit import VisionTransformer
from src.data.dataset import get_cifar100_loaders
from src.engine.trainer import train_one_epoch, evaluate