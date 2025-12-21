"""Training and evaluation logic for federated learning with differential privacy.

This module contains the core training and evaluation functions used by
federated clients during local training rounds.

NOTE: DiceLoss is NOT used during training because it causes SIGFPE (floating point
exceptions) with Opacus due to division operations in per-sample gradient computation.
We use CrossEntropyLoss only for training, and DiceMetric only for evaluation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> float:
    """Train model for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Loss function (default: CrossEntropy only for DP stability)

    Returns:
        Average loss for the epoch
    """
    model.train()

    if criterion is None:
        # Use only CrossEntropyLoss for DP training
        # DiceLoss causes SIGFPE with Opacus due to division operations in per-sample gradients
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # CrossEntropyLoss expects: outputs=[B, C, H, W], labels=[B, H, W]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics (dice, loss)
    """
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            # Compute dice metric
            # Get predictions: [B, C, H, W] -> [B, 1, H, W]
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            # Add channel dimension to labels: [B, H, W] -> [B, 1, H, W]
            labels_with_channel = labels.unsqueeze(1)
            dice_metric(preds, labels_with_channel)

    avg_loss = total_loss / max(num_batches, 1)

    # Aggregate dice scores - returns a Tensor
    dice_tensor = dice_metric.aggregate()
    dice_score = dice_tensor.item() if dice_tensor.numel() > 0 else 0.0  # type: ignore
    dice_metric.reset()

    return {
        "loss": avg_loss,
        "dice": dice_score,
    }
