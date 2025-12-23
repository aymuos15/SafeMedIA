"""Training and evaluation logic for federated learning with differential privacy.

This module contains the core training and evaluation functions used by
federated clients during local training rounds.

NOTE: We use manual training loops instead of MONAI engines because:
1. Opacus (DP) wraps model+optimizer+dataloader together - incompatible with MONAI engines
2. MONAI's SupervisedTrainer causes SIGFPE with Opacus per-sample gradient computation
3. Simple loops are more debuggable for FL + DP use case

Loss functions are now configurable via the [loss] config section. Options:
- cross_entropy: Standard CrossEntropyLoss (default, most stable with DP)
- soft_dice: SoftDiceLoss using softmax probabilities (DP-compatible)
- dice_ce: Combined Dice + CrossEntropy (recommended for segmentation)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric
from loguru import logger

from dp_fedmed.losses.dice import get_loss_function
from dp_fedmed.utils import extract_batch_data


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    loss_config: Optional[Dict] = None,
) -> float:
    """Train model for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Loss function (overrides loss_config if provided)
        loss_config: Loss configuration dict (used if criterion is None)

    Returns:
        Average loss for the epoch
    """
    model.train()

    if criterion is None:
        if loss_config:
            criterion = get_loss_function(loss_config)
        else:
            # Default to CrossEntropyLoss for backward compatibility
            criterion = nn.CrossEntropyLoss(reduction="mean")

    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        # Handle both tuple and dict batch formats with validation
        batch_data = extract_batch_data(batch, device)
        if batch_data is None:
            continue
        images, labels = batch_data

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss detected: {loss.item()}, skipping batch")
            continue

        loss.backward()

        # Check if gradients were actually computed
        # Use getattr for Opacus-wrapped parameters which might use grad_sample
        has_grads = False
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is not None or getattr(p, "grad_sample", None) is not None:
                    has_grads = True
                    break

        if not has_grads:
            logger.warning("No gradients computed in backward pass!")
            continue

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
        for batch in test_loader:
            # Handle both tuple and dict batch formats with validation
            batch_data = extract_batch_data(batch, device)
            if batch_data is None:
                continue
            images, labels = batch_data

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
    if isinstance(dice_tensor, tuple):
        if len(dice_tensor) > 0:
            dice_tensor = dice_tensor[0]
        else:
            logger.warning("Empty dice metric tuple result")
            dice_tensor = torch.tensor(0.0)
    dice_score = dice_tensor.item() if dice_tensor.numel() > 0 else 0.0
    dice_metric.reset()

    return {
        "loss": avg_loss,
        "dice": dice_score,
    }
