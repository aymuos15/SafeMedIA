"""Training and evaluation logic for federated learning with differential privacy.

This module contains the core training and evaluation functions used by
federated clients during local training rounds.

NOTE: We use manual training loops instead of MONAI engines because:
1. Opacus (DP) wraps model+optimizer+dataloader together - incompatible with MONAI engines
2. MONAI's SupervisedTrainer causes SIGFPE with Opacus per-sample gradient computation
3. Simple loops are more debuggable for FL + DP use case

Checkpointing is handled via the checkpoint module for both best and last models.

Loss functions are now configurable via the [loss] config section. Options:
- cross_entropy: Standard CrossEntropyLoss (default, most stable with DP)
- soft_dice: SoftDiceLoss using softmax probabilities (DP-compatible)
- dice_ce: Combined Dice + CrossEntropy (recommended for segmentation)
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric
from loguru import logger

from dp_fedmed.losses.dice import get_loss_function
from .checkpoint import save_model_checkpoint


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    loss_config: Optional[Dict] = None,
    checkpoint_dir: Optional[Path] = None,
) -> float:
    """Train model for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Loss function (overrides loss_config if provided)
        loss_config: Loss configuration dict (used if criterion is None)
        checkpoint_dir: Optional directory for saving checkpoints

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

    for batch_idx, batch in enumerate(train_loader):
        # Handle both tuple and dict batch formats
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

        # Ensure labels are [B, H, W] and long type for standard losses
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        labels = labels.long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
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

    # Save checkpoint after training epoch
    if checkpoint_dir:
        _save_training_checkpoint(model, checkpoint_dir)

    return avg_loss


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        checkpoint_dir: Optional directory for saving best/last model

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
            # Handle both tuple and dict batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)

            # Compute loss
            # Ensure labels are [B, H, W] and long type
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            labels = labels.long()

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
        dice_tensor = dice_tensor[0]
    dice_score = dice_tensor.item() if dice_tensor.numel() > 0 else 0.0
    dice_metric.reset()

    # Save checkpoints if directory provided
    if checkpoint_dir:
        _save_eval_checkpoint(model, dice_score, checkpoint_dir)

    return {
        "loss": avg_loss,
        "dice": dice_score,
    }


def _save_training_checkpoint(model: nn.Module, checkpoint_dir: Path) -> None:
    """Save model checkpoint after training.

    Args:
        model: Model to checkpoint
        checkpoint_dir: Directory to save to
    """
    save_model_checkpoint(model, checkpoint_dir)


def _save_eval_checkpoint(
    model: nn.Module, dice_score: float, checkpoint_dir: Path
) -> None:
    """Save model checkpoints after evaluation.

    Saves both 'last_model.pt' (always) and 'best_model.pt' (when dice improves).

    Args:
        model: Model to checkpoint
        dice_score: Current dice score
        checkpoint_dir: Directory to save to
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Determine if this is the best model by checking existing best checkpoint
    best_dice = 0.0
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            best_dice = checkpoint.get("dice", 0.0)
        except Exception:
            best_dice = 0.0

    is_best = dice_score > best_dice
    save_model_checkpoint(model, checkpoint_dir, dice_score=dice_score, is_best=is_best)
