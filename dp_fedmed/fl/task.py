"""Training and evaluation logic for federated learning with differential privacy.

This module contains the core training and evaluation functions used by
federated clients during local training rounds.

NOTE: We use manual training loops instead of MONAI engines because:
1. Opacus (DP) wraps model+optimizer+dataloader together - incompatible with MONAI engines
2. MONAI's SupervisedTrainer causes SIGFPE with Opacus per-sample gradient computation
3. Simple loops are more debuggable for FL + DP use case

Checkpointing is added directly to these functions for both best and last models.

NOTE: DiceLoss is NOT used during training because it causes SIGFPE (floating point
exceptions) with Opacus due to division operations in per-sample gradient computation.
We use CrossEntropyLoss only for training, and DiceMetric only for evaluation.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    checkpoint_dir: Optional[Path] = None,
) -> float:
    """Train model for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Loss function (default: CrossEntropy only for DP stability)
        checkpoint_dir: Optional directory for saving checkpoints

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

    for batch_idx, batch in enumerate(train_loader):
        # Handle both tuple and dict batch formats
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # CrossEntropyLoss expects: outputs=[B, C, H, W], labels=[B, H, W]
        # Ensure labels are [B, H, W] and long type
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1).long()

        loss = criterion(outputs, labels)

        loss.backward()
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
                labels = labels.squeeze(1).long()

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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Handle Opacus-wrapped models
    if hasattr(model, "_module"):
        state_dict = model._module.state_dict()
    else:
        state_dict = model.state_dict()

    # Save last model (overwritten each time)
    torch.save({"model": state_dict}, checkpoint_dir / "last_model.pt")


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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Handle Opacus-wrapped models
    if hasattr(model, "_module"):
        state_dict = model._module.state_dict()
    else:
        state_dict = model.state_dict()

    # Track best dice in a file
    best_dice_file = checkpoint_dir / ".best_dice"
    best_dice = 0.0
    if best_dice_file.exists():
        try:
            best_dice = float(best_dice_file.read_text().strip())
        except ValueError:
            best_dice = 0.0

    # Save best model if improved
    if dice_score > best_dice:
        torch.save(
            {"model": state_dict, "dice": dice_score}, checkpoint_dir / "best_model.pt"
        )
        best_dice_file.write_text(str(dice_score))

    # Always save last model
    torch.save({"model": state_dict}, checkpoint_dir / "last_model.pt")
