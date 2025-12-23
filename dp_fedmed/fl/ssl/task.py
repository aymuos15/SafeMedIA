"""SSL-specific training and evaluation functions."""

from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data
from loguru import logger

from dp_fedmed.utils import is_loss_valid


def train_epoch_ssl(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Train for one epoch with SSL contrastive loss.

    Args:
        model: SSL model (SSLUNet with projection head)
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        criterion: Contrastive loss function (e.g., NTXentLoss)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Handle both tuple and list returns
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                images, _ = batch  # Discard dummy labels
            else:
                images = batch[0]
        else:
            images = batch

        # Skip empty batches (can occur with Opacus Poisson sampling)
        if isinstance(images, torch.Tensor) and images.shape[0] == 0:
            continue

        # Get augmented views
        images_1, images_2 = _parse_view_pair(images, device)

        # Forward pass
        optimizer.zero_grad()

        _, z1 = model(images_1)
        _, z2 = model(images_2)

        # Compute contrastive loss
        loss = criterion(z1, z2)
        loss = loss.mean()

        if not is_loss_valid(loss):
            logger.warning(
                f"Non-finite loss detected: {loss.item() if loss.numel() == 1 else 'multiple'}, skipping batch"
            )
            continue

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate_ssl(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Validate SSL model using contrastive loss.

    Args:
        model: SSL model
        val_loader: Validation data loader
        device: Device to evaluate on
        criterion: Contrastive loss function

    Returns:
        Dictionary with 'loss' and 'val_loss' metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Handle batch format
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch[0]
            else:
                images = batch

            # Skip empty batches (can occur with Opacus Poisson sampling)
            if isinstance(images, torch.Tensor) and images.shape[0] == 0:
                continue

            # Get augmented views
            images_1, images_2 = _parse_view_pair(images, device)

            # Forward pass
            _, z1 = model(images_1)
            _, z2 = model(images_2)

            # Compute contrastive loss
            loss = criterion(z1, z2)

            if is_loss_valid(loss):
                total_loss += loss.mean().item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss, "val_loss": avg_loss}


def _parse_view_pair(
    images: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse images into two augmented views.

    Args:
        images: Input tensor (either view pair or stacked tensor)
        device: Device to move tensors to

    Returns:
        Tuple of (view1, view2) tensors
    """
    # Handle different tensor formats from DataLoader:
    # 1. [B, 2, C, H, W] - from flatten_augmented_views=True (5D tensor)
    # 2. [B*2, C, H, W] - legacy format (4D tensor)
    # 3. List/tuple of (view1, view2) tensors
    if isinstance(images, torch.Tensor):
        if len(images.shape) == 5 and images.shape[1] == 2:
            # Format: [B, 2, C, H, W] -> split along dim 1
            images_1 = images[:, 0].to(device)  # [B, C, H, W]
            images_2 = images[:, 1].to(device)  # [B, C, H, W]
        elif len(images.shape) == 4 and images.shape[0] % 2 == 0:
            # Legacy format: [B*2, C, H, W] -> split along dim 0
            try:
                batch_size = images.shape[0] // 2
                images_1 = images[:batch_size].to(device)
                images_2 = images[batch_size:].to(device)
            except Exception:
                logger.warning(
                    "Could not parse Opacus-wrapped tensor structure, using single view"
                )
                images_1 = images_2 = images.to(device)
        else:
            # Fallback: use same image twice
            logger.warning(f"Unexpected tensor shape {images.shape}, using single view")
            images_1 = images_2 = images.to(device)
    elif isinstance(images, (list, tuple)):
        images_1 = images[0].to(device)
        images_2 = images[1].to(device)
    else:
        # Fallback: use same image twice
        logger.warning("Expected augmented view pair, got single image")
        images_1 = images_2 = images.to(device)

    return images_1, images_2
