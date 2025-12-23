"""Utility functions for DP-FedMed.

This module contains common utility functions used across the codebase
to reduce redundancy and improve maintainability.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Get the base model, handling Opacus wrapping.

    Opacus wraps models in a GradSampleModule for per-sample gradient computation.
    This function extracts the original model from the wrapper if present.

    Args:
        model: PyTorch model (possibly wrapped by Opacus)

    Returns:
        Unwrapped base model
    """
    if hasattr(model, "_module"):
        return model._module  # type: ignore
    return model


def extract_batch_data(
    batch, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract images and labels from batch in any format.

    Handles both dictionary format (MONAI) and tuple format (Opacus).
    Also preprocesses labels to ensure correct shape and dtype.

    Args:
        batch: Batch data (dict or tuple/list)
        device: Device to move tensors to

    Returns:
        Tuple of (images, labels) or None if batch format is invalid
    """
    # Handle different batch formats
    if isinstance(batch, dict):
        if "image" not in batch or "label" not in batch:
            logger.warning(f"Batch dict missing required keys: {batch.keys()}")
            return None
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
    else:
        logger.warning(f"Unexpected batch format: {type(batch)}, skipping")
        return None

    # Preprocess labels: squeeze channel dimension if present, ensure long type
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    labels = labels.long()

    return images, labels


def get_dataset_size(dataloader: torch.utils.data.DataLoader) -> int:
    """Safely get dataset size from dataloader.

    Args:
        dataloader: PyTorch DataLoader

    Returns:
        Number of samples in dataset, or 0 if cannot be determined
    """
    try:
        return len(dataloader.dataset)  # type: ignore
    except (TypeError, AttributeError):
        return 0
