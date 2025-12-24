"""Checkpoint utilities for saving and loading pretrained encoders.

This module provides functions for transfer learning workflows:
- Saving pretrained model checkpoints from SSL pretraining
- Loading pretrained encoder weights into downstream models
- Freezing encoder parameters for fine-tuning
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from loguru import logger


def save_pretrained_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_path: Path,
) -> None:
    """Save pretrained model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_pretrained_encoder(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool = False,
) -> Dict:
    """Load pretrained encoder weights into model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce matching keys

    Returns:
        Checkpoint metadata (epoch, metrics, etc.)

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(  # nosec B614 - loading trusted checkpoint
        checkpoint_path, map_location="cpu", weights_only=False
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Loaded pretrained encoder from {checkpoint_path}")
    except RuntimeError as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }

    return metadata


def load_encoder_for_downstream(
    encoder_checkpoint_path: Path,
    downstream_model: nn.Module,
    freeze_encoder: bool = False,
) -> None:
    """Load pretrained encoder into downstream task model.

    Args:
        encoder_checkpoint_path: Path to pretrained encoder checkpoint
        downstream_model: Downstream task model (e.g., UNet for segmentation)
        freeze_encoder: Whether to freeze encoder during downstream training
    """
    encoder_checkpoint_path = Path(encoder_checkpoint_path)

    if not encoder_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {encoder_checkpoint_path}"
        )

    checkpoint = torch.load(  # nosec B614 - loading trusted checkpoint
        encoder_checkpoint_path, map_location="cpu", weights_only=False
    )
    encoder_state = checkpoint.get("model_state_dict")

    if encoder_state is None:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")

    try:
        downstream_model.load_state_dict(encoder_state, strict=False)
        logger.info(f"Loaded pretrained encoder from {encoder_checkpoint_path}")
    except Exception as e:
        logger.warning(f"Could not load as full model, attempting partial load: {e}")
        encoder = getattr(downstream_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "load_state_dict"):
            encoder.load_state_dict(encoder_state, strict=False)
            logger.info("Loaded pretrained encoder into model.encoder")
        else:
            raise

    if freeze_encoder:
        encoder = getattr(downstream_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "parameters"):
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Froze encoder parameters")
        else:
            logger.warning("Cannot freeze encoder: no encoder attribute found")
