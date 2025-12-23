"""Utility functions for DP-FedMed.

This module contains common utility functions used across the codebase
to reduce redundancy and improve maintainability.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def load_client_history(run_dir: Path | None) -> List[Dict[str, Any]]:
    """Load client history from disk.

    Args:
        run_dir: Directory where history is saved

    Returns:
        List of round history dictionaries
    """
    if run_dir:
        history_path = run_dir / "history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                    return data.get("rounds", [])
            except FileNotFoundError:
                logger.debug("No history file found")
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted history file, starting fresh: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading history: {e}")
    return []


def save_client_metrics(
    run_dir: Path | None,
    client_id: int,
    start_time: str,
    round_history: List[Dict[str, Any]],
    train_loader: torch.utils.data.DataLoader,
    privacy_config: Dict[str, Any],
    extra_metrics: Dict[str, Any] | None = None,
) -> None:
    """Save client-specific metrics and history.

    Args:
        run_dir: Directory to save metrics
        client_id: Client partition ID
        start_time: Training start time ISO string
        round_history: List of round history
        train_loader: Training data loader
        privacy_config: Privacy configuration
        extra_metrics: Additional metrics to include in final summary
    """
    if run_dir is None:
        return

    end_time = datetime.now().isoformat()

    # Calculate final metrics from history
    final_train_loss = 0.0
    total_epsilon = 0.0

    if round_history:
        last_round = round_history[-1]
        final_train_loss = last_round.get("train_loss", 0.0)
        total_epsilon = sum(r.get("epsilon", 0.0) for r in round_history)

    num_total_samples = get_dataset_size(train_loader)

    # Save metrics.json
    metrics_data = {
        "client_id": client_id,
        "start_time": start_time,
        "end_time": end_time,
        "num_rounds": len(round_history),
        "final_train_loss": final_train_loss,
        "training_samples": num_total_samples,
        "privacy": {
            "total_epsilon": total_epsilon,
            "delta": privacy_config.get("target_delta", 1e-5),
        },
    }

    if extra_metrics:
        metrics_data.update(extra_metrics)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Save history.json
    history_data = {"client_id": client_id, "rounds": round_history}

    history_path = run_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    logger.debug(f"✓ Client {client_id} metrics saved to {run_dir}")
