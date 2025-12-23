"""I/O operations for checkpointing.

This module handles saving and loading checkpoint files, including
atomic writes and path resolution.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

from ...utils import get_unwrapped_model
from .models import UnifiedCheckpoint


def get_model_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Extract state dict from model, handling Opacus-wrapped models.

    Args:
        model: PyTorch model (possibly wrapped by Opacus)

    Returns:
        Model state dictionary
    """
    unwrapped_model = get_unwrapped_model(model)
    return unwrapped_model.state_dict()


def _atomic_save(data: Dict[str, Any], path: Path) -> None:
    """Save checkpoint atomically using temp file + rename.

    Args:
        data: Data to save
        path: Target path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)  # Atomic on POSIX
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_unified_checkpoint(
    checkpoint: UnifiedCheckpoint,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save unified checkpoint.

    Args:
        checkpoint: Unified checkpoint data
        checkpoint_dir: Directory to save to
        is_best: Whether this is the best model so far

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_data = checkpoint.to_dict()

    # Always save last checkpoint
    last_path = checkpoint_dir / "last.pt"
    _atomic_save(checkpoint_data, last_path)

    # Save best checkpoint if requested
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        _atomic_save(checkpoint_data, best_path)
        logger.info(
            f"New best checkpoint saved (dice={checkpoint.server.best_dice:.4f})"
        )
        return best_path

    return last_path


def load_unified_checkpoint(checkpoint_path: Path) -> UnifiedCheckpoint:
    """Load unified checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Unified checkpoint object

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )  # nosec B614

    # Validate version
    version = checkpoint_data.get("version", "1.0")
    if not version.startswith("2."):
        raise ValueError(f"Incompatible checkpoint version: {version}. Expected 2.x")

    checkpoint = UnifiedCheckpoint.from_dict(checkpoint_data)

    logger.info(
        f"Loaded unified checkpoint from {checkpoint_path} "
        f"(round={checkpoint.round.current}/{checkpoint.round.total}, "
        f"status={checkpoint.round.status})"
    )

    return checkpoint


def resolve_checkpoint_path(
    resume_from: Optional[str],
    run_dir: Path,
) -> Optional[Path]:
    """Resolve checkpoint path from config value.

    Args:
        resume_from: Config value - "last", "best", or absolute path
        run_dir: Run directory (parent of checkpoints/)

    Returns:
        Resolved path or None if no resume

    Raises:
        FileNotFoundError: If specified checkpoint doesn't exist
        ValueError: If resume_from is invalid
    """
    if not resume_from:
        return None

    resume_from = resume_from.strip()
    if not resume_from:
        return None

    checkpoint_dir = run_dir / "checkpoints"

    if resume_from.lower() == "last":
        path = checkpoint_dir / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"No last checkpoint found at {path}")
        return path

    if resume_from.lower() == "best":
        path = checkpoint_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {path}")
        return path

    # Absolute path
    path = Path(resume_from)
    if not path.is_absolute():
        raise ValueError(
            f"resume_from must be 'last', 'best', or absolute path. Got: {resume_from}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path
