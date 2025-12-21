"""Checkpointing utilities for federated learning.

This module provides unified checkpointing for both server-side (FL Parameters)
and client-side (PyTorch models) use cases.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
import torch.nn as nn
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from loguru import logger


def get_model_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Extract state dict from model, handling Opacus-wrapped models.

    Args:
        model: PyTorch model (possibly wrapped by Opacus)

    Returns:
        Model state dictionary
    """
    if hasattr(model, "_module"):
        return cast(nn.Module, model._module).state_dict()
    return model.state_dict()


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    dice_score: Optional[float] = None,
    is_best: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a PyTorch model checkpoint.

    Args:
        model: Model to checkpoint
        checkpoint_dir: Directory to save to
        dice_score: Optional dice score to include in checkpoint
        is_best: Whether this is the best model so far
        extra_metadata: Additional metadata to save

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_dict = get_model_state_dict(model)

    checkpoint_data: Dict[str, Any] = {"model": state_dict}
    if dice_score is not None:
        checkpoint_data["dice"] = dice_score
    if extra_metadata:
        checkpoint_data.update(extra_metadata)

    # Always save last model
    last_path = checkpoint_dir / "last_model.pt"
    torch.save(checkpoint_data, last_path)

    # Save best model if requested
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint_data, best_path)
        logger.info(f"New best model saved (dice={dice_score:.4f})")
        return best_path

    return last_path


def save_server_checkpoint(
    parameters: Parameters,
    checkpoint_dir: Path,
    round_num: int,
    dice_score: float,
    cumulative_epsilon: float,
    is_best: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save server-side FL parameters checkpoint.

    Args:
        parameters: FL Parameters to save
        checkpoint_dir: Directory to save to
        round_num: Current round number
        dice_score: Aggregated dice score
        cumulative_epsilon: Cumulative privacy budget spent
        is_best: Whether this is the best model so far
        extra_metadata: Additional metadata to save

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    params_ndarrays = parameters_to_ndarrays(parameters)

    checkpoint_data = {
        "parameters": params_ndarrays,
        "round": round_num,
        "cumulative_epsilon": cumulative_epsilon,
        "dice": dice_score,
    }
    if extra_metadata:
        checkpoint_data.update(extra_metadata)

    # Always save last model
    last_path = checkpoint_dir / "last_model.pt"
    torch.save(checkpoint_data, last_path)

    # Save best model if requested
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint_data, best_path)
        logger.info(f"New best model saved (dice={dice_score:.4f})")
        return best_path

    return last_path


def load_model_checkpoint(
    checkpoint_path: Path,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a PyTorch model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load weights into
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with 'model' key and any extra metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=True
    )

    if model is not None:
        state_dict = checkpoint.get("model", checkpoint)
        if hasattr(model, "_module"):
            cast(nn.Module, model._module).load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {checkpoint_path}")

    return checkpoint


def load_server_checkpoint(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Load a server-side FL checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with:
            - parameters: FL Parameters object
            - round: Round number
            - cumulative_epsilon: Privacy budget spent
            - dice: Dice score at checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Convert numpy arrays back to FL Parameters
    params_ndarrays = checkpoint.get("parameters")
    if params_ndarrays is not None:
        parameters = ndarrays_to_parameters(params_ndarrays)
        checkpoint["parameters"] = parameters

    logger.info(
        f"Loaded server checkpoint from {checkpoint_path} "
        f"(round={checkpoint.get('round', '?')}, dice={checkpoint.get('dice', 0):.4f})"
    )

    return checkpoint


class CheckpointManager:
    """Manages checkpointing with best model tracking.

    This class provides a stateful wrapper around the checkpoint functions,
    automatically tracking the best dice score.
    """

    def __init__(self, checkpoint_dir: Path, mode: str = "model"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            mode: Either 'model' for client-side or 'server' for server-side
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mode = mode
        self.best_dice = 0.0

        # Try to load existing best dice from checkpoint
        self._load_best_dice()

    def _load_best_dice(self) -> None:
        """Load best dice score from existing checkpoint if available."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            try:
                checkpoint = torch.load(
                    best_path, map_location="cpu", weights_only=True
                )
                self.best_dice = checkpoint.get("dice", 0.0)
                logger.debug(f"Loaded best dice from checkpoint: {self.best_dice:.4f}")
            except Exception as e:
                logger.warning(f"Could not load best dice from checkpoint: {e}")

    def save(
        self,
        model_or_params: Union[nn.Module, Parameters],
        dice_score: float,
        round_num: Optional[int] = None,
        cumulative_epsilon: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint, automatically tracking best model.

        Args:
            model_or_params: Model (client-side) or Parameters (server-side)
            dice_score: Current dice score
            round_num: Round number (required for server mode)
            cumulative_epsilon: Privacy budget spent (required for server mode)
            extra_metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
        is_best = dice_score > self.best_dice
        if is_best:
            self.best_dice = dice_score

        if self.mode == "server":
            if round_num is None or cumulative_epsilon is None:
                raise ValueError(
                    "round_num and cumulative_epsilon required for server mode"
                )
            if not isinstance(model_or_params, Parameters):
                raise ValueError("model_or_params must be Parameters in server mode")
            return save_server_checkpoint(
                parameters=model_or_params,
                checkpoint_dir=self.checkpoint_dir,
                round_num=round_num,
                dice_score=dice_score,
                cumulative_epsilon=cumulative_epsilon,
                is_best=is_best,
                extra_metadata=extra_metadata,
            )
        else:
            if not isinstance(model_or_params, nn.Module):
                raise ValueError("model_or_params must be nn.Module in model mode")
            return save_model_checkpoint(
                model=model_or_params,
                checkpoint_dir=self.checkpoint_dir,
                dice_score=dice_score,
                is_best=is_best,
                extra_metadata=extra_metadata,
            )

    def load_best(
        self, model: Optional[nn.Module] = None, device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the best checkpoint.

        Args:
            model: Optional model to load weights into (client mode only)
            device: Device to load tensors to

        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if self.mode == "server":
            return load_server_checkpoint(best_path)
        return load_model_checkpoint(best_path, model, device)

    def load_last(
        self, model: Optional[nn.Module] = None, device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the last checkpoint.

        Args:
            model: Optional model to load weights into (client mode only)
            device: Device to load tensors to

        Returns:
            Checkpoint dictionary
        """
        last_path = self.checkpoint_dir / "last_model.pt"
        if self.mode == "server":
            return load_server_checkpoint(last_path)
        return load_model_checkpoint(last_path, model, device)

    def has_checkpoint(self, which: str = "last") -> bool:
        """Check if a checkpoint exists.

        Args:
            which: Either 'last' or 'best'

        Returns:
            True if checkpoint exists
        """
        path = self.checkpoint_dir / f"{which}_model.pt"
        return path.exists()
