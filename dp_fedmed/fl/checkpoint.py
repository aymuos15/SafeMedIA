"""Unified checkpointing for federated learning with mid-round recovery.

This module provides a unified checkpoint system that captures both server
and client state in a single file, enabling recovery from mid-round crashes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from flwr.common import Parameters, ndarrays_to_parameters
from loguru import logger


# Checkpoint version for future format migrations
CHECKPOINT_VERSION = "2.0"


@dataclass
class EpochProgress:
    """Tracks epoch progress within a round."""

    current: int
    total: int
    status: str  # "in_progress" | "completed"


@dataclass
class ClientState:
    """Per-client training state for mid-round recovery."""

    client_id: int
    epoch: EpochProgress
    model_state: Optional[Dict[str, torch.Tensor]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    partial_metrics: Dict[str, float] = field(
        default_factory=lambda: {"loss_sum": 0.0, "epochs_done": 0}
    )
    partial_privacy: Dict[str, float] = field(
        default_factory=lambda: {"epsilon": 0.0, "steps": 0}
    )
    # Cached results for completed clients
    final_parameters: Optional[List[np.ndarray]] = None
    final_metrics: Optional[Dict[str, float]] = None
    num_samples: int = 0


@dataclass
class ServerState:
    """Server-side FL state."""

    parameters: List[np.ndarray]
    best_dice: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoundProgress:
    """Tracks round progress."""

    current: int
    total: int
    status: str  # "in_progress" | "completed"


@dataclass
class PrivacyState:
    """Privacy accounting state."""

    target_delta: float
    sample_history: List[Tuple[float, float, int]] = field(
        default_factory=list
    )  # (sigma, q, steps)
    user_history: List[Tuple[float, float, int]] = field(default_factory=list)
    cumulative_sample_epsilon: float = 0.0
    cumulative_user_epsilon: float = 0.0
    partial_round_epsilon: float = 0.0


@dataclass
class UnifiedCheckpoint:
    """Complete FL system state for mid-round recovery."""

    version: str
    timestamp: str
    run_name: str
    round: RoundProgress
    server: ServerState
    clients: Dict[int, ClientState]
    privacy: PrivacyState

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "run_name": self.run_name,
            "round": {
                "current": self.round.current,
                "total": self.round.total,
                "status": self.round.status,
            },
            "server": {
                "parameters": self.server.parameters,
                "best_dice": self.server.best_dice,
                "metrics": self.server.metrics,
            },
            "clients": {
                cid: {
                    "client_id": cs.client_id,
                    "epoch": {
                        "current": cs.epoch.current,
                        "total": cs.epoch.total,
                        "status": cs.epoch.status,
                    },
                    "model_state": cs.model_state,
                    "optimizer_state": cs.optimizer_state,
                    "partial_metrics": cs.partial_metrics,
                    "partial_privacy": cs.partial_privacy,
                    "final_parameters": cs.final_parameters,
                    "final_metrics": cs.final_metrics,
                    "num_samples": cs.num_samples,
                }
                for cid, cs in self.clients.items()
            },
            "privacy": {
                "target_delta": self.privacy.target_delta,
                "sample_history": self.privacy.sample_history,
                "user_history": self.privacy.user_history,
                "cumulative_sample_epsilon": self.privacy.cumulative_sample_epsilon,
                "cumulative_user_epsilon": self.privacy.cumulative_user_epsilon,
                "partial_round_epsilon": self.privacy.partial_round_epsilon,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedCheckpoint":
        """Create from dictionary."""
        round_data = data["round"]
        server_data = data["server"]
        privacy_data = data["privacy"]

        clients = {}
        for cid_str, cs_data in data["clients"].items():
            try:
                cid = int(cid_str)
            except ValueError:
                raise ValueError(
                    f"Checkpoint corrupted: invalid client ID '{cid_str}'. "
                    "Expected numeric client ID."
                )
            epoch_data = cs_data["epoch"]
            clients[cid] = ClientState(
                client_id=cs_data["client_id"],
                epoch=EpochProgress(
                    current=epoch_data["current"],
                    total=epoch_data["total"],
                    status=epoch_data["status"],
                ),
                model_state=cs_data.get("model_state"),
                optimizer_state=cs_data.get("optimizer_state"),
                partial_metrics=cs_data.get(
                    "partial_metrics", {"loss_sum": 0.0, "epochs_done": 0}
                ),
                partial_privacy=cs_data.get(
                    "partial_privacy", {"epsilon": 0.0, "steps": 0}
                ),
                final_parameters=cs_data.get("final_parameters"),
                final_metrics=cs_data.get("final_metrics"),
                num_samples=cs_data.get("num_samples", 0),
            )

        return cls(
            version=data["version"],
            timestamp=data["timestamp"],
            run_name=data["run_name"],
            round=RoundProgress(
                current=round_data["current"],
                total=round_data["total"],
                status=round_data["status"],
            ),
            server=ServerState(
                parameters=server_data["parameters"],
                best_dice=server_data.get("best_dice", 0.0),
                metrics=server_data.get("metrics", {}),
            ),
            clients=clients,
            privacy=PrivacyState(
                target_delta=privacy_data["target_delta"],
                sample_history=privacy_data.get("sample_history", []),
                user_history=privacy_data.get("user_history", []),
                cumulative_sample_epsilon=privacy_data.get(
                    "cumulative_sample_epsilon", 0.0
                ),
                cumulative_user_epsilon=privacy_data.get(
                    "cumulative_user_epsilon", 0.0
                ),
                partial_round_epsilon=privacy_data.get("partial_round_epsilon", 0.0),
            ),
        )


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


class UnifiedCheckpointManager:
    """Manages unified checkpoints for FL runs."""

    def __init__(
        self,
        checkpoint_dir: Path,
        run_name: str,
        num_rounds: int,
        target_delta: float = 1e-5,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            run_name: Name of the run
            num_rounds: Total number of rounds
            target_delta: Privacy delta value
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.num_rounds = num_rounds
        self.target_delta = target_delta

        self.best_dice = 0.0
        self._current_checkpoint: Optional[UnifiedCheckpoint] = None

        # Try to load existing checkpoint
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing checkpoint if available."""
        last_path = self.checkpoint_dir / "last.pt"
        if last_path.exists():
            try:
                checkpoint = load_unified_checkpoint(last_path)
                self._current_checkpoint = checkpoint
                self.best_dice = checkpoint.server.best_dice
                logger.debug(f"Loaded existing checkpoint: {self.best_dice:.4f}")
            except Exception as e:
                logger.warning(f"Could not load existing checkpoint: {e}")

    def create_initial_checkpoint(
        self,
        parameters: List[np.ndarray],
        num_clients: int,
        local_epochs: int,
    ) -> UnifiedCheckpoint:
        """Create initial checkpoint for a fresh run.

        Args:
            parameters: Initial global model parameters
            num_clients: Number of clients
            local_epochs: Local epochs per round

        Returns:
            Initial unified checkpoint
        """
        clients = {
            i: ClientState(
                client_id=i,
                epoch=EpochProgress(current=0, total=local_epochs, status="pending"),
            )
            for i in range(num_clients)
        }

        checkpoint = UnifiedCheckpoint(
            version=CHECKPOINT_VERSION,
            timestamp=datetime.now().isoformat(),
            run_name=self.run_name,
            round=RoundProgress(current=1, total=self.num_rounds, status="in_progress"),
            server=ServerState(parameters=parameters),
            clients=clients,
            privacy=PrivacyState(target_delta=self.target_delta),
        )

        self._current_checkpoint = checkpoint
        return checkpoint

    def update_client_epoch(
        self,
        client_id: int,
        epoch: int,
        total_epochs: int,
        model_state: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        epoch_loss: float = 0.0,
        partial_epsilon: float = 0.0,
        partial_steps: int = 0,
    ) -> None:
        """Update client's epoch progress.

        Args:
            client_id: Client ID
            epoch: Current epoch (0-indexed, after completion)
            total_epochs: Total epochs
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            epoch_loss: Loss for this epoch
            partial_epsilon: Epsilon spent so far
            partial_steps: DP steps so far
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint initialized")

        client = self._current_checkpoint.clients.get(client_id)
        if client is None:
            # Create new client state
            client = ClientState(
                client_id=client_id,
                epoch=EpochProgress(
                    current=0, total=total_epochs, status="in_progress"
                ),
            )
            self._current_checkpoint.clients[client_id] = client

        # Update epoch progress
        client.epoch.current = epoch + 1  # Store 1-indexed
        client.epoch.total = total_epochs
        client.epoch.status = (
            "completed" if epoch + 1 >= total_epochs else "in_progress"
        )

        # Update model state if provided
        if model_state is not None:
            client.model_state = model_state
        if optimizer_state is not None:
            client.optimizer_state = optimizer_state

        # Accumulate metrics
        client.partial_metrics["loss_sum"] += epoch_loss
        client.partial_metrics["epochs_done"] = epoch + 1
        client.partial_privacy["epsilon"] = partial_epsilon
        client.partial_privacy["steps"] = partial_steps

        # Update timestamp
        self._current_checkpoint.timestamp = datetime.now().isoformat()

    def mark_client_completed(
        self,
        client_id: int,
        final_parameters: List[np.ndarray],
        final_metrics: Dict[str, float],
        num_samples: int,
    ) -> None:
        """Mark client's round as completed with final results.

        Args:
            client_id: Client ID
            final_parameters: Final model parameters
            final_metrics: Final metrics dict
            num_samples: Number of training samples
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint initialized")

        client = self._current_checkpoint.clients.get(client_id)
        if client is None:
            raise KeyError(f"Client {client_id} not found in checkpoint")

        client.epoch.status = "completed"
        client.final_parameters = final_parameters
        client.final_metrics = final_metrics
        client.num_samples = num_samples

        # Clear intermediate state (no longer needed)
        client.model_state = None
        client.optimizer_state = None

    def update_server_state(
        self,
        parameters: List[np.ndarray],
        metrics: Dict[str, float],
        round_num: int,
        cumulative_sample_epsilon: float = 0.0,
        cumulative_user_epsilon: float = 0.0,
    ) -> None:
        """Update server state after aggregation.

        Args:
            parameters: Aggregated global parameters
            metrics: Aggregated metrics
            round_num: Current round number
            cumulative_sample_epsilon: Cumulative sample epsilon
            cumulative_user_epsilon: Cumulative user epsilon
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint initialized")

        self._current_checkpoint.server.parameters = parameters
        self._current_checkpoint.server.metrics = metrics
        self._current_checkpoint.round.current = round_num

        # Update privacy state
        self._current_checkpoint.privacy.cumulative_sample_epsilon = (
            cumulative_sample_epsilon
        )
        self._current_checkpoint.privacy.cumulative_user_epsilon = (
            cumulative_user_epsilon
        )

        # Update timestamp
        self._current_checkpoint.timestamp = datetime.now().isoformat()

    def mark_round_completed(self, dice_score: float) -> None:
        """Mark current round as completed.

        Args:
            dice_score: Final dice score for the round
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint initialized")

        self._current_checkpoint.round.status = "completed"
        self._current_checkpoint.server.metrics["dice"] = dice_score

        # Check if best
        if dice_score > self.best_dice:
            self.best_dice = dice_score
            self._current_checkpoint.server.best_dice = dice_score

    def start_next_round(self, round_num: int, local_epochs: int) -> None:
        """Start a new round.

        Args:
            round_num: New round number
            local_epochs: Local epochs for this round
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint initialized")

        self._current_checkpoint.round.current = round_num
        self._current_checkpoint.round.status = "in_progress"

        # Reset client epoch progress
        for client in self._current_checkpoint.clients.values():
            client.epoch = EpochProgress(
                current=0, total=local_epochs, status="in_progress"
            )
            client.model_state = None
            client.optimizer_state = None
            client.partial_metrics = {"loss_sum": 0.0, "epochs_done": 0}
            client.partial_privacy = {"epsilon": 0.0, "steps": 0}
            client.final_parameters = None
            client.final_metrics = None

    def save(self) -> Path:
        """Save current checkpoint.

        Returns:
            Path to saved checkpoint
        """
        if self._current_checkpoint is None:
            raise RuntimeError("No checkpoint to save")

        is_best = (
            self._current_checkpoint.server.metrics.get("dice", 0) >= self.best_dice
            and self._current_checkpoint.round.status == "completed"
        )

        return save_unified_checkpoint(
            self._current_checkpoint,
            self.checkpoint_dir,
            is_best=is_best,
        )

    def get_current_checkpoint(self) -> Optional[UnifiedCheckpoint]:
        """Get current checkpoint object."""
        return self._current_checkpoint

    def get_client_state(self, client_id: int) -> Optional[ClientState]:
        """Get client state from current checkpoint."""
        if self._current_checkpoint is None:
            return None
        return self._current_checkpoint.clients.get(client_id)

    def has_checkpoint(self, which: str = "last") -> bool:
        """Check if a checkpoint exists.

        Args:
            which: Either 'last' or 'best'

        Returns:
            True if checkpoint exists
        """
        path = self.checkpoint_dir / f"{which}.pt"
        return path.exists()

    def is_mid_round_resume(self) -> bool:
        """Check if this is a mid-round resume scenario.

        Returns:
            True if resuming from mid-round
        """
        if self._current_checkpoint is None:
            return False
        return self._current_checkpoint.round.status == "in_progress"

    def get_resume_round(self) -> int:
        """Get the round to resume from.

        Returns:
            Round number to resume from
        """
        if self._current_checkpoint is None:
            return 1
        if self._current_checkpoint.round.status == "completed":
            # Resume from next round
            return self._current_checkpoint.round.current + 1
        # Resume same round (mid-round)
        return self._current_checkpoint.round.current

    def get_parameters_as_fl(self) -> Optional[Parameters]:
        """Get server parameters as FL Parameters object.

        Returns:
            FL Parameters or None
        """
        if self._current_checkpoint is None:
            return None
        return ndarrays_to_parameters(self._current_checkpoint.server.parameters)
