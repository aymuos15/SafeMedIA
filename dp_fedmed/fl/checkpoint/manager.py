"""Checkpoint manager for federated learning runs.

This module provides the UnifiedCheckpointManager class which coordinates
checkpoint operations across training rounds.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from flwr.common import Parameters, ndarrays_to_parameters
from loguru import logger

from .models import (
    CHECKPOINT_VERSION,
    UnifiedCheckpoint,
    ClientState,
    ServerState,
    PrivacyState,
    RoundProgress,
    EpochProgress,
)
from .io import save_unified_checkpoint, load_unified_checkpoint


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
