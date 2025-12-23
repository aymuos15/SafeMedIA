"""Data structures for unified checkpointing.

This module defines the dataclasses that represent the state of
federated learning runs, including client, server, and privacy state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


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
