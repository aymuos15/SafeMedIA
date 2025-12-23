"""Federated Learning module for DP-FedMed.

This module contains the client and server components for federated learning
with differential privacy.

Submodules:
    - fl.task: Training and evaluation functions
    - fl.checkpoint: Unified checkpointing for mid-round recovery
    - fl.client: Client-side components (DPFlowerClient, client_fn, app)
    - fl.server: Server-side components (DPFedAvg, weighted_average, server_fn, app)

Note: Imports are not done at module level to avoid deadlock issues when
Flower loads server and client apps in parallel threads during simulation.

Usage:
    from dp_fedmed.fl.task import train_one_epoch, evaluate
    from dp_fedmed.fl.checkpoint import UnifiedCheckpointManager, load_unified_checkpoint
    from dp_fedmed.fl.client import DPFlowerClient, client_fn
    from dp_fedmed.fl.server import DPFedAvg, server_fn
"""

# Only export task functions at module level (no dependencies on client/server)
from .task import train_one_epoch, evaluate
from .checkpoint import (
    UnifiedCheckpointManager,
    UnifiedCheckpoint,
    ClientState,
    ServerState,
    PrivacyState,
    RoundProgress,
    EpochProgress,
    save_unified_checkpoint,
    load_unified_checkpoint,
    resolve_checkpoint_path,
    get_model_state_dict,
)

__all__ = [
    "train_one_epoch",
    "evaluate",
    "UnifiedCheckpointManager",
    "UnifiedCheckpoint",
    "ClientState",
    "ServerState",
    "PrivacyState",
    "RoundProgress",
    "EpochProgress",
    "save_unified_checkpoint",
    "load_unified_checkpoint",
    "resolve_checkpoint_path",
    "get_model_state_dict",
]
