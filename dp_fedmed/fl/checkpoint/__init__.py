"""Unified checkpointing for federated learning with mid-round recovery.

This module provides a unified checkpoint system that captures both server
and client state in a single file, enabling recovery from mid-round crashes.
"""

from .models import (
    CHECKPOINT_VERSION,
    EpochProgress,
    ClientState,
    ServerState,
    RoundProgress,
    PrivacyState,
    UnifiedCheckpoint,
)
from .io import (
    get_model_state_dict,
    save_unified_checkpoint,
    load_unified_checkpoint,
    resolve_checkpoint_path,
)
from .manager import UnifiedCheckpointManager

__all__ = [
    "CHECKPOINT_VERSION",
    "EpochProgress",
    "ClientState",
    "ServerState",
    "RoundProgress",
    "PrivacyState",
    "UnifiedCheckpoint",
    "get_model_state_dict",
    "save_unified_checkpoint",
    "load_unified_checkpoint",
    "resolve_checkpoint_path",
    "UnifiedCheckpointManager",
]
