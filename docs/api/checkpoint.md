# Checkpoint API

The Checkpoint API provides unified state persistence for federated learning with mid-round recovery support.

## Overview

The checkpoint system captures both server and client state in a single file, enabling:

- **Mid-round recovery**: Resume from any epoch within a round
- **Atomic saves**: Crash-safe checkpoint writes
- **State versioning**: Forward-compatible checkpoint format
- **Best model tracking**: Automatic best checkpoint based on Dice score

## Main Components

::: dp_fedmed.fl.checkpoint.UnifiedCheckpoint
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.fl.checkpoint.UnifiedCheckpointManager
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## State Dataclasses

::: dp_fedmed.fl.checkpoint.ClientState
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.ServerState
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.RoundProgress
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.PrivacyState
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.EpochProgress
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Utility Functions

::: dp_fedmed.fl.checkpoint.save_unified_checkpoint
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.load_unified_checkpoint
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.fl.checkpoint.resolve_checkpoint_path
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Creating Initial Checkpoint

```python
from dp_fedmed.fl.checkpoint import UnifiedCheckpointManager
from pathlib import Path

# Initialize checkpoint manager
manager = UnifiedCheckpointManager(
    checkpoint_dir=Path("./checkpoints"),
    run_name="my_experiment",
    num_rounds=10,
    target_delta=1e-5,
)

# Create initial checkpoint
checkpoint = manager.create_initial_checkpoint(
    parameters=initial_model_params,
    num_clients=10,
    local_epochs=5,
)

# Save to disk
checkpoint_path = manager.save()  # Saves to checkpoints/last.pt
```

### Updating During Training

```python
# After aggregation, update server state
manager.update_server_state(
    parameters=aggregated_params,
    metrics={"dice": 0.85},
    round_num=3,
    cumulative_sample_epsilon=2.5,
    cumulative_user_epsilon=0.0,
)

# Mark round as completed
manager.mark_round_completed(dice_score=0.85)

# Save checkpoint (saves last + best if dice improved)
manager.save()
```

### Mid-Round Checkpointing

```python
# Client saves progress after each epoch
manager.update_client_epoch(
    client_id=0,
    epoch=2,  # Just completed epoch 2
    total_epochs=5,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    epoch_loss=0.35,
    partial_epsilon=0.5,
    partial_steps=50,
)

# Save checkpoint mid-round
manager.save()  # Can resume from this state if crash occurs
```

### Resuming from Checkpoint

```python
from dp_fedmed.fl.checkpoint import (
    load_unified_checkpoint,
    resolve_checkpoint_path,
)
from pathlib import Path

# Resolve checkpoint path from config
checkpoint_path = resolve_checkpoint_path(
    resume_from="last",  # or "best", or "/path/to/checkpoint.pt"
    run_dir=Path("./results/my_experiment"),
)

# Load checkpoint
checkpoint = load_unified_checkpoint(checkpoint_path)

# Check if mid-round resume
if checkpoint.round.status == "in_progress":
    print(f"Resuming from round {checkpoint.round.current}, mid-round")
    start_round = checkpoint.round.current
    client_states = checkpoint.clients
else:
    print(f"Resuming from round {checkpoint.round.current + 1}")
    start_round = checkpoint.round.current + 1
    client_states = {}

# Create manager with loaded state
manager = UnifiedCheckpointManager(
    checkpoint_dir=checkpoint_path.parent,
    run_name=checkpoint.run_name,
    num_rounds=checkpoint.round.total,
    target_delta=checkpoint.privacy.target_delta,
)
# Manager automatically loads last.pt if it exists

# Resume training with client states
for client_id, client_state in client_states.items():
    if client_state.epoch.status == "completed":
        # Client already finished this round
        print(f"Client {client_id} already completed")
    else:
        # Resume from saved epoch
        start_epoch = client_state.epoch.current
        print(f"Client {client_id} resuming from epoch {start_epoch}")
```

### Accessing Checkpoint Data

```python
# Load checkpoint
checkpoint = load_unified_checkpoint(Path("./checkpoints/best.pt"))

# Access server state
server_params = checkpoint.server.parameters  # List[np.ndarray]
best_dice = checkpoint.server.best_dice  # float

# Access client state
client_0_state = checkpoint.clients[0]
if client_0_state.epoch.status == "completed":
    # Client finished training
    final_params = client_0_state.final_parameters
    final_metrics = client_0_state.final_metrics
else:
    # Client was interrupted
    current_epoch = client_0_state.epoch.current
    model_state = client_0_state.model_state

# Access privacy state
cumulative_eps = checkpoint.privacy.cumulative_sample_epsilon
rdp_history = checkpoint.privacy.sample_history

# Round progress
current_round = checkpoint.round.current
total_rounds = checkpoint.round.total
status = checkpoint.round.status  # "in_progress" or "completed"
```

## Checkpoint File Structure

Checkpoints are saved as PyTorch `.pt` files with the following structure:

```
checkpoints/
├── last.pt      # Most recent checkpoint (any status)
└── best.pt      # Best model (highest Dice score, completed rounds only)
```

Each checkpoint file contains:

```python
{
    "version": "2.0",
    "timestamp": "2024-01-01T12:34:56",
    "run_name": "my_experiment",
    "round": {
        "current": 5,
        "total": 10,
        "status": "completed"
    },
    "server": {
        "parameters": [...],  # List[np.ndarray]
        "best_dice": 0.87,
        "metrics": {"dice": 0.87}
    },
    "clients": {
        0: {
            "client_id": 0,
            "epoch": {
                "current": 5,
                "total": 5,
                "status": "completed"
            },
            "model_state": {...},      # Optional
            "optimizer_state": {...},  # Optional
            "partial_metrics": {...},
            "partial_privacy": {...},
            "final_parameters": [...], # Cached results
            "final_metrics": {...},
            "num_samples": 100
        },
        # ... other clients
    },
    "privacy": {
        "target_delta": 1e-5,
        "sample_history": [(sigma, q, steps), ...],
        "user_history": [],
        "cumulative_sample_epsilon": 5.2,
        "cumulative_user_epsilon": 0.0,
        "partial_round_epsilon": 0.0
    }
}
```

See [Checkpoint Structure Schema](../schemas/checkpoint.md) for complete format details.

## Atomic Saves

Checkpoints are saved atomically using temp file + rename to prevent corruption:

1. Write checkpoint to temporary file: `checkpoints/.tmp_XXXXX.tmp`
2. Atomically rename to target: `checkpoints/last.pt`
3. If crash occurs during write, original checkpoint remains intact

This ensures checkpoints are always in a valid state.

## Version Compatibility

Checkpoint format version: `2.0`

- Version 2.x checkpoints include mid-round recovery support
- Version 1.x checkpoints are not compatible (missing client states)
- Future versions will provide migration paths

## See Also

- [Client API](client.md) - Client-side checkpoint usage
- [Server API](server.md) - Server-side checkpoint management
- [Checkpoint Structure Schema](../schemas/checkpoint.md) - Complete JSON format
- [Communication Flow](../protocol/communication.md) - Resume workflow
