# Checkpoint Structure

Complete reference for unified checkpoint file format.

## Overview

Checkpoints are saved as PyTorch `.pt` files containing the complete FL system state, enabling mid-round recovery. Checkpoints use atomic saves (temp file + rename) to prevent corruption.

## File Locations

```
results/
└── {run_name}/
    └── checkpoints/
        ├── last.pt     # Most recent checkpoint (any status)
        └── best.pt     # Best model (highest Dice, completed only)
```

## Checkpoint Structure

### Top-Level Schema

```python
{
    "version": str,           # Checkpoint format version ("2.0")
    "timestamp": str,         # ISO 8601 timestamp
    "run_name": str,          # Experiment name
    "round": RoundProgress,   # Round state
    "server": ServerState,    # Server/global model state
    "clients": Dict[int, ClientState],  # Per-client states
    "privacy": PrivacyState,  # Privacy accounting state
}
```

### RoundProgress

```python
{
    "current": int,    # Current round number (1-indexed)
    "total": int,      # Total number of rounds
    "status": str,     # "in_progress" or "completed"
}
```

### ServerState

```python
{
    "parameters": List[np.ndarray],  # Global model parameters
    "best_dice": float,              # Best Dice score so far
    "metrics": Dict[str, float],     # Latest metrics {"dice": ...}
}
```

### ClientState

```python
{
    "client_id": int,
    "epoch": EpochProgress,
    "model_state": Optional[Dict[str, torch.Tensor]],    # For mid-epoch resume
    "optimizer_state": Optional[Dict[str, Any]],         # For mid-epoch resume
    "partial_metrics": Dict[str, float],                 # Accumulated metrics
    "partial_privacy": Dict[str, float],                 # Partial privacy
    "final_parameters": Optional[List[np.ndarray]],      # Cached if completed
    "final_metrics": Optional[Dict[str, float]],         # Cached if completed
    "num_samples": int,                                  # Training samples
}
```

**EpochProgress:**
```python
{
    "current": int,    # Current epoch (0-indexed during, 1-indexed after)
    "total": int,      # Total local epochs
    "status": str,     # "in_progress", "completed", or "pending"
}
```

**partial_metrics:**
```python
{
    "loss_sum": float,      # Accumulated loss across completed epochs
    "epochs_done": int,     # Number of completed epochs
}
```

**partial_privacy:**
```python
{
    "epsilon": float,  # Privacy spent so far this round
    "steps": int,      # DP-SGD steps completed
}
```

### PrivacyState

```python
{
    "target_delta": float,
    "sample_history": List[Tuple[float, float, int]],  # (sigma, q, steps)
    "user_history": List[Tuple[float, float, int]],    # (sigma, q, steps)
    "cumulative_sample_epsilon": float,
    "cumulative_user_epsilon": float,
    "partial_round_epsilon": float,  # For mid-round resume
}
```

## Complete Example

### Fresh Training Checkpoint

```python
{
    "version": "2.0",
    "timestamp": "2024-01-15T10:30:45.123456",
    "run_name": "experiment_001",
    "round": {
        "current": 3,
        "total": 10,
        "status": "completed"
    },
    "server": {
        "parameters": [
            np.array(...),  # Model weights
            np.array(...),  # Model biases
            # ... more parameters
        ],
        "best_dice": 0.85,
        "metrics": {
            "dice": 0.85
        }
    },
    "clients": {
        0: {
            "client_id": 0,
            "epoch": {
                "current": 5,
                "total": 5,
                "status": "completed"
            },
            "model_state": None,
            "optimizer_state": None,
            "partial_metrics": {
                "loss_sum": 0.0,
                "epochs_done": 0
            },
            "partial_privacy": {
                "epsilon": 0.0,
                "steps": 0
            },
            "final_parameters": None,
            "final_metrics": None,
            "num_samples": 0
        },
        1: {
            # ... similar structure
        }
    },
    "privacy": {
        "target_delta": 1e-5,
        "sample_history": [
            (1.0, 0.1, 100),  # Round 1
            (1.0, 0.1, 100),  # Round 2
            (1.0, 0.1, 100),  # Round 3
        ],
        "user_history": [],
        "cumulative_sample_epsilon": 2.8,
        "cumulative_user_epsilon": 0.0,
        "partial_round_epsilon": 0.0
    }
}
```

### Mid-Round Checkpoint

Checkpoint saved after client 0 completed 3/5 epochs before crash:

```python
{
    "version": "2.0",
    "timestamp": "2024-01-15T11:15:30.456789",
    "run_name": "experiment_001",
    "round": {
        "current": 4,
        "total": 10,
        "status": "in_progress"  # ← Mid-round!
    },
    "server": {
        "parameters": [...],  # From round 3
        "best_dice": 0.85,
        "metrics": {"dice": 0.85}
    },
    "clients": {
        0: {
            "client_id": 0,
            "epoch": {
                "current": 3,       # Completed 3 epochs
                "total": 5,
                "status": "in_progress"  # ← Mid-training!
            },
            "model_state": {
                "conv1.weight": torch.Tensor(...),
                "conv1.bias": torch.Tensor(...),
                # ... full model state dict
            },
            "optimizer_state": {
                "state": {...},
                "param_groups": [...]
            },
            "partial_metrics": {
                "loss_sum": 1.05,  # Sum of 3 epoch losses
                "epochs_done": 3
            },
            "partial_privacy": {
                "epsilon": 0.5,    # Epsilon for 3 epochs
                "steps": 60       # 60 out of 100 steps
            },
            "final_parameters": None,
            "final_metrics": None,
            "num_samples": 100
        },
        1: {
            "client_id": 1,
            "epoch": {
                "current": 0,
                "total": 5,
                "status": "pending"  # ← Not started yet
            },
            # ... rest is None/0
        }
    },
    "privacy": {
        "target_delta": 1e-5,
        "sample_history": [
            (1.0, 0.1, 100),  # Rounds 1-3 completed
            (1.0, 0.1, 100),
            (1.0, 0.1, 100),
        ],
        "user_history": [],
        "cumulative_sample_epsilon": 2.8,
        "cumulative_user_epsilon": 0.0,
        "partial_round_epsilon": 0.5  # Partial epsilon for round 4
    }
}
```

## Loading Checkpoints

### Python API

```python
from dp_fedmed.fl.checkpoint import load_unified_checkpoint
from pathlib import Path

# Load checkpoint
checkpoint = load_unified_checkpoint(Path("checkpoints/last.pt"))

# Access fields
print(f"Round: {checkpoint.round.current}/{checkpoint.round.total}")
print(f"Status: {checkpoint.round.status}")
print(f"Best Dice: {checkpoint.server.best_dice:.4f}")

# Check for mid-round resume
if checkpoint.round.status == "in_progress":
    print("Mid-round resume required")
    for client_id, client_state in checkpoint.clients.items():
        print(f"Client {client_id}: epoch {client_state.epoch.current}/{client_state.epoch.total}")
```

### Resume Workflow

1. **Load checkpoint:**
   ```python
   checkpoint = load_unified_checkpoint(checkpoint_path)
   ```

2. **Determine resume type:**
   ```python
   if checkpoint.round.status == "completed":
       # Normal resume: start next round
       start_round = checkpoint.round.current + 1
       client_states = {}
   else:
       # Mid-round resume: continue current round
       start_round = checkpoint.round.current
       client_states = checkpoint.clients
   ```

3. **Create strategy with resume state:**
   ```python
   strategy = DPFedAvg(
       start_round=start_round,
       is_mid_round_resume=(checkpoint.round.status == "in_progress"),
       client_resume_states=client_states,
       # ... other params
   )
   ```

4. **Clients resume training:**
   - If `resume_from_checkpoint=True` in config, client loads saved state
   - Continues from saved epoch
   - Accumulates partial metrics

## Checkpoint Versioning

### Version 2.0 (Current)

- **Features:** Full mid-round recovery support
- **Format:** Includes client states, partial metrics, privacy
- **Migration:** Not backward compatible with v1.x

### Version 1.0 (Legacy)

- **Features:** Round-level checkpoints only
- **Limitation:** Cannot resume mid-round
- **Status:** Deprecated

## See Also

- [Checkpoint API](../api/checkpoint.md) - Manager and utilities
- [Server API](../api/server.md) - Server-side checkpointing
- [Client API](../api/client.md) - Client-side resume
- [Communication Flow](../protocol/communication.md) - Resume workflow
