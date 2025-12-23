# Server API

The Server API provides the federated aggregation strategy with privacy budget tracking and checkpointing.

## Overview

The `DPFedAvg` class extends Flower's `FedAvg` strategy to:

- Track privacy budget consumption across rounds (sample-level and user-level)
- Save unified checkpoints for mid-round recovery
- Aggregate client metrics (loss, Dice score, privacy)
- Persist round history and final results

## Main Components

::: dp_fedmed.fl.server.strategy.DPFedAvg
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Factory Function

::: dp_fedmed.fl.server.factory.server_fn
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Aggregation Functions

::: dp_fedmed.fl.server.aggregation.weighted_average
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Example

### Basic Strategy Creation

```python
from dp_fedmed.fl.server import DPFedAvg
from pathlib import Path

strategy = DPFedAvg(
    # Privacy parameters
    target_delta=1e-5,
    noise_multiplier=1.0,  # Pre-computed for sample-level DP
    max_grad_norm=1.0,
    user_noise_multiplier=0.0,  # Set > 0 for user-level DP
    user_max_grad_norm=0.0,

    # Federated learning parameters
    num_rounds=10,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,

    # Checkpointing
    run_dir=Path("./results/my_run"),
    run_name="my_run",
    save_metrics=True,

    # Client population
    total_clients=10,
    local_epochs=5,
)
```

### Training Round Lifecycle

```python
# 1. Configure fit: Send config to selected clients
fit_instructions = strategy.configure_fit(
    server_round=1,
    parameters=global_parameters,
    client_manager=client_manager,
)
# Returns: [(client, FitIns(parameters, config)), ...]

# 2. Clients train locally and return results
# (handled by Flower framework)

# 3. Aggregate fit results
aggregated_params, aggregated_metrics = strategy.aggregate_fit(
    server_round=1,
    results=client_fit_results,
    failures=[],
)
# Aggregates parameters via FedAvg
# Records privacy budget in PrivacyAccountant
# Returns aggregated metrics:
# {
#     "round_sample_epsilon": 0.85,
#     "cumulative_sample_epsilon": 2.5,
#     "cumulative_user_epsilon": 0.0,
# }

# 4. Configure evaluate: Send config to clients
eval_instructions = strategy.configure_evaluate(
    server_round=1,
    parameters=aggregated_params,
    client_manager=client_manager,
)

# 5. Aggregate evaluation results
loss, metrics = strategy.aggregate_evaluate(
    server_round=1,
    results=client_eval_results,
    failures=[],
)
# Computes weighted average of Dice scores
# Saves checkpoint (last + best)
# Returns aggregated metrics:
# {
#     "dice": 0.87,
# }
```

### Checkpoint Resumption

```python
from dp_fedmed.fl.checkpoint import (
    UnifiedCheckpointManager,
    load_unified_checkpoint,
)

# Load existing checkpoint
checkpoint_path = Path("./results/my_run/checkpoints/last.pt")
checkpoint = load_unified_checkpoint(checkpoint_path)

# Create checkpoint manager with loaded state
checkpoint_manager = UnifiedCheckpointManager(
    checkpoint_dir=checkpoint_path.parent,
    run_name=checkpoint.run_name,
    num_rounds=checkpoint.round.total,
    target_delta=checkpoint.privacy.target_delta,
)

# Determine resume parameters
is_mid_round = checkpoint.round.status == "in_progress"
start_round = checkpoint.round.current
client_resume_states = checkpoint.clients if is_mid_round else {}

# Create strategy with resume state
strategy = DPFedAvg(
    target_delta=checkpoint.privacy.target_delta,
    start_round=start_round,
    is_mid_round_resume=is_mid_round,
    client_resume_states=client_resume_states,
    checkpoint_manager=checkpoint_manager,
    # ... other parameters
)
```

## Methods

### configure_fit()

Prepares configuration for client training rounds.

**Configuration sent to clients:**

```python
config = {
    "noise_multiplier": float,      # Pre-computed DP noise
    "server_round": int,            # Current round number
    "local_epochs": int,            # Epochs to train
    "resume_from_checkpoint": bool, # Mid-round resume flag
    "checkpoint_path": str,         # Path to checkpoint (if resuming)
}
```

### aggregate_fit()

Aggregates client training results and updates privacy accounting.

**Process:**

1. Collect per-client metrics (loss, epsilon, sample_rate, steps)
2. Calculate average epsilon and sample rate
3. Record round in `PrivacyAccountant`
4. Aggregate parameters using FedAvg
5. Store server-level round metrics

**Returns:**

- Aggregated parameters (FedAvg weighted average)
- Aggregated metrics with privacy budget

### aggregate_evaluate()

Aggregates client evaluation results and saves checkpoints.

**Process:**

1. Collect per-client Dice scores and losses
2. Calculate weighted average (by number of samples)
3. Update server round metrics
4. Save unified checkpoint (last + best)
5. If final round, save logs to disk

**Returns:**

- Aggregated loss (weighted average)
- Aggregated metrics (Dice score)

## Privacy Accounting

The server strategy integrates with `PrivacyAccountant` to track cumulative privacy budget:

### Sample-Level Privacy

From local DP-SGD training:

- **Noise**: Applied by Opacus on client side
- **Tracking**: Client reports per-round epsilon
- **Composition**: Server uses RDP accountant to compose across rounds

### User-Level Privacy

From client sampling (optional):

- **Noise**: Applied to aggregated parameters on server
- **Tracking**: Based on fraction of clients participating
- **Composition**: RDP composition for client-level privacy

## Metrics and Logging

The strategy saves three types of outputs:

### 1. metrics.json (Final Summary)

```json
{
  "config": "my_run",
  "start_time": "2024-01-01T12:00:00",
  "end_time": "2024-01-01T14:30:00",
  "num_rounds": 10,
  "final_dice": 0.87,
  "final_loss": 0.23,
  "privacy": {
    "target_delta": 1e-5,
    "cumulative_sample_epsilon": 8.5,
    "cumulative_user_epsilon": 0.0,
    "num_rounds": 10
  }
}
```

### 2. history.json (Per-Round Data)

```json
{
  "server": {
    "rounds": [
      {
        "round": 1,
        "sample_epsilon": 0.85,
        "cumulative_sample_epsilon": 0.85,
        "cumulative_user_epsilon": 0.0,
        "num_clients": 10,
        "num_failures": 0,
        "aggregated_dice": 0.75,
        "aggregated_loss": 0.45
      }
    ]
  },
  "clients": [
    {
      "client_id": "0",
      "rounds": [
        {
          "round": 1,
          "train_loss": 0.42,
          "sample_epsilon": 0.83,
          "delta": 1e-5,
          "num_samples": 100,
          "dice": 0.76,
          "eval_loss": 0.38
        }
      ]
    }
  ]
}
```

### 3. Checkpoints (State Files)

See [Checkpoint API](checkpoint.md) for details.

## See Also

- [Client API](client.md) - Client-side training
- [Privacy API](privacy.md) - Privacy budget accounting
- [Checkpoint API](checkpoint.md) - State persistence
- [Communication Flow](../protocol/communication.md) - Round lifecycle details
