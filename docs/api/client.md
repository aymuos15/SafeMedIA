# Client API

The Client API provides components for privacy-preserving local training in federated learning.

## Overview

The `DPFlowerClient` class extends Flower's `NumPyClient` to support differential privacy using Opacus. It handles:

- Local model training with DP-SGD (sample-level DP)
- Privacy budget tracking per round
- Checkpoint-based mid-round recovery
- Metrics logging and persistence

## Main Components

::: dp_fedmed.fl.client.dp_client.DPFlowerClient
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Factory Function

::: dp_fedmed.fl.client.factory.client_fn
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Usage Example

### Basic Client Creation

```python
from dp_fedmed.fl.client import DPFlowerClient
import torch

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    drop_last=True,  # Required for DP
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

# Create DP client
client = DPFlowerClient(
    train_loader=train_loader,
    test_loader=test_loader,
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "channels": [16, 32, 64, 128],
        "strides": [2, 2, 2],
        "num_res_units": 2,
    },
    training_config={
        "local_epochs": 5,
        "learning_rate": 0.001,
        "momentum": 0.9,
    },
    privacy_config={
        "style": "sample",  # Enable sample-level DP
        "target_delta": 1e-5,
        "sample": {
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
        },
    },
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    client_id=0,
)
```

### Training Round

```python
# Receive global model parameters from server
global_parameters = server.get_parameters()

# Configuration from server
config = {
    "noise_multiplier": 1.0,
    "server_round": 1,
    "local_epochs": 5,
}

# Train locally
updated_params, num_samples, metrics = client.fit(global_parameters, config)

# Metrics returned:
# {
#     "loss": 0.245,
#     "epsilon": 0.85,
#     "delta": 1e-5,
#     "sample_rate": 0.1,
#     "steps": 100,
# }
```

### Evaluation Round

```python
# Evaluate on local test data
loss, num_samples, metrics = client.evaluate(global_parameters, config)

# Metrics returned:
# {
#     "dice": 0.89,
# }
```

### Mid-Round Resume

When a client crashes mid-training, it can resume from the last saved epoch:

```python
# Server signals resume with checkpoint path
resume_config = {
    "noise_multiplier": 1.0,
    "server_round": 3,
    "local_epochs": 5,
    "resume_from_checkpoint": True,
    "checkpoint_path": "/path/to/checkpoints/last.pt",
}

# Client automatically loads saved state and continues training
params, num_samples, metrics = client.fit(global_parameters, resume_config)
```

## Configuration Keys

### fit() Config Dictionary

Sent from server to client in `configure_fit()`:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `noise_multiplier` | float | Yes | Pre-computed noise multiplier for DP-SGD |
| `server_round` | int | Yes | Current round number (for logging) |
| `local_epochs` | int | Yes | Number of local training epochs |
| `resume_from_checkpoint` | bool | No | Whether to resume from saved state (default: False) |
| `checkpoint_path` | str | No | Path to checkpoint file (required if resuming) |

### fit() Return Metrics

Sent from client to server after training:

| Key | Type | Description |
|-----|------|-------------|
| `loss` | float | Average training loss across all epochs |
| `epsilon` | float | Privacy budget (ε) spent in this round |
| `delta` | float | Privacy parameter δ (target_delta from config) |
| `sample_rate` | float | Batch sampling rate for DP-SGD |
| `steps` | int | Number of DP-SGD steps performed |

### evaluate() Return Metrics

Sent from client to server after evaluation:

| Key | Type | Description |
|-----|------|-------------|
| `dice` | float | Dice coefficient score on local test data |

## Privacy Guarantees

The `DPFlowerClient` provides **sample-level differential privacy** using Opacus:

- **Gradient clipping**: Per-sample gradients clipped to `max_grad_norm`
- **Noise addition**: Gaussian noise scaled by `noise_multiplier`
- **Privacy accounting**: RDP-based composition via Opacus PrivacyEngine

**Privacy budget** (ε) is computed per round and sent to the server for global accounting.

## See Also

- [Server API](server.md) - Server-side aggregation and strategy
- [Privacy API](privacy.md) - Privacy budget accounting
- [Checkpoint API](checkpoint.md) - State persistence and recovery
- [Message Schemas](../schemas/messages.md) - Complete message format reference
