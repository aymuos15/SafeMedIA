# Configuration Schema

Complete reference for TOML configuration files.

## Overview

DP-FedMed uses TOML configuration files with validation via Pydantic. All configuration values are validated at load time with clear error messages.

## Configuration Sections

### [data]

Dataset and data loading configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `data_dir` | Path | - | Yes | Must exist | Path to dataset directory |
| `image_size` | int | 256 | No | 16-2048 | Image resize dimension (square) |
| `batch_size` | int | 8 | No | 1-512 | Batch size for training |
| `num_workers` | int | 2 | No | 0-32 | DataLoader worker processes |

**Example:**
```toml
[data]
data_dir = "/path/to/Dataset001_Cellpose"
image_size = 256
batch_size = 8
num_workers = 2
```

### [model]

UNet architecture configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `in_channels` | int | 1 | No | ≥ 1 | Input image channels (1=grayscale, 3=RGB) |
| `out_channels` | int | 2 | No | ≥ 1 | Output segmentation classes |
| `channels` | list[int] | [16, 32, 64, 128] | No | Non-empty, all > 0 | Feature channels at each level |
| `strides` | list[int] | [2, 2, 2] | No | len = len(channels)-1 | Downsampling strides |
| `num_res_units` | int | 2 | No | ≥ 0 | Residual units per level |
| `dropout` | float | 0.0 | No | 0.0-1.0 | Dropout probability |

**Example:**
```toml
[model]
in_channels = 1
out_channels = 2
channels = [16, 32, 64, 128]
strides = [2, 2, 2]
num_res_units = 2
dropout = 0.0
```

**Architecture Notes:**
- `channels` length determines network depth
- `strides` must have exactly len(channels)-1 elements
- Larger `channels` values increase model capacity but also memory usage

### [federated]

Federated learning parameters.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `num_clients` | int | 2 | No | ≥ 1 | Total number of clients |
| `num_rounds` | int | 5 | No | ≥ 1 | Number of FL rounds |
| `fraction_fit` | float | 1.0 | No | 0.0-1.0 | Fraction of clients for training |
| `fraction_evaluate` | float | 1.0 | No | 0.0-1.0 | Fraction of clients for evaluation |
| `min_fit_clients` | int | 2 | No | ≥ 1 | Minimum clients required for training |
| `min_evaluate_clients` | int | 2 | No | ≥ 1 | Minimum clients for evaluation |
| `min_available_clients` | int | 2 | No | ≥ 1 | Minimum available to start |

**Example:**
```toml
[federated]
num_clients = 10
num_rounds = 20
fraction_fit = 1.0
fraction_evaluate = 1.0
min_fit_clients = 8
min_evaluate_clients = 8
min_available_clients = 8
```

**Sampling Notes:**
- `fraction_fit * num_clients` clients selected per round
- Must satisfy: `fraction_fit * num_clients ≥ min_fit_clients`

**User-Level DP Warning:**
- For `privacy.style = "user"` or `"hybrid"`: **min_fit_clients ≥ 10 recommended**
- With <10 clients, noise overwhelms signal → poor convergence/utility
- See [Types of DP - Practical Considerations](../types_of_dp.md#practical-considerations-for-user-level-dp) for details

### [client_resources]

Per-client resource allocation (for simulation).

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `num_cpus` | int | 1 | No | ≥ 1 | CPU cores per client |
| `num_gpus` | float | 0.3 | No | 0.0-8.0 | GPU fraction per client |

**Example:**
```toml
[client_resources]
num_cpus = 2
num_gpus = 0.5  # 2 clients can share 1 GPU
```

### [training]

Training hyperparameters.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `local_epochs` | int | 5 | No | ≥ 1 | Epochs per round per client |
| `learning_rate` | float | 0.001 | No | > 0.0 | Optimizer learning rate |
| `optimizer` | str | "sgd" | No | {"sgd", "adam", "adamw"} | Optimizer type |
| `momentum` | float | 0.9 | No | 0.0-1.0 | SGD momentum (if optimizer="sgd") |

**Example:**
```toml
[training]
local_epochs = 5
learning_rate = 0.001
optimizer = "sgd"
momentum = 0.9
```

### [privacy]

Differential privacy configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `style` | str | "sample" | No | {"none", "sample", "user", "hybrid"} | DP mechanism to use |
| `target_delta` | float | 1e-5 | No | 0.0 < x < 1.0 | Privacy parameter δ |
| `client_dataset_size` | int | None | No | ≥ 1 | Samples per client (for budget calc) |

**DP Styles:**
- `"none"`: No differential privacy (baseline)
- `"sample"`: Sample-level DP via Opacus DP-SGD
- `"user"`: User-level DP via client sampling noise
- `"hybrid"`: Both sample and user-level DP

**Example:**
```toml
[privacy]
style = "sample"
target_delta = 1.0e-5
client_dataset_size = 100
```

### [privacy.sample]

Sample-level DP configuration (Opacus).

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `noise_multiplier` | float | 1.0 | No | > 0.0 | Gaussian noise scale σ |
| `max_grad_norm` | float | 1.0 | No | > 0.0 | Per-sample gradient clipping bound |

**Example:**
```toml
[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0
```

**Privacy-Utility Tradeoff:**
- Higher `noise_multiplier` → stronger privacy, lower utility
- Lower `max_grad_norm` → stronger privacy, may hurt convergence

### [privacy.user]

User-level DP configuration (server-side).

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `noise_multiplier` | float | 0.5 | No | > 0.0 | Server-side noise scale |
| `max_grad_norm` | float | 0.1 | No | > 0.0 | Client update clipping bound |

**Example:**
```toml
[privacy.user]
noise_multiplier = 0.5
max_grad_norm = 0.1
```

### [loss]

Loss function configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `type` | str | "cross_entropy" | No | {"cross_entropy", "soft_dice", "dice_ce"} | Loss function type |
| `dice_smooth` | float | 1.0 | No | ≥ 0.0 | Laplace smoothing for Dice |
| `dice_include_background` | bool | false | No | - | Include background in Dice |
| `dice_weight` | float | 0.5 | No | 0.0-1.0 | Weight for Dice term (if dice_ce) |
| `ce_weight` | float | 0.5 | No | 0.0-1.0 | Weight for CE term (if dice_ce) |

**Example:**
```toml
[loss]
type = "dice_ce"
dice_smooth = 1.0
dice_include_background = false
dice_weight = 0.5
ce_weight = 0.5
```

**Loss Types:**
- `"cross_entropy"`: Standard pixel-wise classification
- `"soft_dice"`: Optimizes Dice coefficient directly
- `"dice_ce"`: Weighted combination of Dice and CE

### [checkpointing]

Checkpoint configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `enabled` | bool | true | No | - | Enable checkpointing |
| `checkpoint_dir` | str | "checkpoints" | No | - | Directory name (relative to run_dir) |
| `resume_from` | str | "" | No | {"", "last", "best", path} | Checkpoint to resume from |

**Example:**
```toml
[checkpointing]
enabled = true
checkpoint_dir = "checkpoints"
resume_from = "last"  # or "best", or "/path/to/checkpoint.pt", or ""
```

**Resume Options:**
- `""` (empty): Start fresh run
- `"last"`: Resume from last checkpoint
- `"best"`: Resume from best checkpoint (highest Dice)
- Absolute path: Resume from specific checkpoint file

### [logging]

Logging configuration.

| Key | Type | Default | Required | Constraints | Description |
|-----|------|---------|----------|-------------|-------------|
| `save_model` | bool | true | No | - | Save model checkpoints |
| `save_metrics` | bool | true | No | - | Save metrics JSON files |
| `level` | str | "INFO" | No | {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} | Log level |

**Example:**
```toml
[logging]
save_model = true
save_metrics = true
level = "INFO"
```

## Complete Example

```toml
# Complete DP-FedMed Configuration

[data]
data_dir = "/path/to/Dataset001_Cellpose"
image_size = 256
batch_size = 8
num_workers = 2

[model]
in_channels = 1
out_channels = 2
channels = [16, 32, 64, 128]
strides = [2, 2, 2]
num_res_units = 2
dropout = 0.0

[federated]
num_clients = 10
num_rounds = 20
fraction_fit = 1.0
fraction_evaluate = 1.0
min_fit_clients = 8
min_evaluate_clients = 8
min_available_clients = 8

[client_resources]
num_cpus = 2
num_gpus = 0.5

[training]
local_epochs = 5
learning_rate = 0.001
optimizer = "sgd"
momentum = 0.9

[privacy]
style = "sample"
target_delta = 1.0e-5
client_dataset_size = 100

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[privacy.user]
noise_multiplier = 0.0
max_grad_norm = 0.0

[loss]
type = "dice_ce"
dice_smooth = 1.0
dice_include_background = false
dice_weight = 0.5
ce_weight = 0.5

[checkpointing]
enabled = true
checkpoint_dir = "checkpoints"
resume_from = ""

[logging]
save_model = true
save_metrics = true
level = "INFO"
```

## Preset Configurations

DP-FedMed includes several preset configurations:

### configs/default.toml
- **Purpose**: Balanced privacy-utility tradeoff
- **DP Style**: Hybrid (sample + user)
- **Settings**: 2 clients, 5 rounds, moderate privacy

### configs/none.toml
- **Purpose**: Baseline (no privacy)
- **DP Style**: None
- **Settings**: Fast training, maximum utility

### configs/sample.toml
- **Purpose**: Sample-level DP only
- **DP Style**: Sample
- **Settings**: Opacus DP-SGD

### configs/user.toml
- **Purpose**: User-level DP only
- **DP Style**: User
- **Settings**: Client sampling noise

## Validation

All configuration values are validated at load time:

```python
from dp_fedmed.config import load_config

try:
    config = load_config("configs/my_config.toml")
except ValidationError as e:
    # Detailed error messages
    for error in e.errors():
        print(f"{error['loc']}: {error['msg']}")
```

Common validation errors:
- Missing required fields (data_dir)
- Out-of-range values (batch_size < 1)
- Invalid types (channels must be list[int])
- Constraint violations (strides length mismatch)

## See Also

- [Configuration API](../api/config.md) - Config loading and validation
- [Privacy API](../api/privacy.md) - Privacy accounting details
- [Types of DP](../types_of_dp.md) - DP style explanations
