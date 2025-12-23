# Configuration API

The Configuration API provides type-safe configuration loading and validation using Pydantic.

## Overview

DP-FedMed uses TOML files for configuration with automatic validation via Pydantic models. This ensures:

- **Type safety**: All configuration values are validated at load time
- **Default values**: Missing values use sensible defaults
- **Error messages**: Clear validation errors with field names
- **IDE support**: Full autocomplete and type hints

## Main Components

::: dp_fedmed.config.Config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.config.load_config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Configuration Sections

::: dp_fedmed.config.DataConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.ModelConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.FederatedConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.TrainingConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.PrivacyConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.LossConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.CheckpointingConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.config.LoggingConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Enums

::: dp_fedmed.config.DPStyle
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Loading Configuration

```python
from dp_fedmed.config import load_config
from pathlib import Path

# Load from TOML file
config = load_config("configs/default.toml")

# Access configuration values
data_dir = config.data.data_dir  # Path
batch_size = config.data.batch_size  # int
num_rounds = config.federated.num_rounds  # int
dp_style = config.privacy.style  # DPStyle enum
```

### Accessing Nested Values

```python
# Direct attribute access (type-safe)
learning_rate = config.training.learning_rate  # float

# Dot notation (backward compatible)
learning_rate = config.get("training.learning_rate", 0.001)  # Any

# Get entire section as dict
training_dict = config.get_section("training")
# {"local_epochs": 5, "learning_rate": 0.001, ...}
```

### Creating Configuration Programmatically

```python
from dp_fedmed.config import (
    Config,
    DataConfig,
    ModelConfig,
    PrivacyConfig,
    DPStyle,
)
from pathlib import Path

config = Config(
    data=DataConfig(
        data_dir=Path("/path/to/dataset"),
        image_size=256,
        batch_size=8,
    ),
    model=ModelConfig(
        in_channels=1,
        out_channels=2,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
    ),
    privacy=PrivacyConfig(
        style=DPStyle.SAMPLE,
        target_delta=1e-5,
    ),
    # ... other sections
)
```

### Validation Examples

```python
from pydantic import ValidationError

try:
    config = load_config("configs/invalid.toml")
except ValidationError as e:
    # Detailed error messages
    print(e)
    # [
    #   {
    #     'loc': ('data', 'batch_size'),
    #     'msg': 'ensure this value is greater than or equal to 1',
    #     'type': 'value_error.number.not_ge',
    #   }
    # ]
```

### Merging Configurations

```python
from dp_fedmed.config import merge_configs
import tomli

# Load base config
with open("configs/default.toml", "rb") as f:
    base = tomli.load(f)

# Load override config
with open("configs/override.toml", "rb") as f:
    override = tomli.load(f)

# Merge (override takes precedence)
merged = merge_configs(base, override)

# Create validated config from merged dict
config = Config(**merged)
```

## Validation Rules

### Data Configuration

- `data_dir`: Must exist on filesystem
- `image_size`: 16 ≤ value ≤ 2048
- `batch_size`: 1 ≤ value ≤ 512
- `num_workers`: 0 ≤ value ≤ 32

### Model Configuration

- `channels`: Non-empty list of positive integers
- `strides`: Length must be len(channels) - 1
- `dropout`: 0.0 ≤ value ≤ 1.0

### Federated Configuration

- `fraction_fit`, `fraction_evaluate`: 0.0 ≤ value ≤ 1.0
- `min_fit_clients`, `min_evaluate_clients`: ≥ 1

### Training Configuration

- `learning_rate`: > 0.0
- `momentum`: 0.0 ≤ value ≤ 1.0
- `optimizer`: Must be "sgd", "adam", or "adamw"

### Privacy Configuration

- `target_delta`: 0.0 < value < 1.0
- `style`: Must be DPStyle enum value
- `noise_multiplier`: > 0.0
- `max_grad_norm`: > 0.0

### Loss Configuration

- `type`: Must be "cross_entropy", "soft_dice", or "dice_ce"
- `dice_weight`, `ce_weight`: 0.0 ≤ value ≤ 1.0

### Checkpointing Configuration

- `resume_from`: Must be "last", "best", or absolute path that exists
  - Keywords are resolved at runtime by server factory

## Configuration Files

DP-FedMed includes several preset configurations:

### configs/default.toml

Standard configuration for most use cases:
- 2 clients, 5 rounds
- Sample-level DP (ε=8.0, δ=1e-5)
- Batch size 8, local epochs 5

### configs/none.toml

No differential privacy (for baseline):
- DP style: "none"
- Faster training, no privacy guarantees

### configs/user.toml

User-level DP only:
- DP style: "user"
- Privacy from client sampling

### configs/sample.toml

Sample-level DP only (Opacus):
- DP style: "sample"
- Privacy from DP-SGD

## See Also

- [Configuration Schema](../schemas/config.md) - Complete TOML reference
- [Client API](client.md) - Using config in client
- [Server API](server.md) - Using config in server
