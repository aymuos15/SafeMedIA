# Models API

The Models API provides utilities for creating and managing medical image segmentation models.

## Overview

DP-FedMed uses MONAI's UNet architecture with utilities for parameter conversion between PyTorch tensors and NumPy arrays (required for Flower's network serialization).

## Main Components

::: dp_fedmed.models.unet2d.create_unet2d
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.models.unet2d.get_parameters
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.models.unet2d.set_parameters
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Usage Examples

### Creating a UNet Model

```python
from dp_fedmed.models import create_unet2d
import torch

# Create UNet for binary segmentation
model = create_unet2d(
    in_channels=1,      # Grayscale input
    out_channels=2,     # Background + foreground
    channels=[16, 32, 64, 128],  # Feature channels at each level
    strides=[2, 2, 2],  # Downsampling strides
    num_res_units=2,    # Residual units per level
)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Model summary
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Parameter Conversion for Flower

```python
from dp_fedmed.models import get_parameters, set_parameters
from flwr.common import NDArrays

# Extract model parameters as NumPy arrays (for sending to server)
params: NDArrays = get_parameters(model)
# Returns: List[np.ndarray]

# Send params to server (Flower handles serialization)
# ...

# Restore model parameters from NumPy arrays (from server)
global_params: NDArrays = receive_from_server()
set_parameters(model, global_params)
```

### Working with Opacus-Wrapped Models

```python
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule

# Create model
model = create_unet2d(...)

# Wrap with Opacus
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Model is now GradSampleModule
assert isinstance(model, GradSampleModule)

# get_parameters() handles wrapped models automatically
params = get_parameters(model)  # Extracts from wrapped model

# Unwrap if needed
original_model = model._module  # Access underlying model
```

## Model Architecture

The UNet architecture consists of:

### Encoder (Downsampling Path)

- Convolutional blocks with residual connections
- Downsampling via strided convolutions
- Feature channels increase at each level

### Decoder (Upsampling Path)

- Transposed convolutions for upsampling
- Skip connections from encoder
- Feature channels decrease at each level

### Output Layer

- 1×1 convolution to output channels
- No activation (raw logits for loss functions)

### Example Architecture

For `channels=[16, 32, 64, 128]` and `strides=[2, 2, 2]`:

```
Input (1×256×256)
    ↓
Conv Block (16 channels)
    ↓ stride 2
Conv Block (32 channels)
    ↓ stride 2
Conv Block (64 channels)
    ↓ stride 2
Conv Block (128 channels) [Bottleneck]
    ↑ stride 2
Conv Block + Skip (64 channels)
    ↑ stride 2
Conv Block + Skip (32 channels)
    ↑ stride 2
Conv Block + Skip (16 channels)
    ↓
Output (2×256×256)
```

## Parameter Management

### get_parameters()

Extracts model parameters as NumPy arrays for Flower communication.

**Handles:**
- Regular PyTorch models (`nn.Module`)
- Opacus-wrapped models (`GradSampleModule`)
- DataParallel models

**Returns:**
- `List[np.ndarray]`: One array per model parameter (weights, biases)
- Maintains parameter order for correct restoration

### set_parameters()

Restores model parameters from NumPy arrays.

**Process:**
1. Iterates through model parameters and NumPy arrays in parallel
2. Converts NumPy array to PyTorch tensor
3. Copies data into model parameter (in-place)
4. Preserves gradient tracking settings

**Safety:**
- Validates parameter count matches
- Handles device placement (CPU/CUDA)
- Works with any model architecture

## Integration with Training

### Client-Side Usage

```python
from dp_fedmed.fl.client import DPFlowerClient
from dp_fedmed.models import create_unet2d

class MyClient(DPFlowerClient):
    def __init__(self, ...):
        # Model created internally from config
        self.model = create_unet2d(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            channels=model_config["channels"],
            strides=model_config["strides"],
            num_res_units=model_config["num_res_units"],
        )

    def get_parameters(self, config):
        # Extract parameters for server
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        # Receive parameters from server
        set_parameters(self.model, parameters)
```

### Server-Side Usage

```python
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Server works with NDArrays via Flower's Parameters object
# (never needs direct model instance)

# FedAvg automatically handles parameter aggregation
strategy = FedAvg(
    # ... config
)

# Parameters are aggregated as weighted averages
# No need for server-side model instantiation
```

## See Also

- [Client API](client.md) - Using models in client training
- [Configuration API](config.md) - Model configuration options
- [Tasks API](tasks.md) - Training and evaluation loops
