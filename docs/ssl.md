# Self-Supervised Learning (SSL)

Federated SSL pretraining for medical image segmentation using SimCLR contrastive learning with differential privacy.

## Overview

The SSL module provides federated pretraining of UNet encoders on unlabeled medical images. This enables better downstream segmentation performance, especially with limited labeled data.

**Stack:**

- **Lightly** - SimCLR contrastive learning with augmented view pairs
- **Opacus** - Differential privacy via per-sample gradient clipping (DP-SGD)
- **Flower** - Federated learning with FedAvg aggregation

## Quick Start

```bash
python3 scripts/run_ssl_pretraining.py
```

This runs federated SSL pretraining with the default config (`configs/pretraining.toml`).

### Custom Config

```bash
python3 scripts/run_ssl_pretraining.py configs/my_config.toml
```

## Architecture

### Training Pipeline

1. **Data Loading** - Unlabeled images from nnUNet format dataset
2. **Augmentation** - Lightly's `SimCLRTransform` generates two augmented views per image
3. **Forward Pass** - SSLUNet extracts features and projects to embedding space
4. **Contrastive Loss** - NTXentLoss compares embeddings of augmented pairs
5. **DP-SGD** - Opacus `GradSampleModule` enables per-sample gradient clipping
6. **Federation** - Flower aggregates model updates across clients

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `FederatedSSLClient` | `dp_fedmed/pretraining/federated_ssl_client.py` | Flower client with SSL training loop |
| `SSLUNet` | `dp_fedmed/pretraining/federated_ssl_client.py` | UNet + projection head for contrastive learning |
| `DPFedAvgSSL` | `dp_fedmed/pretraining/federated_ssl_strategy.py` | FedAvg strategy with privacy tracking |
| `SimCLRTransform` | `dp_fedmed/pretraining/transforms.py` | Medical image-friendly augmentations |

### Opacus + Lightly Compatibility

Opacus's `DPDataLoader` doesn't support tuple outputs from Lightly transforms. The solution:

```python
# Instead of full make_private():
model = GradSampleModule(model)  # Per-sample gradient computation
# Use original dataloader (not DPDataLoader)
```

This enables:

- Per-sample gradient clipping for privacy
- Compatibility with Lightly's `(view1, view2)` output format
- NTXentLoss contrastive learning

## Configuration

### Example Config

```toml
[data]
data_dir = "/path/to/nnUNet_raw/Dataset001_Cellpose"
image_size = [224, 224]

[ssl]
method = "simclr"
epochs = 2              # Local epochs per round
batch_size = 32
learning_rate = 0.001
temperature = 0.07      # Contrastive loss temperature
projection_dim = 128    # Projection head output

[federated]
num_clients = 2
num_rounds = 2
fraction_fit = 1.0

[privacy]
style = "sample"        # none, sample, user, hybrid
target_epsilon = 8.0
target_delta = 1e-5

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[checkpointing]
save_dir = "results/pretrained"
```

### Privacy Styles

| Style | Description |
|-------|-------------|
| `none` | No differential privacy |
| `sample` | Sample-level DP via Opacus DP-SGD |
| `user` | User-level DP with per-client clipping |
| `hybrid` | Combined sample + user-level DP |

## Output Files

After training:

```
results/pretrained/
├── best_model.pt           # Best encoder checkpoint
├── final_model.pt          # Final encoder checkpoint
├── server_metrics.json     # Server-side metrics
└── client_metrics/         # Per-client metrics
```

## Using Pretrained Models

Load the pretrained encoder for downstream federated segmentation:

```python
import torch
from dp_fedmed.models.unet2d import create_unet2d

# Load pretrained weights
pretrained_path = "results/pretrained/best_model.pt"
pretrained_weights = torch.load(pretrained_path)

# Create model for downstream task
model = create_unet2d(
    in_channels=1,
    out_channels=2,  # Number of segmentation classes
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
)

# Load pretrained encoder weights
model.load_state_dict(pretrained_weights, strict=False)
```

## Troubleshooting

### CUDA out of memory

Reduce batch size:

```toml
[ssl]
batch_size = 16  # or 8
```

### No data found

Verify data directory:

```bash
ls /path/to/nnUNet_raw/Dataset001_Cellpose/imagesTr/
```

### Ray initialization errors

Restart Ray:

```bash
python3 -c "import ray; ray.shutdown()"
python3 scripts/run_ssl_pretraining.py
```

## API Reference

### Main Exports

```python
from dp_fedmed.pretraining import (
    SSLConfig,
    save_pretrained_checkpoint,
    load_pretrained_encoder,
    server_app,
    client_app,
)
```

### Programmatic Usage

```python
from dp_fedmed.pretraining import SSLConfig

config = SSLConfig.from_toml("configs/pretraining.toml")
config.validate()
```
