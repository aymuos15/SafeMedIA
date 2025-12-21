# DP-FedMed: Differential Privacy + Federated Learning for Medical Imaging

Privacy-preserving federated learning framework combining **Flower + Opacus + MONAI** for medical image segmentation with differential privacy guarantees.

## Features

- **Differential Privacy**: Per-sample gradient clipping with Opacus DP-SGD
- **Federated Learning**: Decentralized training using Flower framework  
- **Medical Imaging**: MONAI-based 2D UNet optimized for medical image segmentation
- **Privacy Accounting**: Track cumulative privacy budget (ε, δ) across federated rounds
- **YAML Configuration**: All settings managed through clean YAML config files
- **No Centralized Training**: Pure federated architecture - all training happens on clients

## Quick Start

### 1. Installation

```bash
cd dp_fl_unet
pip install -e .
```

### 2. Configure Your Dataset

Edit `configs/default.yaml` and set your data directory:

```yaml
data:
  data_dir: "/path/to/your/Dataset001_Cellpose"
```

This is the **only required configuration**. All other settings have sensible defaults.

### 3. Run Federated Learning

```bash
# Default configuration (ε = 8.0, batch=2, rounds=2, epochs=2)
flwr run .
```

That's it! Results will be saved to `./results/`.

## Configuration Profiles

Choose different privacy-utility tradeoffs with pre-configured YAML files:

| Configuration | Privacy Level | Target ε | Noise | Use Case |
|--------------|---------------|----------|-------|----------|
| `default.yaml` | Low | 8.0 | 1.0 | Quick training with reasonable utility |
| `high_privacy.yaml` | High | ~2.0 | 2.0 | Maximum privacy protection |
| `low_privacy.yaml` | Low | ~8.0 | 0.8 | Better model performance |
| `no_dp.yaml` | None | ∞ | 0.0 | Baseline/debugging |
| `quick_test.yaml` | Medium | ~4.0 | 1.0 | Fast testing (2 clients, 2 rounds) |

### Usage Examples

```bash
# High privacy (stronger guarantees)
flwr run . --run-config configs/high_privacy.yaml

# Low privacy (better accuracy)
flwr run . --run-config configs/low_privacy.yaml

# No differential privacy (baseline comparison)
flwr run . --run-config configs/no_dp.yaml

# Quick test (fast iteration during development)
flwr run . --run-config configs/quick_test.yaml
```

## Configuration Guide

All configurations are in `configs/*.yaml`. Key sections:

### Data Settings (Required)

```yaml
data:
  data_dir: "/path/to/dataset"  # REQUIRED: Must be set by user
  image_size: 256                # Image dimensions (256x256)
  batch_size: 8                  # Batch size per client
  num_workers: 2                 # Data loading workers
```

### Model Architecture (Configurable)

```yaml
model:
  in_channels: 1                 # Input channels (1 for grayscale)
  out_channels: 2                # Output classes
  channels: [16, 32, 64, 128]    # Channel dimensions per layer
  strides: [2, 2, 2]             # Downsampling strides
  num_res_units: 2               # Residual units per block
```

### Federated Learning Settings

```yaml
federated:
  num_clients: 3                 # Number of federated clients
  num_rounds: 5                  # Federated training rounds
  min_fit_clients: 2             # Minimum clients for training
  min_evaluate_clients: 2        # Minimum clients for evaluation
```

### Client Resources (GPU Configuration)

```yaml
client_resources:
  num_cpus: 1                    # CPUs per client
  num_gpus: 0.3                  # GPU fraction per client (0.3 = 30%)
```

**Note**: For simulation, multiple clients share available GPUs. Set `num_gpus: 0` to use CPU only.

### Training Settings

```yaml
training:
  local_epochs: 1                # Local training epochs per round
  learning_rate: 0.001           # Learning rate
  optimizer: "sgd"               # Optimizer type
  momentum: 0.9                  # SGD momentum
```

### Differential Privacy Settings

```yaml
privacy:
  enable_dp: true                # Enable/disable DP
  noise_multiplier: 1.0          # Noise level (higher = more privacy)
  max_grad_norm: 1.0             # Gradient clipping threshold
  target_epsilon: 4.0            # Privacy budget (lower = stronger privacy)
  target_delta: 1.0e-5           # DP delta parameter
```

### Logging Settings

```yaml
logging:
  log_dir: "./results"           # Output directory
  save_model: true               # Save final model
  save_metrics: true             # Save training metrics
  level: "INFO"                  # Log level (DEBUG, INFO, WARNING, ERROR)
```

## Understanding Differential Privacy

### Privacy-Utility Tradeoff

| Parameter | Effect on Privacy | Effect on Utility |
|-----------|-------------------|-------------------|
| ↑ `noise_multiplier` | ↑ **Stronger privacy** | ↓ Lower accuracy |
| ↓ `max_grad_norm` | ↑ **Stronger privacy** | ↓ Lower accuracy |
| ↓ `target_epsilon` | ↑ **Stronger privacy** | ↓ Lower accuracy |
| ↑ `num_rounds` | ↓ Weaker privacy | ↑ Higher accuracy |

### Privacy Budget (ε)

- **ε < 1**: Very strong privacy (significant utility loss)
- **ε ≈ 2-4**: Good privacy (moderate utility)
- **ε ≈ 6-10**: Weak privacy (better utility)
- **ε = ∞**: No privacy (baseline)

**Lower ε = Stronger Privacy Guarantee**

## Results and Logging

After running `flwr run`, results are saved to the configured log directory (default: `./results/`):

### Log Files

- **`dp_fedmed_YYYY-MM-DD.log`**: Full training logs with timestamps
- **`errors_YYYY-MM-DD.log`**: Error logs only
- **`training_log_<timestamp>.json`**: Structured metrics in JSON format

### JSON Metrics Structure

```json
{
  "start_time": "2025-12-21T10:30:00",
  "end_time": "2025-12-21T11:15:00",
  "privacy_summary": {
    "target_epsilon": 4.0,
    "cumulative_epsilon": 3.8,
    "budget_exceeded": false,
    "rounds": [...]
  },
  "rounds": [
    {
      "round": 1,
      "round_epsilon": 0.75,
      "cumulative_epsilon": 0.75,
      "dice": 0.82,
      "loss": 0.45
    },
    ...
  ]
}
```

## Project Structure

```
dp_fl_unet/
├── dp_fedmed/
│   ├── client_app.py          # Flower client with Opacus DP
│   ├── server_app.py          # Flower server with privacy tracking
│   ├── task.py                # Training and evaluation functions
│   ├── config.py              # YAML configuration loader
│   ├── logging_config.py      # Loguru logging setup
│   ├── data/
│   │   ├── cellpose.py        # Dataset loader
│   │   └── partitioner.py     # Data partitioning utilities
│   ├── models/
│   │   └── unet2d.py          # 2D UNet with InstanceNorm (DP-compatible)
│   └── privacy/
│       └── accountant.py      # Privacy budget tracking
├── configs/
│   ├── default.yaml           # Default configuration
│   ├── high_privacy.yaml      # High privacy (ε ≈ 2.0)
│   ├── low_privacy.yaml       # Low privacy (ε ≈ 8.0)
│   ├── no_dp.yaml             # No DP (baseline)
│   └── quick_test.yaml        # Fast testing
├── results/                   # Training results and logs
├── pyproject.toml             # Package and Flower configuration
└── README.md                  # This file
```

## Key Implementation Details

### DP-Compatible Model

The MONAI UNet uses **InstanceNorm** instead of BatchNorm for Opacus compatibility:

```python
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    norm="instance",  # Required for Opacus (BatchNorm not compatible)
)
```

### Opacus Integration

Opacus provides differential privacy through:
1. **Per-sample gradient clipping**: Limits gradient contribution from each sample
2. **Noise addition**: Adds calibrated Gaussian noise to gradients
3. **Privacy accounting**: Tracks cumulative privacy budget across training

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# After training
epsilon = privacy_engine.get_epsilon(delta=1e-5)
```

## Customizing Configurations

### Creating a New Configuration

1. Copy an existing config file:
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   ```

2. Edit the parameters:
   ```yaml
   data:
     data_dir: "/my/dataset/path"
     batch_size: 16
   
   federated:
     num_clients: 5
     num_rounds: 10
   
   privacy:
     target_epsilon: 6.0
     noise_multiplier: 1.2
   ```

3. Run with your config:
   ```bash
   flwr run . --run-config configs/my_config.yaml
   ```

### Common Adjustments

**For faster training:**
- Increase `batch_size`
- Decrease `num_rounds`
- Increase `learning_rate`

**For better privacy:**
- Decrease `target_epsilon`
- Increase `noise_multiplier`
- Decrease `max_grad_norm`

**For better accuracy:**
- Increase `num_rounds`
- Increase `local_epochs`
- Decrease `noise_multiplier`

## Troubleshooting

### Out of Memory (OOM) Errors

Reduce memory usage by:
- Decreasing `batch_size`
- Decreasing `image_size`
- Reducing `client_resources.num_gpus` (run more clients on CPU)

### Slow Training

Speed up training by:
- Increasing `client_resources.num_gpus` (if GPUs available)
- Increasing `batch_size` (if memory allows)
- Using `quick_test.yaml` for rapid iteration

### Poor Model Performance

Improve accuracy by:
- Increasing `num_rounds`
- Decreasing `noise_multiplier` (weakens privacy)
- Increasing `local_epochs`
- Using `low_privacy.yaml` or `no_dp.yaml` to establish baseline

### Privacy Budget Exceeded

If you see "Privacy budget EXCEEDED" warnings:
- Decrease `num_rounds`
- Increase `target_epsilon` (weakens privacy)
- This is a warning, not an error - training continues

## References

- [Flower: Federated Learning Framework](https://flower.ai/)
- [Opacus: Differential Privacy Library](https://opacus.ai/)
- [MONAI: Medical Open Network for AI](https://monai.io/)
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

## License

Apache 2.0

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dp_fedmed2025,
  title = {DP-FedMed: Differential Privacy + Federated Learning for Medical Imaging},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/dp_fl_unet}
}
```
