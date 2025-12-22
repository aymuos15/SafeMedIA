# Self-Supervised Learning (SSL) Implementation Plan for DP-FedMed

## Overview

Add reconstruction-based self-supervised learning to the existing DP-FL medical image segmentation framework. SSL will pretrain the full UNet encoder-decoder using masked autoencoder or denoising tasks, with full support for all 4 DP styles (none, sample, user, hybrid).

**Training Flow**: SSL Pretraining → Supervised Fine-tuning (two-phase)

---

## New Files to Create

### 1. SSL Core Module

| File | Purpose |
|------|---------|
| `dp_fedmed/ssl/__init__.py` | Package exports |
| `dp_fedmed/ssl/task.py` | `ssl_train_one_epoch()`, `ssl_evaluate()` - reconstruction training loops |
| `dp_fedmed/ssl/transforms.py` | `MaskingTransform`, `DenoisingTransform` - corruption augmentations |
| `dp_fedmed/ssl/losses.py` | `ReconstructionLoss` (MSE/L1), `get_ssl_loss_function()` factory |

### 2. SSL FL Integration

| File | Purpose |
|------|---------|
| `dp_fedmed/fl/client/ssl_client.py` | `SSLFlowerClient` - FL client for SSL pretraining |
| `dp_fedmed/fl/client/ssl_factory.py` | `ssl_client_fn()` - creates SSL clients with image-only loaders |
| `dp_fedmed/fl/server/ssl_strategy.py` | `SSLDPFedAvg` - FL strategy tracking reconstruction metrics |
| `dp_fedmed/fl/server/ssl_factory.py` | `ssl_server_fn()` - creates SSL server components |

### 3. SSL Data Loading

| File | Purpose |
|------|---------|
| `dp_fedmed/data/ssl_cellpose.py` | `build_ssl_data_list()` - images-only loading (ignores labels) |

### 4. Configuration

| File | Purpose |
|------|---------|
| `configs/ssl_pretrain.toml` | Example SSL pretraining configuration |
| `configs/ssl_finetune.toml` | Example fine-tuning with pretrained weights |

---

## Files to Modify

### `dp_fedmed/config.py`
Add Pydantic models for SSL configuration:

```python
class SSLMethod(str, Enum):
    MASKING = "masking"
    DENOISING = "denoising"

class MaskingConfig(BaseModel):
    mask_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
    patch_size: int = Field(default=16, ge=4, le=64)

class DenoisingConfig(BaseModel):
    noise_type: str = Field(default="gaussian")  # gaussian, salt_pepper
    noise_level: float = Field(default=0.2, ge=0.0, le=1.0)

class SSLLossConfig(BaseModel):
    type: str = Field(default="mse")  # mse, l1, mse_l1
    mse_weight: float = Field(default=0.5)
    l1_weight: float = Field(default=0.5)
    mask_weighted: bool = True  # Only compute loss on corrupted regions

class SSLConfig(BaseModel):
    enabled: bool = False
    method: SSLMethod = Field(default=SSLMethod.MASKING)
    num_rounds: int = Field(default=10, ge=1)
    local_epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    privacy_style: DPStyle = Field(default=DPStyle.SAMPLE)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    denoising: DenoisingConfig = Field(default_factory=DenoisingConfig)
    loss: SSLLossConfig = Field(default_factory=SSLLossConfig)
    pretrained_checkpoint: Optional[str] = None  # For fine-tuning phase
```

Add to `Config` class:
```python
ssl: SSLConfig = Field(default_factory=SSLConfig)
```

### `dp_fedmed/fl/client/factory.py`
Add mode detection to route to SSL client:
```python
training_mode = str(cfg.get("training-mode", "supervised"))
if training_mode == "ssl":
    from .ssl_factory import ssl_client_fn
    return ssl_client_fn(context)
```

### `dp_fedmed/fl/server/factory.py`
Add mode detection and pretrained checkpoint loading:
```python
training_mode = str(cfg.get("training-mode", "supervised"))
if training_mode == "ssl":
    from .ssl_factory import ssl_server_fn
    return ssl_server_fn(context)

# Load SSL pretrained weights for fine-tuning
if config.ssl.pretrained_checkpoint:
    # Load and set initial_parameters
```

### `dp_fedmed/fl/checkpoint.py`
Add SSL-specific checkpoint functions:
- `save_ssl_checkpoint()` - saves reconstruction_loss, PSNR metrics
- `load_ssl_checkpoint()` - loads pretrained weights for fine-tuning

---

## Implementation Details

### SSL Training Loop (`dp_fedmed/ssl/task.py`)

```python
def ssl_train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,  # Returns (corrupted, original) tuples
    optimizer,
    device: torch.device,
    ssl_config: Dict,
    checkpoint_dir: Optional[Path] = None,
) -> float:
    """
    1. Load batch: (corrupted_image, original_image)
    2. Forward: reconstructed = model(corrupted_image)
    3. Loss: MSE/L1 between reconstructed and original
    4. Optional: mask-weighted loss (only on corrupted regions)
    5. Backward + optimizer step
    """

def ssl_evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    ssl_config: Dict,
) -> Dict[str, float]:
    """Returns: {reconstruction_loss, psnr, ssim}"""
```

### SSL Transforms (`dp_fedmed/ssl/transforms.py`)

```python
class MaskingTransform:
    """Random patch masking (MAE-style)"""
    def __init__(self, mask_ratio: float = 0.5, patch_size: int = 16):
        ...
    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (masked_image, mask, original_image)

class DenoisingTransform:
    """Noise injection"""
    def __init__(self, noise_type: str = "gaussian", noise_level: float = 0.2):
        ...
    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns: (noisy_image, original_image)
```

### SSL Loss (`dp_fedmed/ssl/losses.py`)

```python
class ReconstructionLoss(nn.Module):
    """MSE/L1 reconstruction loss with optional mask weighting"""
    def __init__(self, loss_type: str = "mse", mask_weighted: bool = True):
        ...
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # DP-compatible: simple reduction operations
```

---

## Two-Phase Workflow

### Phase 1: SSL Pretraining
```bash
flwr run . --run-config "config-file=configs/ssl_pretrain.toml,training-mode=ssl"
```

**Output**: `results/<run>/server/checkpoints/best_model.pt`

### Phase 2: Supervised Fine-tuning
```bash
flwr run . --run-config "config-file=configs/ssl_finetune.toml"
```

**Config points to pretrained checkpoint**:
```toml
[ssl]
pretrained_checkpoint = "results/ssl_pretrain/server/checkpoints/best_model.pt"
```

---

## TOML Configuration Example

```toml
# configs/ssl_pretrain.toml
[ssl]
enabled = true
method = "masking"
num_rounds = 20
local_epochs = 5
learning_rate = 0.001
privacy_style = "sample"  # Supports: none, sample, user, hybrid

[ssl.masking]
mask_ratio = 0.5
patch_size = 16

[ssl.loss]
type = "mse"
mask_weighted = true

[privacy]
style = "sample"

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0
```

---

## Implementation Order

1. **Config schema** (`config.py`) - Add SSL Pydantic models
2. **SSL transforms** (`ssl/transforms.py`) - Masking and denoising
3. **SSL losses** (`ssl/losses.py`) - Reconstruction loss
4. **SSL task** (`ssl/task.py`) - Training/evaluation loops
5. **SSL data** (`data/ssl_cellpose.py`) - Images-only loading
6. **SSL client** (`fl/client/ssl_client.py`, `ssl_factory.py`)
7. **SSL server** (`fl/server/ssl_strategy.py`, `ssl_factory.py`)
8. **Factory modifications** - Mode switching
9. **Checkpoint enhancements** - SSL-specific save/load
10. **Example configs** - SSL pretrain and finetune
11. **Tests** - Unit and integration tests

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate SSLFlowerClient** | Clean separation; SSL has different metrics, losses, data formats |
| **Full encoder-decoder pretraining** | Reconstruction requires decoder; skip connections make encoder-only complex |
| **Same 4 DP styles** | Consistency with fine-tuning; MSE/L1 are Opacus-compatible |
| **Masking as default** | Better for learning local structure in medical images |
| **Mask-weighted loss optional** | Forces learning on corrupted regions (MAE-style) |

---

## DP Compatibility Notes

### Verified Compatible with Opacus:
- **MSE Loss**: Simple reduction, works with per-sample gradients
- **L1 Loss**: Simple reduction, works with per-sample gradients
- **UNet with InstanceNorm**: Already verified in existing codebase
- **Random masking**: Deterministic per-sample transform (no cross-sample dependencies)

### Implementation Requirements:
1. SSL transforms must be applied per-sample (not batch-level operations)
2. Mask generation must be deterministic given the input (use seeded RNG per sample)
3. Loss reduction must be "mean" (Opacus requirement)
4. Data loader must return tuples `(corrupted, original)` for Opacus compatibility

---

## Critical Files Reference

| File | Description |
|------|-------------|
| `dp_fedmed/config.py` | Pydantic config patterns (lines 21-218) |
| `dp_fedmed/fl/task.py` | Training loop pattern (lines 31-108) |
| `dp_fedmed/losses/dice.py` | Loss factory pattern (lines 146-186) |
| `dp_fedmed/fl/client/dp_client.py` | Client pattern with Opacus |
| `dp_fedmed/fl/client/factory.py` | Factory routing pattern |
| `dp_fedmed/fl/server/strategy.py` | Server strategy pattern |
| `dp_fedmed/data/cellpose.py` | Data loading patterns |
