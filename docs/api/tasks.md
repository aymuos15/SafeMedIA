# Tasks API

The Tasks API provides training and evaluation loops for federated learning.

## Overview

The tasks module contains the core training and evaluation logic used by clients during federated rounds:

- **train_one_epoch()**: Single epoch of training with differential privacy support
- **evaluate()**: Model evaluation with Dice score computation

## Main Components

::: dp_fedmed.fl.task.train_one_epoch
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.fl.task.evaluate
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Usage Examples

### Training One Epoch

```python
from dp_fedmed.fl.task import train_one_epoch
import torch

# Standard training (no DP)
epoch_loss = train_one_epoch(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=torch.device("cuda"),
    loss_config={"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5},
)

print(f"Average loss: {epoch_loss:.4f}")
```

### Training with Differential Privacy

```python
from opacus import PrivacyEngine

# Wrap model with Opacus
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train (DP-SGD applied automatically)
epoch_loss = train_one_epoch(
    model=model,  # Now wrapped with GradSampleModule
    train_loader=train_loader,  # Now DP-enabled
    optimizer=optimizer,  # Now DP-enabled
    device=device,
    loss_config=loss_config,
)

# Check privacy spent
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy spent: ε = {epsilon:.4f}")
```

### Model Evaluation

```python
from dp_fedmed.fl.task import evaluate

# Evaluate on test set
metrics = evaluate(
    model=model,
    test_loader=test_loader,
    device=device,
)

# Metrics returned:
# {
#     "dice": 0.87,     # Dice coefficient
#     "loss": 0.23,     # Average loss
# }

print(f"Dice score: {metrics['dice']:.4f}")
print(f"Test loss: {metrics['loss']:.4f}")
```

### Full Training Loop

```python
from dp_fedmed.fl.task import train_one_epoch, evaluate

# Train for multiple epochs
for epoch in range(num_epochs):
    # Training
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_config=loss_config,
    )

    # Evaluation
    eval_metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f}, "
          f"dice={eval_metrics['dice']:.4f}")
```

## Function Details

### train_one_epoch()

**Process:**
1. Set model to training mode
2. Iterate through training batches
3. Forward pass: compute predictions
4. Compute loss
5. Backward pass: compute gradients
6. Optimizer step: update weights
7. Return average loss

**DP Handling:**
- Automatically uses DP-SGD if model wrapped with Opacus
- Gradient clipping and noise addition handled by Opacus
- No code changes needed for DP vs non-DP training

**Loss Computation:**
- Loss function created from `loss_config`
- Supports: "cross_entropy", "soft_dice", "dice_ce"
- See [Losses API](losses.md) for details

### evaluate()

**Process:**
1. Set model to evaluation mode (disables dropout/batchnorm updates)
2. Disable gradient computation
3. Iterate through test batches
4. Compute predictions
5. Calculate Dice score and loss
6. Return aggregated metrics

**Dice Computation:**
```python
# For each batch
pred_labels = torch.argmax(logits, dim=1)  # (B, H, W)
intersection = (pred_labels * labels).sum()
union = pred_labels.sum() + labels.sum()
dice = 2 * intersection / (union + 1e-6)

# Average across batches
avg_dice = total_dice / num_batches
```

## Integration with Client

The client's `fit()` and `evaluate()` methods use these functions:

```python
class DPFlowerClient(NumPyClient):
    def fit(self, parameters, config):
        # Set global parameters
        set_parameters(self.model, parameters)

        # Train for local_epochs
        total_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                loss_config=self.loss_config,
            )
            total_loss += epoch_loss

        # Return updated parameters
        return get_parameters(self.model), num_samples, metrics

    def evaluate(self, parameters, config):
        # Set parameters to evaluate
        set_parameters(self.model, parameters)

        # Evaluate
        metrics = evaluate(
            self.model,
            self.test_loader,
            self.device,
        )

        return metrics["loss"], num_samples, {"dice": metrics["dice"]}
```

## Device Handling

Both functions support CPU and GPU:

```python
# Automatic device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = model.to(device)

# Training/evaluation handles batch device placement automatically
train_one_epoch(model, train_loader, optimizer, device, loss_config)
```

## Error Handling

Both functions include error handling:

- **Empty batches**: Skipped with warning
- **NaN losses**: Logged but training continues
- **Device mismatches**: Data automatically moved to model's device

## Performance Considerations

### Training

- **Gradient accumulation**: Not currently supported
- **Mixed precision**: Not enabled (can be added via torch.cuda.amp)
- **DataLoader workers**: Configure via `num_workers` in data config

### Evaluation

- **No gradient computation**: Saves memory via `torch.no_grad()`
- **Batch size**: Can use larger batches than training
- **Deterministic**: Results are reproducible

## See Also

- [Client API](client.md) - Using tasks in client training
- [Losses API](losses.md) - Loss function details
- [Data API](data.md) - DataLoader configuration
- [Models API](models.md) - Model creation
