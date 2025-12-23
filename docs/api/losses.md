# Losses API

The Losses API provides DP-compatible loss functions for segmentation tasks.

## Overview

DP-FedMed includes loss functions that are compatible with Opacus differential privacy:

- Soft Dice Loss using softmax probabilities (avoids SIGFPE)
- Combined Dice + CrossEntropy loss
- Numerical stability for per-sample gradient computation
- Factory function for configuration-based loss creation

## Main Components

::: dp_fedmed.losses.dice.SoftDiceLoss
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.losses.dice.DiceCELoss
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: dp_fedmed.losses.dice.get_loss_function
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## See Also

- [Tasks API](tasks.md) - Training and evaluation loops
- [Configuration API](config.md) - Loss configuration
- [Client API](client.md) - Using losses in training
