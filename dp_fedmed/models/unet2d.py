"""2D UNet model with GroupNorm for differential privacy compatibility.

Opacus requires GroupNorm instead of BatchNorm because:
- BatchNorm computes statistics across the batch, violating per-sample privacy
- GroupNorm computes statistics within each sample, making it DP-compatible
"""

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets.unet import UNet

from dp_fedmed.utils import get_unwrapped_model


def create_unet2d(
    in_channels: int = 1,
    out_channels: int = 2,
    channels: Sequence[int] = (16, 32, 64, 128),
    strides: Sequence[int] = (2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    """Create a DP-compatible 2D UNet with InstanceNorm.

    Note: We use InstanceNorm instead of GroupNorm/BatchNorm because:
    - BatchNorm is not compatible with Opacus (computes batch statistics)
    - InstanceNorm works like GroupNorm with num_groups=num_channels
    - InstanceNorm is always compatible regardless of channel count

    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output classes (2 for binary segmentation)
        channels: Number of channels at each level
        strides: Stride for each downsampling layer
        num_res_units: Number of residual units per level
        dropout: Dropout probability

    Returns:
        MONAI UNet with InstanceNorm normalization (DP-compatible)
    """
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm="instance",  # InstanceNorm is DP-compatible!
        dropout=dropout,
    )

    return model


def get_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of numpy arrays.

    Handles Opacus-wrapped modules correctly.
    """
    unwrapped_model = get_unwrapped_model(model)
    return [val.cpu().numpy() for val in unwrapped_model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Set model parameters from a list of numpy arrays.

    Handles Opacus-wrapped modules correctly.
    """
    unwrapped_model = get_unwrapped_model(model)
    state_dict = unwrapped_model.state_dict()
    params_dict = zip(state_dict.keys(), parameters)

    # Build new state dict
    new_state_dict = {k: torch.as_tensor(v).detach().clone() for k, v in params_dict}

    # Load into unwrapped model
    unwrapped_model.load_state_dict(new_state_dict, strict=True)
