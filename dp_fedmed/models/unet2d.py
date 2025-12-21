"""2D UNet model with GroupNorm for differential privacy compatibility.

Opacus requires GroupNorm instead of BatchNorm because:
- BatchNorm computes statistics across the batch, violating per-sample privacy
- GroupNorm computes statistics within each sample, making it DP-compatible
"""

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets.unet import UNet
from typing import cast


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


class UNet2D(nn.Module):
    """Wrapper class for 2D UNet with additional utilities."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (16, 32, 64, 128),
        strides: Sequence[int] = (2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = create_unet2d(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )
        self.out_channels: int = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def get_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of numpy arrays.

    Handles Opacus-wrapped modules correctly.
    """
    # Check if model is wrapped by Opacus GradSampleModule
    if hasattr(model, "_module"):
        state_dict = cast(nn.Module, model._module).state_dict()
    else:
        state_dict = model.state_dict()

    return [val.cpu().numpy() for val in state_dict.values()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Set model parameters from a list of numpy arrays.

    Handles Opacus-wrapped modules correctly.
    """
    # Check if model is wrapped by Opacus GradSampleModule
    if hasattr(model, "_module"):
        state_dict = cast(nn.Module, model._module).state_dict()
    else:
        state_dict = model.state_dict()

    params_dict = zip(state_dict.keys(), parameters)

    # Build new state dict
    new_state_dict = {k: torch.tensor(v) for k, v in params_dict}

    # Load into appropriate target
    if hasattr(model, "_module"):
        cast(nn.Module, model._module).load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)
