"""DP-compatible loss functions for semantic segmentation."""

from dp_fedmed.losses.dice import DiceCELoss, SoftDiceLoss, get_loss_function

__all__ = ["SoftDiceLoss", "DiceCELoss", "get_loss_function"]
