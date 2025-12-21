"""DP-compatible loss functions for semantic segmentation.

This module provides loss functions that are compatible with Opacus differential
privacy training. Standard DiceLoss causes SIGFPE (floating point exceptions)
with Opacus due to division operations in per-sample gradient computation.

SoftDiceLoss uses softmax probabilities instead of hard predictions, avoiding
the division instability that causes issues with per-sample gradients.
"""

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss using softmax probabilities.

    DP-Compatible: Uses softmax probabilities instead of hard predictions,
    avoiding the division instability that causes SIGFPE with Opacus.

    The soft dice loss is computed as:
        loss = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)

    Where:
        - p = softmax probabilities (not argmax)
        - y = one-hot encoded labels
        - smooth = smoothing factor to prevent division by zero

    Args:
        smooth: Smoothing factor for numerical stability (default: 1.0)
        include_background: Whether to include background class in loss (default: False)
        reduction: Reduction mode - 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        smooth: float = 1.0,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.smooth: float = smooth
        self.include_background: bool = include_background
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft dice loss.

        Args:
            logits: Raw model outputs of shape [B, C, H, W]
            targets: Integer class labels of shape [B, H, W]

        Returns:
            Soft Dice loss value
        """
        num_classes = logits.shape[1]

        # Convert logits to softmax probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.long(), num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Optionally exclude background
        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]  # Skip class 0
            targets_one_hot = targets_one_hot[:, 1:]

        # Flatten spatial dimensions
        probs_flat = probs.flatten(2)  # [B, C, H*W]
        targets_flat = targets_one_hot.flatten(2)  # [B, C, H*W]

        # Compute Dice per class
        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        # Average over classes
        dice_loss = dice_loss.mean(dim=1)  # [B]

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy Loss.

    DP-Compatible: Uses SoftDiceLoss for the Dice component.

    The combined loss is computed as:
        loss = alpha * SoftDiceLoss + beta * CrossEntropyLoss

    This combination provides:
    - Dice: Good for class imbalance, focuses on overlap
    - CrossEntropy: Stable gradients, per-pixel classification

    Args:
        alpha: Weight for Dice loss component (default: 0.5)
        beta: Weight for CrossEntropy loss component (default: 0.5)
        smooth: Smoothing factor for Dice loss (default: 1.0)
        include_background: Include background in Dice calculation (default: False)
        ce_weight: Optional class weights for CrossEntropy loss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        include_background: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.dice_loss: SoftDiceLoss = SoftDiceLoss(
            smooth=smooth,
            include_background=include_background,
        )
        self.ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=ce_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CrossEntropy loss.

        Args:
            logits: Raw model outputs of shape [B, C, H, W]
            targets: Integer class labels of shape [B, H, W]

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets.long())
        return self.alpha * dice + self.beta * ce


def get_loss_function(loss_config: Dict) -> nn.Module:
    """Factory function to create loss by configuration.

    Args:
        loss_config: Dictionary with loss configuration:
            - type: One of 'cross_entropy', 'soft_dice', 'dice_ce'
            - dice_smooth: Smoothing factor for dice (default: 1.0)
            - dice_include_background: Include background class (default: False)
            - dice_weight: Weight for dice in combined loss (default: 0.5)
            - ce_weight: Weight for CE in combined loss (default: 0.5)

    Returns:
        Configured loss module

    Raises:
        ValueError: If loss type is unknown
    """
    loss_type = loss_config.get("type", "cross_entropy").lower()

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif loss_type == "soft_dice":
        return SoftDiceLoss(
            smooth=loss_config.get("dice_smooth", 1.0),
            include_background=loss_config.get("dice_include_background", False),
        )

    elif loss_type == "dice_ce":
        return DiceCELoss(
            alpha=loss_config.get("dice_weight", 0.5),
            beta=loss_config.get("ce_weight", 0.5),
            smooth=loss_config.get("dice_smooth", 1.0),
            include_background=loss_config.get("dice_include_background", False),
        )

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: cross_entropy, soft_dice, dice_ce"
        )
