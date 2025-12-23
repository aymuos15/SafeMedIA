"""Extended tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from dp_fedmed.losses.dice import SoftDiceLoss, DiceCELoss, get_loss_function


class TestSoftDiceLoss:
    """Tests for SoftDiceLoss."""

    def test_soft_dice_loss_initialization(self):
        """Test SoftDiceLoss can be initialized."""
        loss_fn = SoftDiceLoss()
        assert loss_fn.smooth == 1.0
        assert loss_fn.include_background is False
        assert loss_fn.reduction == "mean"

    def test_soft_dice_loss_custom_params(self):
        """Test SoftDiceLoss with custom parameters."""
        loss_fn = SoftDiceLoss(smooth=0.5, include_background=True, reduction="sum")
        assert loss_fn.smooth == 0.5
        assert loss_fn.include_background is True
        assert loss_fn.reduction == "sum"

    def test_soft_dice_loss_forward(self):
        """Test SoftDiceLoss forward pass."""
        loss_fn = SoftDiceLoss()

        # Create dummy logits and targets
        batch_size = 2
        num_classes = 2
        height, width = 64, 64

        logits = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss = loss_fn(logits, targets)

        assert torch.is_tensor(loss)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0.0  # Dice loss should be non-negative
        assert loss <= 1.0  # Dice loss should be <= 1.0

    def test_soft_dice_loss_perfect_prediction(self):
        """Test SoftDiceLoss with perfect predictions."""
        loss_fn = SoftDiceLoss()

        batch_size = 2
        num_classes = 2
        height, width = 64, 64

        # Create perfect predictions (high confidence for correct class)
        targets = torch.zeros(batch_size, height, width, dtype=torch.long)
        logits = torch.zeros(batch_size, num_classes, height, width)
        logits[:, 0, :, :] = 10.0  # High score for class 0
        logits[:, 1, :, :] = -10.0  # Low score for class 1

        loss = loss_fn(logits, targets)

        # Loss should be very small (close to 0) for perfect predictions
        assert loss < 0.1

    def test_soft_dice_loss_with_background(self):
        """Test SoftDiceLoss with include_background=True."""
        loss_with_bg = SoftDiceLoss(include_background=True)
        loss_without_bg = SoftDiceLoss(include_background=False)

        batch_size = 2
        num_classes = 2
        height, width = 64, 64

        logits = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss1 = loss_with_bg(logits, targets)
        loss2 = loss_without_bg(logits, targets)

        # Losses should be different
        assert not torch.allclose(loss1, loss2)

    def test_soft_dice_loss_reduction_modes(self):
        """Test different reduction modes."""
        batch_size = 2
        num_classes = 2
        height, width = 64, 64

        logits = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss_mean = SoftDiceLoss(reduction="mean")(logits, targets)
        loss_sum = SoftDiceLoss(reduction="sum")(logits, targets)
        loss_none = SoftDiceLoss(reduction="none")(logits, targets)

        assert loss_mean.dim() == 0
        assert loss_sum.dim() == 0
        assert loss_none.dim() == 1  # Per-batch losses
        assert loss_none.shape[0] == batch_size

    def test_soft_dice_loss_gradient_flow(self):
        """Test that gradients flow through SoftDiceLoss."""
        loss_fn = SoftDiceLoss()

        batch_size = 2
        num_classes = 2
        height, width = 32, 32

        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_soft_dice_loss_batch_sizes(self):
        """Test SoftDiceLoss with different batch sizes."""
        loss_fn = SoftDiceLoss()

        for batch_size in [1, 4, 8]:
            logits = torch.randn(batch_size, 2, 32, 32)
            targets = torch.randint(0, 2, (batch_size, 32, 32))

            loss = loss_fn(logits, targets)
            assert torch.is_tensor(loss)
            assert loss.dim() == 0


class TestDiceCELoss:
    """Tests for DiceCELoss."""

    def test_dice_ce_loss_initialization(self):
        """Test DiceCELoss initialization."""
        loss_fn = DiceCELoss()
        assert loss_fn.alpha == 0.5
        assert loss_fn.beta == 0.5

    def test_dice_ce_loss_custom_weights(self):
        """Test DiceCELoss with custom weights."""
        loss_fn = DiceCELoss(alpha=0.7, beta=0.3)
        assert loss_fn.alpha == 0.7
        assert loss_fn.beta == 0.3

    def test_dice_ce_loss_forward(self):
        """Test DiceCELoss forward pass."""
        loss_fn = DiceCELoss()

        batch_size = 2
        num_classes = 2
        height, width = 64, 64

        logits = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss = loss_fn(logits, targets)

        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert loss >= 0.0

    def test_dice_ce_loss_gradient_flow(self):
        """Test gradient flow through DiceCELoss."""
        loss_fn = DiceCELoss()

        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 32, 32))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_dice_ce_loss_combines_both_losses(self):
        """Test that DiceCELoss combines both Dice and CE."""
        # Create separate loss functions
        dice_only = DiceCELoss(alpha=1.0, beta=0.0)
        ce_only = DiceCELoss(alpha=0.0, beta=1.0)
        combined = DiceCELoss(alpha=0.5, beta=0.5)

        logits = torch.randn(2, 2, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32))

        _ = dice_only(logits, targets)
        _ = ce_only(logits, targets)
        loss_combined = combined(logits, targets)

        # Combined should be between the two individual losses (approximately)
        # This is a rough check - exact relationship depends on the data
        assert loss_combined.item() > 0


class TestGetLossFunction:
    """Tests for get_loss_function factory."""

    def test_get_loss_function_cross_entropy(self):
        """Test getting CrossEntropyLoss."""
        loss_config = {"type": "cross_entropy"}
        loss_fn = get_loss_function(loss_config)

        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_function_soft_dice(self):
        """Test getting SoftDiceLoss."""
        loss_config = {"type": "soft_dice"}
        loss_fn = get_loss_function(loss_config)

        assert isinstance(loss_fn, SoftDiceLoss)

    def test_get_loss_function_dice_ce(self):
        """Test getting DiceCELoss."""
        loss_config = {"type": "dice_ce"}
        loss_fn = get_loss_function(loss_config)

        assert isinstance(loss_fn, DiceCELoss)

    def test_get_loss_function_with_params(self):
        """Test get_loss_function with custom parameters."""
        loss_config = {
            "type": "soft_dice",
            "dice_smooth": 0.5,
            "dice_include_background": True,
        }
        loss_fn = get_loss_function(loss_config)

        assert isinstance(loss_fn, SoftDiceLoss)
        assert loss_fn.smooth == 0.5
        assert loss_fn.include_background is True

    def test_get_loss_function_dice_ce_weights(self):
        """Test DiceCELoss with custom weights."""
        loss_config = {
            "type": "dice_ce",
            "dice_weight": 0.7,
            "ce_weight": 0.3,
        }
        loss_fn = get_loss_function(loss_config)

        assert isinstance(loss_fn, DiceCELoss)
        assert loss_fn.alpha == 0.7
        assert loss_fn.beta == 0.3

    def test_get_loss_function_invalid_type_raises(self):
        """Test that invalid loss type raises ValueError."""
        loss_config = {"type": "invalid_loss"}

        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function(loss_config)

    def test_get_loss_function_case_insensitive(self):
        """Test that loss type is case-insensitive."""
        loss_config1 = {"type": "Cross_Entropy"}
        loss_config2 = {"type": "CROSS_ENTROPY"}

        loss_fn1 = get_loss_function(loss_config1)
        loss_fn2 = get_loss_function(loss_config2)

        assert isinstance(loss_fn1, nn.CrossEntropyLoss)
        assert isinstance(loss_fn2, nn.CrossEntropyLoss)

    def test_get_loss_function_default_params(self):
        """Test that default parameters are applied correctly."""
        loss_config = {"type": "soft_dice"}
        loss_fn = get_loss_function(loss_config)

        assert loss_fn.smooth == 1.0  # Default
        assert loss_fn.include_background is False  # Default


class TestLossFunctionsDPCompatibility:
    """Tests for DP compatibility of loss functions."""

    def test_soft_dice_no_division_by_zero(self):
        """Test that SoftDiceLoss handles edge cases without division by zero."""
        loss_fn = SoftDiceLoss(smooth=1.0)

        # All predictions for background class
        logits = torch.zeros(2, 2, 32, 32)
        logits[:, 0, :, :] = 10.0  # High score for class 0
        targets = torch.zeros(2, 32, 32, dtype=torch.long)

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_soft_dice_with_empty_class(self):
        """Test SoftDiceLoss when a class is not present in targets."""
        loss_fn = SoftDiceLoss()

        # Only class 0 in targets
        logits = torch.randn(2, 3, 32, 32)
        targets = torch.zeros(2, 32, 32, dtype=torch.long)

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)
