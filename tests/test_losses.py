"""Tests for DP-compatible loss functions."""

import pytest
import torch
import torch.nn as nn

from dp_fedmed.losses.dice import SoftDiceLoss, DiceCELoss, get_loss_function


class TestSoftDiceLoss:
    """Tests for SoftDiceLoss."""

    def test_output_is_scalar(self):
        """Test that loss returns a scalar tensor."""
        loss_fn = SoftDiceLoss()

        # [B, C, H, W]
        logits = torch.randn(4, 2, 32, 32)
        # [B, H, W]
        targets = torch.randint(0, 2, (4, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_perfect_prediction_low_loss(self):
        """Test that perfect prediction gives very low loss."""
        loss_fn = SoftDiceLoss()

        # Create "perfect" prediction: high logit for correct class
        targets = torch.zeros(2, 16, 16, dtype=torch.long)
        targets[:, 8:, 8:] = 1  # Right-bottom quadrant is class 1

        # Create logits that strongly predict the targets
        logits = torch.zeros(2, 2, 16, 16)
        logits[:, 0, :8, :] = 10.0  # High logit for class 0 in top half
        logits[:, 0, :, :8] = 10.0  # High logit for class 0 in left half
        logits[:, 1, 8:, 8:] = 10.0  # High logit for class 1 in right-bottom

        loss = loss_fn(logits, targets)

        # Perfect prediction should have very low loss (close to 0)
        assert loss.item() < 0.1

    def test_random_prediction_higher_loss(self):
        """Test that random prediction gives higher loss than perfect."""
        loss_fn = SoftDiceLoss()

        targets = torch.randint(0, 2, (4, 32, 32))

        # Random logits
        random_logits = torch.randn(4, 2, 32, 32)
        random_loss = loss_fn(random_logits, targets)

        # "Perfect" logits
        perfect_logits = torch.zeros(4, 2, 32, 32)
        perfect_logits.scatter_(1, targets.unsqueeze(1), 10.0)
        perfect_loss = loss_fn(perfect_logits, targets)

        # Random should have higher loss
        assert random_loss.item() > perfect_loss.item()

    def test_gradient_computation(self):
        """Test that gradients are computable."""
        loss_fn = SoftDiceLoss()

        logits = torch.randn(2, 2, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 16, 16))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()

    def test_include_background_option(self):
        """Test include_background option."""
        loss_with_bg = SoftDiceLoss(include_background=True)
        loss_without_bg = SoftDiceLoss(include_background=False)

        logits = torch.randn(2, 2, 16, 16)
        targets = torch.randint(0, 2, (2, 16, 16))

        loss1 = loss_with_bg(logits, targets)
        loss2 = loss_without_bg(logits, targets)

        # Both should be valid losses, but may differ
        assert loss1.item() >= 0
        assert loss2.item() >= 0

    def test_reduction_modes(self):
        """Test different reduction modes."""
        logits = torch.randn(4, 2, 16, 16)
        targets = torch.randint(0, 2, (4, 16, 16))

        # Mean reduction (default)
        loss_mean = SoftDiceLoss(reduction="mean")
        result_mean = loss_mean(logits, targets)
        assert result_mean.dim() == 0

        # Sum reduction
        loss_sum = SoftDiceLoss(reduction="sum")
        result_sum = loss_sum(logits, targets)
        assert result_sum.dim() == 0

        # No reduction
        loss_none = SoftDiceLoss(reduction="none")
        result_none = loss_none(logits, targets)
        assert result_none.shape == (4,)


class TestDiceCELoss:
    """Tests for DiceCELoss."""

    def test_output_is_scalar(self):
        """Test that combined loss returns a scalar."""
        loss_fn = DiceCELoss()

        logits = torch.randn(4, 2, 32, 32)
        targets = torch.randint(0, 2, (4, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_alpha_beta_weighting(self):
        """Test that alpha/beta weights are applied correctly."""
        logits = torch.randn(2, 2, 16, 16)
        targets = torch.randint(0, 2, (2, 16, 16))

        # Equal weights
        loss_equal = DiceCELoss(alpha=0.5, beta=0.5)
        result_equal = loss_equal(logits, targets)

        # Dice-only (alpha=1, beta=0)
        loss_dice_only = DiceCELoss(alpha=1.0, beta=0.0)
        result_dice = loss_dice_only(logits, targets)

        # CE-only (alpha=0, beta=1)
        loss_ce_only = DiceCELoss(alpha=0.0, beta=1.0)
        result_ce = loss_ce_only(logits, targets)

        # All should be valid losses
        assert result_equal.item() >= 0
        assert result_dice.item() >= 0
        assert result_ce.item() >= 0

        # Pure dice should match SoftDiceLoss
        soft_dice = SoftDiceLoss()
        expected_dice = soft_dice(logits, targets)
        assert torch.isclose(result_dice, expected_dice, atol=1e-5)

        # Pure CE should match CrossEntropyLoss
        ce_loss = nn.CrossEntropyLoss()
        expected_ce = ce_loss(logits, targets.long())
        assert torch.isclose(result_ce, expected_ce, atol=1e-5)

    def test_gradient_computation(self):
        """Test that gradients are computable for combined loss."""
        loss_fn = DiceCELoss()

        logits = torch.randn(2, 2, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 16, 16))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()


class TestGetLossFunction:
    """Tests for get_loss_function factory."""

    def test_cross_entropy_type(self):
        """Test creating CrossEntropyLoss."""
        loss_fn = get_loss_function({"type": "cross_entropy"})
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_soft_dice_type(self):
        """Test creating SoftDiceLoss."""
        loss_fn = get_loss_function({"type": "soft_dice"})
        assert isinstance(loss_fn, SoftDiceLoss)

    def test_dice_ce_type(self):
        """Test creating DiceCELoss."""
        loss_fn = get_loss_function({"type": "dice_ce"})
        assert isinstance(loss_fn, DiceCELoss)

    def test_dice_ce_with_config(self):
        """Test creating DiceCELoss with custom config."""
        config = {
            "type": "dice_ce",
            "dice_smooth": 2.0,
            "dice_include_background": True,
            "dice_weight": 0.7,
            "ce_weight": 0.3,
        }
        loss_fn = get_loss_function(config)

        assert isinstance(loss_fn, DiceCELoss)
        assert loss_fn.alpha == 0.7
        assert loss_fn.beta == 0.3

    def test_unknown_type_raises_error(self):
        """Test that unknown loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function({"type": "unknown"})

    def test_default_type(self):
        """Test that empty config defaults to cross_entropy."""
        loss_fn = get_loss_function({})
        assert isinstance(loss_fn, nn.CrossEntropyLoss)


class TestOpacusCompatibility:
    """Tests for Opacus differential privacy compatibility.

    These tests verify that the loss functions work correctly with
    Opacus's per-sample gradient computation.
    """

    @pytest.fixture
    def simple_model(self):
        """Create a simple CNN for testing."""
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 3, padding=1),
        )

    def test_soft_dice_with_model_training(self, simple_model):
        """Test SoftDiceLoss in a training loop."""
        model = simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = SoftDiceLoss()

        # Simulate training step
        images = torch.randn(4, 1, 32, 32)
        labels = torch.randint(0, 2, (4, 32, 32))

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() >= 0

    def test_dice_ce_with_model_training(self, simple_model):
        """Test DiceCELoss in a training loop."""
        model = simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = DiceCELoss()

        # Simulate training step
        images = torch.randn(4, 1, 32, 32)
        labels = torch.randint(0, 2, (4, 32, 32))

        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() >= 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Opacus test requires GPU or is slow on CPU",
    )
    def test_soft_dice_with_opacus(self, simple_model):
        """Test SoftDiceLoss with Opacus PrivacyEngine.

        This test verifies that SoftDiceLoss does NOT cause SIGFPE
        when used with Opacus's per-sample gradient computation.
        """
        from opacus import PrivacyEngine
        from torch.utils.data import DataLoader, TensorDataset

        model = simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create dummy dataset
        images = torch.randn(8, 1, 32, 32)
        labels = torch.randint(0, 2, (8, 32, 32))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # Wrap with Opacus
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        loss_fn = SoftDiceLoss()

        # Training step with Opacus - this should NOT cause SIGFPE
        for images_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            break  # One batch is enough to test

        # If we reach here without SIGFPE, the test passes
        assert True
