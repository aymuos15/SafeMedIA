"""Tests for training and evaluation in task.py."""

import pytest
import torch

from dp_fedmed.fl.task import train_one_epoch, evaluate


class TestTrainOneEpoch:
    """Tests for train_one_epoch function."""

    def test_runs_without_error(self, dummy_model, dummy_dataloader, optimizer, device):
        """Verify training completes one epoch without errors."""
        dummy_model.to(device)

        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_with_dict_dataloader(
        self, dummy_model, dummy_dict_dataloader, optimizer, device
    ):
        """Verify training works with dict-format data loader."""
        dummy_model.to(device)

        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dict_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert isinstance(loss, float)

    def test_returns_average_loss(
        self, dummy_model, dummy_dataloader, optimizer, device
    ):
        """Verify training returns average loss for the epoch."""
        dummy_model.to(device)

        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Loss should be a reasonable value (positive, not NaN/Inf)
        assert loss >= 0
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))

    def test_model_weights_change(self, dummy_model, dummy_dataloader, device):
        """Verify model weights actually change during training."""
        dummy_model.to(device)

        # Use a higher learning rate to ensure weights change
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)

        # Get initial weights (just first parameter for speed)
        first_param = list(dummy_model.parameters())[0]
        initial_sum = first_param.sum().item()

        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Check at least one weight changed
        final_sum = first_param.sum().item()
        assert abs(initial_sum - final_sum) > 1e-6, (
            "Weights should change during training"
        )


class TestEvaluate:
    """Tests for evaluate function."""

    def test_returns_metrics(self, dummy_model, dummy_dataloader, device):
        """Verify evaluate returns expected metrics."""
        dummy_model.to(device)

        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
        )

        assert "loss" in metrics
        assert "dice" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["dice"], float)
        assert 0 <= metrics["dice"] <= 1

    def test_with_dict_dataloader(self, dummy_model, dummy_dict_dataloader, device):
        """Verify evaluate works with dict-format data loader."""
        dummy_model.to(device)

        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dict_dataloader,
            device=device,
        )

        assert "loss" in metrics
        assert "dice" in metrics

    def test_dice_in_valid_range(self, dummy_model, dummy_dataloader, device):
        """Verify dice score is in valid range [0, 1]."""
        dummy_model.to(device)

        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
        )

        assert 0 <= metrics["dice"] <= 1

    def test_model_in_eval_mode(self, dummy_model, dummy_dataloader, device):
        """Verify model is in eval mode after evaluation."""
        dummy_model.to(device)

        evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
        )

        assert not dummy_model.training


class TestOpacusCompatibility:
    """Tests for Opacus compatibility.

    Note: Opacus tests can cause SIGFPE due to pytest fixture interaction.
    These tests work when run directly (python -c "...") but may fail in pytest.
    Run with: pytest tests/test_task.py::TestOpacusCompatibility --forked
    """

    @pytest.mark.skip(
        reason="Opacus + pytest fixture interaction causes SIGFPE. Run separately: python3 -c 'from tests.test_task import test_opacus_manually; test_opacus_manually()'"
    )
    def test_opacus_training(self, dummy_model, dummy_dataloader, device):
        """Verify training works with Opacus-wrapped components."""
        pytest.importorskip("opacus")
        from opacus import PrivacyEngine

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        privacy_engine = PrivacyEngine()
        model, optimizer, dp_loader = privacy_engine.make_private(
            module=dummy_model,
            optimizer=optimizer,
            data_loader=dummy_dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Test training works
        loss = train_one_epoch(
            model=model,
            train_loader=dp_loader,
            optimizer=optimizer,
            device=device,
        )

        assert isinstance(loss, float)


def test_opacus_manually():
    """Standalone test for Opacus. Run with: python3 -c 'from tests.test_task import test_opacus_manually; test_opacus_manually()'"""
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset
    from dp_fedmed.models.unet2d import create_unet2d

    model = create_unet2d(
        in_channels=1, out_channels=2, channels=(16, 32), strides=(2,), num_res_units=1
    )
    images = torch.rand(8, 1, 64, 64)
    labels = torch.randint(0, 2, (8, 64, 64))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    privacy_engine = PrivacyEngine()
    model, optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    loss = train_one_epoch(model, dp_loader, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    print(f"SUCCESS! Loss: {loss}")
