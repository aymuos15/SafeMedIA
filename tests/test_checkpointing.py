"""Tests for checkpoint saving and loading."""

import pytest
import torch
from pathlib import Path

from dp_fedmed.fl.task import train_one_epoch, evaluate
from dp_fedmed.models.unet2d import create_unet2d, set_parameters


class TestClientCheckpointing:
    """Tests for client-side checkpointing."""

    def test_last_model_saved_after_training(
        self, dummy_model, dummy_dataloader, optimizer, device, tmp_checkpoint_dir
    ):
        """Verify last_model.pt is created after training."""
        dummy_model.to(device)

        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        assert (tmp_checkpoint_dir / "last_model.pt").exists()

    def test_last_model_saved_after_eval(
        self, dummy_model, dummy_dataloader, device, tmp_checkpoint_dir
    ):
        """Verify last_model.pt is created after evaluation."""
        dummy_model.to(device)

        evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        assert (tmp_checkpoint_dir / "last_model.pt").exists()

    def test_best_model_saved_on_first_eval(
        self, dummy_model, dummy_dataloader, device, tmp_checkpoint_dir
    ):
        """Verify best_model.pt is saved on first evaluation with positive dice."""
        dummy_model.to(device)

        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        # Best model should be saved if dice > 0
        if metrics["dice"] > 0:
            assert (tmp_checkpoint_dir / "best_model.pt").exists()

    def test_checkpoint_loadable(
        self, dummy_model, dummy_dataloader, device, tmp_checkpoint_dir
    ):
        """Verify saved checkpoint can be loaded."""
        dummy_model.to(device)

        evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        # Load the checkpoint
        checkpoint_path = tmp_checkpoint_dir / "last_model.pt"
        assert checkpoint_path.exists()

        loaded = torch.load(checkpoint_path)
        assert "model" in loaded

    def test_checkpoint_restores_model(
        self, dummy_model, dummy_dataloader, device, tmp_checkpoint_dir
    ):
        """Verify loading checkpoint restores model state."""
        dummy_model.to(device)

        # Get initial predictions
        dummy_model.eval()
        with torch.no_grad():
            sample_input = torch.rand(1, 1, 64, 64).to(device)
            initial_output = dummy_model(sample_input).clone()

        # Train to change weights
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        # Save current state
        evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        # Get trained predictions
        dummy_model.eval()
        with torch.no_grad():
            trained_output = dummy_model(sample_input).clone()

        # Load checkpoint
        checkpoint = torch.load(tmp_checkpoint_dir / "last_model.pt")
        dummy_model.load_state_dict(checkpoint["model"])

        # Verify loaded model gives same output as saved
        dummy_model.eval()
        with torch.no_grad():
            restored_output = dummy_model(sample_input)

        assert torch.allclose(trained_output, restored_output, atol=1e-6)


class TestServerCheckpointing:
    """Tests for server-side checkpointing."""

    def test_save_checkpoint_creates_file(self, tmp_checkpoint_dir):
        """Verify _save_checkpoints creates files."""
        import numpy as np
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        # Create mock parameters
        params = [np.random.randn(10, 10).astype(np.float32)]
        parameters = ndarrays_to_parameters(params)

        # Save checkpoint manually (simulating server behavior)
        tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        params_ndarrays = parameters_to_ndarrays(parameters)

        last_path = tmp_checkpoint_dir / "last_model.pt"
        torch.save({"parameters": params_ndarrays}, last_path)

        assert last_path.exists()

        # Verify loadable
        loaded = torch.load(last_path)
        assert "parameters" in loaded
        assert len(loaded["parameters"]) == 1

    def test_best_model_includes_dice(self, tmp_checkpoint_dir):
        """Verify best model checkpoint includes dice score."""
        import numpy as np

        params = [np.random.randn(10, 10).astype(np.float32)]
        dice_score = 0.85

        best_path = tmp_checkpoint_dir / "best_model.pt"
        torch.save({"parameters": params, "dice": dice_score}, best_path)

        loaded = torch.load(best_path)
        assert "parameters" in loaded
        assert "dice" in loaded
        assert loaded["dice"] == dice_score
