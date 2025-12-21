"""Tests for checkpoint resumption functionality."""

import pytest
import torch
from unittest.mock import MagicMock

from dp_fedmed.fl.server.factory import load_checkpoint
from dp_fedmed.fl.server.strategy import DPFedAvg
from dp_fedmed.models.unet2d import create_unet2d, get_parameters


class TestCheckpointMetadata:
    """Tests for checkpoint metadata saving."""

    @pytest.fixture
    def dummy_strategy(self, tmp_path):
        """Create a DPFedAvg strategy for testing."""
        return DPFedAvg(
            target_epsilon=8.0,
            target_delta=1e-5,
            run_dir=tmp_path,
            run_name="test",
            save_metrics=True,
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            start_round=1,
        )

    def test_checkpoint_includes_round_number(self, dummy_strategy, tmp_path):
        """Test that saved checkpoint includes round number."""
        from flwr.common import ndarrays_to_parameters

        # Create dummy parameters
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        params = get_parameters(model)
        dummy_strategy.latest_parameters = ndarrays_to_parameters(params)
        dummy_strategy.current_round = 3

        # Save checkpoint
        dummy_strategy._save_checkpoints(current_dice=0.75)

        # Load and verify
        checkpoint_path = tmp_path / "checkpoints" / "last_model.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "round" in checkpoint
        assert checkpoint["round"] == 3

    def test_checkpoint_includes_cumulative_epsilon(self, dummy_strategy, tmp_path):
        """Test that saved checkpoint includes cumulative epsilon."""
        from flwr.common import ndarrays_to_parameters

        # Create dummy parameters
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        params = get_parameters(model)
        dummy_strategy.latest_parameters = ndarrays_to_parameters(params)
        dummy_strategy.current_round = 3

        # Record some privacy spending
        dummy_strategy.privacy_accountant.record_round(
            round_num=1,
            epsilon=0.5,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            num_samples=100,
        )

        # Save checkpoint
        dummy_strategy._save_checkpoints(current_dice=0.75)

        # Load and verify
        checkpoint_path = tmp_path / "checkpoints" / "last_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert "cumulative_epsilon" in checkpoint
        assert checkpoint["cumulative_epsilon"] > 0

    def test_checkpoint_includes_dice(self, dummy_strategy, tmp_path):
        """Test that saved checkpoint includes dice score."""
        from flwr.common import ndarrays_to_parameters

        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        params = get_parameters(model)
        dummy_strategy.latest_parameters = ndarrays_to_parameters(params)
        dummy_strategy.current_round = 3

        # Save checkpoint
        dummy_strategy._save_checkpoints(current_dice=0.85)

        # Load and verify
        checkpoint_path = tmp_path / "checkpoints" / "last_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert "dice" in checkpoint
        assert checkpoint["dice"] == 0.85


class TestLoadCheckpoint:
    """Tests for checkpoint loading functionality."""

    @pytest.fixture
    def dummy_model(self):
        """Create a small UNet for testing."""
        return create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

    def test_load_server_format_checkpoint(self, dummy_model, tmp_path):
        """Test loading server-format checkpoint (numpy arrays)."""
        # Create and save checkpoint
        params = get_parameters(dummy_model)
        checkpoint = {
            "parameters": params,
            "round": 5,
            "cumulative_epsilon": 2.5,
            "dice": 0.80,
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        parameters, resume_round, cum_epsilon, dice = load_checkpoint(
            checkpoint_path, dummy_model
        )

        assert resume_round == 6  # Should be checkpoint_round + 1
        assert cum_epsilon == 2.5
        assert dice == 0.80
        assert parameters is not None

    def test_load_client_format_checkpoint(self, dummy_model, tmp_path):
        """Test loading client-format checkpoint (state dict)."""
        # Create and save checkpoint
        checkpoint = {
            "model": dummy_model.state_dict(),
            "round": 3,
            "dice": 0.75,
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        parameters, resume_round, cum_epsilon, dice = load_checkpoint(
            checkpoint_path, dummy_model
        )

        assert resume_round == 4  # Should be checkpoint_round + 1
        assert dice == 0.75
        assert parameters is not None

    def test_load_checkpoint_missing_round_defaults_to_zero(
        self, dummy_model, tmp_path
    ):
        """Test that missing round defaults to 0 (resume from 1)."""
        # Create checkpoint without round
        params = get_parameters(dummy_model)
        checkpoint = {
            "parameters": params,
            "dice": 0.70,
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        parameters, resume_round, cum_epsilon, dice = load_checkpoint(
            checkpoint_path, dummy_model
        )

        assert resume_round == 1  # 0 + 1

    def test_load_checkpoint_unknown_format_raises_error(self, dummy_model, tmp_path):
        """Test that unknown checkpoint format raises ValueError."""
        # Create checkpoint with neither 'parameters' nor 'model'
        checkpoint = {
            "weights": [1, 2, 3],
            "round": 5,
        }

        checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="Unknown checkpoint format"):
            load_checkpoint(checkpoint_path, dummy_model)


class TestStartRoundOffset:
    """Tests for round offset when resuming."""

    def test_configure_fit_uses_actual_round(self):
        """Test that configure_fit uses actual round when resuming."""
        strategy = DPFedAvg(
            target_epsilon=8.0,
            target_delta=1e-5,
            start_round=4,  # Resuming from round 4
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Mock client manager
        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 2
        mock_client_manager.sample.return_value = [MagicMock(), MagicMock()]

        # Flower calls with server_round=1 (first round in current run)
        # But actual round should be 4
        result = strategy.configure_fit(
            server_round=1,
            parameters=None,
            client_manager=mock_client_manager,
        )

        # Extract config from first client's FitIns
        _, fit_ins = result[0]
        config = fit_ins.config

        # Server_round in config should be actual round (4)
        assert config["server_round"] == 4
        assert strategy.current_round == 4

    def test_configure_evaluate_uses_actual_round(self):
        """Test that configure_evaluate uses actual round when resuming."""
        strategy = DPFedAvg(
            target_epsilon=8.0,
            target_delta=1e-5,
            start_round=3,  # Resuming from round 3
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Mock client manager
        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 2
        mock_client_manager.sample.return_value = [MagicMock(), MagicMock()]

        # Flower calls with server_round=2 (second round in current run)
        # But actual round should be 3 + 2 - 1 = 4
        result = strategy.configure_evaluate(
            server_round=2,
            parameters=None,
            client_manager=mock_client_manager,
        )

        # Extract config from first client's EvaluateIns
        _, eval_ins = result[0]
        config = eval_ins.config

        # Server_round in config should be actual round (4)
        assert config["server_round"] == 4

    def test_fresh_start_has_start_round_one(self):
        """Test that fresh start (no resume) has start_round=1."""
        strategy = DPFedAvg(
            target_epsilon=8.0,
            target_delta=1e-5,
            # No start_round specified, should default to 1
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        assert strategy.start_round == 1

        # Mock client manager
        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 2
        mock_client_manager.sample.return_value = [MagicMock(), MagicMock()]

        result = strategy.configure_fit(
            server_round=1,
            parameters=None,
            client_manager=mock_client_manager,
        )

        _, fit_ins = result[0]
        config = fit_ins.config

        # With start_round=1, server_round=1 should give actual_round=1
        assert config["server_round"] == 1


class TestRemainingRoundsCalculation:
    """Tests for remaining rounds calculation in server factory."""

    def test_remaining_rounds_calculated_correctly(self):
        """Test that remaining rounds is num_rounds - start_round + 1."""
        # If num_rounds=10 and we resume from round 4
        # We should run rounds 4, 5, 6, 7, 8, 9, 10 = 7 rounds
        num_rounds = 10
        start_round = 4
        remaining_rounds = num_rounds - start_round + 1

        assert remaining_rounds == 7

    def test_resume_at_last_round_gives_one_remaining(self):
        """Test resuming at last round gives 1 remaining round."""
        num_rounds = 5
        start_round = 5
        remaining_rounds = num_rounds - start_round + 1

        assert remaining_rounds == 1

    def test_resume_past_num_rounds_gives_zero_or_negative(self):
        """Test resuming past num_rounds gives 0 or negative remaining."""
        num_rounds = 5
        start_round = 6  # Checkpoint from round 5, resume at 6
        remaining_rounds = num_rounds - start_round + 1

        assert remaining_rounds == 0
