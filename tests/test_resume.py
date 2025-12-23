"""Tests for checkpoint resumption functionality with unified checkpoints."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from dp_fedmed.fl.checkpoint import (
    UnifiedCheckpoint,
    UnifiedCheckpointManager,
    ClientState,
    ServerState,
    PrivacyState,
    RoundProgress,
    EpochProgress,
    load_unified_checkpoint,
    resolve_checkpoint_path,
    save_unified_checkpoint,
)
from dp_fedmed.fl.base.strategy import DPStrategy
from dp_fedmed.models.unet2d import create_unet2d, get_parameters


class TestCheckpointMetadata:
    """Tests for checkpoint metadata in unified format."""

    @pytest.fixture
    def dummy_strategy(self, tmp_path):
        """Create a DPStrategy strategy for testing."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_manager = UnifiedCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_name="test",
            num_rounds=10,
            target_delta=1e-5,
        )
        return DPStrategy(
            target_delta=1e-5,
            run_dir=tmp_path / "server",
            run_name="test",
            save_metrics=True,
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            start_round=1,
            local_epochs=5,
            checkpoint_manager=checkpoint_manager,
        )

    def test_checkpoint_includes_round_status(self, dummy_strategy, tmp_path):
        """Test that checkpoint includes round status (in_progress/completed)."""
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

        # Initialize checkpoint
        dummy_strategy.initialize_checkpoint(dummy_strategy.latest_parameters)

        # Save checkpoint
        dummy_strategy._save_checkpoints(0.75)

        # Load and verify
        checkpoint_path = tmp_path / "checkpoints" / "last.pt"
        assert checkpoint_path.exists()

        checkpoint = load_unified_checkpoint(checkpoint_path)
        assert checkpoint.round.current == 3
        assert checkpoint.round.status == "completed"

    def test_checkpoint_includes_privacy_state(self, dummy_strategy, tmp_path):
        """Test that checkpoint includes cumulative epsilon."""
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

        # Record some privacy spending
        dummy_strategy.privacy_accountant.record_round(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=10,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=100,
        )

        # Initialize and save checkpoint
        dummy_strategy.initialize_checkpoint(dummy_strategy.latest_parameters)
        dummy_strategy._save_checkpoints(0.75)

        # Load and verify
        checkpoint_path = tmp_path / "checkpoints" / "last.pt"
        checkpoint = load_unified_checkpoint(checkpoint_path)

        assert checkpoint.privacy.cumulative_sample_epsilon > 0


class TestMidRoundResume:
    """Tests for mid-round resume functionality."""

    def test_mid_round_checkpoint_has_in_progress_status(self, tmp_path):
        """Test that mid-round checkpoint has status='in_progress'."""
        params = [np.random.randn(10, 10).astype(np.float32)]

        # Create checkpoint with in_progress status
        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="in_progress"),
            server=ServerState(parameters=params, best_dice=0.75),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=5, total=10, status="in_progress"),
                ),
                1: ClientState(
                    client_id=1,
                    epoch=EpochProgress(current=3, total=10, status="in_progress"),
                ),
            },
            privacy=PrivacyState(target_delta=1e-5),
        )

        # Save and load
        save_unified_checkpoint(checkpoint, tmp_path)
        loaded = load_unified_checkpoint(tmp_path / "last.pt")

        assert loaded.round.status == "in_progress"
        assert loaded.clients[0].epoch.current == 5
        assert loaded.clients[1].epoch.current == 3

    def test_client_resume_from_epoch(self, tmp_path):
        """Test that client can resume from specific epoch."""
        model_state = {"layer.weight": torch.randn(10, 10)}

        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="in_progress"),
            server=ServerState(parameters=[], best_dice=0.75),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=7, total=10, status="in_progress"),
                    model_state=model_state,
                    partial_metrics={"loss_sum": 3.5, "epochs_done": 7},
                    partial_privacy={"epsilon": 1.2, "steps": 70},
                ),
            },
            privacy=PrivacyState(target_delta=1e-5),
        )

        save_unified_checkpoint(checkpoint, tmp_path)
        loaded = load_unified_checkpoint(tmp_path / "last.pt")

        client = loaded.clients[0]
        assert client.epoch.current == 7
        assert client.partial_metrics["epochs_done"] == 7
        assert client.partial_privacy["epsilon"] == 1.2
        assert client.model_state is not None


class TestStartRoundOffset:
    """Tests for round offset when resuming."""

    def test_configure_fit_uses_actual_round(self):
        """Test that configure_fit uses actual round when resuming."""
        strategy = DPStrategy(
            target_delta=1e-5,
            start_round=4,  # Resuming from round 4
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            local_epochs=5,
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

    def test_configure_fit_includes_local_epochs(self):
        """Test that configure_fit includes local_epochs in config."""
        strategy = DPStrategy(
            target_delta=1e-5,
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            local_epochs=7,
        )

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

        assert config["local_epochs"] == 7

    def test_mid_round_resume_signals_clients(self):
        """Test that mid-round resume sends signal to clients."""
        strategy = DPStrategy(
            target_delta=1e-5,
            start_round=4,
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            local_epochs=5,
            is_mid_round_resume=True,
        )

        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 2
        mock_client_manager.sample.return_value = [MagicMock(), MagicMock()]

        # First round of current run (but round 4 overall)
        result = strategy.configure_fit(
            server_round=1,
            parameters=None,
            client_manager=mock_client_manager,
        )

        _, fit_ins = result[0]
        config = fit_ins.config

        assert config["resume_from_checkpoint"] is True
        assert "checkpoint_path" in config

    def test_fresh_start_has_start_round_one(self):
        """Test that fresh start has start_round=1."""
        strategy = DPStrategy(
            target_delta=1e-5,
            num_rounds=10,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            local_epochs=5,
        )

        assert strategy.start_round == 1

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

        assert config["server_round"] == 1
        assert config["resume_from_checkpoint"] is False


class TestRemainingRoundsCalculation:
    """Tests for remaining rounds calculation."""

    def test_remaining_rounds_calculated_correctly(self):
        """Test that remaining rounds is num_rounds - start_round + 1."""
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

    def test_resume_past_num_rounds_gives_zero(self):
        """Test resuming past num_rounds gives 0 remaining."""
        num_rounds = 5
        start_round = 6
        remaining_rounds = num_rounds - start_round + 1

        assert remaining_rounds == 0


class TestCheckpointPathResolution:
    """Tests for unified checkpoint path resolution."""

    def test_resolve_last_keyword_unified(self, tmp_path):
        """Test resolving 'last' keyword to unified checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        last_checkpoint = checkpoint_dir / "last.pt"
        last_checkpoint.touch()

        resolved = resolve_checkpoint_path("last", tmp_path)
        assert resolved == last_checkpoint

    def test_resolve_best_keyword_unified(self, tmp_path):
        """Test resolving 'best' keyword to unified checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        best_checkpoint = checkpoint_dir / "best.pt"
        best_checkpoint.touch()

        resolved = resolve_checkpoint_path("best", tmp_path)
        assert resolved == best_checkpoint

    def test_resolve_keyword_case_insensitive(self, tmp_path):
        """Test that keywords are case-insensitive."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.pt").touch()

        assert resolve_checkpoint_path("LAST", tmp_path) is not None
        assert resolve_checkpoint_path("Last", tmp_path) is not None
        assert resolve_checkpoint_path("  last  ", tmp_path) is not None

    def test_resolve_missing_checkpoint_raises(self, tmp_path):
        """Test that missing checkpoint raises FileNotFoundError."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path("last", tmp_path)

    def test_resolve_absolute_path(self, tmp_path):
        """Test resolving absolute path unchanged."""
        checkpoint = tmp_path / "my_checkpoint.pt"
        checkpoint.touch()

        resolved = resolve_checkpoint_path(str(checkpoint), tmp_path)
        assert resolved == checkpoint

    def test_resolve_none_returns_none(self, tmp_path):
        """Test that None/empty input returns None."""
        assert resolve_checkpoint_path(None, tmp_path) is None
        assert resolve_checkpoint_path("", tmp_path) is None
        assert resolve_checkpoint_path("   ", tmp_path) is None


class TestCheckpointResumeIntegration:
    """Integration tests for full unified checkpoint resume flow."""

    @pytest.fixture
    def dummy_model(self):
        """Create a small UNet for testing."""
        return create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

    def test_resume_completed_round(self, tmp_path, dummy_model):
        """Test resuming from a completed round."""
        params = get_parameters(dummy_model)

        # Create checkpoint with completed round
        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=5, total=10, status="completed"),
            server=ServerState(parameters=params, best_dice=0.85),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=10, total=10, status="completed"),
                ),
            },
            privacy=PrivacyState(
                target_delta=1e-5,
                cumulative_sample_epsilon=2.5,
            ),
        )

        checkpoint_dir = tmp_path / "checkpoints"
        save_unified_checkpoint(checkpoint, checkpoint_dir)

        # Load and verify
        loaded = load_unified_checkpoint(checkpoint_dir / "last.pt")

        # Should resume from round 6 (checkpoint_round + 1)
        assert loaded.round.current == 5
        assert loaded.round.status == "completed"
        # Resume round should be 6
        resume_round = loaded.round.current + 1
        assert resume_round == 6

    def test_resume_mid_round(self, tmp_path, dummy_model):
        """Test resuming from mid-round checkpoint."""
        params = get_parameters(dummy_model)
        model_state = dummy_model.state_dict()

        # Create checkpoint with in_progress round
        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="in_progress"),
            server=ServerState(parameters=params, best_dice=0.70),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=7, total=10, status="in_progress"),
                    model_state=model_state,
                    partial_metrics={"loss_sum": 3.5, "epochs_done": 7},
                ),
                1: ClientState(
                    client_id=1,
                    epoch=EpochProgress(current=5, total=10, status="in_progress"),
                    model_state=model_state,
                    partial_metrics={"loss_sum": 2.5, "epochs_done": 5},
                ),
            },
            privacy=PrivacyState(
                target_delta=1e-5,
                cumulative_sample_epsilon=1.2,
                partial_round_epsilon=0.3,
            ),
        )

        checkpoint_dir = tmp_path / "checkpoints"
        save_unified_checkpoint(checkpoint, checkpoint_dir)

        # Load and verify
        loaded = load_unified_checkpoint(checkpoint_dir / "last.pt")

        # Should resume from same round (mid-round)
        assert loaded.round.current == 3
        assert loaded.round.status == "in_progress"

        # Clients should resume from their saved epochs
        assert loaded.clients[0].epoch.current == 7
        assert loaded.clients[1].epoch.current == 5

        # Model state should be preserved
        assert loaded.clients[0].model_state is not None
