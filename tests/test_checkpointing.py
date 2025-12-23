"""Tests for unified checkpoint system."""

import numpy as np
import pytest
import torch

from dp_fedmed.fl.checkpoint import (
    UnifiedCheckpoint,
    UnifiedCheckpointManager,
    ClientState,
    ServerState,
    PrivacyState,
    RoundProgress,
    EpochProgress,
    save_unified_checkpoint,
    load_unified_checkpoint,
    resolve_checkpoint_path,
    get_model_state_dict,
)


class TestUnifiedCheckpointDataclasses:
    """Tests for checkpoint dataclasses."""

    def test_epoch_progress_creation(self):
        """Test EpochProgress creation."""
        progress = EpochProgress(current=5, total=10, status="in_progress")
        assert progress.current == 5
        assert progress.total == 10
        assert progress.status == "in_progress"

    def test_client_state_creation(self):
        """Test ClientState creation with defaults."""
        client = ClientState(
            client_id=0,
            epoch=EpochProgress(current=3, total=5, status="in_progress"),
        )
        assert client.client_id == 0
        assert client.epoch.current == 3
        assert client.model_state is None
        assert client.partial_metrics == {"loss_sum": 0.0, "epochs_done": 0}

    def test_server_state_creation(self):
        """Test ServerState creation."""
        params = [np.random.randn(10, 10).astype(np.float32)]
        server = ServerState(parameters=params, best_dice=0.85)
        assert len(server.parameters) == 1
        assert server.best_dice == 0.85

    def test_unified_checkpoint_to_dict(self):
        """Test UnifiedCheckpoint serialization to dict."""
        params = [np.random.randn(10, 10).astype(np.float32)]

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
                )
            },
            privacy=PrivacyState(target_delta=1e-5),
        )

        data = checkpoint.to_dict()

        assert data["version"] == "2.0"
        assert data["run_name"] == "test"
        assert data["round"]["current"] == 3
        assert data["round"]["status"] == "in_progress"
        assert data["server"]["best_dice"] == 0.75
        assert 0 in data["clients"]
        assert data["privacy"]["target_delta"] == 1e-5

    def test_unified_checkpoint_from_dict_roundtrip(self):
        """Test UnifiedCheckpoint from_dict reconstructs correctly."""
        params = [np.random.randn(10, 10).astype(np.float32)]

        original = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="completed"),
            server=ServerState(parameters=params, best_dice=0.85),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=10, total=10, status="completed"),
                ),
                1: ClientState(
                    client_id=1,
                    epoch=EpochProgress(current=8, total=10, status="in_progress"),
                ),
            },
            privacy=PrivacyState(
                target_delta=1e-5,
                cumulative_sample_epsilon=2.5,
            ),
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = UnifiedCheckpoint.from_dict(data)

        assert restored.version == original.version
        assert restored.run_name == original.run_name
        assert restored.round.current == original.round.current
        assert restored.round.status == original.round.status
        assert restored.server.best_dice == original.server.best_dice
        assert len(restored.clients) == 2
        assert restored.clients[0].epoch.status == "completed"
        assert restored.clients[1].epoch.status == "in_progress"
        assert restored.privacy.cumulative_sample_epsilon == 2.5


class TestUnifiedCheckpointSaveLoad:
    """Tests for saving and loading unified checkpoints."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading unified checkpoint."""
        params = [np.random.randn(10, 10).astype(np.float32)]

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
                )
            },
            privacy=PrivacyState(target_delta=1e-5),
        )

        # Save
        path = save_unified_checkpoint(checkpoint, tmp_path)
        assert path.exists()
        assert path.name == "last.pt"

        # Load
        loaded = load_unified_checkpoint(path)
        assert loaded.version == "2.0"
        assert loaded.round.current == 3
        assert loaded.server.best_dice == 0.75

    def test_save_best_checkpoint(self, tmp_path):
        """Test saving best checkpoint."""
        params = [np.random.randn(10, 10).astype(np.float32)]

        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="completed"),
            server=ServerState(parameters=params, best_dice=0.95),
            clients={},
            privacy=PrivacyState(target_delta=1e-5),
        )

        save_unified_checkpoint(checkpoint, tmp_path, is_best=True)
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "last.pt").exists()

    def test_load_nonexistent_checkpoint_raises(self, tmp_path):
        """Test that loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_unified_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_incompatible_version_raises(self, tmp_path):
        """Test that loading incompatible version raises ValueError."""
        # Save with old version
        checkpoint_data = {"version": "1.0", "data": "old"}
        path = tmp_path / "old_checkpoint.pt"
        torch.save(checkpoint_data, path)

        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            load_unified_checkpoint(path)


class TestUnifiedCheckpointManager:
    """Tests for UnifiedCheckpointManager."""

    def test_create_initial_checkpoint(self, tmp_path):
        """Test creating initial checkpoint for fresh run."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=10,
            target_delta=1e-5,
        )

        params = [np.random.randn(10, 10).astype(np.float32)]
        checkpoint = manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        assert checkpoint.round.current == 1
        assert checkpoint.round.total == 10
        assert checkpoint.round.status == "in_progress"
        assert len(checkpoint.clients) == 2
        assert checkpoint.clients[0].epoch.total == 5

    def test_update_client_epoch(self, tmp_path):
        """Test updating client epoch progress."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=10,
        )

        params = [np.random.randn(10, 10).astype(np.float32)]
        manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        # Update client 0's epoch
        manager.update_client_epoch(
            client_id=0,
            epoch=2,  # 0-indexed, so epoch 3 (1-indexed)
            total_epochs=5,
            epoch_loss=0.5,
        )

        client = manager.get_client_state(0)
        assert client is not None
        assert client.epoch.current == 3  # Stored as 1-indexed
        assert client.partial_metrics["epochs_done"] == 3

    def test_mark_round_completed(self, tmp_path):
        """Test marking round as completed."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=10,
        )

        params = [np.random.randn(10, 10).astype(np.float32)]
        manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        manager.mark_round_completed(dice_score=0.85)

        checkpoint = manager.get_current_checkpoint()
        assert checkpoint is not None
        assert checkpoint.round.status == "completed"
        assert checkpoint.server.best_dice == 0.85

    def test_start_next_round_resets_clients(self, tmp_path):
        """Test starting next round resets client progress."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=10,
        )

        params = [np.random.randn(10, 10).astype(np.float32)]
        manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        # Complete round 1
        manager.update_client_epoch(0, epoch=4, total_epochs=5)
        manager.mark_round_completed(0.75)

        # Start round 2
        manager.start_next_round(round_num=2, local_epochs=5)

        checkpoint = manager.get_current_checkpoint()
        assert checkpoint is not None
        assert checkpoint.round.current == 2
        assert checkpoint.round.status == "in_progress"

        # Client progress should be reset
        client = manager.get_client_state(0)
        assert client is not None
        assert client.epoch.current == 0
        assert client.model_state is None

    def test_is_mid_round_resume(self, tmp_path):
        """Test mid-round resume detection."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=10,
        )

        params = [np.random.randn(10, 10).astype(np.float32)]
        manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=5,
        )

        # Should be mid-round (in_progress)
        assert manager.is_mid_round_resume() is True

        # Complete the round
        manager.mark_round_completed(0.80)
        assert manager.is_mid_round_resume() is False


class TestResolveCheckpointPath:
    """Tests for checkpoint path resolution."""

    def test_resolve_last_keyword(self, tmp_path):
        """Test resolving 'last' keyword."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "last.pt").touch()

        resolved = resolve_checkpoint_path("last", tmp_path)
        assert resolved == checkpoint_dir / "last.pt"

    def test_resolve_best_keyword(self, tmp_path):
        """Test resolving 'best' keyword."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "best.pt").touch()

        resolved = resolve_checkpoint_path("best", tmp_path)
        assert resolved == checkpoint_dir / "best.pt"

    def test_resolve_keyword_case_insensitive(self, tmp_path):
        """Test keywords are case-insensitive."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "last.pt").touch()

        assert resolve_checkpoint_path("LAST", tmp_path) is not None
        assert resolve_checkpoint_path("Last", tmp_path) is not None

    def test_resolve_none_returns_none(self, tmp_path):
        """Test None/empty returns None."""
        assert resolve_checkpoint_path(None, tmp_path) is None
        assert resolve_checkpoint_path("", tmp_path) is None
        assert resolve_checkpoint_path("   ", tmp_path) is None

    def test_resolve_missing_checkpoint_raises(self, tmp_path):
        """Test missing checkpoint raises FileNotFoundError."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path("last", tmp_path)

    def test_resolve_absolute_path(self, tmp_path):
        """Test resolving absolute path."""
        checkpoint = tmp_path / "my_checkpoint.pt"
        checkpoint.touch()

        resolved = resolve_checkpoint_path(str(checkpoint), tmp_path)
        assert resolved == checkpoint


class TestGetModelStateDict:
    """Tests for get_model_state_dict helper."""

    def test_get_state_dict_regular_model(self, dummy_model, device):
        """Test getting state dict from regular model."""
        dummy_model.to(device)
        state_dict = get_model_state_dict(dummy_model)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_get_state_dict_opacus_wrapped(self, dummy_model, device):
        """Test getting state dict from Opacus-wrapped model."""
        from opacus.grad_sample.grad_sample_module import GradSampleModule

        dummy_model.to(device)
        wrapped = GradSampleModule(dummy_model)

        state_dict = get_model_state_dict(wrapped)
        assert isinstance(state_dict, dict)
        # Should get inner module's state dict
        assert len(state_dict) > 0
