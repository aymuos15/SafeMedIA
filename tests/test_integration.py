"""Integration tests for FL training and unified checkpointing."""

import numpy as np
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
)


class TestFullRoundWithCheckpointing:
    """End-to-end tests for FL round with unified checkpointing."""

    def test_client_training_returns_metrics(
        self, dummy_model, dummy_dataloader, device, tmp_path
    ):
        """Test client training round completes and returns metrics."""
        from dp_fedmed.fl.task import train_one_epoch, evaluate
        from dp_fedmed.models.unet2d import get_parameters

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Simulate FL client fit
        get_parameters(dummy_model)

        # Train
        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Evaluate
        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
        )

        # Verify training completed
        assert isinstance(loss, float)
        assert loss >= 0

        # Verify metrics
        assert "dice" in metrics
        assert "loss" in metrics

    def test_unified_checkpoint_saves_server_and_clients(self, tmp_path):
        """Test unified checkpoint contains both server and client state."""
        checkpoint_dir = tmp_path / "checkpoints"

        # Create checkpoint with server and client state
        params = [np.random.randn(10, 10).astype(np.float32)]

        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=3, total=10, status="completed"),
            server=ServerState(parameters=params, best_dice=0.85),
            clients={
                0: ClientState(
                    client_id=0,
                    epoch=EpochProgress(current=5, total=5, status="completed"),
                ),
                1: ClientState(
                    client_id=1,
                    epoch=EpochProgress(current=5, total=5, status="completed"),
                ),
            },
            privacy=PrivacyState(
                target_delta=1e-5,
                cumulative_sample_epsilon=2.5,
            ),
        )

        # Save
        save_unified_checkpoint(checkpoint, checkpoint_dir)

        # Verify files created
        assert (checkpoint_dir / "last.pt").exists()

        # Load and verify
        loaded = load_unified_checkpoint(checkpoint_dir / "last.pt")
        assert loaded.round.current == 3
        assert loaded.server.best_dice == 0.85
        assert len(loaded.clients) == 2
        assert loaded.privacy.cumulative_sample_epsilon == 2.5

    def test_checkpoint_manager_tracks_round_progress(self, tmp_path):
        """Test checkpoint manager properly tracks round and client progress."""
        manager = UnifiedCheckpointManager(
            checkpoint_dir=tmp_path,
            run_name="test",
            num_rounds=5,
            target_delta=1e-5,
        )

        # Initialize
        params = [np.random.randn(10, 10).astype(np.float32)]
        manager.create_initial_checkpoint(
            parameters=params,
            num_clients=2,
            local_epochs=3,
        )

        # Simulate client training progress
        manager.update_client_epoch(
            client_id=0,
            epoch=1,
            total_epochs=3,
            epoch_loss=0.5,
        )

        manager.update_client_epoch(
            client_id=1,
            epoch=2,
            total_epochs=3,
            epoch_loss=0.4,
        )

        # Verify client states
        client_0 = manager.get_client_state(0)
        client_1 = manager.get_client_state(1)

        assert client_0 is not None
        assert client_1 is not None
        assert client_0.epoch.current == 2  # 1 + 1 (0-indexed to 1-indexed)
        assert client_1.epoch.current == 3

        # Complete round
        manager.mark_round_completed(dice_score=0.75)
        manager.save()

        # Verify saved
        assert (tmp_path / "last.pt").exists()

        # Load and verify
        loaded = load_unified_checkpoint(tmp_path / "last.pt")
        assert loaded.round.status == "completed"
        assert loaded.server.best_dice == 0.75

    def test_model_restoration_from_unified_checkpoint(
        self, dummy_model, dummy_dataloader, device, tmp_path
    ):
        """Test that model can be restored from unified checkpoint."""
        from dp_fedmed.fl.task import train_one_epoch
        from dp_fedmed.models.unet2d import get_parameters, create_unet2d

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Train
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Get trained parameters
        trained_params = get_parameters(dummy_model)

        # Create unified checkpoint with trained parameters
        checkpoint = UnifiedCheckpoint(
            version="2.0",
            timestamp="2025-01-01T00:00:00",
            run_name="test",
            round=RoundProgress(current=1, total=5, status="completed"),
            server=ServerState(parameters=trained_params, best_dice=0.75),
            clients={},
            privacy=PrivacyState(target_delta=1e-5),
        )

        checkpoint_dir = tmp_path / "checkpoints"
        save_unified_checkpoint(checkpoint, checkpoint_dir)

        # Create fresh model
        fresh_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
            num_res_units=1,
        )
        fresh_model.to(device)

        # Load checkpoint and restore parameters
        loaded = load_unified_checkpoint(checkpoint_dir / "last.pt")

        # Restore parameters using set_parameters
        from dp_fedmed.models.unet2d import set_parameters

        set_parameters(fresh_model, loaded.server.parameters)

        # Verify same predictions
        dummy_model.eval()
        fresh_model.eval()
        with torch.no_grad():
            sample = torch.rand(1, 1, 64, 64).to(device)
            orig_out = dummy_model(sample)
            restored_out = fresh_model(sample)

        assert torch.allclose(orig_out, restored_out, atol=1e-6)


class TestDataFormatCompatibility:
    """Tests for different data format handling."""

    def test_tuple_batch_format(self, dummy_model, dummy_dataloader, device):
        """Test with tuple batch format (images, labels)."""
        from dp_fedmed.fl.task import train_one_epoch

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Should work with tuple format
        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert isinstance(loss, float)

    def test_dict_batch_format(self, dummy_model, dummy_dict_dataloader, device):
        """Test with dict batch format {"image": ..., "label": ...}."""
        from dp_fedmed.fl.task import train_one_epoch

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Should work with dict format
        loss = train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dict_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert isinstance(loss, float)
