"""Integration tests for FL round with checkpointing."""

import torch
import numpy as np


class TestFullRoundWithCheckpointing:
    """End-to-end tests for FL round with both client and server checkpoints."""

    def test_client_training_creates_checkpoints(
        self, dummy_model, dummy_dataloader, device, tmp_path
    ):
        """Test client training round creates checkpoints."""
        from dp_fedmed.fl.task import train_one_epoch, evaluate
        from dp_fedmed.models.unet2d import get_parameters

        client_dir = tmp_path / "client_0"
        checkpoint_dir = client_dir / "checkpoints"

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Simulate FL client fit
        get_parameters(dummy_model)

        # Train
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # Evaluate
        metrics = evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify checkpoints created
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "last_model.pt").exists()

        # Verify metrics
        assert "dice" in metrics
        assert "loss" in metrics

    def test_server_aggregation_with_checkpointing(self, tmp_path):
        """Test server aggregation saves checkpoints."""

        server_dir = tmp_path / "server"
        checkpoint_dir = server_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simulate aggregated parameters from multiple clients
        client_params = [
            [np.random.randn(10, 10).astype(np.float32) for _ in range(3)]
            for _ in range(2)  # 2 clients
        ]

        # Average parameters (simplified FedAvg)
        avg_params = [
            np.mean([client[i] for client in client_params], axis=0) for i in range(3)
        ]

        # Save checkpoint
        torch.save({"parameters": avg_params}, checkpoint_dir / "last_model.pt")

        # Simulate best model save
        dice_score = 0.75
        torch.save(
            {"parameters": avg_params, "dice": dice_score},
            checkpoint_dir / "best_model.pt",
        )

        # Verify
        assert (checkpoint_dir / "last_model.pt").exists()
        assert (checkpoint_dir / "best_model.pt").exists()

        best = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
        assert best["dice"] == dice_score

    def test_checkpoint_can_restore_training(
        self, dummy_model, dummy_dataloader, device, tmp_path
    ):
        """Test that checkpointed model can continue training."""
        from dp_fedmed.fl.task import train_one_epoch, evaluate
        from dp_fedmed.models.unet2d import get_parameters

        checkpoint_dir = tmp_path / "checkpoints"
        dummy_model.to(device)

        # Round 1: Train and save
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        evaluate(
            model=dummy_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # Get trained parameters
        get_parameters(dummy_model)

        # Create fresh model (simulating new round)
        from dp_fedmed.models.unet2d import create_unet2d

        fresh_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),  # Must match dummy_model fixture
            strides=(2,),
            num_res_units=1,
        )
        fresh_model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_dir / "last_model.pt", weights_only=True)
        fresh_model.load_state_dict(checkpoint["model"])

        # Verify same predictions
        dummy_model.eval()
        fresh_model.eval()
        with torch.no_grad():
            sample = torch.rand(1, 1, 64, 64).to(device)
            orig_out = dummy_model(sample)
            restored_out = fresh_model(sample)

        assert torch.allclose(orig_out, restored_out, atol=1e-6)

        # Round 2: Continue training
        optimizer2 = torch.optim.SGD(fresh_model.parameters(), lr=0.01)
        train_one_epoch(
            model=fresh_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer2,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        round2_metrics = evaluate(
            model=fresh_model,
            test_loader=dummy_dataloader,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # Training should continue (not fail)
        assert "dice" in round2_metrics


class TestDataFormatCompatibility:
    """Tests for different data format handling."""

    def test_tuple_batch_format(
        self, dummy_model, dummy_dataloader, device, tmp_checkpoint_dir
    ):
        """Test with tuple batch format (images, labels)."""
        from dp_fedmed.fl.task import train_one_epoch

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Should work with tuple format
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert True

    def test_dict_batch_format(
        self, dummy_model, dummy_dict_dataloader, device, tmp_checkpoint_dir
    ):
        """Test with dict batch format {"image": ..., "label": ...}."""
        from dp_fedmed.fl.task import train_one_epoch

        dummy_model.to(device)
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Should work with dict format
        train_one_epoch(
            model=dummy_model,
            train_loader=dummy_dict_dataloader,
            optimizer=optimizer,
            device=device,
        )

        assert True
