"""Integration tests for pretrained federated learning.

This module tests the integration between SSL pretraining and federated learning,
ensuring that:
- Pretrained checkpoints can be loaded into federated clients
- Training works with pretrained encoder initialization
- Encoder freezing works correctly
"""

from typing import Any
import torch
import torch.nn as nn

from dp_fedmed.models.unet2d import create_unet2d
from dp_fedmed.fl.ssl.checkpoint import save_pretrained_checkpoint
from dp_fedmed.fl.ssl.config import SSLConfig
from dp_fedmed.fl.ssl.task import train_epoch_ssl
from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.fl.client.dp_client import DPFlowerClient


class TestPretrainedEncoderLoading:
    """Test loading pretrained encoders into DP clients."""

    def test_client_with_pretrained_checkpoint(self, tmp_path):
        """Test DPFlowerClient initialization with pretrained checkpoint."""
        # Create a dummy checkpoint
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={"loss": 0.5},
            checkpoint_path=checkpoint_path,
        )

        # Create dummy dataloaders
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Create client with pretrained checkpoint
        device = torch.device("cpu")
        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=device,
            pretrained_checkpoint_path=checkpoint_path,
        )

        # Verify client was created
        assert client is not None
        assert client.model is not None

    def test_client_without_pretrained_checkpoint(self, tmp_path):
        """Test DPFlowerClient initialization without pretrained checkpoint."""
        # Create dummy dataloaders
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        device = torch.device("cpu")
        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=device,
            pretrained_checkpoint_path=None,
        )

        assert client is not None
        assert client.model is not None

    def test_client_with_pretrained_encoder_freeze(self, tmp_path):
        """Test DPFlowerClient with frozen encoder."""
        # Create a dummy checkpoint
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={"loss": 0.5},
            checkpoint_path=checkpoint_path,
        )

        # Create dummy dataloaders
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        device = torch.device("cpu")
        # Client should accept freeze_encoder parameter
        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=device,
            pretrained_checkpoint_path=checkpoint_path,
            freeze_encoder=True,
        )

        assert client is not None


class TestFullPipelineIntegration:
    """Test full pipeline: pretrain -> load -> federated train."""

    def test_pretrain_and_load_into_client(self, tmp_path):
        """Test pretraining a model and loading it into a federated client."""
        # Phase 1: Pretrain with SSL
        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        config = SSLConfig(
            data_dir=tmp_path,
            method="simclr",
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            device="cpu",
        )

        ssl_model = SSLUNet(
            base_model,
            projection_dim=config.projection_dim,
            hidden_dim=config.hidden_dim,
        ).to("cpu")
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=config.learning_rate)
        criterion = nn.CosineSimilarity(dim=1)

        # Create minimal SSL dataloader
        class SSLDataset(torch.utils.data.Dataset[Any]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: Any):
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        ssl_loader = torch.utils.data.DataLoader(SSLDataset(), batch_size=2)

        # Train for one epoch
        loss = train_epoch_ssl(
            ssl_model,
            ssl_loader,
            optimizer,
            torch.device("cpu"),
            criterion,
        )
        assert loss >= 0.0

        # Save pretrained checkpoint
        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=ssl_model,
            optimizer=optimizer,
            epoch=1,
            metrics={"loss": loss},
            checkpoint_path=checkpoint_path,
        )
        assert checkpoint_path.exists()

        # Phase 2: Load into federated client
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=torch.device("cpu"),
            pretrained_checkpoint_path=checkpoint_path,
        )

        assert client is not None
        assert client.model is not None

    def test_weight_comparison_pretrained_vs_random(self, tmp_path):
        """Verify that pretrained and random initialized models have different weights."""
        # Create and save a model
        model1 = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model1,
            optimizer=optimizer,
            epoch=10,
            metrics={"loss": 0.5},
            checkpoint_path=checkpoint_path,
        )

        # Create a second model for comparison
        base_model2 = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        # Get weights - they should differ before loading
        checkpoint_weights = torch.load(checkpoint_path, map_location="cpu")[
            "model_state_dict"
        ]
        trainer_weights = base_model2.state_dict()

        # Find at least one parameter that differs
        weights_differ = False
        for key in list(trainer_weights.keys())[:3]:  # Check first few layers
            if key in checkpoint_weights:
                if not torch.allclose(
                    checkpoint_weights[key], trainer_weights[key], atol=1e-6
                ):
                    weights_differ = True
                    break

        # New models should have different random initialization
        assert weights_differ, "Different models should have different weights"

    def test_client_preserves_pretrained_weights(self, tmp_path):
        """Verify that loading pretrained weights into client preserves them."""
        # Create initial model
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Save checkpoint
        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={},
            checkpoint_path=checkpoint_path,
        )

        # Get weights from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_weights = checkpoint["model_state_dict"]

        # Load into client
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=torch.device("cpu"),
            pretrained_checkpoint_path=checkpoint_path,
        )

        # Get client model weights
        # Need to unwrap if using GradSampleModule
        from dp_fedmed.utils import get_unwrapped_model

        unwrapped_client_model = get_unwrapped_model(client.model)
        client_weights = unwrapped_client_model.state_dict()

        # Verify they match
        # UNet has 2 main submodules: 'model' (Sequential) and potentially 'encoders' if custom
        # Monai UNet structure might have different naming.
        # Let's check common keys instead of specific ones if we want to be robust,
        # or use a more flexible check.
        checkpoint_keys = set(checkpoint_weights.keys())
        client_keys = set(client_weights.keys())

        common_keys = checkpoint_keys.intersection(client_keys)
        assert len(common_keys) > 0, (
            "No common keys found between checkpoint and client model"
        )

        for key in common_keys:
            assert torch.allclose(
                checkpoint_weights[key],
                client_weights[key],
                atol=1e-6,
            ), f"Weight mismatch for {key}"


class TestFederatedTrainingWithPretrained:
    """Test actual federated training with pretrained encoder."""

    def test_client_fit_with_pretrained(self, tmp_path):
        """Test running fit() on client with pretrained weights."""
        # Create and save a pretrained checkpoint
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={},
            checkpoint_path=checkpoint_path,
        )

        # Create client with pretrained checkpoint
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        device = torch.device("cpu")
        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={
                "learning_rate": 0.001,
                "local_epochs": 1,
                "momentum": 0.9,
            },
            privacy_config={"style": "none", "target_delta": 1e-5},
            device=device,
            pretrained_checkpoint_path=checkpoint_path,
        )

        # Create dummy global parameters
        from dp_fedmed.models.unet2d import get_parameters

        global_params = get_parameters(client.model)

        # Run one fit round
        updated_params, num_samples, metrics = client.fit(
            global_params,
            {
                "server_round": 1,
                "local_epochs": 1,
                "resume_from_checkpoint": False,
                "noise_multiplier": 1.0,
            },
        )

        # Verify results
        assert updated_params is not None
        assert len(updated_params) > 0
        assert num_samples == 8
        assert "loss" in metrics
        assert "epsilon" in metrics

    def test_client_evaluate_with_pretrained(self, tmp_path):
        """Test running evaluate() on client with pretrained weights."""
        # Create and save pretrained checkpoint
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "pretrained.pt"
        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={},
            checkpoint_path=checkpoint_path,
        )

        # Create client
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 1, 64, 64),
            torch.randint(0, 2, (8, 64, 64)),
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        client = DPFlowerClient(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config={
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 1,
            },
            training_config={"learning_rate": 0.001, "local_epochs": 1},
            privacy_config={"style": "none"},
            device=torch.device("cpu"),
            pretrained_checkpoint_path=checkpoint_path,
        )

        from dp_fedmed.models.unet2d import get_parameters

        params = get_parameters(client.model)

        # Run evaluation
        loss, num_samples, metrics = client.evaluate(params, {"server_round": 1})

        # Verify results
        assert loss >= 0.0
        assert num_samples == 8
        assert "dice" in metrics
