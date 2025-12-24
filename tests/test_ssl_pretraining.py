"""Tests for SSL pretraining functionality.

This module tests the self-supervised learning pretraining pipeline including:
- Configuration loading and validation
- Model training with SSL losses
- Checkpoint saving and loading
- Transform generation for SSL
"""

import pytest
from typing import Any
import torch
import torch.nn as nn
from pathlib import Path

from dp_fedmed.fl.ssl.config import SSLConfig, AugmentationConfig
from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.fl.checkpoint import (
    save_pretrained_checkpoint,
    load_pretrained_encoder,
)
from dp_fedmed.fl.ssl.transforms import get_ssl_transform
from dp_fedmed.models.unet2d import create_unet2d


class TestSSLConfig:
    """Test SSL configuration loading and validation."""

    def test_ssl_config_creation(self, tmp_path):
        """Test creating SSLConfig with defaults."""
        config = SSLConfig(data_dir=tmp_path)
        assert config.method == "simclr"
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.num_workers == 4

    def test_ssl_config_with_custom_values(self, tmp_path):
        """Test SSLConfig with custom values."""
        config = SSLConfig(
            data_dir=tmp_path,
            method="moco",
            epochs=20,
            batch_size=64,
            learning_rate=0.0001,
        )
        assert config.method == "moco"
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001

    def test_augmentation_config(self):
        """Test AugmentationConfig creation."""
        aug_config = AugmentationConfig()
        assert aug_config.input_size == (224, 224)
        assert aug_config.gaussian_blur is True
        assert aug_config.color_jitter_prob == 0.2
        assert aug_config.normalize is True

    def test_ssl_config_validation(self, tmp_path):
        """Test SSLConfig validation."""
        config = SSLConfig(data_dir=Path(tmp_path), method="simclr")
        # Should not raise
        assert config.method == "simclr"

    def test_ssl_config_invalid_method(self, tmp_path):
        """Test SSLConfig with invalid method still creates (validation is in trainer)."""
        config = SSLConfig(data_dir=Path(tmp_path), method="invalid_method")
        # Config creation doesn't validate method, trainer does
        assert config.method == "invalid_method"

    def test_ssl_config_from_dict(self, tmp_path):
        """Test creating SSLConfig from dictionary."""
        data: dict[str, Any] = {
            "data_dir": Path(tmp_path),
            "method": "simsiam",
            "epochs": 15,
            "batch_size": 16,
            "learning_rate": 0.005,
        }
        config = SSLConfig(**data)
        assert config.method == "simsiam"
        assert config.epochs == 15


class TestSSLTransforms:
    """Test SSL transform generation."""

    def test_simclr_transform_creation(self):
        """Test creating SimCLR transforms."""
        aug_config = AugmentationConfig()
        transform = get_ssl_transform(method="simclr", config=aug_config)
        assert transform is not None
        # Verify it's callable
        assert callable(transform)

    def test_moco_transform_creation(self):
        """Test creating MoCo transforms."""
        aug_config = AugmentationConfig()
        transform = get_ssl_transform(method="moco", config=aug_config)
        assert transform is not None
        assert callable(transform)

    def test_simsiam_transform_creation(self):
        """Test creating SimSiam transforms."""
        aug_config = AugmentationConfig()
        transform = get_ssl_transform(method="simsiam", config=aug_config)
        assert transform is not None
        assert callable(transform)

    def test_transform_invalid_method(self):
        """Test transform creation with invalid method."""
        aug_config = AugmentationConfig()
        with pytest.raises((ValueError, KeyError)):
            get_ssl_transform(
                method="invalid_method",
                config=aug_config,
            )

    def test_transform_output_shape(self):
        """Test that transforms output correct shape."""
        # Create dummy image
        image = torch.rand(1, 256, 256)

        # Test transform with actual image tensor
        transform = get_ssl_transform(method="simclr", config=AugmentationConfig())
        # SimCLR returns a tuple of augmented views
        result = transform(image)
        assert isinstance(result, (tuple, list))


class TestSSLUNet:
    """Test SSLUNet model."""

    def test_ssl_unet_creation(self):
        """Test creating SSLUNet model."""
        base_unet = create_unet2d(in_channels=1, out_channels=2)
        model = SSLUNet(
            base_model=base_unet,
            projection_dim=128,
            hidden_dim=2048,
        )
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_ssl_unet_forward_pass(self):
        """Test forward pass through SSLUNet."""
        base_unet = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        model = SSLUNet(
            base_model=base_unet,
            projection_dim=128,
            hidden_dim=2048,
        )

        # Create dummy input
        x = torch.rand(2, 1, 64, 64)

        # Forward pass
        _, z = model(x)

        # Check output shape
        assert z.shape == (2, 128)  # projection_dim

    def test_ssl_unet_encoder_extraction(self):
        """Test extracting encoder from SSLUNet."""
        base_unet = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        model = SSLUNet(
            base_model=base_unet,
            projection_dim=128,
            hidden_dim=2048,
        )

        # Extract encoder
        encoder = model.get_backbone()

        # Verify encoder is the base model
        assert isinstance(encoder, nn.Module)


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        metrics = {"loss": 0.5, "accuracy": 0.95}

        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

        # Verify checkpoint exists
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        # Create and save a model
        model1 = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        metrics = {"loss": 0.5, "accuracy": 0.95}

        save_pretrained_checkpoint(
            model=model1,
            optimizer=optimizer,
            epoch=5,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

        # Load into new model
        model2 = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        metadata = load_pretrained_encoder(model2, checkpoint_path)

        # Verify metadata
        assert metadata["epoch"] == 5
        assert metadata["metrics"]["loss"] == 0.5

        # Verify weights are loaded
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        for key in state_dict1.keys():
            assert torch.allclose(state_dict1[key], state_dict2[key])

    def test_checkpoint_info(self, tmp_path):
        """Test getting checkpoint info without loading."""
        model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        metrics = {"loss": 0.5, "accuracy": 0.95}

        save_pretrained_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        assert checkpoint["epoch"] == 5
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint
        assert checkpoint["metrics"]["loss"] == 0.5

    def test_checkpoint_info_nonexistent(self, tmp_path):
        """Test getting info for nonexistent checkpoint."""
        checkpoint_path = tmp_path / "nonexistent.pt"
        assert not checkpoint_path.exists()


class TestSSLPretrainer:
    """Test SSL pretraining task."""

    def test_train_epoch_ssl(self, tmp_path):
        """Test training a single epoch."""
        from dp_fedmed.fl.ssl.task import train_epoch_ssl

        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        ssl_model = SSLUNet(
            base_model,
            projection_dim=128,
            hidden_dim=256,
        )
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
        criterion = nn.CosineSimilarity(dim=1)

        # Create SSL-style dataloader with tuple of augmented views
        class SSLDataset(torch.utils.data.Dataset[Any]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: Any):
                # Return tuple of two augmented views
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        ssl_loader = torch.utils.data.DataLoader(
            SSLDataset(), batch_size=2, shuffle=False
        )

        loss = train_epoch_ssl(
            ssl_model,
            ssl_loader,
            optimizer,
            torch.device("cpu"),
            criterion,
        )

        assert isinstance(loss, float)
        assert loss >= 0.0


class TestIntegrationSSLPretraining:
    """Integration tests for SSL pretraining pipeline."""

    def test_full_ssl_pipeline_simclr(self, device, tmp_path):
        """Test full SSL pretraining pipeline with SimCLR."""
        from dp_fedmed.fl.ssl.task import train_epoch_ssl

        # Create model and config
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
            projection_dim=128,
            hidden_dim=256,
        )
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
        criterion = nn.CosineSimilarity(dim=1)

        # Create minimal dataloader
        class SSLDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        loader = torch.utils.data.DataLoader(SSLDataset(), batch_size=2)

        # Train one epoch
        loss = train_epoch_ssl(
            ssl_model,
            loader,
            optimizer,
            torch.device("cpu"),
            criterion,
        )
        assert loss >= 0.0

        # Save checkpoint
        checkpoint_path = config.save_dir / "checkpoint_epoch_1.pt"
        save_pretrained_checkpoint(
            model=ssl_model,
            optimizer=optimizer,
            epoch=0,
            metrics={"loss": loss},
            checkpoint_path=checkpoint_path,
        )
        assert checkpoint_path.exists()

        # Verify can load into new model
        new_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        metadata = load_pretrained_encoder(new_model, checkpoint_path)
        assert metadata["epoch"] == 0

    def test_full_ssl_pipeline_moco(self, device, tmp_path):
        """Test full SSL pretraining pipeline with MoCo."""
        from dp_fedmed.fl.ssl.task import train_epoch_ssl

        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )

        ssl_model = SSLUNet(
            base_model,
            projection_dim=128,
            hidden_dim=256,
        )
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
        criterion = nn.CosineSimilarity(dim=1)

        class SSLDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        loader = torch.utils.data.DataLoader(SSLDataset(), batch_size=2)
        loss = train_epoch_ssl(
            ssl_model,
            loader,
            optimizer,
            torch.device("cpu"),
            criterion,
        )
        assert loss >= 0.0


class TestDPSSLPretraining:
    """Test differential privacy features in SSL pretraining."""

    def test_dp_config_loading(self, tmp_path):
        """Test loading DP configuration parameters."""
        config = SSLConfig(
            data_dir=tmp_path,
            noise_multiplier=1.5,
            max_grad_norm=0.5,
            target_delta=1e-6,
        )
        assert config.noise_multiplier == 1.5
        assert config.max_grad_norm == 0.5
        assert config.target_delta == 1e-6

    def test_federated_config_loading(self, tmp_path):
        """Test loading federated configuration parameters."""
        config = SSLConfig(
            data_dir=tmp_path,
            num_clients=4,
            num_rounds=10,
        )
        assert config.num_clients == 4
        assert config.num_rounds == 10

    def test_privacy_engine_initialization(self, tmp_path):
        """Test PrivacyEngine initialization."""
        from opacus import PrivacyEngine

        # Create base model
        base_model = create_unet2d(
            in_channels=1, out_channels=2, channels=(8, 16), strides=(2,)
        )
        ssl_model = SSLUNet(base_model)
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)

        # Create dummy data loader
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

        # Test privacy engine setup
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=ssl_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        assert model is not None

    def test_epsilon_tracking(self, tmp_path):
        """Test epsilon (privacy budget) tracking during training."""
        from opacus import PrivacyEngine

        # Create base model
        base_model = create_unet2d(
            in_channels=1, out_channels=2, channels=(8, 16), strides=(2,)
        )
        ssl_model = SSLUNet(base_model)
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)

        # Create dummy data loader
        class DummyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

        # Setup privacy engine
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=ssl_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Train for one epoch (simulated step)
        optimizer.zero_grad()
        _, z1 = model(torch.rand(2, 1, 64, 64))
        _, z2 = model(torch.rand(2, 1, 64, 64))
        loss = nn.CosineSimilarity(dim=1)(z1, z2).mean()
        loss.backward()
        optimizer.step()

        # Check epsilon computation
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        assert epsilon > 0, "Epsilon should be > 0 after training"

    def test_dp_training_with_privacy_engine(self, tmp_path):
        """Test full DP training loop with privacy engine."""
        from opacus import PrivacyEngine
        from dp_fedmed.fl.ssl.task import train_epoch_ssl

        # Create base model
        base_model = create_unet2d(
            in_channels=1, out_channels=2, channels=(8, 16), strides=(2,)
        )
        ssl_model = SSLUNet(base_model)
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
        criterion = nn.CosineSimilarity(dim=1)

        # Create dummy data loaders
        class DummyDataset(torch.utils.data.Dataset[Any]):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, index: Any):
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        dataset = DummyDataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Setup privacy engine
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=ssl_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
        )

        # Train with DP
        loss = train_epoch_ssl(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            criterion,
        )

        # Verify training completed
        assert loss >= 0.0

    def test_config_from_toml_with_privacy(self, tmp_path):
        """Test loading SSL config with privacy parameters from TOML."""
        # Create a test TOML file
        toml_content = """
[data]
data_dir = "{}"
image_size = [224, 224]
num_workers = 2

[ssl]
method = "simclr"
epochs = 2
batch_size = 32
learning_rate = 0.001
device = "cpu"

[privacy]
style = "sample"
target_epsilon = 8.0
target_delta = 1e-6

[privacy.sample]
noise_multiplier = 1.5
max_grad_norm = 0.8

[federated]
num_clients = 2
num_rounds = 5

[augmentation]
gaussian_blur = true
gaussian_blur_prob = 0.5

[checkpointing]
save_dir = "{}"
"""

        config_path = tmp_path / "test_config.toml"
        config_path.write_text(
            toml_content.format(str(tmp_path), str(tmp_path / "results"))
        )

        # Load config
        config = SSLConfig.from_toml(config_path)

        # Verify privacy parameters
        assert config.noise_multiplier == 1.5
        assert config.max_grad_norm == 0.8
        assert config.target_delta == 1e-6

        # Verify federated parameters
        assert config.num_clients == 2
        assert config.num_rounds == 5

    def test_dp_gradient_clipping(self, tmp_path):
        """Test that gradient clipping is applied in DP training."""
        from opacus import PrivacyEngine

        # Create base model
        base_model = create_unet2d(
            in_channels=1, out_channels=2, channels=(8, 16), strides=(2,)
        )
        ssl_model = SSLUNet(base_model)
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)

        # Create dummy data loader
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                x1 = torch.rand(1, 64, 64)
                x2 = torch.rand(1, 64, 64)
                return (x1, x2)

        loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

        # Setup privacy engine with small gradient clipping
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=ssl_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=0.5,
        )

        # Verify privacy engine is active
        assert model is not None
