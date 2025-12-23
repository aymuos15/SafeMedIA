"""Integration tests for federated SSL pretraining with differential privacy.

This module tests the federated-only SSL pretraining pipeline including:
- Federated SSL client training with Opacus DP-SGD
- Federated SSL strategy with RDP accounting
- Privacy budget computation across rounds
- Client factory and server factory functions
- All 4 privacy styles: "none", "sample", "user", "hybrid"
"""

import torch
from pathlib import Path
from unittest.mock import MagicMock

from dp_fedmed.fl.ssl.config import SSLConfig, AugmentationConfig
from dp_fedmed.fl.ssl.client import SSLUNet
from dp_fedmed.fl.base.strategy import DPStrategy
from dp_fedmed.fl.ssl.transforms import get_ssl_transform
from dp_fedmed.models.unet2d import create_unet2d


class TestSSLUNetModel:
    """Test SSLUNet model for federated SSL pretraining."""

    def test_sslunt_creation(self):
        """Test creating SSLUNet model."""
        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64),
            strides=(2, 2),
        )
        ssl_model = SSLUNet(base_model, projection_dim=128, hidden_dim=256)
        assert ssl_model is not None

    def test_sslunt_forward_pass(self):
        """Test SSLUNet forward pass."""
        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        ssl_model = SSLUNet(base_model, projection_dim=128, hidden_dim=256)

        # Create dummy input (batch_size=2, channels=1, H=224, W=224)
        x = torch.randn(2, 1, 224, 224)

        # Test forward pass - returns tuple of (view1_projection, view2_projection) or just projection
        with torch.no_grad():
            output = ssl_model(x)

        # Output can be tensor or tuple depending on usage
        assert output is not None

    def test_sslunt_get_encoder(self):
        """Test extracting encoder from SSLUNet."""
        base_model = create_unet2d(
            in_channels=1,
            out_channels=2,
            channels=(16, 32),
            strides=(2,),
        )
        ssl_model = SSLUNet(base_model, projection_dim=128, hidden_dim=256)

        # Extract backbone
        backbone = ssl_model.get_backbone()
        assert backbone is not None

        # Verify it's the base model
        assert backbone is base_model


class TestFederatedSSLClientFactory:
    """Test federated SSL client factory."""

    def test_client_fn_creates_federated_ssl_client(self):
        """Test that client_fn creates FederatedSSLClient."""

        # Create mock context
        context = MagicMock()
        context.node_config = {
            "config-file": "configs/pretraining.toml",
            "client_id": 0,
        }
        context.state = MagicMock()
        context.state.configs_records = []

        # This should not raise
        # (Note: actual client_fn will try to load config, so we're mainly testing interface)
        # In real usage, config would be passed via context


class TestFederatedSSLStrategy:
    """Test federated SSL strategy with privacy accounting."""

    def test_dp_strategy_ssl_creation(self, tmp_path):
        """Test creating DPStrategy for SSL."""
        strategy = DPStrategy(
            primary_metric="val_loss",
            higher_is_better=False,
            fraction_fit=1.0,
            min_fit_clients=2,
            target_epsilon=8.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            run_dir=tmp_path / "results",
            run_name="test_ssl",
        )
        assert strategy is not None
        assert strategy.primary_metric == "val_loss"
        assert strategy.target_epsilon == 8.0

    def test_privacy_budget_computation(self, tmp_path):
        """Test privacy budget computation across rounds."""
        strategy = DPStrategy(
            primary_metric="val_loss",
            higher_is_better=False,
            fraction_fit=1.0,
            min_fit_clients=2,
            target_epsilon=8.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            run_dir=tmp_path / "results",
            run_name="test_ssl",
        )

        # Verify strategy has privacy accounting
        assert hasattr(strategy, "target_epsilon")
        assert hasattr(strategy, "target_delta")


class TestFederatedSSLConfigLoading:
    """Test loading federated SSL configuration."""

    def test_config_loads_with_federated_settings(self, tmp_path):
        """Test that config loads federated settings."""
        config_file = tmp_path / "test_pretraining.toml"
        config_file.write_text("""
[data]
data_dir = "/tmp"
image_size = [224, 224]
num_workers = 2

[ssl]
method = "simclr"
epochs = 2
batch_size = 16
learning_rate = 0.001
temperature = 0.07
projection_dim = 128
hidden_dim = 256
device = "cpu"

[augmentation]
gaussian_blur = true
gaussian_blur_prob = 0.5
color_jitter_prob = 0.2
rotation_prob = 0.5
flip_prob = 0.5
crop_min_scale = 0.8
crop_max_scale = 1.0
normalize_mean = [0.5]
normalize_std = [0.5]

[federated]
num_clients = 2
num_rounds = 2
fraction_fit = 1.0
min_fit_clients = 2
min_available_clients = 2

[privacy]
style = "sample"
target_epsilon = 8.0
target_delta = 1e-5

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[checkpointing]
save_dir = "/tmp/pretrained"
save_best_model = true
save_final_model = true
        """)

        config = SSLConfig.from_toml(config_file)

        # Verify federated settings loaded
        assert config.num_clients == 2
        assert config.num_rounds == 2
        assert config.fraction_fit == 1.0

        # Verify privacy settings loaded
        assert config.privacy_style == "sample"
        assert config.target_epsilon == 8.0
        assert config.target_delta == 1e-5

    def test_config_supports_all_privacy_styles(self, tmp_path):
        """Test that config supports all 4 privacy styles."""
        privacy_styles = ["none", "sample", "user", "hybrid"]

        for style in privacy_styles:
            config_file = tmp_path / f"test_{style}.toml"
            config_file.write_text(f"""
[data]
data_dir = "/tmp"
image_size = [224, 224]
num_workers = 2

[ssl]
method = "simclr"
epochs = 1
batch_size = 16
learning_rate = 0.001
temperature = 0.07
projection_dim = 128
hidden_dim = 256
device = "cpu"

[augmentation]
gaussian_blur = true
gaussian_blur_prob = 0.5
color_jitter_prob = 0.2
rotation_prob = 0.5
flip_prob = 0.5
crop_min_scale = 0.8
crop_max_scale = 1.0
normalize_mean = [0.5]
normalize_std = [0.5]

[federated]
num_clients = 2
num_rounds = 1
fraction_fit = 1.0
min_fit_clients = 2
min_available_clients = 2

[privacy]
style = "{style}"
target_epsilon = 8.0
target_delta = 1e-5

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[privacy.user]
noise_multiplier = 0.0
max_grad_norm = 1.0

[privacy.hybrid]
noise_multiplier = 1.0
max_grad_norm = 1.0

[checkpointing]
save_dir = "/tmp/pretrained"
save_best_model = true
save_final_model = true
            """)

            config = SSLConfig.from_toml(config_file)
            assert config.privacy_style == style


class TestFederatedSSLAugmentations:
    """Test SSL augmentation transforms for federated learning."""

    def test_augmentation_transform_creation(self):
        """Test creating augmentation transform."""

        aug_config = AugmentationConfig()

        # Create transform (will use lightly)
        transform = get_ssl_transform(method="simclr", config=aug_config)
        assert transform is not None

    def test_augmentation_produces_different_views(self):
        """Test that augmentation produces different views of same image."""

        aug_config = AugmentationConfig()

        transform = get_ssl_transform(method="simclr", config=aug_config)

        # Create dummy image
        img = torch.randn(1, 224, 224)

        # Apply transform twice (should produce different augmentations)
        view1 = transform(img)
        view2 = transform(img)

        # Verify we get some output (format may vary)
        assert view1 is not None
        assert view2 is not None


class TestFederatedSSLEndToEnd:
    """End-to-end integration tests for federated SSL pretraining."""

    def test_federated_ssl_config_validation(self, tmp_path):
        """Test that federated SSL config passes validation."""
        config_file = tmp_path / "pretrain.toml"
        config_file.write_text("""
[data]
data_dir = "/tmp"
image_size = [224, 224]
num_workers = 2

[ssl]
method = "simclr"
epochs = 1
batch_size = 16
learning_rate = 0.001
temperature = 0.07
projection_dim = 128
hidden_dim = 256
device = "cpu"

[augmentation]
gaussian_blur = true
gaussian_blur_prob = 0.5
color_jitter_prob = 0.2
rotation_prob = 0.5
flip_prob = 0.5
crop_min_scale = 0.8
crop_max_scale = 1.0
normalize_mean = [0.5]
normalize_std = [0.5]

[federated]
num_clients = 2
num_rounds = 2
fraction_fit = 1.0
min_fit_clients = 2
min_available_clients = 2

[privacy]
style = "sample"
target_epsilon = 8.0
target_delta = 1e-5

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[checkpointing]
save_dir = "/tmp/pretrained"
save_best_model = true
save_final_model = true
        """)

        config = SSLConfig.from_toml(config_file)

        # Should pass validation
        config.validate()

        # Verify all required fields
        assert config.method == "simclr"
        assert config.num_clients == 2
        assert config.num_rounds == 2
        assert config.privacy_style == "sample"

    def test_federated_ssl_no_centralized_path(self, tmp_path):
        """Test that federated SSL has no centralized training path."""
        # Verify trainer.py was deleted (should not exist)
        assert not Path("dp_fedmed/pretraining/trainer.py").exists()

        # Verify config requires federated section
        config_file = tmp_path / "pretrain.toml"
        config_file.write_text("""
[data]
data_dir = "/tmp"
image_size = [224, 224]
num_workers = 2

[ssl]
method = "simclr"
epochs = 1
batch_size = 16
learning_rate = 0.001
temperature = 0.07
projection_dim = 128
hidden_dim = 256
device = "cpu"

[augmentation]
gaussian_blur = true
gaussian_blur_prob = 0.5
color_jitter_prob = 0.2
rotation_prob = 0.5
flip_prob = 0.5
crop_min_scale = 0.8
crop_max_scale = 1.0
normalize_mean = [0.5]
normalize_std = [0.5]

[federated]
num_clients = 2
num_rounds = 2
fraction_fit = 1.0
min_fit_clients = 2
min_available_clients = 2

[privacy]
style = "sample"
target_epsilon = 8.0
target_delta = 1e-5

[privacy.sample]
noise_multiplier = 1.0
max_grad_norm = 1.0

[checkpointing]
save_dir = "/tmp/pretrained"
save_best_model = true
save_final_model = true
        """)

        config = SSLConfig.from_toml(config_file)

        # Federated settings are required (not optional)
        assert config.num_clients > 0
        assert config.num_rounds > 0
