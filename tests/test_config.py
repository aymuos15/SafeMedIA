"""Tests for Pydantic configuration system."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from dp_fedmed.config import (
    Config,
    DataConfig,
    ModelConfig,
    FederatedConfig,
    TrainingConfig,
    PrivacyConfig,
    LossConfig,
    CheckpointingConfig,
    load_config,
)


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_valid_data_config(self, tmp_path):
        """Test valid data configuration."""
        # Create a temp directory to use as data_dir
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = DataConfig(data_dir=data_dir)
        assert config.data_dir == data_dir
        assert config.image_size == 256  # default
        assert config.batch_size == 8  # default

    def test_invalid_data_dir_raises_error(self):
        """Test that non-existent data_dir raises error."""
        with pytest.raises(ValidationError):
            DataConfig(data_dir=Path("/nonexistent/path"))

    def test_batch_size_bounds(self, tmp_path):
        """Test batch_size validation bounds."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Valid batch size
        config = DataConfig(data_dir=data_dir, batch_size=32)
        assert config.batch_size == 32

        # Invalid batch size (too low)
        with pytest.raises(ValidationError):
            DataConfig(data_dir=data_dir, batch_size=0)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = ModelConfig()
        assert config.in_channels == 1
        assert config.out_channels == 2
        assert config.channels == [16, 32, 64, 128]
        assert config.strides == [2, 2, 2]

    def test_strides_length_validation(self):
        """Test that strides length must be channels length - 1."""
        with pytest.raises(ValidationError):
            ModelConfig(channels=[16, 32, 64], strides=[2, 2, 2])  # Wrong length

    def test_empty_channels_raises_error(self):
        """Test that empty channels list raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(channels=[])


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_valid_training_config(self):
        """Test valid training configuration."""
        config = TrainingConfig()
        assert config.local_epochs == 5
        assert config.learning_rate == 0.001
        assert config.optimizer == "sgd"

    def test_optimizer_normalization(self):
        """Test that optimizer name is normalized to lowercase."""
        config = TrainingConfig(optimizer="SGD")
        assert config.optimizer == "sgd"

    def test_invalid_optimizer_raises_error(self):
        """Test that invalid optimizer raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="rmsprop")


class TestLossConfig:
    """Tests for LossConfig validation."""

    def test_valid_loss_config(self):
        """Test valid loss configuration."""
        config = LossConfig()
        assert config.type == "cross_entropy"

    def test_loss_type_validation(self):
        """Test loss type validation."""
        # Valid types
        for loss_type in ["cross_entropy", "soft_dice", "dice_ce"]:
            config = LossConfig(type=loss_type)
            assert config.type == loss_type

        # Invalid type
        with pytest.raises(ValidationError):
            LossConfig(type="mse")

    def test_dice_weights(self):
        """Test dice weight configuration."""
        config = LossConfig(dice_weight=0.7, ce_weight=0.3)
        assert config.dice_weight == 0.7
        assert config.ce_weight == 0.3


class TestCheckpointingConfig:
    """Tests for CheckpointingConfig validation."""

    def test_valid_checkpointing_config(self):
        """Test valid checkpointing configuration."""
        config = CheckpointingConfig()
        assert config.enabled is True
        assert config.checkpoint_dir == "checkpoints"
        assert config.resume_from is None

    def test_resume_from_validation(self, tmp_path):
        """Test resume_from path validation."""
        # Empty string should be converted to None
        config = CheckpointingConfig(resume_from="")
        assert config.resume_from is None

        # Non-existent path should raise error
        with pytest.raises(ValidationError):
            CheckpointingConfig(resume_from="/nonexistent/checkpoint.pt")

        # Existing path should work
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.touch()
        config = CheckpointingConfig(resume_from=str(checkpoint))
        assert config.resume_from == str(checkpoint)


class TestConfigBackwardCompatibility:
    """Tests for backward compatibility of Config class."""

    def test_get_method(self, tmp_path):
        """Test config.get() method for dot notation access."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = Config(
            data=DataConfig(data_dir=data_dir),
            model=ModelConfig(),
            federated=FederatedConfig(),
            training=TrainingConfig(),
            privacy=PrivacyConfig(),
        )

        # Test nested access
        assert config.get("data.batch_size") == 8
        assert config.get("model.in_channels") == 1
        assert config.get("training.learning_rate") == 0.001

        # Test with default
        assert config.get("nonexistent.key", "default") == "default"

    def test_get_section_method(self, tmp_path):
        """Test config.get_section() method."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = Config(
            data=DataConfig(data_dir=data_dir),
            model=ModelConfig(),
            federated=FederatedConfig(),
            training=TrainingConfig(),
            privacy=PrivacyConfig(),
        )

        model_section = config.get_section("model")
        assert isinstance(model_section, dict)
        assert model_section["in_channels"] == 1
        assert model_section["out_channels"] == 2

    def test_to_dict_method(self, tmp_path):
        """Test config.to_dict() method."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = Config(
            data=DataConfig(data_dir=data_dir),
            model=ModelConfig(),
            federated=FederatedConfig(),
            training=TrainingConfig(),
            privacy=PrivacyConfig(),
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid TOML config file."""
        # Create a minimal valid config
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config_content = f"""
[data]
data_dir = "{data_dir}"
image_size = 128
batch_size = 4

[model]
in_channels = 1
out_channels = 2
channels = [16, 32]
strides = [2]

[federated]
num_clients = 2
num_rounds = 3

[training]
local_epochs = 2
learning_rate = 0.01

[privacy]
enable_dp = true
noise_multiplier = 1.0
max_grad_norm = 1.0
target_epsilon = 4.0
target_delta = 1e-5
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config.data.image_size == 128
        assert config.federated.num_rounds == 3
        assert config.privacy.target_epsilon == 4.0

    def test_load_nonexistent_config_raises_error(self):
        """Test that loading non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.toml")

    def test_optional_sections_have_defaults(self, tmp_path):
        """Test that optional sections (loss, checkpointing, logging) have defaults."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Minimal config without optional sections
        config_content = f"""
[data]
data_dir = "{data_dir}"

[model]
in_channels = 1
out_channels = 2
channels = [16, 32]
strides = [2]

[federated]
num_clients = 2
num_rounds = 3

[training]
local_epochs = 2
learning_rate = 0.01

[privacy]
enable_dp = true
"""
        config_file = tmp_path / "minimal_config.toml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        # Optional sections should have defaults
        assert config.loss.type == "cross_entropy"
        assert config.checkpointing.enabled is True
        assert config.logging.level == "INFO"
