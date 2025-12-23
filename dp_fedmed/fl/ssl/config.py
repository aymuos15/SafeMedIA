"""Configuration management for SSL pretraining.

This module defines the SSLConfig dataclass for managing all SSL training parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tomli


@dataclass
class AugmentationConfig:
    """Configuration for SSL augmentations."""

    input_size: Tuple[int, int] = (224, 224)
    gaussian_blur: bool = True
    gaussian_blur_prob: float = 0.5
    color_jitter_prob: float = 0.2
    color_jitter_strength: float = 0.3
    rotation_prob: float = 0.5
    rotation_degrees: int = 30
    flip_prob: float = 0.5
    crop_min_scale: float = 0.8
    crop_max_scale: float = 1.0
    normalize: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.5])
    normalize_std: List[float] = field(default_factory=lambda: [0.5])


@dataclass
class SSLConfig:
    """Configuration for self-supervised learning pretraining."""

    # Data configuration
    data_dir: Path
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4

    # SSL method and hyperparameters
    method: str = "simclr"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # SSL-specific parameters
    temperature: float = 0.07
    projection_dim: int = 128
    hidden_dim: int = 256

    # Checkpoint and logging
    save_dir: Path = Path("results/pretrained")
    save_interval: int = 10
    log_interval: int = 10

    # Device and computation
    device: str = "cuda"

    # Augmentation config
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Differential Privacy parameters
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_delta: float = 1e-5
    target_epsilon: Optional[float] = None
    privacy_style: str = "sample"

    # Federated pretraining parameters
    num_clients: int = 2
    num_rounds: int = 2
    fraction_fit: float = 1.0
    min_fit_clients: int = 2
    min_available_clients: int = 2

    # Resume from checkpoint
    resume_from: Optional[Path] = None
    start_epoch: int = 0

    @classmethod
    def from_toml(cls, config_path: Path) -> "SSLConfig":
        """Load configuration from TOML file.

        Args:
            config_path: Path to TOML configuration file

        Returns:
            SSLConfig instance
        """
        config_path = Path(config_path)

        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)

        # Extract sections
        data_config = config_dict.get("data", {})
        ssl_config = config_dict.get("ssl", {})
        aug_config = config_dict.get("augmentation", {})
        checkpoint_config = config_dict.get("checkpointing", {})
        privacy_config = config_dict.get("privacy", {})
        federated_config = config_dict.get("federated", {})

        # Parse data directory
        data_dir = Path(data_config.get("data_dir", "."))

        # Parse image size
        image_size_list = data_config.get("image_size", [224, 224])
        image_size = (
            tuple(image_size_list)
            if isinstance(image_size_list, list)
            else image_size_list
        )

        # Create augmentation config
        augmentation = AugmentationConfig(
            input_size=image_size,
            gaussian_blur=aug_config.get("gaussian_blur", True),
            gaussian_blur_prob=aug_config.get("gaussian_blur_prob", 0.5),
            color_jitter_prob=aug_config.get("color_jitter_prob", 0.2),
            color_jitter_strength=aug_config.get("color_jitter_strength", 0.3),
            rotation_prob=aug_config.get("rotation_prob", 0.5),
            rotation_degrees=aug_config.get("rotation_degrees", 30),
            flip_prob=aug_config.get("flip_prob", 0.5),
            crop_min_scale=aug_config.get("crop_min_scale", 0.8),
            crop_max_scale=aug_config.get("crop_max_scale", 1.0),
        )

        # Parse privacy config
        style = privacy_config.get("style", "sample")
        privacy_params = privacy_config.get(style, {})

        # Create main config
        return cls(
            data_dir=data_dir,
            image_size=image_size,
            num_workers=data_config.get("num_workers", 4),
            method=ssl_config.get("method", "simclr"),
            epochs=ssl_config.get("epochs", 100),
            batch_size=ssl_config.get("batch_size", 32),
            learning_rate=ssl_config.get("learning_rate", 0.001),
            weight_decay=ssl_config.get("weight_decay", 1e-4),
            momentum=ssl_config.get("momentum", 0.9),
            temperature=ssl_config.get("temperature", 0.07),
            projection_dim=ssl_config.get("projection_dim", 128),
            hidden_dim=ssl_config.get("hidden_dim", 256),
            save_dir=Path(checkpoint_config.get("save_dir", "results/pretrained")),
            save_interval=checkpoint_config.get("save_interval", 10),
            log_interval=checkpoint_config.get("log_interval", 10),
            device=ssl_config.get("device", "cuda"),
            augmentation=augmentation,
            noise_multiplier=privacy_params.get("noise_multiplier", 1.0),
            max_grad_norm=privacy_params.get("max_grad_norm", 1.0),
            target_delta=privacy_config.get("target_delta", 1e-5),
            target_epsilon=privacy_config.get("target_epsilon", None),
            privacy_style=style,
            num_clients=federated_config.get("num_clients", 2),
            num_rounds=federated_config.get("num_rounds", 2),
            fraction_fit=federated_config.get("fraction_fit", 1.0),
            min_fit_clients=federated_config.get("min_fit_clients", 2),
            min_available_clients=federated_config.get("min_available_clients", 2),
            resume_from=Path(checkpoint_config.get("resume_from"))
            if checkpoint_config.get("resume_from")
            else None,
            start_epoch=checkpoint_config.get("start_epoch", 0),
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        result = asdict(self)
        result["data_dir"] = str(result["data_dir"])
        result["save_dir"] = str(result["save_dir"])
        if result.get("resume_from"):
            result["resume_from"] = str(result["resume_from"])
        return result

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.method not in ["simclr", "moco", "simsiam"]:
            raise ValueError(
                f"method must be one of ['simclr', 'moco', 'simsiam'], got {self.method}"
            )

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

        if self.projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {self.projection_dim}"
            )
