"""Configuration loading and validation for DP-FedMed using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tomli
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data configuration."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    image_size: int = Field(default=256, ge=16, le=2048)
    batch_size: int = Field(default=8, ge=1, le=512)
    num_workers: int = Field(default=2, ge=0, le=32)

    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, v: Any) -> Path:
        path = Path(v)
        if not path.exists():
            raise ValueError(f"data_dir does not exist: {v}")
        return path


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_config = ConfigDict(extra="forbid")

    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=2, ge=1)
    channels: List[int] = Field(default=[16, 32, 64, 128])
    strides: List[int] = Field(default=[2, 2, 2])
    num_res_units: int = Field(default=2, ge=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("channels must not be empty")
        if any(c < 1 for c in v):
            raise ValueError("all channel values must be positive")
        return v

    @model_validator(mode="after")
    def validate_strides_length(self) -> "ModelConfig":
        if len(self.strides) != len(self.channels) - 1:
            raise ValueError(
                f"strides length ({len(self.strides)}) must be "
                f"channels length - 1 ({len(self.channels) - 1})"
            )
        return self


class FederatedConfig(BaseModel):
    """Federated learning configuration."""

    model_config = ConfigDict(extra="forbid")

    num_clients: int = Field(default=2, ge=1)
    num_rounds: int = Field(default=5, ge=1)
    fraction_fit: float = Field(default=1.0, ge=0.0, le=1.0)
    fraction_evaluate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_fit_clients: int = Field(default=2, ge=1)
    min_evaluate_clients: int = Field(default=2, ge=1)
    min_available_clients: int = Field(default=2, ge=1)


class ClientResourcesConfig(BaseModel):
    """Client resource configuration."""

    model_config = ConfigDict(extra="forbid")

    num_cpus: int = Field(default=1, ge=1)
    num_gpus: float = Field(default=0.3, ge=0.0, le=8.0)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    local_epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    optimizer: str = Field(default="sgd")
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        allowed = ["sgd", "adam", "adamw"]
        if v.lower() not in allowed:
            raise ValueError(f"optimizer must be one of {allowed}")
        return v.lower()


class PrivacyConfig(BaseModel):
    """Differential privacy configuration."""

    model_config = ConfigDict(extra="forbid")

    enable_dp: bool = True
    noise_multiplier: float = Field(default=1.0, gt=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    target_epsilon: float = Field(default=8.0, gt=0.0)
    target_delta: float = Field(default=1e-5, gt=0.0, lt=1.0)
    client_dataset_size: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_dp_params(self) -> "PrivacyConfig":
        if self.enable_dp:
            if self.noise_multiplier <= 0:
                raise ValueError("noise_multiplier must be positive when DP is enabled")
            if self.max_grad_norm <= 0:
                raise ValueError("max_grad_norm must be positive when DP is enabled")
        return self


class LossConfig(BaseModel):
    """Loss function configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="cross_entropy")
    dice_smooth: float = Field(default=1.0, ge=0.0)
    dice_include_background: bool = False
    dice_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    ce_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def validate_loss_type(cls, v: str) -> str:
        allowed = ["cross_entropy", "soft_dice", "dice_ce"]
        if v.lower() not in allowed:
            raise ValueError(f"loss type must be one of {allowed}")
        return v.lower()


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = Field(default=None)

    @field_validator("resume_from", mode="before")
    @classmethod
    def validate_resume_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip():
            path = Path(v)
            if not path.exists():
                raise ValueError(f"resume_from path does not exist: {v}")
            return v
        return None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    save_model: bool = True
    save_metrics: bool = True
    level: str = Field(default="INFO")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log level must be one of {allowed}")
        return v.upper()


class Config(BaseModel):
    """Root configuration model for DP-FedMed."""

    model_config = ConfigDict(extra="allow")

    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    client_resources: ClientResourcesConfig = Field(
        default_factory=ClientResourcesConfig
    )
    training: TrainingConfig
    privacy: PrivacyConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation (backward compatibility).

        Args:
            key: Dot-separated key path (e.g., "data.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("data.batch_size", 8)
            8
        """
        keys = key.split(".")
        value: Any = self

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self.model_dump()

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section as dict.

        Args:
            section: Section name (e.g., "data", "model", "privacy")

        Returns:
            Dictionary with section contents
        """
        if hasattr(self, section):
            obj = getattr(self, section)
            if isinstance(obj, BaseModel):
                return obj.model_dump()
        return {}


def load_config(config_path: Union[str, Path]) -> Config:
    """Load and validate TOML configuration file.

    Args:
        config_path: Path to TOML config file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        pydantic.ValidationError: If config validation fails
        tomli.TOMLDecodeError: If TOML parsing fails
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs in configs/: default.toml"
        )

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(path, "rb") as f:
            config_dict = tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Failed to parse TOML config: {e}")

    if config_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Create Pydantic model (validation happens automatically)
    config = Config(**config_dict)

    logger.info("Configuration validated successfully")
    logger.info(f"  Data directory: {config.data.data_dir}")
    logger.info(f"  Clients: {config.federated.num_clients}")
    logger.info(f"  Rounds: {config.federated.num_rounds}")
    logger.info(f"  DP enabled: {config.privacy.enable_dp}")

    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two configuration dictionaries.

    Override values take precedence over base values.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
