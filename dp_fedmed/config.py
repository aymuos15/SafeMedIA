"""Configuration loading and validation for DP-FedMed."""

from pathlib import Path
from typing import Any, Dict, Union

import tomli

from loguru import logger


class Config:
    """Configuration manager for DP-FedMed with strict validation."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration with strict validation.

        Args:
            config_dict: Configuration dictionary from YAML file

        Raises:
            ValueError: If required fields are missing or invalid
        """
        self._config = config_dict
        self._validate()

    def _validate(self):
        """Validate all required fields strictly."""
        errors = []

        # Required: data.data_dir
        data_dir = self.get("data.data_dir")
        if data_dir is None:
            errors.append("data.data_dir is REQUIRED but not specified in config")
        elif not Path(data_dir).exists():
            errors.append(f"data.data_dir does not exist: {data_dir}")

        # Required: data settings with types
        if not isinstance(self.get("data.image_size"), int):
            errors.append("data.image_size must be an integer")
        if not isinstance(self.get("data.batch_size"), int):
            errors.append("data.batch_size must be an integer")

        # Required: model settings
        if self.get("model.in_channels") is None:
            errors.append("model.in_channels is required")
        if self.get("model.out_channels") is None:
            errors.append("model.out_channels is required")
        if not isinstance(self.get("model.channels"), list):
            errors.append("model.channels must be a list")
        if not isinstance(self.get("model.strides"), list):
            errors.append("model.strides must be a list")

        # Required: federated settings
        if not isinstance(self.get("federated.num_clients"), int):
            errors.append("federated.num_clients must be an integer")
        if not isinstance(self.get("federated.num_rounds"), int):
            errors.append("federated.num_rounds must be an integer")

        # Required: training settings
        if not isinstance(self.get("training.local_epochs"), int):
            errors.append("training.local_epochs must be an integer")
        if not isinstance(self.get("training.learning_rate"), (int, float)):
            errors.append("training.learning_rate must be a number")

        # Required: privacy settings
        enable_dp = self.get("privacy.enable_dp")
        if not isinstance(enable_dp, bool):
            errors.append("privacy.enable_dp must be a boolean (true/false)")

        if enable_dp:
            # If DP is enabled, validate DP parameters
            if not isinstance(self.get("privacy.noise_multiplier"), (int, float)):
                errors.append("privacy.noise_multiplier must be a number")
            if not isinstance(self.get("privacy.max_grad_norm"), (int, float)):
                errors.append("privacy.max_grad_norm must be a number")
            if not isinstance(self.get("privacy.target_epsilon"), (int, float)):
                errors.append("privacy.target_epsilon must be a number")
            if not isinstance(self.get("privacy.target_delta"), (int, float)):
                errors.append("privacy.target_delta must be a number")
            # Optional but validated if present
            client_dataset_size = self.get("privacy.client_dataset_size")
            if client_dataset_size is not None and not isinstance(
                client_dataset_size, int
            ):
                errors.append("privacy.client_dataset_size must be an integer")

        # If any errors, raise them all at once
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  ✗ {e}" for e in errors
            )
            raise ValueError(error_msg)

        logger.info("✓ Configuration validated successfully")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Clients: {self.get('federated.num_clients')}")
        logger.info(f"  Rounds: {self.get('federated.num_rounds')}")
        logger.info(f"  DP enabled: {self.get('privacy.enable_dp')}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation.

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
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self._config

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., "data", "model", "privacy")

        Returns:
            Dictionary with section contents
        """
        return self._config.get(section, {})


def load_config(config_path: Union[str, Path]) -> Config:
    """Load and validate TOML configuration file.

    Args:
        config_path: Path to TOML config file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
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

    return Config(config_dict)


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
