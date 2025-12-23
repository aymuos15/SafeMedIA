"""Factory function for creating Flower clients.

This module contains the client_fn function that creates configured
client instances for both supervised and SSL federated learning.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
import torch.utils.data
import flwr as fl
from flwr.common import Context
from monai.data.dataset import Dataset
from loguru import logger

from dp_fedmed.config import load_config
from dp_fedmed.logging import setup_logging
from dp_fedmed.data.cellpose import build_data_list, get_transforms
from dp_fedmed.fl.base.dataset import TupleDataset, UnlabeledImageDataset
from dp_fedmed.fl.client.dp_client import DPFlowerClient


class TrainingMode(Enum):
    """Training mode for federated learning."""

    SUPERVISED = "supervised"
    SSL = "ssl"


def create_client_fn(
    config_file: str = "configs/default.toml",
    mode: TrainingMode = TrainingMode.SUPERVISED,
) -> Callable[[Context], fl.client.Client]:
    """Create a client factory function with config file and mode baked in.

    This is needed because Flower's ClientApp requires a client_fn with
    signature `def client_fn(context: Context)`. We use a closure to
    capture the config file path and training mode.

    Args:
        config_file: Path to the config TOML file
        mode: Training mode (SUPERVISED or SSL)

    Returns:
        A client factory function that can be passed to ClientApp
    """

    def _client_fn(context: Context) -> fl.client.Client:
        return client_fn(context, config_file=config_file, mode=mode)

    return _client_fn


def client_fn(
    context: Context,
    config_file: str = "configs/default.toml",
    mode: TrainingMode = TrainingMode.SUPERVISED,
) -> fl.client.Client:
    """Create a Flower client instance.

    This function is called by Flower to create clients. It dispatches
    to the appropriate client type based on the training mode.

    Args:
        context: Flower context containing configuration
        config_file: Path to config file (passed via closure from create_client_fn)
        mode: Training mode (SUPERVISED or SSL)

    Returns:
        Configured Flower client
    """
    # Load TOML configuration
    cfg = context.run_config
    config_file = str(cfg.get("config-file", config_file))

    # Get training mode from run_config (allows runtime override)
    mode_str = str(cfg.get("training-mode", mode.value))
    try:
        mode = TrainingMode(mode_str)
    except ValueError:
        logger.warning(f"Unknown training mode '{mode_str}', defaulting to supervised")
        mode = TrainingMode.SUPERVISED

    try:
        config = load_config(config_file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        raise

    # Get partition info
    partition_id = int(context.node_config.get("partition-id", 0))  # type: ignore
    num_partitions = int(
        context.node_config.get(
            "num-partitions", config.get("federated.num_clients", 2)
        )  # type: ignore
    )

    # Setup logging with hierarchical structure
    log_level = str(config.get("logging.level", "INFO"))
    run_name = Path(config_file).stem
    role = f"client_{partition_id}"
    run_dir = setup_logging(run_name=run_name, level=log_level, role=role)

    logger.info("=" * 60)
    logger.info(
        f"Initializing {mode.value.upper()} Client {partition_id}/{num_partitions}"
    )
    logger.info("=" * 60)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dispatch to appropriate client creation function
    if mode == TrainingMode.SSL:
        return _create_ssl_client(
            context, config, partition_id, num_partitions, device, run_dir
        )
    else:
        return _create_supervised_client(
            context, config, partition_id, num_partitions, device, run_dir, config_file
        )


def _create_supervised_client(
    context: Context,
    config: Any,
    partition_id: int,
    num_partitions: int,
    device: torch.device,
    run_dir: Path,
    config_file: str,
) -> fl.client.Client:
    """Create a supervised learning client.

    Args:
        context: Flower context
        config: Loaded configuration
        partition_id: Client partition ID
        num_partitions: Total number of partitions
        device: Device to train on
        run_dir: Directory for logging
        config_file: Path to config file (for compatibility)

    Returns:
        Configured supervised FL client
    """
    # Get data settings from config
    data_dir = config.get("data.data_dir")
    batch_size = int(config.get("data.batch_size", 8))
    image_size = int(config.get("data.image_size", 256))
    num_workers = int(config.get("data.num_workers", 2))

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Batch size: {batch_size}, Image size: {image_size}x{image_size}")

    # Build data lists
    full_train_data = build_data_list(data_dir, "train")
    test_data = build_data_list(data_dir, "test")

    # Partition training data for this client
    total_samples = len(full_train_data)
    samples_per_client = total_samples // num_partitions
    start_idx = partition_id * samples_per_client
    end_idx = (
        start_idx + samples_per_client
        if partition_id < num_partitions - 1
        else total_samples
    )

    # Create subset for this client
    train_indices = list(range(start_idx, end_idx))
    client_train_data = [full_train_data[i] for i in train_indices]

    logger.info(f"Client {partition_id} data: {len(train_indices)} training samples")

    # Create datasets with transforms
    train_transforms = get_transforms((image_size, image_size), is_train=True)
    test_transforms = get_transforms((image_size, image_size), is_train=False)

    client_train_dataset = Dataset(data=client_train_data, transform=train_transforms)
    test_dataset = Dataset(data=test_data, transform=test_transforms)

    # Wrap training dataset to return tuples for Opacus compatibility
    client_train_dataset = TupleDataset(client_train_dataset)
    train_num_workers = 0  # Avoid Opacus multiprocessing issues

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        client_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Get config sections
    model_config = config.get_section("model")
    training_config = config.get_section("training")
    privacy_config = config.get_section("privacy")
    loss_config = config.get_section("loss")
    ssl_config = config.get_section("ssl")

    # Log privacy settings
    if privacy_config.get("enable_dp", True):
        logger.info(
            f"Privacy: ENABLED (epsilon target = {privacy_config.get('target_epsilon', 4.0)})"
        )
    else:
        logger.warning("Privacy: DISABLED")

    # Log loss settings
    loss_type = loss_config.get("type", "cross_entropy")
    logger.info(f"Loss function: {loss_type}")

    # Log SSL/transfer learning settings
    pretrained_checkpoint_path = ssl_config.get("pretrained_checkpoint_path", "")
    freeze_encoder = bool(ssl_config.get("freeze_encoder", False))
    if pretrained_checkpoint_path:
        logger.info(
            f"Transfer Learning: ENABLED (checkpoint: {pretrained_checkpoint_path}, "
            f"freeze_encoder: {freeze_encoder})"
        )
    else:
        logger.info("Transfer Learning: DISABLED (random initialization)")

    # Create and return client
    return DPFlowerClient(
        train_loader=train_loader,
        test_loader=test_loader,
        model_config=model_config,
        training_config=training_config,
        privacy_config=privacy_config,
        device=device,
        client_id=partition_id,
        run_dir=run_dir,
        loss_config=loss_config,
        pretrained_checkpoint_path=pretrained_checkpoint_path
        if pretrained_checkpoint_path
        else None,
        freeze_encoder=freeze_encoder,
    ).to_client()


def _create_ssl_client(
    context: Context,
    config: Any,
    partition_id: int,
    num_partitions: int,
    device: torch.device,
    run_dir: Path,
) -> fl.client.Client:
    """Create an SSL pretraining client.

    Args:
        context: Flower context
        config: Loaded configuration
        partition_id: Client partition ID
        num_partitions: Total number of partitions
        device: Device to train on
        run_dir: Directory for logging

    Returns:
        Configured SSL FL client
    """
    # Import SSL-specific modules
    from dp_fedmed.fl.ssl.client import SSLFlowerClient
    from dp_fedmed.fl.ssl.config import AugmentationConfig
    from dp_fedmed.fl.ssl.transforms import get_ssl_transform

    # Get data settings from config
    data_dir = config.get("data.data_dir")
    image_size_val = config.get("data.image_size", 224)
    if isinstance(image_size_val, (list, tuple)):
        image_size = tuple(image_size_val)
    else:
        image_size = (image_size_val, image_size_val)
    num_workers = int(config.get("data.num_workers", 4))
    batch_size = int(config.get("ssl.batch_size", 32))

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Image size: {image_size}, Batch size: {batch_size}")

    # Build data list of unlabeled images
    full_train_data = build_data_list(data_dir, "train")
    logger.info(f"Total unlabeled images: {len(full_train_data)}")

    # Support limiting samples for testing
    max_samples_per_client = int(config.get("data.max_samples_per_client", 0))

    # Partition training data for this client
    total_samples = len(full_train_data)
    samples_per_client = total_samples // num_partitions
    start_idx = partition_id * samples_per_client
    end_idx = (
        start_idx + samples_per_client
        if partition_id < num_partitions - 1
        else total_samples
    )

    # Create subset for this client
    train_indices = list(range(start_idx, end_idx))
    client_train_data = [full_train_data[i] for i in train_indices]

    # Optionally limit samples (for testing)
    if max_samples_per_client > 0:
        client_train_data = client_train_data[:max_samples_per_client]
        logger.info(
            f"Limited to {max_samples_per_client} samples per client (for testing)"
        )

    logger.info(f"Client {partition_id} assigned {len(client_train_data)} images")

    # Create transforms using SSL augmentation
    augmentation_dict = config.get_section("augmentation")
    if augmentation_dict:
        augmentation_config = AugmentationConfig(**augmentation_dict)
    else:
        augmentation_config = AugmentationConfig()

    ssl_method = config.get("ssl.method", "simclr")
    ssl_transform = get_ssl_transform(ssl_method, augmentation_config)

    # Create datasets
    # Enable flatten_augmented_views for Opacus DPDataLoader compatibility
    # This stacks (view1, view2) -> [2, C, H, W] tensor that Opacus can handle
    client_train_dataset = UnlabeledImageDataset(
        [item["image"] for item in client_train_data],
        transform=ssl_transform,
        flatten_augmented_views=True,
    )

    # Split into train/validation
    train_size = int(0.9 * len(client_train_dataset))
    val_size = len(client_train_dataset) - train_size

    if val_size > 0:
        from torch.utils.data import random_split

        train_dataset, val_dataset = random_split(
            client_train_dataset, [train_size, val_size]
        )
    else:
        train_dataset = client_train_dataset
        val_dataset = None

    logger.info(
        f"Client {partition_id}: Train={len(train_dataset)}, "
        f"Val={len(val_dataset) if val_dataset else 0}"
    )

    # Create data loaders
    # Note: Opacus DPDataLoader uses Poisson sampling, so drop_last is ignored
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid Opacus multiprocessing issues
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # Get config sections
    model_config = config.get_section("model")
    training_config = config.get_section("training") or {}
    privacy_config = config.get_section("privacy")
    ssl_config = config.get_section("ssl")

    # Log privacy settings
    privacy_style = privacy_config.get("style", "sample")
    logger.info(f"Privacy Style: {privacy_style}")
    if privacy_style in ["sample", "hybrid"]:
        sample_config = privacy_config.get("sample", {})
        logger.info(
            f"  Sample DP: ENABLED (clip={sample_config.get('max_grad_norm', 1.0)})"
        )
    if privacy_style in ["user", "hybrid"]:
        user_config = privacy_config.get("user", {})
        logger.info(
            f"  User DP: ENABLED (noise={user_config.get('noise_multiplier', 0.0)})"
        )

    # Log SSL settings
    logger.info(f"SSL Method: {ssl_config.get('method', 'simclr')}")
    logger.info(f"Projection Dim: {ssl_config.get('projection_dim', 128)}")

    # Create and return client
    ssl_client = SSLFlowerClient(
        train_loader=train_loader,
        val_loader=val_loader or train_loader,  # Use train as fallback if no val
        model_config=model_config,
        training_config=training_config,
        privacy_config=privacy_config,
        ssl_config=ssl_config,
        device=device,
        client_id=partition_id,
        run_dir=run_dir,
    )
    return ssl_client.to_client()
