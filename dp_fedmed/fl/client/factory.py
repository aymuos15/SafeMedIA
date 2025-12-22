"""Factory function for creating Flower clients.

This module contains the client_fn function that creates configured
DPFlowerClient instances for federated learning.
"""

from pathlib import Path

from typing import Any
import torch.utils.data
import flwr as fl
from flwr.common import Context
from monai.data.dataset import Dataset
from loguru import logger

from ...config import load_config
from ...logging import setup_logging
from ...data.cellpose import build_data_list, get_transforms
from .dp_client import DPFlowerClient


class TupleDataset(torch.utils.data.Dataset):
    """Wrapper for MONAI Dataset to return (image, label) tuples.

    Opacus's DPDataLoader expects standard (data, target) tuples rather than
    the dictionaries returned by MONAI datasets.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data = self.dataset[index]
        return data["image"], data["label"]


def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client instance.

    This function is called by Flower to create clients. It loads configuration
    from the YAML file specified in run_config.

    Args:
        context: Flower context containing configuration

    Returns:
        Configured Flower client
    """
    # Load YAML configuration
    cfg = context.run_config
    config_file = str(cfg.get("config-file", "configs/default.toml"))

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
    run_name = Path(config_file).stem  # Extract "default" from "configs/default.toml"
    role = f"client_{partition_id}"
    run_dir = setup_logging(run_name=run_name, level=log_level, role=role)

    logger.info("=" * 60)
    logger.info(f"Initializing Client {partition_id}/{num_partitions}")
    logger.info("=" * 60)

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
    # Also set num_workers=0 for training to avoid Opacus multiprocessing issues
    client_train_dataset = TupleDataset(client_train_dataset)
    train_num_workers = 0

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders
    # Note: Do NOT use custom collate_fn - Opacus replaces it and expects default format
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

    # Log privacy settings
    if privacy_config.get("enable_dp", True):
        logger.info(
            f"Privacy: ENABLED (ε target = {privacy_config.get('target_epsilon', 4.0)})"
        )
    else:
        logger.warning("Privacy: DISABLED")

    # Log loss settings
    loss_type = loss_config.get("type", "cross_entropy")
    logger.info(f"Loss function: {loss_type}")

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
    ).to_client()
