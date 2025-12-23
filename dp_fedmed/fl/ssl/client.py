"""Flower client for federated SSL pretraining with differential privacy.

This module defines the SSLFlowerClient class that performs distributed
self-supervised learning with differential privacy using Opacus.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.data
from lightly.loss import NTXentLoss

from dp_fedmed.models.unet2d import create_unet2d
from dp_fedmed.fl.base.client import BaseFlowerClient
from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.fl.ssl.task import train_epoch_ssl, validate_ssl


class SSLFlowerClient(BaseFlowerClient):
    """Flower client for federated SSL pretraining with differential privacy.

    This client extends BaseFlowerClient with SSL-specific functionality:
    - Contrastive loss (NTXentLoss)
    - SSLUNet model with projection head
    - Validation loss as primary metric
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model_config: Dict,
        training_config: Dict,
        privacy_config: Dict,
        ssl_config: Dict,
        device: torch.device,
        client_id: int = 0,
        run_dir: Optional[Path] = None,
    ):
        """Initialize the Federated SSL client.

        Args:
            train_loader: Training data loader (unlabeled)
            val_loader: Validation data loader (unlabeled)
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            privacy_config: Differential privacy settings
            ssl_config: SSL-specific configuration
            device: Device to train on
            client_id: Client partition ID
            run_dir: Directory to save client metrics
        """
        self.ssl_config = ssl_config

        # Call parent init (creates model via _create_model)
        super().__init__(
            train_loader=train_loader,
            val_loader=val_loader,
            model_config=model_config,
            training_config=training_config,
            privacy_config=privacy_config,
            device=device,
            client_id=client_id,
            run_dir=run_dir,
        )

        # SSL loss function
        self.criterion = NTXentLoss(temperature=ssl_config.get("temperature", 0.07))

    def _create_model(self, model_config: Dict) -> nn.Module:
        """Create an SSLUNet model with projection head.

        Args:
            model_config: Model architecture configuration

        Returns:
            Configured SSLUNet model
        """
        # Create base UNet
        base_model = create_unet2d(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 2),
            channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
            strides=tuple(model_config.get("strides", [2, 2, 2])),
            num_res_units=model_config.get("num_res_units", 2),
        )

        # Wrap with SSL projection head
        return SSLUNet(
            base_model,
            projection_dim=self.ssl_config.get("projection_dim", 128),
            hidden_dim=self.ssl_config.get("hidden_dim", 256),
        )

    def _train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch with SSL contrastive loss.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss for the epoch
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")

        return train_epoch_ssl(
            self.model,
            train_loader,
            self.optimizer,
            self.device,
            self.criterion,
        )

    def _evaluate_model(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model with contrastive loss.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with 'loss' and 'val_loss' metrics
        """
        return validate_ssl(
            self.model,
            val_loader,
            self.device,
            self.criterion,
        )

    def _get_primary_metric_name(self) -> str:
        """Get the primary evaluation metric name.

        Returns:
            'val_loss' for SSL pretraining
        """
        return "val_loss"
