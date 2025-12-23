"""Flower client with differential privacy for supervised learning.

This module defines the DPFlowerClient class that performs local training
with differential privacy using Opacus for supervised segmentation tasks.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.data
from loguru import logger

from dp_fedmed.models.unet2d import create_unet2d
from dp_fedmed.fl.base.client import BaseFlowerClient
from dp_fedmed.fl.task import train_one_epoch, evaluate
from dp_fedmed.fl.ssl.checkpoint import load_pretrained_encoder


class DPFlowerClient(BaseFlowerClient):
    """Flower client with differential privacy for supervised learning.

    This client extends BaseFlowerClient with supervised learning specifics:
    - Dice/cross-entropy loss for segmentation
    - Transfer learning from pretrained encoders
    - Encoder freezing options
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_config: Dict,
        training_config: Dict,
        privacy_config: Dict,
        device: torch.device,
        client_id: int = 0,
        run_dir: Optional[Path] = None,
        loss_config: Optional[Dict] = None,
        pretrained_checkpoint_path: Optional[Path | str] = None,
        freeze_encoder: bool = False,
    ):
        """Initialize the DP Flower client for supervised learning.

        Args:
            train_loader: Training data loader (must have drop_last=True for DP)
            test_loader: Test data loader
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            privacy_config: Differential privacy settings
            device: Device to train on
            client_id: Client partition ID
            run_dir: Directory to save client metrics
            loss_config: Loss function configuration
            pretrained_checkpoint_path: Path to pretrained model checkpoint
            freeze_encoder: Whether to freeze encoder parameters during training
        """
        self.loss_config = loss_config or {}
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.freeze_encoder = freeze_encoder

        # Call parent init (creates model via _create_model)
        super().__init__(
            train_loader=train_loader,
            val_loader=test_loader,
            model_config=model_config,
            training_config=training_config,
            privacy_config=privacy_config,
            device=device,
            client_id=client_id,
            run_dir=run_dir,
        )

        # Load pretrained encoder if provided
        if pretrained_checkpoint_path is not None:
            self._load_pretrained_encoder(pretrained_checkpoint_path, freeze_encoder)

    def _create_model(self, model_config: Dict) -> nn.Module:
        """Create a UNet model for supervised segmentation.

        Args:
            model_config: Model architecture configuration

        Returns:
            Configured UNet model
        """
        return create_unet2d(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 2),
            channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
            strides=tuple(model_config.get("strides", [2, 2, 2])),
            num_res_units=model_config.get("num_res_units", 2),
        )

    def _train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch with supervised loss.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss for the epoch
        """
        return train_one_epoch(
            self.model,
            train_loader,
            self.optimizer,
            self.device,
            loss_config=self.loss_config,
        )

    def _evaluate_model(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model with Dice metric.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with 'loss' and 'dice' metrics
        """
        return evaluate(
            self.model,
            val_loader,
            self.device,
        )

    def _get_primary_metric_name(self) -> str:
        """Get the primary evaluation metric name.

        Returns:
            'dice' for supervised segmentation
        """
        return "dice"

    def _load_pretrained_encoder(
        self, checkpoint_path: Path | str, freeze_encoder: bool = False
    ) -> None:
        """Load pretrained encoder weights into the model.

        Args:
            checkpoint_path: Path to the pretrained checkpoint
            freeze_encoder: Whether to freeze encoder parameters
        """
        try:
            checkpoint_path_obj = Path(checkpoint_path)
            metadata = load_pretrained_encoder(
                self.model, checkpoint_path_obj, strict=False
            )
            logger.info(
                f"Client {self.client_id} loaded pretrained encoder from {checkpoint_path}"
            )

            # Freeze encoder parameters if requested
            if freeze_encoder:
                if hasattr(self.model, "encoders"):
                    encoders = getattr(self.model, "encoders")
                    if isinstance(encoders, (list, torch.nn.ModuleList)):
                        for encoder in encoders:
                            if hasattr(encoder, "parameters"):
                                for param in encoder.parameters():
                                    param.requires_grad = False
                        logger.info(f"Client {self.client_id} froze encoder parameters")
                else:
                    logger.warning(
                        f"Client {self.client_id}: Cannot identify encoder to freeze"
                    )

            logger.info(
                f"Client {self.client_id} pretraining metadata: epoch={metadata.get('epoch')}, "
                f"metrics={metadata.get('metrics')}"
            )
        except Exception as e:
            logger.error(
                f"Client {self.client_id} failed to load pretrained encoder: {e}"
            )
            raise
