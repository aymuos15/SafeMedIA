"""Abstract base class for Flower clients with differential privacy.

This module defines the BaseFlowerClient class that provides shared
functionality for both supervised and SSL federated learning clients.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule
from loguru import logger

from dp_fedmed.models.unet2d import get_parameters, set_parameters
from dp_fedmed.fl.checkpoint import load_unified_checkpoint, ClientState
from dp_fedmed.utils import (
    get_dataset_size,
    load_client_history,
    save_client_metrics,
)


class BaseFlowerClient(NumPyClient, ABC):
    """Abstract base class for Flower clients with differential privacy.

    This class provides shared functionality for parameter management,
    differential privacy setup, and metric tracking. Subclasses implement
    the model creation and training/evaluation logic.
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model_config: Dict,
        training_config: Dict,
        privacy_config: Dict,
        device: torch.device,
        client_id: int = 0,
        run_dir: Optional[Path] = None,
    ):
        """Initialize the base Flower client.

        Args:
            train_loader: Training data loader
            val_loader: Validation/test data loader
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            privacy_config: Differential privacy settings
            device: Device to train on
            client_id: Client partition ID
            run_dir: Directory to save client metrics
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.training_config = training_config
        self.privacy_config = privacy_config
        self.device = device
        self.client_id = client_id
        self.run_dir = run_dir

        # Track metrics across rounds
        self.round_history: list = []
        self._load_history()
        self.start_time: str = datetime.now().isoformat()

        # Create model (delegated to subclass)
        self.model: nn.Module | GradSampleModule = self._create_model(model_config)
        self.model.to(self.device)

        # Will be set up each round
        self.optimizer: Optional[optim.Optimizer] = None
        self.privacy_engine: Optional[PrivacyEngine] = None
        self._dp_train_loader: Optional[torch.utils.data.DataLoader] = None

        logger.info(
            f"Client {self.client_id} initialized: "
            f"{sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

    @abstractmethod
    def _create_model(self, model_config: Dict) -> nn.Module:
        """Create the model for this training mode.

        Args:
            model_config: Model architecture configuration

        Returns:
            Configured PyTorch model
        """
        pass

    @abstractmethod
    def _train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss for the epoch
        """
        pass

    @abstractmethod
    def _evaluate_model(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def _get_primary_metric_name(self) -> str:
        """Get the name of the primary evaluation metric.

        Returns:
            Metric name (e.g., 'dice' for supervised, 'val_loss' for SSL)
        """
        pass

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters."""
        return get_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters."""
        set_parameters(self.model, parameters)

    def _load_resume_state(self, config: Dict[str, Scalar]) -> Optional[ClientState]:
        """Load client state from checkpoint for mid-round resume.

        Args:
            config: Training configuration with checkpoint info

        Returns:
            ClientState if resuming, None otherwise
        """
        resume_from_checkpoint = bool(config.get("resume_from_checkpoint", False))
        if not resume_from_checkpoint:
            return None

        checkpoint_path = config.get("checkpoint_path")
        if not checkpoint_path:
            logger.warning("Resume signaled but no checkpoint path provided")
            return None

        try:
            checkpoint = load_unified_checkpoint(Path(str(checkpoint_path)))
            client_state = checkpoint.clients.get(self.client_id)
            if client_state is None:
                logger.warning(f"No state for client {self.client_id} in checkpoint")
                return None
            return client_state
        except Exception as e:
            logger.error(f"Failed to load resume state: {e}")
            return None

    def _setup_optimizer(self, lr: float, momentum: float) -> optim.Optimizer:
        """Set up the optimizer.

        Args:
            lr: Learning rate
            momentum: Momentum

        Returns:
            Configured optimizer
        """
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
        )

    def _setup_dp_if_enabled(
        self,
        config: Dict[str, Scalar],
    ) -> Tuple[bool, torch.utils.data.DataLoader, float, int]:
        """Set up differential privacy if enabled.

        Args:
            config: Training configuration from server

        Returns:
            Tuple of (dp_enabled, train_loader, sample_rate, steps)
        """
        privacy_style = self.privacy_config.get("style", "sample")
        enable_sample_dp = privacy_style in ["sample", "hybrid"]

        sample_config = self.privacy_config.get("sample", {})
        max_grad_norm = float(sample_config.get("max_grad_norm", 1.0))
        noise_multiplier = float(config.get("noise_multiplier", 1.0))

        local_epochs = int(
            config.get("local_epochs", self.training_config.get("local_epochs", 1))
        )

        if not enable_sample_dp:
            return (
                False,
                self.train_loader,
                0.0,
                len(self.train_loader) * local_epochs,
            )

        # Set up privacy engine
        privacy_engine = PrivacyEngine()
        self.privacy_engine = privacy_engine

        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized")

        try:
            model, optimizer, self._dp_train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            self.model = model
            self.optimizer = optimizer
        except Exception as e:
            logger.error(f"Failed to make model private: {e}")
            raise

        train_loader = self._dp_train_loader
        sample_rate = getattr(self._dp_train_loader, "sample_rate", 0.0)
        if train_loader is None:
            raise ValueError("DP training loader is None after make_private")

        steps = len(train_loader) * local_epochs

        logger.info(
            f"Sample-level DP enabled: noise={noise_multiplier}, clip={max_grad_norm}"
        )

        return True, train_loader, sample_rate, steps

    def _unwrap_model_if_needed(self) -> None:
        """Unwrap GradSampleModule if model was wrapped."""
        if isinstance(self.model, GradSampleModule):
            logger.debug("Unwrapping GradSampleModule for next round")
            self.model = self.model.to_standard_module()

        self.model.zero_grad(set_to_none=True)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data.

        Args:
            parameters: Global model parameters
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Check for mid-round resume
        resume_state = self._load_resume_state(config)
        start_epoch = 0
        accumulated_loss = 0.0

        if resume_state is not None:
            # Check if client already completed this round
            if (
                resume_state.epoch.status == "completed"
                and resume_state.final_parameters is not None
            ):
                logger.info(
                    f"Client {self.client_id} already completed round, "
                    "returning cached results"
                )
                metrics: Dict[str, Scalar] = dict(resume_state.final_metrics or {})
                return (
                    resume_state.final_parameters,
                    resume_state.num_samples,
                    metrics,
                )

            # Resume from saved epoch
            start_epoch = resume_state.epoch.current
            accumulated_loss = resume_state.partial_metrics.get("loss_sum", 0.0)

            # Load saved model state
            if resume_state.model_state is not None:
                try:
                    self._unwrap_model_if_needed()
                    self.model.load_state_dict(resume_state.model_state)
                    logger.info(
                        f"Resumed client {self.client_id} from epoch {start_epoch}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load model state, starting fresh: {e}")
                    start_epoch = 0
                    accumulated_loss = 0.0
                    self.set_parameters(parameters)
            else:
                self.set_parameters(parameters)
        else:
            self.set_parameters(parameters)

        # Get training parameters
        local_epochs = int(
            config.get("local_epochs", self.training_config.get("local_epochs", 1))
        )
        lr = float(self.training_config.get("learning_rate", 0.001))
        momentum = float(self.training_config.get("momentum", 0.9))

        # Privacy setup
        privacy_style = self.privacy_config.get("style", "sample")
        target_delta = float(self.privacy_config.get("target_delta", 1e-5))
        noise_multiplier = float(config.get("noise_multiplier", 1.0))

        logger.info(
            f"Round {config.get('server_round', '?')}: Style={privacy_style}, "
            f"noise_multiplier={noise_multiplier:.4f}, epochs={start_epoch}/{local_epochs}"
        )

        # Unwrap model if already wrapped from previous round
        self._unwrap_model_if_needed()

        # Set up optimizer (must be created AFTER unwrapping)
        self.optimizer = self._setup_optimizer(lr, momentum)

        # Set up DP if enabled
        dp_enabled, train_loader, sample_rate, steps = self._setup_dp_if_enabled(config)

        # Adjust steps for resume
        if start_epoch > 0:
            remaining_epochs = local_epochs - start_epoch
            steps = len(train_loader) * remaining_epochs

        # Training loop
        total_loss = accumulated_loss
        for epoch in range(start_epoch, local_epochs):
            epoch_loss = self._train_one_epoch(train_loader)
            total_loss += epoch_loss
            logger.debug(
                f"Local epoch {epoch + 1}/{local_epochs}, Loss: {epoch_loss:.4f}"
            )

        # Get privacy spent
        epsilon = 0.0
        if dp_enabled and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=target_delta)
            logger.info(
                f"Training complete. Privacy spent: epsilon = {epsilon:.4f}, "
                f"delta = {target_delta}"
            )
        else:
            completed_epochs = local_epochs - start_epoch
            avg_loss = (total_loss - accumulated_loss) / max(completed_epochs, 1)
            logger.info(f"Training complete. Average loss: {avg_loss:.4f}")

        # Store round metrics
        num_train_samples = get_dataset_size(self.train_loader)

        round_metrics = {
            "round": int(config.get("server_round", 0)),
            "train_loss": float(total_loss / local_epochs),
            "epsilon": float(epsilon),
            "delta": float(target_delta if dp_enabled else 0.0),
            "num_samples": num_train_samples,
        }

        self.round_history.append(round_metrics)
        self._save_client_metrics()

        # Return updated parameters and metrics
        num_fit_samples = get_dataset_size(self.train_loader)

        return (
            get_parameters(self.model),
            num_fit_samples,
            {
                "loss": float(total_loss / local_epochs),
                "epsilon": float(epsilon),
                "delta": float(target_delta if dp_enabled else 0.0),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)

        metrics = self._evaluate_model(self.val_loader)

        primary_metric = self._get_primary_metric_name()
        logger.debug(
            f"Evaluation: {primary_metric} = {metrics.get(primary_metric, 0.0):.4f}, "
            f"Loss = {metrics.get('loss', 0.0):.4f}"
        )

        # Update last round with eval metrics
        server_round = int(config.get("server_round", 0))
        self._load_history()

        found = False
        for entry in reversed(self.round_history):
            if entry["round"] == server_round:
                for key, value in metrics.items():
                    entry[f"eval_{key}"] = float(value)
                found = True
                break

        if not found:
            placeholder: Dict[str, Any] = {
                "round": server_round,
                "train_loss": 0.0,
                "epsilon": 0.0,
                "delta": 0.0,
                "num_samples": 0,
            }
            for key, value in metrics.items():
                placeholder[f"eval_{key}"] = float(value)
            self.round_history.append(placeholder)

        self._save_client_metrics()

        num_val_samples = get_dataset_size(self.val_loader)

        # Return loss and the primary metric
        return_metrics = {primary_metric: metrics.get(primary_metric, 0.0)}

        return (
            metrics.get("loss", 0.0),
            num_val_samples,
            return_metrics,
        )

    def _load_history(self) -> None:
        """Load client history from disk."""
        self.round_history = load_client_history(self.run_dir)

    def _save_client_metrics(self) -> None:
        """Save client-specific metrics and history."""
        primary_metric = self._get_primary_metric_name()
        final_metric_value = 0.0

        if self.round_history:
            last_round = self.round_history[-1]
            final_metric_value = last_round.get(f"eval_{primary_metric}", 0.0)

        save_client_metrics(
            run_dir=self.run_dir,
            client_id=self.client_id,
            start_time=self.start_time,
            round_history=self.round_history,
            train_loader=self.train_loader,
            privacy_config=self.privacy_config,
            extra_metrics={f"final_{primary_metric}": final_metric_value},
        )
