"""Flower client with differential privacy using Opacus.

This module defines the DPFlowerClient class that performs local training
with differential privacy using Opacus.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.utils.data
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from opacus import PrivacyEngine
from loguru import logger


from ...models.unet2d import create_unet2d, get_parameters, set_parameters
from ..task import train_one_epoch, evaluate


class DPFlowerClient(NumPyClient):
    """Flower client with differential privacy using Opacus."""

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_config: Dict,
        training_config: Dict,
        privacy_config: Dict,
        device: torch.device,
        client_id: int = 0,
        run_dir: Path | None = None,
    ):
        """Initialize the DP Flower client.

        Args:
            train_loader: Training data loader (must have drop_last=True for DP)
            test_loader: Test data loader
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            privacy_config: Differential privacy settings
            device: Device to train on
            client_id: Client partition ID
            run_dir: Directory to save client metrics
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_config = training_config
        self.privacy_config = privacy_config
        self.device = device
        self.client_id = client_id
        self.run_dir = run_dir

        # Track metrics across rounds
        self.round_history = []
        self._load_history()
        self.start_time = datetime.now().isoformat()

        # Create model from config
        self.model = create_unet2d(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 2),
            channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
            strides=tuple(model_config.get("strides", [2, 2, 2])),
            num_res_units=model_config.get("num_res_units", 2),
        )
        self.model.to(self.device)

        # Will be set up each round
        self.optimizer = None
        self.privacy_engine = None
        self._dp_train_loader = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters."""
        return get_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters."""
        set_parameters(self.model, parameters)

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
        # Set global parameters
        self.set_parameters(parameters)

        # Get training parameters
        local_epochs = int(self.training_config.get("local_epochs", 1))
        lr = float(self.training_config.get("learning_rate", 0.001))
        momentum = float(self.training_config.get("momentum", 0.9))

        # Get privacy parameters
        enable_dp = bool(self.privacy_config.get("enable_dp", True))
        max_grad_norm = float(self.privacy_config.get("max_grad_norm", 1.0))
        target_delta = float(self.privacy_config.get("target_delta", 1e-5))

        # USE SERVER-PROVIDED NOISE MULTIPLIER (pre-computed)
        noise_multiplier = float(config.get("noise_multiplier", 1.0))

        logger.info(
            f"Round {config.get('server_round', '?')}: Using noise_multiplier = {noise_multiplier:.4f}"
        )

        # Validate privacy parameters
        if enable_dp:
            if max_grad_norm <= 0:
                logger.warning(
                    f"Invalid max_grad_norm {max_grad_norm}, using default 1.0"
                )
                max_grad_norm = 1.0
            if noise_multiplier <= 0:
                logger.warning(
                    f"Invalid noise_multiplier {noise_multiplier}, using default 1.0"
                )
                noise_multiplier = 1.0
            if target_delta <= 0 or target_delta >= 1:
                logger.warning(
                    f"Invalid target_delta {target_delta}, using default 1e-5"
                )
                target_delta = 1e-5

        # Set up optimizer
        self.optimizer = torch.optim.SGD(  # type: ignore
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
        )

        epsilon = 0.0  # Will be updated if DP is enabled

        if enable_dp:
            # Set up privacy engine
            self.privacy_engine = PrivacyEngine()

            # Wrap model, optimizer, and data loader with DP
            self.model, self.optimizer, self._dp_train_loader = (
                self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )
            )

            train_loader = self._dp_train_loader
            logger.info(f"DP enabled: noise={noise_multiplier}, clip={max_grad_norm}")
        else:
            train_loader = self.train_loader
            logger.info("DP disabled - training without privacy")

        # Set up checkpoint directory
        checkpoint_dir = None
        if self.run_dir:
            checkpoint_dir = self.run_dir / "checkpoints"

        # Training loop
        total_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.device,
                checkpoint_dir=checkpoint_dir,
            )
            total_loss += epoch_loss
            logger.debug(
                f"Local epoch {epoch + 1}/{local_epochs}, Loss: {epoch_loss:.4f}"
            )

        # Get privacy spent
        if enable_dp and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=target_delta)
            logger.info(
                f"Training complete. Privacy spent: ε = {epsilon:.4f}, δ = {target_delta}"
            )
        else:
            logger.info(
                f"Training complete. Average loss: {total_loss / local_epochs:.4f}"
            )

        # Store round metrics
        try:
            num_train_samples = len(self.train_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            num_train_samples = 0

        round_metrics = {
            "round": int(config.get("server_round", 0)),
            "train_loss": float(total_loss / local_epochs),
            "epsilon": float(epsilon),
            "delta": float(target_delta if enable_dp else 0.0),
            "num_samples": num_train_samples,
        }

        self.round_history.append(round_metrics)

        # Save metrics after fit to ensure persistence
        self._save_client_metrics()

        # Return updated parameters and metrics
        try:
            num_fit_samples = len(self.train_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            num_fit_samples = 0

        return (
            get_parameters(self.model),
            num_fit_samples,
            {
                "loss": float(total_loss / local_epochs),
                "epsilon": float(epsilon),
                "delta": float(target_delta if enable_dp else 0.0),
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local test data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)

        # Set up checkpoint directory
        checkpoint_dir = None
        if self.run_dir:
            checkpoint_dir = self.run_dir / "checkpoints"

        metrics = evaluate(
            self.model, self.test_loader, self.device, checkpoint_dir=checkpoint_dir
        )

        logger.debug(
            f"Evaluation: Dice = {metrics['dice']:.4f}, Loss = {metrics['loss']:.4f}"
        )

        # Update last round with eval metrics
        server_round = int(config.get("server_round", 0))

        # Ensure latest history is loaded before updating
        self._load_history()

        found = False
        for entry in reversed(self.round_history):
            if entry["round"] == server_round:
                entry["eval_dice"] = float(metrics["dice"])
                entry["eval_loss"] = float(metrics["loss"])
                found = True
                break

        if not found:
            # If round not found in history, create a placeholder entry
            # This happens in Flower simulation because actors are stateless
            placeholder = {
                "round": server_round,
                "train_loss": 0.0,
                "epsilon": 0.0,
                "delta": 0.0,
                "num_samples": 0,
                "eval_dice": float(metrics["dice"]),
                "eval_loss": float(metrics["loss"]),
            }
            self.round_history.append(placeholder)
            # logger.debug(f"Created placeholder for Round {server_round} in evaluate") # Silenced to avoid log noise

        # Save client metrics after evaluation
        self._save_client_metrics()

        try:
            num_test_samples = len(self.test_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            num_test_samples = 0

        return (
            metrics["loss"],
            num_test_samples,
            {
                "dice": metrics["dice"],
            },
        )

        return (
            metrics["loss"],
            num_test_samples,
            {
                "dice": metrics["dice"],
            },
        )

    def _load_history(self) -> None:
        """Load client history from disk."""
        if self.run_dir:
            history_path = self.run_dir / "history.json"
            if history_path.exists():
                try:
                    with open(history_path, "r") as f:
                        data = json.load(f)
                        self.round_history = data.get("rounds", [])
                except Exception:
                    logger.debug("No existing history found to load")

    def _save_client_metrics(self) -> None:
        """Save client-specific metrics and history."""
        if self.run_dir is None:
            return

        end_time = datetime.now().isoformat()

        # Calculate final metrics from last round
        final_train_loss = 0.0
        final_dice = 0.0
        total_epsilon = 0.0

        if self.round_history:
            last_round = self.round_history[-1]
            final_train_loss = last_round.get("train_loss", 0.0)
            final_dice = last_round.get("eval_dice", 0.0)
            total_epsilon = sum(r.get("epsilon", 0.0) for r in self.round_history)

        try:
            num_total_samples = len(self.train_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            num_total_samples = 0

        # Save metrics.json
        metrics_data = {
            "client_id": self.client_id,
            "start_time": self.start_time,
            "end_time": end_time,
            "num_rounds": len(self.round_history),
            "final_train_loss": final_train_loss,
            "final_dice": final_dice,
            "training_samples": num_total_samples,
            "privacy": {
                "total_epsilon": total_epsilon,
                "delta": self.privacy_config.get("target_delta", 1e-5),
            },
        }

        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Save history.json
        history_data = {"client_id": self.client_id, "rounds": self.round_history}

        history_path = self.run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.debug(f"✓ Client {self.client_id} metrics saved to {self.run_dir}")
