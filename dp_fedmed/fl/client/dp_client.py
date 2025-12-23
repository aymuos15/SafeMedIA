"""Flower client with differential privacy using Opacus.

This module defines the DPFlowerClient class that performs local training
with differential privacy using Opacus.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule
from loguru import logger


from ...models.unet2d import create_unet2d, get_parameters, set_parameters
from ..task import train_one_epoch, evaluate
from ..checkpoint import load_unified_checkpoint, ClientState
from ...utils import get_dataset_size


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
        loss_config: Dict | None = None,
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
            loss_config: Loss function configuration
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_config = training_config
        self.privacy_config = privacy_config
        self.device = device
        self.client_id = client_id
        self.run_dir = run_dir
        self.loss_config = loss_config or {}

        # Track metrics across rounds
        self.round_history: list = []
        self._load_history()
        self.start_time: str = datetime.now().isoformat()

        # Create model from config
        self.model: nn.Module | GradSampleModule = create_unet2d(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 2),
            channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
            strides=tuple(model_config.get("strides", [2, 2, 2])),
            num_res_units=model_config.get("num_res_units", 2),
        )
        self.model.to(self.device)

        # Will be set up each round
        self.optimizer: optim.Optimizer | None = None
        self.privacy_engine: PrivacyEngine | None = None
        self._dp_train_loader: torch.utils.data.DataLoader | None = None

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
                    f"Client {self.client_id} already completed round, returning cached results"
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
                    if isinstance(self.model, GradSampleModule):
                        self.model = self.model.to_standard_module()
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
                # No model state saved, start fresh
                self.set_parameters(parameters)
        else:
            # Normal case: use global parameters
            self.set_parameters(parameters)

        # Get training parameters
        local_epochs = int(
            config.get("local_epochs", self.training_config.get("local_epochs", 1))
        )
        lr = float(self.training_config.get("learning_rate", 0.001))
        momentum = float(self.training_config.get("momentum", 0.9))

        # Get privacy style
        privacy_style = self.privacy_config.get("style", "sample")
        enable_sample_dp = privacy_style in ["sample", "hybrid"]

        # Get privacy parameters
        sample_config = self.privacy_config.get("sample", {})
        max_grad_norm = float(sample_config.get("max_grad_norm", 1.0))
        target_delta = float(self.privacy_config.get("target_delta", 1e-5))

        # USE SERVER-PROVIDED NOISE MULTIPLIER (pre-computed)
        noise_multiplier = float(config.get("noise_multiplier", 1.0))

        logger.info(
            f"Round {config.get('server_round', '?')}: Style={privacy_style}, "
            f"noise_multiplier={noise_multiplier:.4f}, epochs={start_epoch}/{local_epochs}"
        )

        # CRITICAL: Unwrap model if already wrapped from previous round
        if isinstance(self.model, GradSampleModule):
            logger.debug("Unwrapping GradSampleModule for next round")
            self.model = self.model.to_standard_module()

        # Ensure all gradients are cleared
        self.model.zero_grad(set_to_none=True)

        # Set up optimizer (must be created AFTER unwrapping)
        self.optimizer = torch.optim.SGD(  # type: ignore
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
        )

        epsilon = 0.0  # Will be updated if DP is enabled
        sample_rate = 0.0
        steps = 0

        if enable_sample_dp:
            # Set up privacy engine
            privacy_engine = PrivacyEngine()
            self.privacy_engine = privacy_engine

            # Wrap model, optimizer, and data loader with DP
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
            # Steps from start_epoch onwards
            steps = len(train_loader) * (local_epochs - start_epoch)
            logger.info(
                f"Sample-level DP enabled: noise={noise_multiplier}, clip={max_grad_norm}"
            )
        else:
            train_loader = self.train_loader
            sample_rate = 0.0
            steps = len(train_loader) * (local_epochs - start_epoch)
            logger.info(f"Sample-level DP disabled (style: {privacy_style})")

        # Training loop (from start_epoch to local_epochs)
        total_loss = accumulated_loss
        for epoch in range(start_epoch, local_epochs):
            epoch_loss = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.device,
                loss_config=self.loss_config,
            )
            total_loss += epoch_loss
            logger.debug(
                f"Local epoch {epoch + 1}/{local_epochs}, Loss: {epoch_loss:.4f}"
            )

        # Get privacy spent
        if enable_sample_dp and self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=target_delta)
            logger.info(
                f"Training complete. Privacy spent: ε = {epsilon:.4f}, δ = {target_delta}"
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
            "delta": float(target_delta if enable_sample_dp else 0.0),
            "num_samples": num_train_samples,
        }

        self.round_history.append(round_metrics)

        # Save metrics after fit to ensure persistence
        self._save_client_metrics()

        # Return updated parameters and metrics
        num_fit_samples = get_dataset_size(self.train_loader)

        return (
            get_parameters(self.model),
            num_fit_samples,
            {
                "loss": float(total_loss / local_epochs),
                "epsilon": float(epsilon),
                "delta": float(target_delta if enable_sample_dp else 0.0),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
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

        metrics = evaluate(
            self.model,
            self.test_loader,
            self.device,
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

        # Save client metrics after evaluation
        self._save_client_metrics()

        num_test_samples = get_dataset_size(self.test_loader)

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
                except FileNotFoundError:
                    logger.debug("No history file found")
                except json.JSONDecodeError as e:
                    logger.warning(f"Corrupted history file, starting fresh: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error loading history: {e}")

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

        num_total_samples = get_dataset_size(self.train_loader)

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
