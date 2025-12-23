"""Abstract base class for DP-aware federated strategies.

This module defines the BaseDPStrategy class that provides shared
functionality for both supervised and SSL federated averaging strategies.
"""

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from loguru import logger

from dp_fedmed.privacy.accountant import PrivacyAccountant
from dp_fedmed.fl.checkpoint import UnifiedCheckpointManager, ClientState


class BaseDPStrategy(FedAvg, ABC):
    """Abstract base class for DP-aware federated strategies.

    This class provides shared functionality for privacy accounting,
    checkpointing, and metric aggregation. Subclasses implement
    the evaluation metric aggregation logic.
    """

    def __init__(
        self,
        target_delta: float = 1e-5,
        target_epsilon: Optional[float] = None,
        run_dir: Optional[Path] = None,
        run_name: str = "default",
        save_metrics: bool = True,
        num_rounds: Optional[int] = None,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        user_noise_multiplier: float = 0.0,
        user_max_grad_norm: float = 0.0,
        total_clients: int = 10,
        start_round: int = 1,
        local_epochs: int = 5,
        checkpoint_manager: Optional[UnifiedCheckpointManager] = None,
        client_resume_states: Optional[Dict[int, ClientState]] = None,
        is_mid_round_resume: bool = False,
        **kwargs,
    ):
        """Initialize DP-aware strategy.

        Args:
            target_delta: Target delta for DP
            target_epsilon: Target epsilon for DP (optional, for tracking)
            run_dir: Directory to save results
            run_name: Name of this run (config name)
            save_metrics: Whether to save metrics to file
            num_rounds: Total number of rounds (including completed rounds)
            noise_multiplier: Pre-computed noise multiplier for sample-level DP
            max_grad_norm: Maximum gradient norm for sample-level DP
            user_noise_multiplier: Noise multiplier for user-level DP
            user_max_grad_norm: Clipping norm for user-level DP
            total_clients: Total number of clients in the population
            start_round: Starting round number (for checkpoint resumption)
            local_epochs: Number of local epochs per round
            checkpoint_manager: Unified checkpoint manager (created externally)
            client_resume_states: Client states for mid-round resume
            is_mid_round_resume: Whether this is a mid-round resume
            **kwargs: Additional arguments for FedAvg
        """
        # Filter out DPFedAvg/DPFedAvgSSL specific kwargs before passing to FedAvg
        fedavg_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "fraction_fit",
                "fraction_evaluate",
                "min_fit_clients",
                "min_evaluate_clients",
                "min_available_clients",
                "evaluate_fn",
                "on_fit_config_fn",
                "on_evaluate_config_fn",
                "accept_failures",
                "initial_parameters",
                "fit_metrics_aggregation_fn",
                "evaluate_metrics_aggregation_fn",
            ]
        }
        super().__init__(**fedavg_kwargs)
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.privacy_accountant = PrivacyAccountant(target_delta=target_delta)
        self.run_dir = run_dir or Path("./results/default")
        self.run_name = run_name
        self.save_metrics = save_metrics
        self.num_rounds = num_rounds
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.user_noise_multiplier = user_noise_multiplier
        self.user_max_grad_norm = user_max_grad_norm
        self.total_clients = total_clients
        self.start_round = start_round
        self.local_epochs = local_epochs

        # Server-level round metrics
        self.server_rounds: List[Dict[str, Any]] = []
        # Per-client metrics: {client_id: [round_metrics]}
        self.client_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Unified checkpointing
        self.checkpoint_dir = self.run_dir.parent / "checkpoints"
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            self.checkpoint_manager = UnifiedCheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                run_name=run_name,
                num_rounds=num_rounds or 5,
                target_delta=target_delta,
            )

        self.latest_parameters: Optional[Parameters] = None
        self.current_round: int = 0

        # Resume state
        self.client_resume_states = client_resume_states or {}
        self.is_mid_round_resume = is_mid_round_resume
        self._resume_round = start_round if is_mid_round_resume else 0

        self.start_time = datetime.now().isoformat()

        logger.info(
            f"{self.__class__.__name__} initialized. Target delta = {target_delta}"
        )
        if start_round > 1:
            logger.info(f"Resuming from round {start_round}")
        if is_mid_round_resume:
            logger.info(
                f"Mid-round resume: {len(self.client_resume_states)} client states loaded"
            )
        logger.info(f"Results will be saved to: {self.run_dir.absolute()}")

    @abstractmethod
    def _get_primary_metric_name(self) -> str:
        """Get the name of the primary evaluation metric.

        Returns:
            Metric name (e.g., 'dice' for supervised, 'val_loss' for SSL)
        """
        pass

    @abstractmethod
    def _aggregate_evaluation_metrics(
        self,
        results: List[Tuple],
        metrics_aggregated: Dict[str, Scalar],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from client results.

        Args:
            results: List of (client_proxy, eval_result) tuples
            metrics_aggregated: Pre-aggregated metrics from parent class

        Returns:
            Tuple of (weighted_metric_value, updated_metrics_dict)
        """
        pass

    def initialize_checkpoint(self, parameters: Parameters) -> None:
        """Initialize checkpoint for a fresh run.

        Args:
            parameters: Initial global model parameters
        """
        if self.checkpoint_manager.get_current_checkpoint() is None:
            params_ndarrays = parameters_to_ndarrays(parameters)
            self.checkpoint_manager.create_initial_checkpoint(
                parameters=params_ndarrays,
                num_clients=self.total_clients,
                local_epochs=self.local_epochs,
            )
            logger.info("Initialized unified checkpoint for fresh run")

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit round with pre-computed noise multiplier."""
        # Calculate actual round number when resuming from checkpoint
        actual_round = self.start_round + server_round - 1
        self.current_round = actual_round

        # Initialize or update checkpoint for this round
        if parameters is not None:
            if server_round == 1 and not self.is_mid_round_resume:
                # Fresh run - initialize checkpoint
                self.initialize_checkpoint(parameters)
            elif not self.is_mid_round_resume:
                # New round (not mid-round resume) - start next round in checkpoint
                self.checkpoint_manager.start_next_round(
                    actual_round, self.local_epochs
                )

        # Ensure values are Scalar (int, float, str, bytes, bool)
        config: Dict[str, Scalar] = {
            "noise_multiplier": float(self.noise_multiplier),
            "server_round": int(actual_round),
            "local_epochs": int(self.local_epochs),
        }

        # Check for mid-round resume
        if self.is_mid_round_resume and actual_round == self._resume_round:
            config["resume_from_checkpoint"] = True
            checkpoint_path = self.checkpoint_dir / "last.pt"
            config["checkpoint_path"] = str(checkpoint_path)
            logger.info(f"Mid-round resume signaled for round {actual_round}")
        else:
            config["resume_from_checkpoint"] = False

        # Get standard sample
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return clients and config
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure evaluate round with server_round in config."""
        # Calculate actual round number when resuming from checkpoint
        actual_round = self.start_round + server_round - 1

        config: Dict[str, Scalar] = {
            "server_round": int(actual_round),
        }

        # Calculate sample size based on fraction_evaluate
        num_available = client_manager.num_available()
        sample_size = max(
            int(num_available * self.fraction_evaluate), self.min_evaluate_clients
        )
        sample_size = min(sample_size, num_available)

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )

        # Return clients and config
        return [
            (client, fl.common.EvaluateIns(parameters, config)) for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and track privacy."""

        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_metrics is None:
            aggregated_metrics = {}
        else:
            aggregated_metrics = dict(aggregated_metrics)

        # Collect per-client metrics
        sample_rates = []
        step_counts = []
        epsilons = []
        for client_proxy, fit_res in results:
            client_id = str(client_proxy.cid)
            metrics = fit_res.metrics or {}

            # Extract client metrics
            eps_sample = float(metrics.get("epsilon", 0.0))
            sample_rate = float(metrics.get("sample_rate", 0.0))
            steps = int(metrics.get("steps", 0))

            if eps_sample > 0:
                epsilons.append(eps_sample)
            if sample_rate > 0:
                sample_rates.append(sample_rate)
            if steps > 0:
                step_counts.append(steps)

            # Store per-client training metrics
            client_round_data = {
                "round": server_round,
                "train_loss": float(metrics.get("loss", 0.0)),
                "sample_epsilon": eps_sample,
                "delta": float(metrics.get("delta", 1e-5)),
                "num_samples": fit_res.num_examples,
            }
            self.client_metrics[client_id].append(client_round_data)

        # Calculate metadata for RDP recording
        avg_epsilon = sum(epsilons) / len(epsilons) if epsilons else 0.0
        avg_sample_rate = sum(sample_rates) / len(sample_rates) if sample_rates else 0.0
        avg_steps = int(sum(step_counts) / len(step_counts)) if step_counts else 0

        # User-level sampling rate: (participating clients / total clients)
        user_sample_rate = (
            len(results) / self.total_clients if self.total_clients > 0 else 0.0
        )

        # Record in privacy accountant
        if avg_sample_rate > 0 or self.user_noise_multiplier > 0:
            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            self.privacy_accountant.record_round(
                round_num=server_round,
                noise_multiplier_sample=self.noise_multiplier,
                sample_rate_sample=avg_sample_rate,
                steps_sample=avg_steps,
                noise_multiplier_user=self.user_noise_multiplier,
                sample_rate_user=user_sample_rate
                if self.user_noise_multiplier > 0
                else 0.0,
                steps_user=1 if self.user_noise_multiplier > 0 else 0,
                num_samples=total_samples,
            )

        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return aggregated_parameters, aggregated_metrics

        # Add privacy metrics
        aggregated_metrics["round_sample_epsilon"] = float(avg_epsilon)
        aggregated_metrics["cumulative_sample_epsilon"] = float(
            self.privacy_accountant.get_cumulative_sample_epsilon()
        )
        aggregated_metrics["cumulative_user_epsilon"] = float(
            self.privacy_accountant.get_cumulative_user_epsilon()
        )

        # Store server-level round metrics
        server_round_data = {
            "round": server_round,
            "sample_epsilon": float(avg_epsilon),
            "cumulative_sample_epsilon": float(
                self.privacy_accountant.get_cumulative_sample_epsilon()
            ),
            "cumulative_user_epsilon": float(
                self.privacy_accountant.get_cumulative_user_epsilon()
            ),
            "num_clients": len(results),
            "num_failures": len(failures),
        }
        self.server_rounds.append(server_round_data)

        # Store latest parameters for checkpointing
        self.latest_parameters = aggregated_parameters

        logger.info(
            f"Round {server_round} complete: "
            f"epsilon_sample = {avg_epsilon:.4f}, "
            f"cumulative epsilon_sample = {self.privacy_accountant.get_cumulative_sample_epsilon():.4f}, "
            f"clients = {len(results)}"
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""

        # Call parent aggregate_evaluate
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if metrics_aggregated is None:
            metrics_aggregated = {}
        else:
            metrics_aggregated = dict(metrics_aggregated)

        # Delegate metric-specific aggregation to subclass
        primary_metric, metrics_aggregated = self._aggregate_evaluation_metrics(
            results, metrics_aggregated
        )

        # Update server round metrics
        if primary_metric is not None:
            primary_metric_name = self._get_primary_metric_name()
            server_found = False
            for entry in self.server_rounds:
                if entry["round"] == server_round:
                    entry[f"aggregated_{primary_metric_name}"] = float(primary_metric)
                    entry["aggregated_loss"] = (
                        float(loss_aggregated) if loss_aggregated else 0.0
                    )
                    server_found = True
                    break
            if not server_found:
                logger.warning(
                    f"Round {server_round} not found in server_rounds for aggregation update"
                )

            logger.info(
                f"Round {server_round} eval: "
                f"{primary_metric_name} = {primary_metric:.4f}, "
                f"Loss = {loss_aggregated:.4f}"
            )

            # Save checkpoints
            self._save_checkpoints(primary_metric)

        # Save logs after final round evaluation completes
        if self.num_rounds and server_round >= self.num_rounds:
            self.save_logs()

        return loss_aggregated, metrics_aggregated

    def _save_checkpoints(self, current_metric: float) -> bool:
        """Save unified checkpoint after round evaluation.

        Args:
            current_metric: Current aggregated primary metric

        Returns:
            True if checkpoint was saved successfully, False otherwise
        """
        if self.latest_parameters is None:
            logger.error(
                "No parameters to checkpoint - this may indicate a training failure"
            )
            return False

        params_ndarrays = parameters_to_ndarrays(self.latest_parameters)

        try:
            primary_metric_name = self._get_primary_metric_name()
            self.checkpoint_manager.update_server_state(
                parameters=params_ndarrays,
                metrics={primary_metric_name: current_metric},
                round_num=self.current_round,
                cumulative_sample_epsilon=self.privacy_accountant.get_cumulative_sample_epsilon(),
                cumulative_user_epsilon=self.privacy_accountant.get_cumulative_user_epsilon(),
            )

            # Mark round as completed
            self.checkpoint_manager.mark_round_completed(current_metric)

            # Save checkpoint
            self.checkpoint_manager.save()

            # Clear mid-round resume flag after first successful round
            if self.is_mid_round_resume:
                self.is_mid_round_resume = False
                logger.info("Mid-round resume completed, future rounds start fresh")

            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def save_logs(self) -> None:
        """Save metrics.json and history.json to run directory."""
        if not self.save_metrics:
            logger.info("Metric saving disabled, skipping log save")
            return

        end_time = datetime.now().isoformat()

        # Get final metrics from last round
        primary_metric_name = self._get_primary_metric_name()
        final_metric = 0.0
        final_loss = 0.0
        if self.server_rounds:
            last_round = self.server_rounds[-1]
            final_metric = last_round.get(f"aggregated_{primary_metric_name}", 0.0)
            final_loss = last_round.get("aggregated_loss", 0.0)

        # Save metrics.json (final summary)
        metrics_data = {
            "config": self.run_name,
            "start_time": self.start_time,
            "end_time": end_time,
            "num_rounds": self.num_rounds,
            f"final_{primary_metric_name}": final_metric,
            "final_loss": final_loss,
            "privacy": self.privacy_accountant.get_summary(),
        }
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info("Saved metrics.json")

        # Save history.json (per-round data)
        history_data = {
            "server": {"rounds": self.server_rounds},
            "clients": [
                {"client_id": cid, "rounds": rounds}
                for cid, rounds in sorted(self.client_metrics.items())
            ],
        }
        history_path = self.run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info("Saved history.json")

        logger.info(f"Results saved to: {self.run_dir}")
        logger.info(
            f"Final {primary_metric_name}: {final_metric:.4f}, Final Loss: {final_loss:.4f}"
        )
