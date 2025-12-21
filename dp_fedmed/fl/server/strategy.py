"""FedAvg strategy with privacy budget tracking.

This module defines the DPFedAvg strategy that extends FedAvg with
privacy budget tracking and per-client metrics.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from loguru import logger

from ...privacy.accountant import PrivacyAccountant


class DPFedAvg(FedAvg):
    """FedAvg strategy with privacy budget tracking and per-client metrics."""

    def __init__(
        self,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        run_dir: Optional[Path] = None,
        run_name: str = "default",
        save_metrics: bool = True,
        num_rounds: Optional[int] = None,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        start_round: int = 1,
        **kwargs,
    ):
        """Initialize DP-aware FedAvg.

        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta for DP
            run_dir: Directory to save results
            run_name: Name of this run (config name)
            save_metrics: Whether to save metrics to file
            num_rounds: Total number of rounds (including completed rounds)
            noise_multiplier: Pre-computed noise multiplier to use for all rounds
            max_grad_norm: Maximum gradient norm for DP clipping
            start_round: Starting round number (for checkpoint resumption)
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.privacy_accountant = PrivacyAccountant(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
        )
        self.run_dir = run_dir or Path("./results/default")
        self.run_name = run_name
        self.save_metrics = save_metrics
        self.num_rounds = num_rounds
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.start_round = start_round

        # Server-level round metrics
        self.server_rounds: List[Dict[str, Any]] = []
        # Per-client metrics: {client_id: [round_metrics]}
        self.client_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Checkpointing state
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.best_dice = 0.0
        self.latest_parameters: Optional[Parameters] = None
        self.current_round: int = 0  # Track actual round number

        self.start_time = datetime.now().isoformat()

        logger.info(
            f"DPFedAvg initialized. Target ε = {target_epsilon}, δ = {target_delta}"
        )
        if start_round > 1:
            logger.info(f"Resuming from round {start_round}")
        logger.info(f"Results will be saved to: {self.run_dir.absolute()}")

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit round with pre-computed noise multiplier."""
        # Calculate actual round number when resuming from checkpoint
        actual_round = self.start_round + server_round - 1
        self.current_round = actual_round

        config = {
            "noise_multiplier": self.noise_multiplier,
            "server_round": actual_round,
        }

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

        config = {
            "server_round": actual_round,
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

        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Collect per-client metrics
        epsilons = []
        for client_proxy, fit_res in results:
            client_id = str(client_proxy.cid)
            metrics = fit_res.metrics or {}

            # Extract client metrics
            eps = metrics.get("epsilon", 0.0)
            if isinstance(eps, (int, float)) and eps > 0:
                epsilons.append(float(eps))

            # Store per-client training metrics
            client_round_data = {
                "round": server_round,
                "train_loss": float(metrics.get("loss", 0.0)),
                "epsilon": float(eps),
                "delta": float(metrics.get("delta", 1e-5)),
                "num_samples": fit_res.num_examples,
            }
            self.client_metrics[client_id].append(client_round_data)

        # Calculate average epsilon for this round
        avg_epsilon = sum(epsilons) / len(epsilons) if epsilons else 0.0

        # Record in privacy accountant
        if epsilons:
            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            self.privacy_accountant.record_round(
                round_num=server_round,
                epsilon=avg_epsilon,
                delta=self.privacy_accountant.target_delta,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                num_samples=total_samples,
            )

        # Check if budget exceeded
        if self.privacy_accountant.is_budget_exceeded():
            logger.warning(
                f"⚠ Privacy budget EXCEEDED! "
                f"Cumulative ε = {self.privacy_accountant.get_cumulative_epsilon():.4f} "
                f"> target ε = {self.privacy_accountant.target_epsilon}"
            )
        else:
            remaining = self.privacy_accountant.get_remaining_budget()
            remaining_pct = (remaining / self.privacy_accountant.target_epsilon) * 100
            logger.info(
                f"Privacy budget OK. Remaining: {remaining_pct:.1f}% ({remaining:.4f} ε)"
            )

        # Aggregate parameters using parent class
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Add privacy metrics
        aggregated_metrics["round_epsilon"] = float(avg_epsilon)
        aggregated_metrics["cumulative_epsilon"] = float(
            self.privacy_accountant.get_cumulative_epsilon()
        )
        aggregated_metrics["remaining_budget"] = float(
            self.privacy_accountant.get_remaining_budget()
        )

        # Store server-level round metrics
        server_round_data = {
            "round": server_round,
            "round_epsilon": float(avg_epsilon),
            "cumulative_epsilon": float(
                self.privacy_accountant.get_cumulative_epsilon()
            ),
            "num_clients": len(results),
            "num_failures": len(failures),
        }
        self.server_rounds.append(server_round_data)

        # Store latest parameters for checkpointing
        self.latest_parameters = aggregated_parameters

        logger.info(
            f"Round {server_round} complete: "
            f"ε = {avg_epsilon:.4f}, "
            f"cumulative ε = {self.privacy_accountant.get_cumulative_epsilon():.4f}, "
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

        if not results:
            logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}

        # Call parent aggregate_evaluate
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Collect per-client eval metrics and calculate weighted dice
        dice_values = []
        weights = []
        for client_proxy, eval_res in results:
            client_id = str(client_proxy.cid)
            metrics = eval_res.metrics or {}

            dice = float(metrics.get("dice", 0.0))
            eval_loss = float(metrics.get("loss", 0.0))

            dice_values.append(dice)
            weights.append(eval_res.num_examples)

            # Update client's last round entry with eval metrics
            client_found = False
            for entry in reversed(self.client_metrics[client_id]):
                if entry["round"] == server_round:
                    entry["dice"] = dice
                    entry["eval_loss"] = eval_loss
                    client_found = True
                    break
            if not client_found:
                logger.warning(
                    f"Round {server_round} not found in client_metrics for client {client_id}"
                )

        if dice_values:
            weighted_dice = sum(d * w for d, w in zip(dice_values, weights)) / sum(
                weights
            )
            metrics_aggregated["dice"] = float(weighted_dice)

            # Update server round metrics
            server_found = False
            for entry in self.server_rounds:
                if entry["round"] == server_round:
                    entry["aggregated_dice"] = float(weighted_dice)
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
                f"Dice = {weighted_dice:.4f}, "
                f"Loss = {loss_aggregated:.4f}"
            )

            # Save checkpoints
            self._save_checkpoints(weighted_dice)

        # Save logs after final round evaluation completes
        if self.num_rounds and server_round >= self.num_rounds:
            self.save_logs()

        return loss_aggregated, metrics_aggregated

    def _save_checkpoints(self, current_dice: float) -> None:
        """Save model checkpoints (best + last).

        Args:
            current_dice: Current aggregated dice score
        """
        if self.latest_parameters is None:
            logger.warning("No parameters to checkpoint")
            return

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Convert parameters to numpy arrays for saving
        params_ndarrays = parameters_to_ndarrays(self.latest_parameters)

        # Checkpoint metadata for resumption
        checkpoint_meta = {
            "parameters": params_ndarrays,
            "round": self.current_round,
            "cumulative_epsilon": self.privacy_accountant.get_cumulative_epsilon(),
            "dice": current_dice,
        }

        # Always save last model
        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(checkpoint_meta, last_path)

        # Save best model if improved
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_meta, best_path)
            logger.info(f"New best model saved (dice={current_dice:.4f})")

    def save_logs(self) -> None:
        """Save metrics.json and history.json to run directory."""
        if not self.save_metrics:
            logger.info("Metric saving disabled, skipping log save")
            return

        end_time = datetime.now().isoformat()

        # Get final metrics from last round
        final_dice = 0.0
        final_loss = 0.0
        if self.server_rounds:
            last_round = self.server_rounds[-1]
            final_dice = last_round.get("aggregated_dice", 0.0)
            final_loss = last_round.get("aggregated_loss", 0.0)

        # Save metrics.json (final summary)
        metrics_data = {
            "config": self.run_name,
            "start_time": self.start_time,
            "end_time": end_time,
            "num_rounds": self.num_rounds,
            "final_dice": final_dice,
            "final_loss": final_loss,
            "privacy": self.privacy_accountant.get_summary(),
        }
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info("✓ Saved metrics.json")

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
        logger.info("✓ Saved history.json")

        logger.info(f"✓ Results saved to: {self.run_dir}")
        logger.info(f"✓ Final Dice: {final_dice:.4f}, Final Loss: {final_loss:.4f}")
