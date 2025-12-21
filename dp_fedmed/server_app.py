"""Flower ServerApp for federated learning with privacy tracking.

This module defines the Flower server that coordinates federated learning
across clients and tracks cumulative privacy budget.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from loguru import logger

from .config import load_config
from .logging_config import setup_logging
from .models.unet2d import create_unet2d, get_parameters
from .privacy.accountant import PrivacyAccountant
from .privacy.budget_calculator import validate_privacy_config


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
        **kwargs,
    ):
        """Initialize DP-aware FedAvg.

        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta for DP
            run_dir: Directory to save results
            run_name: Name of this run (config name)
            save_metrics: Whether to save metrics to file
            num_rounds: Total number of rounds
            noise_multiplier: Pre-computed noise multiplier to use for all rounds
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

        # Server-level round metrics
        self.server_rounds: List[Dict[str, Any]] = []
        # Per-client metrics: {client_id: [round_metrics]}
        self.client_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.start_time = datetime.now().isoformat()

        logger.info(
            f"DPFedAvg initialized. Target ε = {target_epsilon}, δ = {target_delta}"
        )
        logger.info(f"Results will be saved to: {self.run_dir.absolute()}")

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit round with pre-computed noise multiplier."""
        config = {
            "noise_multiplier": self.noise_multiplier,
            "server_round": server_round,
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
            if isinstance(eps, (int, float)) and eps != float("inf"):
                epsilons.append(float(eps))

            # Store per-client training metrics
            client_round_data = {
                "round": server_round,
                "train_loss": float(metrics.get("loss", 0.0)),
                "epsilon": float(eps) if eps != float("inf") else 0.0,
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
                noise_multiplier=1.0,
                max_grad_norm=1.0,
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
            logger.info(f"Privacy budget OK. Remaining: {remaining:.2f}%")

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
            for entry in reversed(self.client_metrics[client_id]):
                if entry["round"] == server_round:
                    entry["dice"] = dice
                    entry["eval_loss"] = eval_loss
                    break

        if dice_values:
            weighted_dice = sum(d * w for d, w in zip(dice_values, weights)) / sum(
                weights
            )
            metrics_aggregated["dice"] = float(weighted_dice)

            # Update server round metrics
            for entry in self.server_rounds:
                if entry["round"] == server_round:
                    entry["aggregated_dice"] = float(weighted_dice)
                    entry["aggregated_loss"] = (
                        float(loss_aggregated) if loss_aggregated else 0.0
                    )
                    break

            logger.info(
                f"Round {server_round} eval: "
                f"Dice = {weighted_dice:.4f}, "
                f"Loss = {loss_aggregated:.4f}"
            )

        # Save logs after final round evaluation completes
        if self.num_rounds and server_round >= self.num_rounds:
            self.save_logs()

        return loss_aggregated, metrics_aggregated

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
        logger.info(f"✓ Saved metrics.json")

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
        logger.info(f"✓ Saved history.json")

        logger.info(f"✓ Results saved to: {self.run_dir}")
        logger.info(f"✓ Final Dice: {final_dice:.4f}, Final Loss: {final_loss:.4f}")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)
    aggregated = {}

    metric_keys = set()
    for _, m in metrics:
        metric_keys.update(m.keys())

    for key in metric_keys:
        values = [
            (num_samples, m.get(key, 0.0)) for num_samples, m in metrics if key in m
        ]
        if values:
            numeric_values = []
            for n, v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append((n, float(v)))

            if numeric_values:
                weighted_sum = sum(n * v for n, v in numeric_values)
                aggregated[key] = weighted_sum / total_samples

    return aggregated


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Create server components."""
    cfg = context.run_config
    config_file = cfg.get("config-file", "configs/default.toml")

    try:
        config = load_config(config_file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        raise

    # Extract run name from config file path (e.g., "configs/default.toml" -> "default")
    run_name = Path(config_file).stem

    # Setup logging (creates results/<run_name>/server/ directory)
    log_level = config.get("logging.level", "INFO")
    run_dir = setup_logging(run_name=run_name, level=log_level, role="server")

    logger.info("=" * 60)
    logger.info("DP-FedMed Server Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Run name: {run_name}")

    # Get federated settings
    num_rounds = int(config.get("federated.num_rounds", 5))
    fraction_fit = float(config.get("federated.fraction_fit", 1.0))
    fraction_evaluate = float(config.get("federated.fraction_evaluate", 1.0))
    min_fit_clients = int(config.get("federated.min_fit_clients", 2))
    min_evaluate_clients = int(config.get("federated.min_evaluate_clients", 2))
    min_available_clients = int(config.get("federated.min_available_clients", 2))

    # Get privacy settings
    target_epsilon = float(config.get("privacy.target_epsilon", 8.0))
    target_delta = float(config.get("privacy.target_delta", 1e-5))
    enable_dp = bool(config.get("privacy.enable_dp", True))

    # Get logging settings
    save_metrics = bool(config.get("logging.save_metrics", True))

    # Log configuration
    logger.info(f"Federated rounds: {num_rounds}")
    logger.info(f"Min fit clients: {min_fit_clients}")
    logger.info(f"Min evaluate clients: {min_evaluate_clients}")

    if enable_dp:
        logger.info("Differential Privacy: ENABLED")
        logger.info(f"  Target ε = {target_epsilon}")
        logger.info(f"  Target δ = {target_delta}")

        # PRE-COMPUTE optimal noise multiplier using Opacus
        try:
            computed_noise, projected_epsilon = validate_privacy_config(
                config=config,
                client_dataset_size=270,  # TODO: Get from actual data or config
            )

            # Override config value with pre-computed noise
            logger.info(f"✓ Pre-computed noise_multiplier: {computed_noise:.4f}")
            logger.info(f"✓ Projected final ε: {projected_epsilon:.4f}")

            # Store for use in strategy
            noise_multiplier = computed_noise

        except ValueError as e:
            logger.error(f"Privacy budget validation failed:\n{e}")
            raise SystemExit(1)  # Abort immediately
    else:
        logger.warning("Differential Privacy: DISABLED")
        noise_multiplier = 1.0  # Not used, but set for consistency

    # Get model configuration
    model_config = config.get_section("model")

    # Create initial model parameters
    model = create_unet2d(
        in_channels=int(model_config.get("in_channels", 1)),
        out_channels=int(model_config.get("out_channels", 2)),
        channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
        strides=tuple(model_config.get("strides", [2, 2, 2])),
        num_res_units=int(model_config.get("num_res_units", 2)),
    )
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    logger.info("✓ Initial model created")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create strategy
    strategy = DPFedAvg(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        run_dir=run_dir,
        run_name=run_name,
        save_metrics=save_metrics,
        num_rounds=num_rounds,
        noise_multiplier=noise_multiplier,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    server_config = ServerConfig(num_rounds=num_rounds)

    logger.info("=" * 60)
    logger.info("Server initialization complete. Starting federated learning...")
    logger.info("=" * 60)

    return ServerAppComponents(strategy=strategy, config=server_config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
