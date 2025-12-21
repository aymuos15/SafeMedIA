"""Factory function for creating Flower server components.

This module contains the server_fn function that creates configured
server components for federated learning.
"""

from pathlib import Path

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from loguru import logger

from ...config import load_config
from ...logging import setup_logging
from ...models.unet2d import create_unet2d, get_parameters
from ...privacy.budget_calculator import validate_privacy_config
from .strategy import DPFedAvg
from .aggregation import weighted_average


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
    max_grad_norm = float(config.get("privacy.max_grad_norm", 1.0))

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
            client_dataset_size = int(config.get("privacy.client_dataset_size", 270))
            computed_noise, projected_epsilon = validate_privacy_config(
                config=config,
                client_dataset_size=client_dataset_size,
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
        max_grad_norm=max_grad_norm,
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
