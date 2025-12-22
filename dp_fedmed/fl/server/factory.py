"""Factory function for creating Flower server components.

This module contains the server_fn function that creates configured
server components for federated learning.
"""

from pathlib import Path
from typing import Tuple

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters, Parameters
from flwr.server import ServerAppComponents, ServerConfig
from loguru import logger

from ...config import load_config
from ...logging import setup_logging
from ...models.unet2d import create_unet2d, get_parameters, set_parameters
from .strategy import DPFedAvg
from .aggregation import weighted_average


def load_checkpoint(
    checkpoint_path: Path, model
) -> Tuple[Parameters, int, float, float]:
    """Load checkpoint and return parameters with metadata.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load parameters into (for client-format checkpoints)

    Returns:
        Tuple of (parameters, resume_round, cumulative_epsilon, best_dice)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)  # nosec B614

    # Get round number (resume from next round)
    checkpoint_round = checkpoint.get("round", 0)
    resume_round = checkpoint_round + 1

    # Get privacy state
    cumulative_epsilon = checkpoint.get("cumulative_epsilon", 0.0)
    best_dice = checkpoint.get("dice", 0.0)

    # Handle different checkpoint formats
    if "parameters" in checkpoint:
        # Server-format checkpoint (numpy arrays)
        parameters = ndarrays_to_parameters(checkpoint["parameters"])
        logger.info(f"Loaded server-format checkpoint from round {checkpoint_round}")
    elif "model" in checkpoint:
        # Client-format checkpoint (state dict)
        set_parameters(model, list(checkpoint["model"].values()))
        parameters = ndarrays_to_parameters(get_parameters(model))
        logger.info(f"Loaded client-format checkpoint from round {checkpoint_round}")
    else:
        raise ValueError(
            "Unknown checkpoint format. Expected 'parameters' or 'model' key."
        )

    logger.info(f"  Resume round: {resume_round}")
    logger.info(f"  Cumulative epsilon: {cumulative_epsilon:.4f}")
    logger.info(f"  Best dice: {best_dice:.4f}")

    return parameters, resume_round, cumulative_epsilon, best_dice


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Create server components."""
    cfg = context.run_config
    config_file = str(cfg.get("config-file", "configs/default.toml"))

    try:
        config = load_config(config_file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        raise

    # Extract run name from config file path (e.g., "configs/default.toml" -> "default")
    run_name = Path(config_file).stem

    # Setup logging (creates results/<run_name>/server/ directory)
    log_level = str(config.get("logging.level", "INFO"))
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
    num_clients = int(config.get("federated.num_clients", 10))

    # Get privacy settings
    privacy_style = config.privacy.style
    target_delta = config.privacy.target_delta

    # Validate min_fit_clients for user-level DP (Literature recommendation: N >= 10)
    if privacy_style in ["user", "hybrid"] and min_fit_clients < 10:
        logger.warning(
            f"⚠ UNSAFE CONFIGURATION: min_fit_clients ({min_fit_clients}) is less than 10. "
            "User-level DP requires a larger crowd to be statistically meaningful and provide utility. "
            "Proceeding anyway, but results may be poor or privacy guarantees weak."
        )

    # Sample-level (Opacus) settings
    sample_noise_multiplier = config.privacy.sample.noise_multiplier
    sample_max_grad_norm = config.privacy.sample.max_grad_norm

    # User-level (Server) settings
    user_noise_multiplier = config.privacy.user.noise_multiplier
    user_max_grad_norm = config.privacy.user.max_grad_norm

    # Get logging settings
    save_metrics = bool(config.get("logging.save_metrics", True))

    # Log configuration
    logger.info(f"Federated rounds: {num_rounds}")
    logger.info(f"Min fit clients: {min_fit_clients}")
    logger.info(f"Min evaluate clients: {min_evaluate_clients}")
    logger.info(f"Privacy Style: {privacy_style}")

    if privacy_style in ["sample", "hybrid"]:
        logger.info(
            f"Sample-level DP (Opacus): ENABLED (multiplier={sample_noise_multiplier}, clip={sample_max_grad_norm})"
        )

        # PRE-COMPUTE optimal noise multiplier using Opacus if not hybrid or special handling
        # For now we use the provided noise_multiplier as base
        noise_multiplier = sample_noise_multiplier
    else:
        logger.info("Sample-level DP (Opacus): DISABLED")
        noise_multiplier = 0.0

    if privacy_style in ["user", "hybrid"]:
        logger.info(
            f"User-level DP (Server): ENABLED (multiplier={user_noise_multiplier}, clip={user_max_grad_norm})"
        )
    else:
        logger.info("User-level DP (Server): DISABLED")

    # Get model configuration
    model_config = config.get_section("model")

    # Create initial model
    model = create_unet2d(
        in_channels=int(model_config.get("in_channels", 1)),
        out_channels=int(model_config.get("out_channels", 2)),
        channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
        strides=tuple(model_config.get("strides", [2, 2, 2])),
        num_res_units=int(model_config.get("num_res_units", 2)),
    )

    logger.info("✓ Initial model created")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check for checkpoint resumption
    resume_from = config.get("checkpointing.resume_from")
    start_round = 1
    initial_parameters = None

    if resume_from and resume_from.strip():
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            initial_parameters, start_round, prev_epsilon, prev_dice = load_checkpoint(
                checkpoint_path, model
            )
            logger.info(f"✓ Resuming training from round {start_round}/{num_rounds}")

            # Calculate remaining rounds
            remaining_rounds = num_rounds - start_round + 1
            if remaining_rounds <= 0:
                logger.warning(
                    f"Checkpoint is from round {start_round - 1}, "
                    f"but num_rounds is {num_rounds}. Nothing to do."
                )
                remaining_rounds = 0
        else:
            logger.warning(f"Checkpoint not found at {resume_from}, starting fresh")
            initial_parameters = ndarrays_to_parameters(get_parameters(model))
            remaining_rounds = num_rounds
    else:
        # Fresh start
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        remaining_rounds = num_rounds

    # Create strategy
    strategy = DPFedAvg(
        target_delta=target_delta,
        run_dir=run_dir,
        run_name=run_name,
        save_metrics=save_metrics,
        num_rounds=num_rounds,
        noise_multiplier=noise_multiplier,
        max_grad_norm=sample_max_grad_norm,
        user_noise_multiplier=user_noise_multiplier
        if privacy_style in ["user", "hybrid"]
        else 0.0,
        user_max_grad_norm=user_max_grad_norm
        if privacy_style in ["user", "hybrid"]
        else 0.0,
        start_round=start_round,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        total_clients=num_clients,
    )

    # Wrap strategy with Flower's Server-Side DP if requested
    if privacy_style in ["user", "hybrid"]:
        strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=user_noise_multiplier,
            clipping_norm=user_max_grad_norm,
            num_sampled_clients=min_fit_clients,
        )
        logger.info("✓ Wrapped strategy with Server-Side DP")

    # Flower runs server_round 1..N, so we run remaining_rounds
    server_config = ServerConfig(num_rounds=remaining_rounds)

    logger.info("=" * 60)
    logger.info("Server initialization complete. Starting federated learning...")
    logger.info("=" * 60)

    return ServerAppComponents(strategy=strategy, config=server_config)
