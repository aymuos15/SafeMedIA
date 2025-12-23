"""Factory function for creating Flower server components.

This module contains the server_fn function that creates configured
server components for federated learning.
"""

from pathlib import Path
from typing import Dict

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from loguru import logger

from ...config import load_config
from ...logging import setup_logging
from ...models.unet2d import create_unet2d, get_parameters
from ..checkpoint import (
    resolve_checkpoint_path,
    load_unified_checkpoint,
    UnifiedCheckpointManager,
    ClientState,
)
from ..base.strategy import DPStrategy
from .aggregation import weighted_average


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Create server components."""
    cfg = context.run_config
    config_file = str(cfg.get("config-file", "configs/default.toml"))

    # Check training mode - delegate to SSL server if needed
    mode_str = str(cfg.get("training-mode", "supervised"))
    if mode_str == "ssl":
        from dp_fedmed.fl.ssl.server_factory import server_fn as ssl_server_fn

        return ssl_server_fn(context)

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

    # Get local epochs for checkpoint manager
    local_epochs = int(config.get("training.local_epochs", 5))

    # Unified checkpoint directory (at run level, not server level)
    checkpoint_dir = run_dir.parent / "checkpoints"

    # Check for checkpoint resumption
    resume_from = config.get("checkpointing.resume_from")
    start_round = 1
    initial_parameters = None
    remaining_rounds = num_rounds  # Default for fresh start
    checkpoint_manager = None
    client_resume_states: Dict[int, ClientState] = {}
    is_mid_round_resume = False

    if resume_from:
        try:
            checkpoint_path = resolve_checkpoint_path(resume_from, run_dir.parent)
            if checkpoint_path:
                # Load unified checkpoint
                checkpoint = load_unified_checkpoint(checkpoint_path)
                logger.info(f"✓ Resolved checkpoint: {checkpoint_path}")

                # Get parameters from checkpoint
                initial_parameters = ndarrays_to_parameters(
                    checkpoint.server.parameters
                )

                # Determine resume behavior based on round status
                if checkpoint.round.status == "in_progress":
                    # Mid-round resume
                    is_mid_round_resume = True
                    start_round = checkpoint.round.current
                    client_resume_states = checkpoint.clients
                    logger.info(
                        f"✓ Mid-round resume: round {start_round}, "
                        f"{len(client_resume_states)} clients"
                    )
                else:
                    # Round completed, resume from next round
                    start_round = checkpoint.round.current + 1
                    logger.info(
                        f"✓ Resuming from completed round {checkpoint.round.current}"
                    )

                # Create checkpoint manager with loaded state
                checkpoint_manager = UnifiedCheckpointManager(
                    checkpoint_dir=checkpoint_dir,
                    run_name=run_name,
                    num_rounds=num_rounds,
                    target_delta=target_delta,
                )

                # Calculate remaining rounds
                remaining_rounds = num_rounds - start_round + 1
                if remaining_rounds <= 0:
                    logger.warning(
                        f"Checkpoint is from round {checkpoint.round.current}, "
                        f"but num_rounds is {num_rounds}. Nothing to do."
                    )
                    remaining_rounds = 0

                logger.info(
                    f"✓ Resuming training from round {start_round}/{num_rounds}"
                )
        except FileNotFoundError as e:
            logger.error(f"Checkpoint resolution failed:\n{e}")
            raise

    if initial_parameters is None:
        # Fresh start
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        remaining_rounds = num_rounds
        checkpoint_manager = UnifiedCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            num_rounds=num_rounds,
            target_delta=target_delta,
        )

    # Create strategy
    strategy = DPStrategy(
        primary_metric="dice",
        higher_is_better=True,
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
        local_epochs=local_epochs,
        checkpoint_manager=checkpoint_manager,
        client_resume_states=client_resume_states,
        is_mid_round_resume=is_mid_round_resume,
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
