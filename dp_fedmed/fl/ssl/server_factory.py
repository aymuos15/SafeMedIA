"""Factory function for creating Flower server components for federated SSL pretraining.

This module contains the server_fn function that creates configured
server components for federated SSL pretraining.
"""

from pathlib import Path
from typing import Dict

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from loguru import logger

from dp_fedmed.config import load_config
from dp_fedmed.logging import setup_logging
from dp_fedmed.models.unet2d import create_unet2d, get_parameters
from dp_fedmed.privacy.budget_calculator import compute_noise_multiplier
from dp_fedmed.fl.checkpoint import (
    resolve_checkpoint_path,
    load_unified_checkpoint,
    UnifiedCheckpointManager,
    ClientState,
)
from dp_fedmed.fl.ssl.strategy import DPFedAvgSSL
from dp_fedmed.fl.ssl.model import SSLUNet
from dp_fedmed.fl.server.aggregation import weighted_average


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Create server components for federated SSL pretraining."""
    cfg = context.run_config
    config_file = str(cfg.get("config-file", "configs/pretraining.toml"))

    try:
        config = load_config(config_file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        raise

    # Extract run name from config file path
    run_name = Path(config_file).stem

    # Setup logging
    log_level = str(config.get("logging.level", "INFO"))
    run_dir = setup_logging(run_name=run_name, level=log_level, role="server")

    logger.info("=" * 60)
    logger.info("DP-FedMed SSL Pretraining Server Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Run name: {run_name}")

    # Get federated settings
    num_rounds = int(config.get("federated.num_rounds", 2))
    fraction_fit = float(config.get("federated.fraction_fit", 1.0))
    fraction_evaluate = float(config.get("federated.fraction_evaluate", 1.0))
    min_fit_clients = int(config.get("federated.min_fit_clients", 2))
    min_evaluate_clients = int(config.get("federated.min_evaluate_clients", 2))
    min_available_clients = int(config.get("federated.min_available_clients", 2))
    num_clients = int(config.get("federated.num_clients", 2))

    # Get privacy settings
    privacy_config_dict = config.get("privacy", {})
    privacy_style = privacy_config_dict.get("style", "sample")
    target_delta = privacy_config_dict.get("target_delta", 1e-5)
    target_epsilon = privacy_config_dict.get("target_epsilon", 8.0)

    # Sample-level (Opacus) settings
    sample_config = privacy_config_dict.get("sample", {})
    sample_noise_multiplier = sample_config.get("noise_multiplier", 1.0)
    sample_max_grad_norm = sample_config.get("max_grad_norm", 1.0)

    # User-level (Server) settings
    user_config = privacy_config_dict.get("user", {})
    user_noise_multiplier = user_config.get("noise_multiplier", 0.0)
    user_max_grad_norm = user_config.get("max_grad_norm", 1.0)

    # Get logging settings
    save_metrics = bool(config.get("logging.save_metrics", True))

    # Log configuration
    logger.info(f"Federated rounds: {num_rounds}")
    logger.info(f"Min fit clients: {min_fit_clients}")
    logger.info(f"Privacy Style: {privacy_style}")

    if privacy_style in ["sample", "hybrid"]:
        logger.info(
            f"Sample-level DP (Opacus): ENABLED (multiplier={sample_noise_multiplier}, clip={sample_max_grad_norm})"
        )
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
    ssl_config = config.get_section("ssl")
    batch_size = int(config.get("ssl.batch_size", 32))
    local_epochs = int(config.get("ssl.epochs", 2))

    # Create initial model (base UNet + SSL projection head)
    base_model = create_unet2d(
        in_channels=int(model_config.get("in_channels", 1)),
        out_channels=int(model_config.get("out_channels", 2)),
        channels=tuple(model_config.get("channels", [16, 32, 64, 128])),
        strides=tuple(model_config.get("strides", [2, 2, 2])),
        num_res_units=int(model_config.get("num_res_units", 2)),
    )

    model = SSLUNet(
        base_model,
        projection_dim=ssl_config.get("projection_dim", 128),
        hidden_dim=ssl_config.get("hidden_dim", 256),
    )

    logger.info("Initial model created")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute privacy budget if needed
    if privacy_style in ["sample", "hybrid"] and sample_noise_multiplier == 0.0:
        logger.info("Computing privacy budget for sample-level DP...")
        try:
            estimated_samples_per_client = 500

            noise_multiplier, projected_epsilon = compute_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                dataset_size=estimated_samples_per_client,
                max_grad_norm=sample_max_grad_norm,
            )
            logger.info(
                f"Computed noise_multiplier: {noise_multiplier:.4f} "
                f"(projected epsilon = {projected_epsilon:.4f})"
            )
        except ValueError as e:
            logger.error(f"Privacy budget computation failed: {e}")
            raise
    else:
        logger.info(f"Using configured noise_multiplier: {sample_noise_multiplier:.4f}")

    # Unified checkpoint directory
    checkpoint_dir = run_dir.parent / "checkpoints"

    # Check for checkpoint resumption
    resume_from = config.get("checkpointing.resume_from")
    start_round = 1
    initial_parameters = None
    remaining_rounds = num_rounds
    checkpoint_manager = None
    client_resume_states: Dict[int, ClientState] = {}
    is_mid_round_resume = False

    if resume_from:
        try:
            checkpoint_path = resolve_checkpoint_path(resume_from, run_dir.parent)
            if checkpoint_path:
                checkpoint = load_unified_checkpoint(checkpoint_path)
                logger.info(f"Resolved checkpoint: {checkpoint_path}")

                initial_parameters = ndarrays_to_parameters(
                    checkpoint.server.parameters
                )

                if checkpoint.round.status == "in_progress":
                    is_mid_round_resume = True
                    start_round = checkpoint.round.current
                    client_resume_states = checkpoint.clients
                    logger.info(
                        f"Mid-round resume: round {start_round}, "
                        f"{len(client_resume_states)} clients"
                    )
                else:
                    start_round = checkpoint.round.current + 1
                    logger.info(
                        f"Resuming from completed round {checkpoint.round.current}"
                    )

                checkpoint_manager = UnifiedCheckpointManager(
                    checkpoint_dir=checkpoint_dir,
                    run_name=run_name,
                    num_rounds=num_rounds,
                    target_delta=target_delta,
                )

                remaining_rounds = num_rounds - start_round + 1
                if remaining_rounds <= 0:
                    logger.warning(
                        f"Checkpoint is from round {checkpoint.round.current}, "
                        f"but num_rounds is {num_rounds}. Nothing to do."
                    )
                    remaining_rounds = 0

                logger.info(f"Resuming training from round {start_round}/{num_rounds}")
        except FileNotFoundError as e:
            logger.error(f"Checkpoint resolution failed:\n{e}")
            raise

    if initial_parameters is None:
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        remaining_rounds = num_rounds
        checkpoint_manager = UnifiedCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            num_rounds=num_rounds,
            target_delta=target_delta,
        )

    # Create strategy
    strategy = DPFedAvgSSL(
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
        logger.info("Wrapped strategy with Server-Side DP")

    server_config = ServerConfig(num_rounds=remaining_rounds)

    logger.info("=" * 60)
    logger.info("Server initialization complete. Starting federated SSL pretraining...")
    logger.info("=" * 60)

    return ServerAppComponents(strategy=strategy, config=server_config)
