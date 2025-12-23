#!/usr/bin/env python3
"""Standalone runner for federated SSL pretraining.

This script runs federated SSL pretraining without depending on 'flwr run'.
It directly uses Flower's simulation backend with the legacy API.

Usage:
    python3 run_ssl_pretraining.py [config_file]

Examples:
    python3 run_ssl_pretraining.py
    python3 run_ssl_pretraining.py configs/pretraining.toml
"""

import sys
from pathlib import Path

from loguru import logger


def run_federated_ssl(config_file: str = "configs/pretraining.toml"):
    """Run federated SSL pretraining.

    Args:
        config_file: Path to pretraining config TOML file
    """
    try:
        # Import Flower components
        from flwr.simulation import start_simulation
        from flwr.server.server_config import ServerConfig
        from dp_fedmed.fl.client.factory import (
            create_client_fn,
            TrainingMode,
        )
        from dp_fedmed.config import load_config

        config_path = Path(config_file)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_file}")
            return 1

        # Load config to verify it's valid
        config = load_config(config_file)

        logger.info("=" * 80)
        logger.info("FEDERATED SSL PRETRAINING")
        logger.info("=" * 80)
        logger.info(f"Config file: {config_file}")
        logger.info(f"Clients: {config.federated.num_clients}")
        logger.info(f"Rounds: {config.federated.num_rounds}")
        logger.info(f"Privacy style: {config.privacy.style}")
        logger.info(f"Target epsilon: {config.privacy.target_delta}")
        logger.info("=" * 80)
        logger.info("")

        # Create server config
        server_config = ServerConfig(num_rounds=config.federated.num_rounds)

        logger.info("Starting Flower simulation...")
        logger.info(f"  Number of clients: {config.federated.num_clients}")
        logger.info(f"  Number of rounds: {config.federated.num_rounds}")
        logger.info(f"  Config file: {config_file}")
        logger.info("")

        # Create client factory with config file path and SSL mode baked in
        client_fn_with_config = create_client_fn(str(config_file), TrainingMode.SSL)

        # Run simulation using legacy API (without run_config parameter)
        history = start_simulation(
            client_fn=client_fn_with_config,
            num_clients=config.federated.num_clients,
            config=server_config,
            client_resources={
                "num_cpus": config.client_resources.num_cpus,
                "num_gpus": config.client_resources.num_gpus,
            },
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
            },
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ FEDERATED SSL PRETRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        # Access losses from History object
        if hasattr(history, "losses_distributed") and history.losses_distributed:
            logger.info(f"Total rounds: {len(history.losses_distributed)}")
            logger.info(
                f"Final distributed loss: {history.losses_distributed[-1][1]:.4f}"
            )
        logger.info("Results directory: results/")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Federated SSL pretraining failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/pretraining.toml"

    try:
        exit_code = run_federated_ssl(config_file)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
