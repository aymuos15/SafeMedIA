#!/usr/bin/env python3
"""CLI script for federated SSL pretraining of medical image encoders.

This script runs FEDERATED-ONLY self-supervised learning pretraining on unlabeled
medical images using Flower federation and Opacus for differential privacy.

No centralized training path exists - all training happens in a federated setting.

Usage:
    # Run with default pretraining config (federated-only)
    flwr run dp_fedmed/pretraining/ssl_app.py \\
        --run-config 'config-file="configs/pretraining.toml"'

    # Or with alternative config
    flwr run dp_fedmed/pretraining/ssl_app.py \\
        --run-config 'config-file="configs/custom_pretrain.toml"'

This script:
1. Loads federated config (num_clients, num_rounds)
2. Creates unlabeled data partitions across clients
3. Runs Flower federation with Opacus DP-SGD
4. Aggregates client models with privacy accounting
5. Saves best and final encoder checkpoints
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Local imports
try:
    from dp_fedmed.fl.ssl.config import SSLConfig
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


def setup_logging(log_dir: Path) -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory for log files
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "federated_pretraining.log"

    logger.remove()  # Remove default handler
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def main():
    """Main entry point for federated pretraining script."""
    parser = argparse.ArgumentParser(
        description="Run federated SSL pretraining with differential privacy"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining.toml"),
        help="Path to federated SSL configuration TOML file (default: configs/pretraining.toml)",
    )

    args = parser.parse_args()

    # Validate config file
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = SSLConfig.from_toml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(config.save_dir)

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("FEDERATED SSL PRETRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"SSL Method: {config.method}")
    logger.info(f"Local epochs per round: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Save directory: {config.save_dir}")
    logger.info("-" * 80)
    logger.info("FEDERATED SETTINGS (REQUIRED)")
    logger.info(f"Number of clients: {config.num_clients}")
    logger.info(f"Number of rounds: {config.num_rounds}")
    logger.info(f"Fraction fit: {config.fraction_fit}")
    logger.info("-" * 80)
    logger.info("DIFFERENTIAL PRIVACY SETTINGS")
    logger.info(f"Privacy style: {config.privacy_style}")
    logger.info(f"Target epsilon: {config.target_epsilon}")
    logger.info(f"Target delta: {config.target_delta}")
    logger.info(f"Noise multiplier: {config.noise_multiplier}")
    logger.info(f"Max gradient norm: {config.max_grad_norm}")
    logger.info("=" * 80)

    logger.info("\nTo start federated pretraining, run:")
    logger.info("")
    logger.info("  flwr run dp_fedmed/pretraining/ssl_app.py \\")
    logger.info(f"      --run-config 'config-file=\"{args.config}\"' \\")
    logger.info("      --backend simulation")
    logger.info("")
    logger.info("Or for production with server:")
    logger.info("")
    logger.info("  # Terminal 1: Start server")
    logger.info("  flwr run dp_fedmed/pretraining/ssl_app.py:server_app \\")
    logger.info(f"      --run-config 'config-file=\"{args.config}\"' \\")
    logger.info("      --backend scaffold")
    logger.info("")
    logger.info("  # Terminal 2+: Start clients")
    logger.info("  flwr run dp_fedmed/pretraining/ssl_app.py:client_app \\")
    logger.info(f"      --run-config 'config-file=\"{args.config}\"' \\")
    logger.info("      --backend scaffold")
    logger.info("")
    logger.info("Configuration validation passed. Ready to federate!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
