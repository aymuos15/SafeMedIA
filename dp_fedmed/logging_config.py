"""Centralized logging configuration using loguru."""

import shutil
import sys
from pathlib import Path

from loguru import logger


def setup_logging(run_name: str, level: str = "INFO", role: str = "server") -> Path:
    """Configure loguru for a training run.

    Creates hierarchical directory structure:
    results/<run_name>/<role>/

    Args:
        run_name: Config name (e.g., "default")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        role: Either "server" or "client_X" where X is partition ID

    Returns:
        Path to the role directory (results/<run_name>/<role>/)
    """
    # Remove default handler
    logger.remove()

    # Create hierarchical directory: results/<config_name>/<role>/
    run_dir = Path("./results") / run_name / role
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Single train.log file (no rotation)
    logger.add(
        run_dir / "train.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        enqueue=True,
    )

    logger.info("=" * 60)
    logger.info("DP-FedMed Logging Initialized")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir.absolute()}")
    logger.info(f"Role: {role}")
    logger.info(f"Log level: {level}")

    return run_dir


def get_logger(name: str = __name__):
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)
