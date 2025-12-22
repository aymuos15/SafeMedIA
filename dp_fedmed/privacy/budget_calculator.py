"""Privacy budget calculator using Opacus RDPAccountant.

Computes optimal noise_multiplier to satisfy target epsilon budget.
"""

from typing import Tuple
from opacus.accountants import RDPAccountant
from loguru import logger


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    dataset_size: int,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Compute noise multiplier required to meet privacy budget.

    Uses Opacus RDPAccountant to calculate minimum noise_multiplier
    that keeps cumulative epsilon within target across all federated rounds.

    NOTE: Uses RDP composition to match the server-side accounting.
    While each client creates a fresh PrivacyEngine per round, the server
    tracks the cumulative privacy loss using a single RDP history across
    all rounds, which is much tighter than linear composition.

    Args:
        target_epsilon: Target privacy budget (e.g., 8.0)
        target_delta: Target delta for DP (e.g., 1e-5)
        num_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs per round
        batch_size: Training batch size
        dataset_size: Size of training dataset per client
        max_grad_norm: Gradient clipping threshold

    Returns:
        Tuple of (noise_multiplier, projected_epsilon)

    Raises:
        ValueError: If no feasible noise_multiplier can satisfy the budget
    """
    # Calculate training parameters
    steps_per_epoch = dataset_size // batch_size
    steps_per_round = local_epochs * steps_per_epoch
    total_steps = int(steps_per_round * num_rounds)
    sample_rate = batch_size / dataset_size

    logger.info("=" * 60)
    logger.info("Computing Privacy Budget")
    logger.info("=" * 60)
    logger.info(f"Target: ε = {target_epsilon:.2f}, δ = {target_delta:.0e}")
    logger.info(f"Training: {num_rounds} rounds × {local_epochs} epochs")
    logger.info(f"Data: batch_size={batch_size}, dataset_size={dataset_size}")
    logger.info(f"Total gradient steps: {total_steps}")
    logger.info(f"Steps per round: {steps_per_round}")
    logger.info(f"Sample rate: {sample_rate:.4f}")
    logger.info("Composition: RDP (cumulative history)")

    # Candidate noise multipliers (ascending order for best utility)
    noise_candidates = [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.7,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        7.0,
        10.0,
    ]

    best_noise = None
    best_epsilon = None

    logger.info("-" * 60)
    logger.info("Testing noise multipliers...")

    for noise in noise_candidates:
        # Create accountant
        accountant = RDPAccountant()

        # Simulate privacy consumption for ALL ROUNDS using RDP
        # Since the server tracks the cumulative history, we can
        # just take total_steps in the calculation.
        for _ in range(total_steps):
            accountant.step(noise_multiplier=noise, sample_rate=sample_rate)

        # Get total epsilon
        projected_epsilon = accountant.get_epsilon(delta=target_delta)

        logger.debug(f"  noise={noise:.2f} → projected_total={projected_epsilon:.4f}")

        # Check if this satisfies budget
        if projected_epsilon <= target_epsilon:
            best_noise = noise
            best_epsilon = projected_epsilon
            logger.info(f"✓ Found feasible solution: noise={noise:.3f}")
            logger.info(
                f"  Projected total ε: {projected_epsilon:.4f} / {target_epsilon:.4f}"
            )
            logger.info(
                f"  Privacy budget utilization: {(projected_epsilon / target_epsilon) * 100:.1f}%"
            )
            break

    if best_noise is None or best_epsilon is None:
        # Budget is infeasible - provide detailed error
        logger.error("=" * 60)
        logger.error("❌ PRIVACY BUDGET INFEASIBLE")
        logger.error("=" * 60)

        # Test with maximum noise to show what's achievable
        accountant = RDPAccountant()
        for _ in range(total_steps):
            accountant.step(noise_multiplier=10.0, sample_rate=sample_rate)
        achievable_epsilon = accountant.get_epsilon(delta=target_delta)

        error_msg = (
            f"Cannot satisfy target ε={target_epsilon:.2f} with current configuration.\n\n"
            f"Current settings:\n"
            f"  - {num_rounds} rounds × {local_epochs} epochs = {total_steps} gradient steps\n"
            f"  - batch_size={batch_size}, dataset_size={dataset_size}\n"
            f"  - sample_rate={sample_rate:.4f}\n\n"
            f"Even with maximum noise (10.0), achievable ε ≈ {achievable_epsilon:.2f}\n\n"
            f"Solutions:\n"
            f"  1. Increase target_epsilon to at least {achievable_epsilon:.1f}\n"
            f"  2. Reduce num_rounds (try {max(1, num_rounds // 2)})\n"
            f"  3. Reduce local_epochs (try {max(1, local_epochs // 2)})\n"
            f"  4. Increase batch_size (try {min(batch_size * 2, dataset_size)})\n"
        )

        raise ValueError(error_msg)

    logger.info("=" * 60)

    return float(best_noise), float(best_epsilon)


def validate_privacy_config(
    config, client_dataset_size: int = 270
) -> Tuple[float, float]:
    """Validate privacy configuration and compute optimal noise.

    Args:
        config: Configuration object
        client_dataset_size: Override dataset size (if known)

    Returns:
        Tuple of (noise_multiplier, projected_epsilon)

    Raises:
        ValueError: If budget is infeasible
    """
    # Extract parameters from config
    target_epsilon = float(config.get("privacy.target_epsilon", 8.0))
    target_delta = float(config.get("privacy.target_delta", 1e-5))
    num_rounds = int(config.get("federated.num_rounds", 5))
    local_epochs = int(config.get("training.local_epochs", 5))
    batch_size = int(config.get("data.batch_size", 8))
    max_grad_norm = float(config.get("privacy.max_grad_norm", 1.0))

    # Use provided dataset size or estimate from config
    # Note: Actual dataset size per client will be determined at runtime
    # This is a conservative estimate for validation
    if client_dataset_size is None:
        # We don't know total dataset size, so use a reasonable default
        # This should be updated if you have dataset info in config
        client_dataset_size = 270  # From your current setup

    return compute_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        dataset_size=client_dataset_size,
        max_grad_norm=max_grad_norm,
    )
