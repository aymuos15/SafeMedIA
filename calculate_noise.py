from dp_fedmed.privacy.budget_calculator import compute_noise_multiplier
import sys

try:
    noise, eps = compute_noise_multiplier(
        target_epsilon=8.0,
        target_delta=1e-5,
        num_rounds=10,
        local_epochs=5,
        batch_size=8,
        dataset_size=67,
    )
    print(f"NOISE_MULTIPLIER={noise}")
    print(f"PROJECTED_EPSILON={eps}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
