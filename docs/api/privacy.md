# Privacy API

The Privacy API provides differential privacy budget accounting using Renyi Differential Privacy (RDP) for tight composition bounds.

## Overview

The `PrivacyAccountant` class tracks privacy budget consumption across federated learning rounds, supporting:

- **Sample-level DP**: Privacy from local DP-SGD training (Opacus)
- **User-level DP**: Privacy from client sampling (optional)
- **RDP composition**: Tight privacy bounds using Renyi divergence
- **Partial round tracking**: Mid-round privacy accounting for checkpointing

## Main Components

::: dp_fedmed.privacy.accountant.PrivacyAccountant
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Dataclasses

::: dp_fedmed.privacy.accountant.PrivacyMetrics
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: dp_fedmed.privacy.accountant.PartialRoundState
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Privacy Tracking

```python
from dp_fedmed.privacy import PrivacyAccountant

# Create accountant with target delta
accountant = PrivacyAccountant(target_delta=1e-5)

# Record privacy metrics for each round
for round_num in range(1, 11):
    accountant.record_round(
        round_num=round_num,
        # Sample-level DP (from client training)
        noise_multiplier_sample=1.0,
        sample_rate_sample=0.1,  # Batch sampling rate
        steps_sample=100,        # DP-SGD steps
        # User-level DP (from client sampling, optional)
        noise_multiplier_user=0.0,
        sample_rate_user=0.0,
        steps_user=0,
        # Metadata
        num_samples=1000,
    )

    # Get cumulative privacy budget
    sample_eps = accountant.get_cumulative_sample_epsilon()
    user_eps = accountant.get_cumulative_user_epsilon()
    print(f"Round {round_num}: ε_sample = {sample_eps:.4f}, ε_user = {user_eps:.4f}")
```

### Privacy Budget Summary

```python
# Get complete privacy summary
summary = accountant.get_summary()

# Summary contains:
# {
#     "target_delta": 1e-5,
#     "cumulative_sample_epsilon": 8.5,
#     "cumulative_user_epsilon": 0.0,
#     "num_rounds": 10,
#     "history": [
#         {
#             "round": 1,
#             "sample": {"noise": 1.0, "rate": 0.1, "steps": 100},
#             "user": {"noise": 0.0, "rate": 0.0, "steps": 0}
#         },
#         # ... more rounds
#     ]
# }

print(f"Final privacy guarantee: (ε={summary['cumulative_sample_epsilon']:.2f}, δ={summary['target_delta']})")
```

### Saving and Loading

```python
# Save privacy log to JSON
accountant.save("privacy_log.json")

# Load from saved log
loaded_accountant = PrivacyAccountant.load("privacy_log.json")

# Verify state
assert loaded_accountant.get_cumulative_sample_epsilon() == accountant.get_cumulative_sample_epsilon()
```

### Partial Round Tracking

For mid-round checkpointing, track partial privacy consumption:

```python
# Client completes 3 out of 5 epochs before crash
accountant.record_partial_progress(
    round_num=5,
    noise_multiplier_sample=1.0,
    sample_rate_sample=0.1,
    steps_sample=60,  # 3/5 of 100 steps
)

# Get partial epsilon (not yet committed)
sample_eps, user_eps = accountant.get_total_epsilon_with_partial()
print(f"Epsilon including partial: {sample_eps:.4f}")

# Save partial state in checkpoint
partial_state = accountant.get_partial_state()
# checkpoint.privacy.partial_round_epsilon = partial_state.partial_sample_epsilon

# After round completes successfully
accountant.finalize_partial_round()  # Commits to history

# Or clear if round is restarted
accountant.clear_partial_state()
```

### Restoring Partial State

```python
# Resume from checkpoint with partial round
accountant = PrivacyAccountant(target_delta=1e-5)

# Restore completed rounds from checkpoint
for round_metrics in checkpoint.privacy_history:
    accountant.record_round(...)

# Restore partial round state
if checkpoint.privacy.partial_round_epsilon > 0:
    accountant.restore_partial_state(
        round_num=checkpoint.round.current,
        sample_history=checkpoint.privacy.sample_history,
        user_history=checkpoint.privacy.user_history,
        partial_sample_epsilon=checkpoint.privacy.partial_round_epsilon,
        partial_user_epsilon=0.0,
    )
```

## Privacy Guarantees

### Sample-Level Differential Privacy

**Definition**: Each training sample's contribution is privatized via DP-SGD.

**Mechanism**:
1. **Gradient clipping**: Per-sample gradients clipped to max_grad_norm
2. **Noise addition**: Gaussian noise ~ N(0, σ²) added to clipped gradients
3. **Privacy cost**: Computed using RDP accountant

**Parameters**:
- `noise_multiplier`: σ = noise_multiplier × max_grad_norm
- `sample_rate`: Probability of sampling each example in a batch
- `steps`: Number of DP-SGD iterations

**Epsilon calculation**:
```
ε = RDP(α) - log(δ) / (α - 1)
```
where α is the Renyi order, optimized by Opacus.

### User-Level Differential Privacy

**Definition**: Each client's participation is privatized via server-side noise.

**Mechanism**:
1. **Client sampling**: Random subset of clients selected per round
2. **Gradient clipping**: Aggregated client updates clipped (optional)
3. **Noise addition**: Gaussian noise added to global model update

**Parameters**:
- `noise_multiplier_user`: Server-side noise scale
- `sample_rate_user`: Fraction of clients participating
- `steps_user`: Number of aggregation steps (typically 1 per round)

### Hybrid Privacy

When both sample-level and user-level DP are enabled:

- **Total epsilon** = ε_sample + ε_user (conservative upper bound)
- **Protects against**: Individual sample leakage AND client membership leakage

## RDP Composition

The accountant uses **Renyi Differential Privacy** for composition:

**Advantages over basic composition**:
- Tighter bounds: O(√T) vs O(T) for T rounds
- Optimal conversion to (ε, δ)-DP
- Moment accountant compatibility (Opacus)

**Privacy loss**:
```
RDP_α(M) = 1/(α-1) × log(E[(P/Q)^α])
```

**Composition**:
```
RDP_α(M₁ ∘ M₂ ∘ ... ∘ M_T) = Σ RDP_α(M_i)
```

**Conversion to (ε, δ)-DP**:
```
ε = min_α { RDP_α(M) + log(1/δ) / (α - 1) }
```

## Integration with Server Strategy

The `DPFedAvg` strategy automatically integrates with `PrivacyAccountant`:

```python
# In aggregate_fit()
accountant.record_round(
    round_num=server_round,
    noise_multiplier_sample=self.noise_multiplier,  # Pre-computed
    sample_rate_sample=avg_sample_rate,             # From clients
    steps_sample=avg_steps,                         # From clients
    noise_multiplier_user=self.user_noise_multiplier,
    sample_rate_user=len(results) / self.total_clients,
    steps_user=1 if self.user_noise_multiplier > 0 else 0,
    num_samples=total_samples,
)
```

See [Server API](server.md) for details.

## See Also

- [Server API](server.md) - Server-side privacy accounting integration
- [Client API](client.md) - Client-side DP-SGD implementation
- [Privacy Accounting Protocol](../protocol/privacy_accounting.md) - Detailed privacy flow
- [Types of DP](../types_of_dp.md) - Sample vs User vs Hybrid DP
