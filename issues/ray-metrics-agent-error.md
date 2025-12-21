# Ray Metrics Agent Connection Error

## Status: Known Issue (Non-Fatal)

## Error Message

```
[E] Failed to establish connection to the metrics exporter agent.
Metrics will not be exported.
Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14
```

## Impact

**This error is non-fatal.** Ray and Flower simulation will continue to work correctly. The only impact is that Prometheus metrics will not be exported.

## Root Cause

This is a known issue with Ray 2.51.1 (and some other recent versions) where the metrics agent fails to initialize properly in certain environments. The gRPC connection to the metrics exporter agent times out.

## Why We Can't Simply Downgrade Ray

Flower pins specific Ray versions:

| Flower Version | Required Ray Version |
|----------------|---------------------|
| >= 1.24.0      | ray==2.51.1         |
| 1.15.1 - 1.23.0| ray==2.31.0         |
| 1.11.0 - 1.15.0| ray==2.10.0         |

Attempting to pin a different Ray version (e.g., `ray==2.48.*`) will cause dependency resolution failures.

## Mitigations Applied

In `dp_fedmed/__init__.py`, we set environment variables to minimize Ray overhead:

```python
import os

# Configure Ray before it initializes (via Flower simulation)
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
```

In `pyproject.toml`, we configure Ray backend options:

```toml
[tool.flwr.federations.local-simulation]
options.backend.init-args.include_dashboard = false
```

These reduce but do not fully eliminate the error messages.

## Workarounds

### Option 1: Ignore the Error (Recommended)

The error is cosmetic. Your federated learning runs correctly without metrics export.

### Option 2: Pin Older Flower Version

If you need to eliminate the error entirely, pin Flower to use Ray 2.31.0:

```toml
dependencies = [
    "flwr[simulation]>=1.11.0,<1.24.0",
    ...
]
```

**Trade-off:** You lose access to newer Flower features.

### Option 3: Suppress Log Output (Not Recommended)

You could filter Ray's stderr, but this hides potentially useful errors.

## Related Links

- [GitHub: chemprop discussion #1312](https://github.com/chemprop/chemprop/discussions/1312)
- [Ray GitHub Issue #23799](https://github.com/ray-project/ray/issues/23799)
- [Ray Metrics Documentation](https://docs.ray.io/en/latest/cluster/metrics.html)

## Resolution

Waiting for upstream fix in Ray or Flower. Monitor:
- [Ray GitHub Issues](https://github.com/ray-project/ray/issues)
- [Flower GitHub Issues](https://github.com/adap/flower/issues)

## Date Logged

2025-12-21
