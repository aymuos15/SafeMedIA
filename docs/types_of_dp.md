# Types of Differential Privacy (DP)

This project implements two layers of Differential Privacy in a Federated Learning context.

## 1. Sample-level DP (Local)
* **Goal**: Protect individual data points (e.g., a single medical image).
* **Implementation**: Client-side using **Opacus**.
* **Mechanism**: Clips gradients and adds noise to model updates before they leave the client.

## 2. User-level DP (Global)
* **Goal**: Protect the participation of the user/client itself.
* **Implementation**: Server-side using **Flower**.
* **Mechanism**: Clips aggregated model updates and adds noise to mask the contribution of any single client.

## 3. Hybrid DP
* **Goal**: Maximum privacy protection.
* **Implementation**: Combines both Sample-level and User-level DP.
* **Accounting**: Tracked independently but composed using Renyi Differential Privacy (RDP).

## Practical Considerations for User-Level DP

### Minimum Client Requirements

**TL;DR: User-level DP requires ≥10 clients for reasonable utility. With 2 clients, expect poor convergence.**

#### Why Client Count Matters

The noise added for user-level DP is inversely proportional to the number of sampled clients:

```
σ = (noise_multiplier × clipping_norm) / num_sampled_clients
```

**Impact comparison:**
- **2 clients**: 5× more noise than 10 clients
- **10 clients**: Minimum for meaningful results
- **100+ clients**: Recommended in literature [1]

#### Mathematical vs. Practical Guarantees

- **Mathematical**: DP guarantees (ε, δ) are **valid** with any number of clients ≥1
- **Practical**: With <10 clients, noise overwhelms signal → poor model utility

The formal differential privacy definition holds regardless of cohort size. However, the privacy-utility tradeoff becomes increasingly unfavorable as the number of clients decreases. With very few clients, the noise required to protect individual participation is so large relative to the signal that the model may fail to converge or achieve meaningful accuracy.

#### Recommendations by Use Case

| Use Case | Min Clients | Privacy Style | Notes |
|----------|-------------|---------------|-------|
| Testing/Development | 2-5 | Any | Warning expected, low utility |
| Research/Experiments | 10-50 | `user` or `hybrid` | Minimum for meaningful results |
| Production | 100+ | `user` or `hybrid` | Literature recommendation |
| Small cohorts (<10) | N/A | `sample` | Switch to sample-level DP instead |

#### When You See the Warning

If you see: `⚠ UNSAFE CONFIGURATION: min_fit_clients (N) is less than 10`

**What it means:**
- DP guarantees are still mathematically valid
- Model utility will be significantly degraded
- Convergence may be very slow or fail entirely

**What to do:**
- **For testing**: Proceed anyway, but expect poor results
- **For production**: Increase `min_fit_clients` to ≥10, preferably ≥100
- **Can't add clients**: Switch to `privacy.style = "sample"` instead

**Example adjustment** for better utility with user-level DP:
```toml
[federated]
num_clients = 10        # Increased from 2
min_fit_clients = 10    # Increased from 2

[privacy]
style = "user"

[privacy.user]
noise_multiplier = 0.5
max_grad_norm = 0.1
```

#### References

[1] McMahan et al. (2017): "Learning Differentially Private Recurrent Language Models", ICLR 2018. Experiments used 100-1250 clients per round.

[2] Geyer et al. (2017): "Differentially Private Federated Learning: A Client Level Perspective", NeurIPS 2017 Workshop.
