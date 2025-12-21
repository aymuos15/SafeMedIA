"""Privacy budget accounting for federated learning with differential privacy."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PrivacyMetrics:
    """Privacy metrics for a single round."""

    round_num: int
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    num_samples: int


@dataclass
class PrivacyAccountant:
    """Track privacy budget consumption across federated rounds.

    Uses simple composition for epsilon accumulation.
    For tighter bounds, consider using Opacus's RDP accountant.
    """

    target_epsilon: float
    target_delta: float = 1e-5
    rounds: List[PrivacyMetrics] = field(default_factory=list)

    def record_round(
        self,
        round_num: int,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
        num_samples: int,
    ) -> None:
        """Record privacy metrics for a round."""
        metrics = PrivacyMetrics(
            round_num=round_num,
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            num_samples=num_samples,
        )
        self.rounds.append(metrics)
        logger.info(
            f"Round {round_num}: epsilon={epsilon:.4f}, "
            f"cumulative_epsilon={self.get_cumulative_epsilon():.4f}"
        )

    def get_cumulative_epsilon(self) -> float:
        """Get cumulative epsilon using simple composition.

        Note: This is a loose upper bound. For production, use
        Opacus's RDP accountant for tighter composition.
        """
        return sum(r.epsilon for r in self.rounds)

    def get_cumulative_delta(self) -> float:
        """Get cumulative delta using simple composition."""
        return sum(r.delta for r in self.rounds)

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded."""
        return self.get_cumulative_epsilon() > self.target_epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.target_epsilon - self.get_cumulative_epsilon())

    def get_summary(self) -> Dict:
        """Get summary of privacy consumption."""
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "cumulative_epsilon": self.get_cumulative_epsilon(),
            "cumulative_delta": self.get_cumulative_delta(),
            "remaining_budget": self.get_remaining_budget(),
            "budget_exceeded": self.is_budget_exceeded(),
            "num_rounds": len(self.rounds),
            "per_round_epsilon": [r.epsilon for r in self.rounds],
        }

    def save(self, path: str) -> None:
        """Save privacy log to JSON file."""
        summary = self.get_summary()
        summary["rounds"] = [
            {
                "round_num": r.round_num,
                "epsilon": r.epsilon,
                "delta": r.delta,
                "noise_multiplier": r.noise_multiplier,
                "max_grad_norm": r.max_grad_norm,
                "num_samples": r.num_samples,
            }
            for r in self.rounds
        ]

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Privacy log saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PrivacyAccountant":
        """Load privacy accountant from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        accountant = cls(
            target_epsilon=data["target_epsilon"],
            target_delta=data["target_delta"],
        )

        for r in data.get("rounds", []):
            accountant.record_round(
                round_num=r["round_num"],
                epsilon=r["epsilon"],
                delta=r["delta"],
                noise_multiplier=r["noise_multiplier"],
                max_grad_norm=r["max_grad_norm"],
                num_samples=r["num_samples"],
            )

        return accountant


class ClientPrivacyTracker:
    """Track privacy budget per client in federated setting."""

    def __init__(
        self,
        num_clients: int,
        target_epsilon: float,
        target_delta: float = 1e-5,
    ):
        self.num_clients = num_clients
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.client_accountants: Dict[int, PrivacyAccountant] = {
            i: PrivacyAccountant(target_epsilon, target_delta)
            for i in range(num_clients)
        }

    def record_client_round(
        self,
        client_id: int,
        round_num: int,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
        num_samples: int,
    ) -> None:
        """Record privacy metrics for a client's round."""
        self.client_accountants[client_id].record_round(
            round_num=round_num,
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            num_samples=num_samples,
        )

    def get_client_epsilon(self, client_id: int) -> float:
        """Get cumulative epsilon for a client."""
        return self.client_accountants[client_id].get_cumulative_epsilon()

    def is_client_budget_exceeded(self, client_id: int) -> bool:
        """Check if a client's budget is exceeded."""
        return self.client_accountants[client_id].is_budget_exceeded()

    def get_active_clients(self) -> List[int]:
        """Get list of clients that haven't exceeded their budget."""
        return [
            i for i in range(self.num_clients) if not self.is_client_budget_exceeded(i)
        ]

    def get_summary(self) -> Dict:
        """Get summary of all clients' privacy consumption."""
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "clients": {
                i: self.client_accountants[i].get_summary()
                for i in range(self.num_clients)
            },
            "active_clients": self.get_active_clients(),
        }

    def save(self, path: str) -> None:
        """Save all privacy logs to JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
