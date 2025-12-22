"""Privacy budget accounting for federated learning with differential privacy."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List
from opacus.accountants import RDPAccountant

logger = logging.getLogger(__name__)


@dataclass
class PrivacyMetrics:
    """Privacy metadata for a single round to support RDP composition."""

    round_num: int
    # Sample-level (local) parameters
    noise_multiplier_sample: float
    sample_rate_sample: float
    steps_sample: int
    # User-level (global) parameters
    noise_multiplier_user: float
    sample_rate_user: float
    steps_user: int

    num_samples: int  # Total training samples in this round


class PrivacyAccountant:
    """Track privacy budget consumption across federated rounds.

    Uses Renyi Differential Privacy (RDP) for tight composition bounds.
    Tracks both sample-level (Opacus) and user-level (Server) privacy.
    """

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.rounds: List[PrivacyMetrics] = []

        # Internal Opacus accountants for RDP composition
        self.sample_accountant = RDPAccountant()
        self.user_accountant = RDPAccountant()

    def record_round(
        self,
        round_num: int,
        noise_multiplier_sample: float,
        sample_rate_sample: float,
        steps_sample: int,
        noise_multiplier_user: float,
        sample_rate_user: float,
        steps_user: int,
        num_samples: int,
    ) -> None:
        """Record privacy metrics for a round and update RDP history."""
        metrics = PrivacyMetrics(
            round_num=round_num,
            noise_multiplier_sample=noise_multiplier_sample,
            sample_rate_sample=sample_rate_sample,
            steps_sample=steps_sample,
            noise_multiplier_user=noise_multiplier_user,
            sample_rate_user=sample_rate_user,
            steps_user=steps_user,
            num_samples=num_samples,
        )
        self.rounds.append(metrics)

        # Update sample-level accountant
        if noise_multiplier_sample > 0:
            self.sample_accountant.history.append(
                (noise_multiplier_sample, sample_rate_sample, steps_sample)
            )

        # Update user-level accountant
        if noise_multiplier_user > 0:
            self.user_accountant.history.append(
                (noise_multiplier_user, sample_rate_user, steps_user)
            )

        logger.info(
            f"Round {round_num} recorded: "
            f"cumulative_sample_eps={self.get_cumulative_sample_epsilon():.4f}, "
            f"cumulative_user_eps={self.get_cumulative_user_epsilon():.4f} "
            f"(δ={self.target_delta})"
        )

    def get_cumulative_sample_epsilon(self) -> float:
        """Get cumulative sample-level epsilon using RDP composition."""
        if not self.sample_accountant.history:
            return 0.0
        return self.sample_accountant.get_epsilon(delta=self.target_delta)

    def get_cumulative_user_epsilon(self) -> float:
        """Get cumulative user-level epsilon using RDP composition."""
        if not self.user_accountant.history:
            return 0.0
        return self.user_accountant.get_epsilon(delta=self.target_delta)

    def get_summary(self) -> Dict:
        """Get summary of privacy consumption."""
        return {
            "target_delta": self.target_delta,
            "cumulative_sample_epsilon": self.get_cumulative_sample_epsilon(),
            "cumulative_user_epsilon": self.get_cumulative_user_epsilon(),
            "num_rounds": len(self.rounds),
            "history": [
                {
                    "round": r.round_num,
                    "sample": {
                        "noise": r.noise_multiplier_sample,
                        "rate": r.sample_rate_sample,
                        "steps": r.steps_sample,
                    },
                    "user": {
                        "noise": r.noise_multiplier_user,
                        "rate": r.sample_rate_user,
                        "steps": r.steps_user,
                    },
                }
                for r in self.rounds
            ],
        }

    def save(self, path: str) -> None:
        """Save privacy log to JSON file."""
        summary = self.get_summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Privacy log saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PrivacyAccountant":
        """Load privacy accountant from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        accountant = cls(target_delta=data["target_delta"])
        for entry in data.get("history", []):
            accountant.record_round(
                round_num=entry["round"],
                noise_multiplier_sample=entry["sample"]["noise"],
                sample_rate_sample=entry["sample"]["rate"],
                steps_sample=entry["sample"]["steps"],
                noise_multiplier_user=entry["user"]["noise"],
                sample_rate_user=entry["user"]["rate"],
                steps_user=entry["user"]["steps"],
                num_samples=0,  # Not critical for RDP calculation
            )
        return accountant
