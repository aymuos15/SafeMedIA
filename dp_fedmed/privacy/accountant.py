"""Privacy budget accounting for federated learning with differential privacy."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from opacus.accountants import RDPAccountant

logger = logging.getLogger(__name__)


@dataclass
class PartialRoundState:
    """Tracks partial round progress for mid-round checkpointing."""

    round_num: int
    sample_history: List[Tuple[float, float, int]]  # (sigma, q, steps)
    user_history: List[Tuple[float, float, int]]
    partial_sample_epsilon: float
    partial_user_epsilon: float


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

        # Partial round tracking for mid-round checkpointing
        self._partial_state: Optional[PartialRoundState] = None

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

    # --- Partial Round Tracking for Mid-Round Checkpointing ---

    def record_partial_progress(
        self,
        round_num: int,
        noise_multiplier_sample: float,
        sample_rate_sample: float,
        steps_sample: int,
        noise_multiplier_user: float = 0.0,
        sample_rate_user: float = 0.0,
        steps_user: int = 0,
    ) -> Tuple[float, float]:
        """Record partial progress within a round for checkpointing.

        This tracks epsilon spent so far in the current round without
        committing it to the final history.

        Args:
            round_num: Current round number
            noise_multiplier_sample: Sample-level noise multiplier
            sample_rate_sample: Sample rate for DP-SGD
            steps_sample: Number of DP-SGD steps completed
            noise_multiplier_user: User-level noise multiplier
            sample_rate_user: User sample rate
            steps_user: User-level DP steps

        Returns:
            Tuple of (partial_sample_epsilon, partial_user_epsilon)
        """
        # Build partial history
        sample_history = []
        user_history = []

        if noise_multiplier_sample > 0 and steps_sample > 0:
            sample_history.append(
                (noise_multiplier_sample, sample_rate_sample, steps_sample)
            )

        if noise_multiplier_user > 0 and steps_user > 0:
            user_history.append((noise_multiplier_user, sample_rate_user, steps_user))

        # Compute partial epsilon by creating temp accountants
        partial_sample_eps = 0.0
        if sample_history:
            temp_accountant = RDPAccountant()
            temp_accountant.history = list(self.sample_accountant.history)
            temp_accountant.history.extend(sample_history)
            full_eps = temp_accountant.get_epsilon(delta=self.target_delta)
            partial_sample_eps = full_eps - self.get_cumulative_sample_epsilon()

        partial_user_eps = 0.0
        if user_history:
            temp_accountant = RDPAccountant()
            temp_accountant.history = list(self.user_accountant.history)
            temp_accountant.history.extend(user_history)
            full_eps = temp_accountant.get_epsilon(delta=self.target_delta)
            partial_user_eps = full_eps - self.get_cumulative_user_epsilon()

        # Store partial state
        self._partial_state = PartialRoundState(
            round_num=round_num,
            sample_history=sample_history,
            user_history=user_history,
            partial_sample_epsilon=partial_sample_eps,
            partial_user_epsilon=partial_user_eps,
        )

        return partial_sample_eps, partial_user_eps

    def finalize_partial_round(self) -> None:
        """Finalize partial round by committing to history.

        Called when a round completes successfully. Converts partial
        state into committed history.
        """
        if self._partial_state is None:
            return

        # Commit sample history
        for entry in self._partial_state.sample_history:
            self.sample_accountant.history.append(entry)

        # Commit user history
        for entry in self._partial_state.user_history:
            self.user_accountant.history.append(entry)

        logger.info(
            f"Finalized round {self._partial_state.round_num}: "
            f"partial_sample_eps={self._partial_state.partial_sample_epsilon:.4f}, "
            f"partial_user_eps={self._partial_state.partial_user_epsilon:.4f}"
        )

        self._partial_state = None

    def get_partial_state(self) -> Optional[PartialRoundState]:
        """Get current partial round state for checkpointing."""
        return self._partial_state

    def restore_partial_state(
        self,
        round_num: int,
        sample_history: List[Tuple[float, float, int]],
        user_history: List[Tuple[float, float, int]],
        partial_sample_epsilon: float,
        partial_user_epsilon: float,
    ) -> None:
        """Restore partial round state from checkpoint.

        Args:
            round_num: Round number being resumed
            sample_history: Sample-level RDP history entries
            user_history: User-level RDP history entries
            partial_sample_epsilon: Epsilon spent so far in round
            partial_user_epsilon: User epsilon spent so far
        """
        self._partial_state = PartialRoundState(
            round_num=round_num,
            sample_history=sample_history,
            user_history=user_history,
            partial_sample_epsilon=partial_sample_epsilon,
            partial_user_epsilon=partial_user_epsilon,
        )

        logger.info(
            f"Restored partial state for round {round_num}: "
            f"sample_eps={partial_sample_epsilon:.4f}, "
            f"user_eps={partial_user_epsilon:.4f}"
        )

    def get_total_epsilon_with_partial(self) -> Tuple[float, float]:
        """Get total epsilon including any partial round progress.

        Returns:
            Tuple of (total_sample_epsilon, total_user_epsilon)
        """
        sample_eps = self.get_cumulative_sample_epsilon()
        user_eps = self.get_cumulative_user_epsilon()

        if self._partial_state:
            sample_eps += self._partial_state.partial_sample_epsilon
            user_eps += self._partial_state.partial_user_epsilon

        return sample_eps, user_eps

    def clear_partial_state(self) -> None:
        """Clear partial state without committing."""
        self._partial_state = None
