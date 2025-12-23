"""Tests for privacy accountant and budget tracking."""

from dp_fedmed.privacy.accountant import (
    PrivacyAccountant,
    PrivacyMetrics,
    PartialRoundState,
)


class TestPrivacyAccountantBasic:
    """Tests for basic PrivacyAccountant functionality."""

    def test_initialization(self):
        """Test PrivacyAccountant initialization with default delta."""
        accountant = PrivacyAccountant()
        assert accountant.target_delta == 1e-5
        assert len(accountant.rounds) == 0
        assert accountant.get_cumulative_sample_epsilon() == 0.0
        assert accountant.get_cumulative_user_epsilon() == 0.0

    def test_initialization_custom_delta(self):
        """Test PrivacyAccountant initialization with custom delta."""
        accountant = PrivacyAccountant(target_delta=1e-6)
        assert accountant.target_delta == 1e-6

    def test_record_single_round(self):
        """Test recording a single round of privacy metrics."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        accountant.record_round(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=1000,
        )

        assert len(accountant.rounds) == 1
        assert accountant.rounds[0].round_num == 1
        assert accountant.rounds[0].noise_multiplier_sample == 1.0
        # Epsilon should be positive after recording sample-level DP
        assert accountant.get_cumulative_sample_epsilon() > 0.0
        # User epsilon should be 0 since no user-level DP
        assert accountant.get_cumulative_user_epsilon() == 0.0


class TestPrivacyAccountantComposition:
    """Tests for RDP composition across multiple rounds."""

    def test_epsilon_monotonically_increases(self):
        """Test that epsilon increases monotonically across rounds."""
        accountant = PrivacyAccountant(target_delta=1e-5)
        epsilons = []

        for round_num in range(1, 6):
            accountant.record_round(
                round_num=round_num,
                noise_multiplier_sample=1.0,
                sample_rate_sample=0.01,
                steps_sample=100,
                noise_multiplier_user=0.0,
                sample_rate_user=0.0,
                steps_user=0,
                num_samples=1000,
            )
            epsilons.append(accountant.get_cumulative_sample_epsilon())

        # Check monotonically increasing
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1], (
                f"Epsilon should be monotonically increasing: "
                f"eps[{i}]={epsilons[i]} < eps[{i - 1}]={epsilons[i - 1]}"
            )

    def test_composition_with_both_dp_levels(self):
        """Test composition when both sample and user-level DP are used."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        accountant.record_round(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=1.0,
            sample_rate_user=0.1,  # 10% of users
            steps_user=1,
            num_samples=1000,
        )

        # Both epsilons should be positive
        assert accountant.get_cumulative_sample_epsilon() > 0.0
        assert accountant.get_cumulative_user_epsilon() > 0.0

    def test_noise_multiplier_effect_on_epsilon(self):
        """Test that higher noise multiplier results in lower epsilon."""
        # Lower noise (more privacy leakage)
        accountant_low_noise = PrivacyAccountant(target_delta=1e-5)
        accountant_low_noise.record_round(
            round_num=1,
            noise_multiplier_sample=0.5,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=1000,
        )

        # Higher noise (better privacy)
        accountant_high_noise = PrivacyAccountant(target_delta=1e-5)
        accountant_high_noise.record_round(
            round_num=1,
            noise_multiplier_sample=2.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=1000,
        )

        eps_low = accountant_low_noise.get_cumulative_sample_epsilon()
        eps_high = accountant_high_noise.get_cumulative_sample_epsilon()

        # Higher noise should result in lower epsilon (better privacy)
        assert eps_high < eps_low, (
            f"Higher noise should give lower epsilon: "
            f"eps_high={eps_high} should be < eps_low={eps_low}"
        )


class TestPrivacyAccountantPartialRound:
    """Tests for partial round tracking (mid-round checkpointing)."""

    def test_record_partial_progress(self):
        """Test recording partial progress within a round."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        partial_sample_eps, partial_user_eps = accountant.record_partial_progress(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=50,  # Half the steps
        )

        # Partial epsilon should be positive
        assert partial_sample_eps > 0.0
        # Partial state should be set
        partial_state = accountant.get_partial_state()
        assert partial_state is not None
        assert partial_state.round_num == 1

    def test_finalize_partial_round(self):
        """Test finalizing partial round commits to history."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        # Record partial progress
        accountant.record_partial_progress(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=50,
        )

        assert accountant.get_partial_state() is not None

        # Finalize
        accountant.finalize_partial_round()

        # Partial state should be cleared
        assert accountant.get_partial_state() is None
        # History should be committed
        assert len(accountant.sample_accountant.history) > 0

    def test_restore_partial_state(self):
        """Test restoring partial state from checkpoint."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        # Restore partial state
        accountant.restore_partial_state(
            round_num=3,
            sample_history=[(1.0, 0.01, 50)],
            user_history=[],
            partial_sample_epsilon=0.5,
            partial_user_epsilon=0.0,
        )

        state = accountant.get_partial_state()
        assert state is not None
        assert state.round_num == 3
        assert state.partial_sample_epsilon == 0.5
        assert len(state.sample_history) == 1

    def test_total_epsilon_with_partial(self):
        """Test getting total epsilon including partial progress."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        # Record a full round
        accountant.record_round(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=1000,
        )

        base_epsilon = accountant.get_cumulative_sample_epsilon()

        # Record partial progress
        accountant.record_partial_progress(
            round_num=2,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=50,
        )

        total_sample_eps, total_user_eps = accountant.get_total_epsilon_with_partial()

        # Total should include partial
        assert total_sample_eps > base_epsilon

    def test_clear_partial_state(self):
        """Test clearing partial state without committing."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        accountant.record_partial_progress(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=50,
        )

        assert accountant.get_partial_state() is not None

        accountant.clear_partial_state()

        assert accountant.get_partial_state() is None


class TestPrivacyAccountantPersistence:
    """Tests for saving and loading privacy accountant state."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading privacy accountant."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        # Record some rounds
        for round_num in range(1, 4):
            accountant.record_round(
                round_num=round_num,
                noise_multiplier_sample=1.0,
                sample_rate_sample=0.01,
                steps_sample=100,
                noise_multiplier_user=0.5,
                sample_rate_user=0.1,
                steps_user=1,
                num_samples=1000,
            )

        save_path = tmp_path / "privacy_log.json"
        accountant.save(str(save_path))

        # Load and verify
        loaded = PrivacyAccountant.load(str(save_path))

        assert loaded.target_delta == accountant.target_delta
        assert len(loaded.rounds) == len(accountant.rounds)
        # Epsilons should match
        assert (
            abs(
                loaded.get_cumulative_sample_epsilon()
                - accountant.get_cumulative_sample_epsilon()
            )
            < 1e-6
        )
        assert (
            abs(
                loaded.get_cumulative_user_epsilon()
                - accountant.get_cumulative_user_epsilon()
            )
            < 1e-6
        )

    def test_get_summary(self):
        """Test getting privacy summary."""
        accountant = PrivacyAccountant(target_delta=1e-5)

        accountant.record_round(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.0,
            sample_rate_user=0.0,
            steps_user=0,
            num_samples=1000,
        )

        summary = accountant.get_summary()

        assert "target_delta" in summary
        assert "cumulative_sample_epsilon" in summary
        assert "cumulative_user_epsilon" in summary
        assert "num_rounds" in summary
        assert "history" in summary
        assert summary["num_rounds"] == 1


class TestPrivacyMetricsDataclass:
    """Tests for PrivacyMetrics dataclass."""

    def test_privacy_metrics_creation(self):
        """Test PrivacyMetrics creation."""
        metrics = PrivacyMetrics(
            round_num=1,
            noise_multiplier_sample=1.0,
            sample_rate_sample=0.01,
            steps_sample=100,
            noise_multiplier_user=0.5,
            sample_rate_user=0.1,
            steps_user=1,
            num_samples=1000,
        )

        assert metrics.round_num == 1
        assert metrics.noise_multiplier_sample == 1.0
        assert metrics.sample_rate_sample == 0.01
        assert metrics.steps_sample == 100
        assert metrics.noise_multiplier_user == 0.5
        assert metrics.num_samples == 1000


class TestPartialRoundStateDataclass:
    """Tests for PartialRoundState dataclass."""

    def test_partial_round_state_creation(self):
        """Test PartialRoundState creation."""
        state = PartialRoundState(
            round_num=2,
            sample_history=[(1.0, 0.01, 50)],
            user_history=[],
            partial_sample_epsilon=0.3,
            partial_user_epsilon=0.0,
        )

        assert state.round_num == 2
        assert len(state.sample_history) == 1
        assert state.partial_sample_epsilon == 0.3
        assert state.partial_user_epsilon == 0.0
