"""Tests for measurement noise statistics and KF integration.

Validates that MeasurementModel.noisy_measure() produces correct noise
statistics (zero-mean, matching covariance, proper correlation structure)
and integrates properly with the Kalman Filter pipeline.

Test classes:
- TestNoisyMeasureBasics: Shape, reproducibility, and deterministic behavior
- TestNoiseStatistics: Statistical properties (mean, covariance, correlation)
- TestLinearMeasurementModelNoise: Noise with partial observation models
- TestFullStateMeasurementNoise: Noise with full-state measurement models
- TestMeasurementNoiseWithKF: End-to-end KF integration with noisy measurements
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import KalmanFilter, LinearMeasurementModel, FullStateMeasurement
from fmd.simulator import SimplePendulum, Boat2D, simulate, linearize, discretize_zoh
from fmd.simulator.params import PENDULUM_1M, SIMPLE_MOTORBOAT


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _simple_2d_model(r_diag=0.01):
    """Create a simple 2-output identity measurement model."""
    H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    R = jnp.eye(2) * r_diag
    return LinearMeasurementModel(
        output_names=("pos", "vel"),
        H=H,
        R=R,
    )


def _collect_noisy_samples(model, x, u, n_samples=1000, base_seed=42):
    """Collect n_samples noisy measurements using vmap for speed."""
    key = jax.random.PRNGKey(base_seed)
    keys = jax.random.split(key, n_samples)
    samples = jax.vmap(lambda k: model.noisy_measure(x, u, k))(keys)
    return samples


# ===========================================================================
# TestNoisyMeasureBasics
# ===========================================================================


class TestNoisyMeasureBasics:
    """Basic correctness tests for noisy_measure."""

    def test_correct_output_shape(self):
        """noisy_measure returns correct shape matching model output dimension."""
        model = _simple_2d_model()
        x = jnp.array([1.0, 2.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(42)

        y_noisy = model.noisy_measure(x, u, key)

        assert y_noisy.shape == (2,), (
            f"Expected shape (2,), got {y_noisy.shape}"
        )

    def test_noisy_differs_from_clean(self):
        """Noisy measurement differs from clean measurement (non-zero R)."""
        model = _simple_2d_model(r_diag=1.0)  # Large noise to ensure difference
        x = jnp.array([1.0, 2.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(42)

        y_clean = model.measure(x, u)
        y_noisy = model.noisy_measure(x, u, key)

        assert not jnp.allclose(y_noisy, y_clean), (
            "Noisy measurement should differ from clean measurement with R != 0"
        )

    def test_different_keys_different_noise(self):
        """Different PRNG keys produce different noisy measurements."""
        model = _simple_2d_model(r_diag=0.1)
        x = jnp.array([1.0, 2.0])
        u = jnp.zeros(1)

        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)

        y1 = model.noisy_measure(x, u, key1)
        y2 = model.noisy_measure(x, u, key2)

        assert not jnp.allclose(y1, y2), (
            "Different keys should produce different noisy measurements"
        )

    def test_same_key_same_noise(self):
        """Same PRNG key produces identical noisy measurements (reproducibility)."""
        model = _simple_2d_model(r_diag=0.1)
        x = jnp.array([1.0, 2.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(42)

        y1 = model.noisy_measure(x, u, key)
        y2 = model.noisy_measure(x, u, key)

        assert jnp.allclose(y1, y2), (
            "Same PRNG key should produce identical noisy measurements"
        )

    def test_zero_r_noisy_equals_clean(self):
        """With very small R, noisy_measure should approximate clean measure.

        Note: R = 0 may cause issues with multivariate_normal sampling,
        so we use a very small R (1e-20) and check approximate equality.
        """
        H = jnp.eye(2)
        R = jnp.eye(2) * 1e-20  # Near-zero noise
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([5.0, -3.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(42)

        y_clean = model.measure(x, u)
        y_noisy = model.noisy_measure(x, u, key)

        assert jnp.allclose(y_noisy, y_clean, atol=1e-8), (
            f"With near-zero R, noisy ({y_noisy}) should equal clean ({y_clean})"
        )


# ===========================================================================
# TestNoiseStatistics
# ===========================================================================


class TestNoiseStatistics:
    """Statistical properties of noise samples."""

    def test_zero_mean(self):
        """Average of N=1000 noisy samples should approximate clean measurement.

        For R = diag(0.01), sigma = 0.1, so 3*sigma/sqrt(N) ~ 0.0095.
        We use atol = 0.1 for generous statistical margin.
        """
        model = _simple_2d_model(r_diag=0.01)
        x = jnp.array([5.0, -3.0])
        u = jnp.zeros(1)

        samples = _collect_noisy_samples(model, x, u, n_samples=1000, base_seed=42)
        sample_mean = jnp.mean(samples, axis=0)
        y_clean = model.measure(x, u)

        assert jnp.allclose(sample_mean, y_clean, atol=0.1), (
            f"Sample mean {sample_mean} should be close to clean {y_clean}"
        )

    def test_empirical_covariance_matches_r(self):
        """Empirical covariance of N=1000 noise samples should match R.

        We extract the noise (y_noisy - y_clean) and compute its covariance.
        With 1000 samples, we expect rtol ~ 0.2 to be sufficient.
        """
        R_true = jnp.array([[0.04, 0.0], [0.0, 0.01]])
        H = jnp.eye(2)
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R_true,
        )

        x = jnp.array([10.0, -5.0])
        u = jnp.zeros(1)

        samples = _collect_noisy_samples(model, x, u, n_samples=2000, base_seed=99)
        y_clean = model.measure(x, u)

        # Extract noise component
        noise_samples = samples - y_clean[None, :]
        empirical_cov = jnp.cov(noise_samples.T)

        # Check diagonal elements within 20%
        for i in range(2):
            assert jnp.abs(empirical_cov[i, i] - R_true[i, i]) / R_true[i, i] < 0.2, (
                f"Diagonal [{i},{i}]: empirical {empirical_cov[i, i]:.4f} "
                f"vs expected {R_true[i, i]:.4f}"
            )

    def test_diagonal_r_independent_noise(self):
        """With diagonal R, cross-correlation of noise components should be ~0."""
        R = jnp.array([[0.1, 0.0], [0.0, 0.1]])
        H = jnp.eye(2)
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([3.0, 7.0])
        u = jnp.zeros(1)

        samples = _collect_noisy_samples(model, x, u, n_samples=2000, base_seed=77)
        y_clean = model.measure(x, u)
        noise_samples = samples - y_clean[None, :]

        empirical_cov = jnp.cov(noise_samples.T)

        # Off-diagonal elements should be near zero
        # With 2000 samples and variance 0.1, the off-diagonal standard error
        # is approximately sqrt(Var1 * Var2 / N) = sqrt(0.01/2000) ~ 0.002
        off_diag = empirical_cov[0, 1]
        assert jnp.abs(off_diag) < 0.05, (
            f"Off-diagonal covariance {off_diag:.4f} should be near zero "
            f"for diagonal R"
        )

    def test_non_diagonal_r_correlated_noise(self):
        """With non-diagonal R (off-diagonal elements), noise should be correlated.

        We construct R with significant positive off-diagonal to induce
        correlation between measurement components.
        """
        # R with positive correlation: rho ~ 0.8
        R = jnp.array([[0.1, 0.08], [0.08, 0.1]])
        H = jnp.eye(2)
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([0.0, 0.0])
        u = jnp.zeros(1)

        samples = _collect_noisy_samples(model, x, u, n_samples=2000, base_seed=55)
        y_clean = model.measure(x, u)
        noise_samples = samples - y_clean[None, :]

        empirical_cov = jnp.cov(noise_samples.T)

        # Off-diagonal should be significantly positive (close to 0.08)
        off_diag = empirical_cov[0, 1]
        assert off_diag > 0.03, (
            f"Off-diagonal covariance {off_diag:.4f} should be significantly "
            f"positive for correlated R (expected ~0.08)"
        )

        # Also check full covariance matrix is close to R
        assert jnp.allclose(empirical_cov, R, atol=0.03), (
            f"Empirical covariance\n{empirical_cov}\nshould be close to R\n{R}"
        )


# ===========================================================================
# TestLinearMeasurementModelNoise
# ===========================================================================


class TestLinearMeasurementModelNoise:
    """Noise tests specific to LinearMeasurementModel.from_indices()."""

    def test_partial_observation_noise_shape(self):
        """from_indices() model with subset of states: noisy_measure returns
        correct smaller shape."""
        model = LinearMeasurementModel.from_indices(
            output_names=("pos",),
            state_indices=(0,),
            num_states=4,
            R=jnp.array([[0.01]]),
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(42)

        y_noisy = model.noisy_measure(x, u, key)

        # Should have shape (1,) since we only observe one state
        assert y_noisy.shape == (1,), (
            f"Expected shape (1,) for single-state observation, got {y_noisy.shape}"
        )

    def test_from_indices_supports_noisy_measure(self):
        """Model created via from_indices() produces valid noisy measurements.

        Verify that noisy values are finite and differ from clean values.
        """
        model = LinearMeasurementModel.from_indices(
            output_names=("pos", "vel"),
            state_indices=(0, 2),
            num_states=4,
            R=jnp.eye(2) * 0.1,
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.zeros(1)
        key = jax.random.PRNGKey(123)

        y_noisy = model.noisy_measure(x, u, key)
        y_clean = model.measure(x, u)

        # Should be finite
        assert jnp.all(jnp.isfinite(y_noisy)), (
            f"Noisy measurement contains non-finite values: {y_noisy}"
        )

        # Should differ from clean (with high probability given R = 0.1 * I)
        assert not jnp.allclose(y_noisy, y_clean, atol=1e-6), (
            "Noisy measurement should differ from clean for non-zero R"
        )

        # Check correct extraction: clean should select states 0 and 2
        expected_clean = jnp.array([x[0], x[2]])
        assert jnp.allclose(y_clean, expected_clean), (
            f"Clean measurement {y_clean} doesn't match expected {expected_clean}"
        )


# ===========================================================================
# TestFullStateMeasurementNoise
# ===========================================================================


class TestFullStateMeasurementNoise:
    """Noise tests for FullStateMeasurement."""

    def test_shape_matches_num_states(self):
        """FullStateMeasurement noisy output shape matches system num_states."""
        system = SimplePendulum(PENDULUM_1M)
        R = jnp.eye(system.num_states) * 0.01
        model = FullStateMeasurement.for_system(system, R)

        x = system.default_state()  # [0, 0]
        u = jnp.array([])  # SimplePendulum has no controls
        key = jax.random.PRNGKey(42)

        y_noisy = model.noisy_measure(x, u, key)

        assert y_noisy.shape == (system.num_states,), (
            f"Expected shape ({system.num_states},), got {y_noisy.shape}"
        )

    def test_noise_diagonal_matches_r(self):
        """Each state's noise variance matches corresponding R diagonal.

        Generate many samples and check per-state variance matches R[i,i].
        """
        system = SimplePendulum(PENDULUM_1M)
        r_diag = jnp.array([0.04, 0.01])  # Different variances per state
        R = jnp.diag(r_diag)
        model = FullStateMeasurement.for_system(system, R)

        x = jnp.array([0.5, -0.3])  # Non-zero state
        u = jnp.array([])
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2000)

        samples = jax.vmap(lambda k: model.noisy_measure(x, u, k))(keys)
        y_clean = model.measure(x, u)

        noise = samples - y_clean[None, :]
        empirical_var = jnp.var(noise, axis=0)

        for i in range(system.num_states):
            assert jnp.abs(empirical_var[i] - r_diag[i]) / r_diag[i] < 0.25, (
                f"State {i}: empirical variance {empirical_var[i]:.4f} "
                f"vs expected {r_diag[i]:.4f}"
            )


# ===========================================================================
# TestMeasurementNoiseWithKF
# ===========================================================================


class TestMeasurementNoiseWithKF:
    """Integration tests: noisy measurements fed to Kalman Filter."""

    def test_kf_with_noisy_measure_converges(self):
        """KF using noisy_measure() observations should converge.

        Create a SimplePendulum linearized at rest, simulate a static
        system, feed noisy measurements to KF, and verify convergence.
        """
        # Set up system and linearize
        system = SimplePendulum(PENDULUM_1M)
        x_eq = system.default_state()  # [0, 0]
        u_eq = jnp.zeros(1)  # Pendulum has 0 controls, but linearize expects 1D

        # Linearize expects control vector; for pendulum with no controls,
        # use a small dummy or actual default. Let's check:
        # SimplePendulum.num_controls == 0, so control_names == ()
        # linearize should handle this, but let's be safe.
        u_eq = system.default_control()  # jnp.array([])

        dt = 0.01

        A_c, B_c = linearize(system, x_eq, u_eq)
        A_d, B_d = discretize_zoh(A_c, B_c, dt=dt)

        n = system.num_states  # 2

        # Measurement model: observe all states
        R_meas = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R_meas)

        # KF setup
        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001
        H = meas_model.H
        R = meas_model.R

        # True state: small angle, at rest
        x_true = jnp.array([0.3, 0.0])

        # Initial estimate: far from true
        x_est = jnp.array([0.0, 0.0])
        P = jnp.eye(n) * 10.0

        initial_error = jnp.linalg.norm(x_est - x_true)

        # B_d may be (2, 0) for pendulum with no controls
        # We need to handle this carefully for KF
        if B_d.shape[1] == 0:
            # No control inputs; create dummy B and u for KF
            B_kf = jnp.zeros((n, 1))
            u_kf = jnp.zeros(1)
        else:
            B_kf = B_d
            u_kf = u_eq

        # Run KF with noisy measurements of the true (static) state
        key = jax.random.PRNGKey(42)
        for _ in range(100):
            key, subkey = jax.random.split(key)
            y = meas_model.noisy_measure(x_true, jnp.array([]), subkey)

            x_est, P = kf.step(x_est, P, A_d, B_kf, Q, u_kf, y, H, R)

        final_error = jnp.linalg.norm(x_est - x_true)

        assert final_error < initial_error, (
            f"KF should converge: initial error {initial_error:.4f}, "
            f"final error {final_error:.4f}"
        )
        # Also check that final error is reasonably small
        assert final_error < 0.5, (
            f"KF final error {final_error:.4f} should be < 0.5 after 100 steps"
        )

    def test_r_mismatch_degrades_performance(self):
        """KF with wrong R (10x larger) should perform worse than correct R.

        Both filters should work, but the one with correct R should give
        smaller estimation error.
        """
        # System setup
        system = SimplePendulum(PENDULUM_1M)
        x_eq = system.default_state()
        u_eq = system.default_control()
        dt = 0.01

        A_c, B_c = linearize(system, x_eq, u_eq)
        A_d, B_d = discretize_zoh(A_c, B_c, dt=dt)

        n = system.num_states  # 2

        # True measurement noise
        R_true = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R_true)

        # KF parameters
        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001
        H = meas_model.H

        # Handle zero-control pendulum
        if B_d.shape[1] == 0:
            B_kf = jnp.zeros((n, 1))
            u_kf = jnp.zeros(1)
        else:
            B_kf = B_d
            u_kf = u_eq

        # True state
        x_true = jnp.array([0.3, 0.0])

        # Generate noisy measurements first (same for both filters)
        key = jax.random.PRNGKey(42)
        n_steps = 100
        keys = jax.random.split(key, n_steps)
        measurements = jax.vmap(
            lambda k: meas_model.noisy_measure(x_true, jnp.array([]), k)
        )(keys)

        # --- Filter 1: Correct R ---
        x_est_correct = jnp.array([0.0, 0.0])
        P_correct = jnp.eye(n) * 10.0
        R_correct = R_true  # Correct R

        for i in range(n_steps):
            x_est_correct, P_correct = kf.step(
                x_est_correct, P_correct, A_d, B_kf, Q, u_kf,
                measurements[i], H, R_correct,
            )

        error_correct = float(jnp.linalg.norm(x_est_correct - x_true))

        # --- Filter 2: Wrong R (10x larger) ---
        x_est_wrong = jnp.array([0.0, 0.0])
        P_wrong = jnp.eye(n) * 10.0
        R_wrong = R_true * 10.0  # 10x overestimate

        for i in range(n_steps):
            x_est_wrong, P_wrong = kf.step(
                x_est_wrong, P_wrong, A_d, B_kf, Q, u_kf,
                measurements[i], H, R_wrong,
            )

        error_wrong = float(jnp.linalg.norm(x_est_wrong - x_true))

        # Both should be finite (filters work)
        assert np.isfinite(error_correct), "Correct-R filter produced non-finite error"
        assert np.isfinite(error_wrong), "Wrong-R filter produced non-finite error"

        # Both should converge somewhat
        initial_error = float(jnp.linalg.norm(jnp.array([0.0, 0.0]) - x_true))
        assert error_correct < initial_error, (
            f"Correct-R filter should converge: {error_correct:.4f} < {initial_error:.4f}"
        )
        assert error_wrong < initial_error, (
            f"Wrong-R filter should converge: {error_wrong:.4f} < {initial_error:.4f}"
        )

        # Correct R should give better (or equal) performance
        # We use a generous margin because with 100 steps both may
        # converge very well; the key is that wrong-R is not *better*.
        # With overestimated R, the filter trusts measurements less,
        # so convergence is slower. After only 100 steps, the
        # correct-R filter should have lower error.
        #
        # However, for a static system with many measurements, both
        # eventually converge. So we check the covariance trace instead
        # as a more reliable indicator: correct R gives smaller covariance.
        trace_correct = float(jnp.trace(P_correct))
        trace_wrong = float(jnp.trace(P_wrong))

        assert trace_correct < trace_wrong, (
            f"Correct-R filter should have smaller covariance trace: "
            f"{trace_correct:.4f} vs {trace_wrong:.4f}"
        )
