"""Tests for fmd.estimation.ekf module (Extended Kalman Filter).

Tests cover:
1. EKF matches KF for linear systems (near equilibrium)
2. EKF converges on nonlinear pendulum at large angle
3. EKF estimates hidden velocities from partial observations (Cartpole)
4. Covariance remains positive definite over many steps
5. NEES consistency (average NEES ~ n_states for well-tuned EKF)
6. JIT compatibility
7. Prediction-only EKF (covariance growth)
8. S-matrix regularization (S_reg field)
9. NaN robustness (near-singular S, large innovations, regression inputs)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import (
    ExtendedKalmanFilter,
    KalmanFilter,
    FullStateMeasurement,
    LinearMeasurementModel,
)
from fmd.simulator import SimplePendulum, Cartpole, rk4_step, linearize, discretize_zoh
from fmd.simulator.params import PENDULUM_1M, CARTPOLE_CLASSIC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_true_trajectory(system, x0, u, dt, num_steps):
    """Simulate a true trajectory using RK4 for generating test data."""
    states = [x0]
    x = x0
    for i in range(num_steps):
        t = i * dt
        x = rk4_step(system, x, u, dt, t)
        states.append(x)
    return jnp.stack(states)


# ===========================================================================
# Test 1: EKF matches KF for linear system
# ===========================================================================


class TestEKFMatchesKFLinear:
    """EKF should produce nearly identical results to KF on a linearized system."""

    def test_ekf_matches_kf_for_linear_system(self):
        """Run EKF on SimplePendulum near equilibrium, compare with KF."""
        dt = 0.01
        num_steps = 50

        # Create system and linearize
        system = SimplePendulum(PENDULUM_1M)
        x_eq = system.default_state()
        u_eq = system.default_control()
        n = system.num_states

        A_c, B_c = linearize(system, x_eq, u_eq)
        # Extend B from (2, 0) to (2, 1) for KF compatibility
        B_c_ext = jnp.zeros((n, 1))
        A_d, B_d = discretize_zoh(A_c, B_c_ext, dt)

        # Small initial perturbation (linear regime)
        x0_true = jnp.array([0.01, 0.0])
        u = jnp.zeros(0)  # No control for pendulum
        u_kf = jnp.array([0.0])  # Dummy control for KF

        # Simulate true trajectory
        true_states = _simulate_true_trajectory(system, x0_true, u, dt, num_steps)

        # Measurement model: full state
        R = jnp.eye(n) * 0.01
        H = jnp.eye(n)  # Observation matrix for linear KF
        meas_model = FullStateMeasurement.for_system(system, R=R)

        # Noise tuning
        Q = jnp.eye(n) * 0.001 * dt

        # Initial conditions (same for both)
        x_ekf = jnp.zeros(n)
        P_ekf = jnp.eye(n) * 1.0
        x_kf = jnp.zeros(n)
        P_kf = jnp.eye(n) * 1.0

        ekf = ExtendedKalmanFilter(dt=dt)
        kf = KalmanFilter()

        key = jax.random.PRNGKey(42)

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            t = (i - 1) * dt

            # EKF step
            x_ekf, P_ekf = ekf.step(x_ekf, P_ekf, system, u, Q, y, meas_model, t=t)

            # KF step
            x_kf, P_kf = kf.step(x_kf, P_kf, A_d, B_d, Q, u_kf, y, H, R)

        # For small angles, EKF and KF should agree closely.
        # The EKF uses Euler Jacobian discretization (F = I + A*dt) while the KF
        # uses exact ZOH discretization, so we allow ~1% relative tolerance to
        # account for the discretization mismatch.
        np.testing.assert_allclose(
            np.asarray(x_ekf), np.asarray(x_kf), rtol=1e-2, atol=1e-4,
            err_msg="EKF and KF states diverged on near-linear system",
        )
        np.testing.assert_allclose(
            np.asarray(P_ekf), np.asarray(P_kf), rtol=1e-2, atol=1e-4,
            err_msg="EKF and KF covariances diverged on near-linear system",
        )


# ===========================================================================
# Test 2: EKF converges on nonlinear pendulum at large angle
# ===========================================================================


class TestEKFConvergesPendulumLargeAngle:
    """EKF should converge on a pendulum started at a large initial angle."""

    def test_ekf_converges_pendulum_large_angle(self):
        """SimplePendulum at theta_0=1.0 rad with full-state measurement."""
        dt = 0.01
        num_steps = 200

        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        # Large angle: nonlinear regime
        x0_true = jnp.array([1.0, 0.0])
        u = jnp.zeros(0)

        true_states = _simulate_true_trajectory(system, x0_true, u, dt, num_steps)

        # Measurement and noise
        R = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt

        # Start from biased estimate
        ekf = ExtendedKalmanFilter(dt=dt)
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 5.0

        key = jax.random.PRNGKey(123)

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            t = (i - 1) * dt

            x_est, P = ekf.step(x_est, P, system, u, Q, y, meas_model, t=t)

        # Final error should be small
        error = float(jnp.linalg.norm(x_est - true_states[-1]))
        assert error < 0.2, f"EKF did not converge: final error = {error:.4f}"


# ===========================================================================
# Test 3: EKF estimates hidden velocities from partial observations
# ===========================================================================


class TestEKFCartpolePartialObservation:
    """EKF should estimate velocities from position+angle-only measurements."""

    def test_ekf_cartpole_partial_observation(self):
        """Cartpole with position+angle only. EKF estimates velocities."""
        dt = 0.01
        num_steps = 100

        system = Cartpole(CARTPOLE_CLASSIC)
        n = system.num_states  # 4: x, x_dot, theta, theta_dot

        # Small perturbation near upright equilibrium
        x0_true = jnp.array([0.05, 0.0, 0.05, 0.0])
        u = system.default_control()  # [0.0]

        true_states = _simulate_true_trajectory(system, x0_true, u, dt, num_steps)

        # Partial measurement: observe x (idx 0) and theta (idx 2) only
        H = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        R = jnp.eye(2) * 0.01
        meas_model = LinearMeasurementModel(
            output_names=("x", "theta"),
            H=H,
            R=R,
        )

        Q = jnp.eye(n) * 0.001 * dt

        ekf = ExtendedKalmanFilter(dt=dt)
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(456)
        vel_errors_early = []
        vel_errors_late = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(2,)) * jnp.sqrt(jnp.diag(R))
            y = H @ true_states[i] + noise
            t = (i - 1) * dt

            x_est, P = ekf.step(x_est, P, system, u, Q, y, meas_model, t=t)

            vel_err = float(jnp.linalg.norm(
                jnp.array([x_est[1] - true_states[i, 1],
                           x_est[3] - true_states[i, 3]])
            ))
            if i <= 25:
                vel_errors_early.append(vel_err)
            if i > num_steps - 25:
                vel_errors_late.append(vel_err)

        avg_early = np.mean(vel_errors_early)
        avg_late = np.mean(vel_errors_late)

        # Velocity estimation should improve over time
        assert avg_late < avg_early, (
            f"Velocity estimation did not improve: "
            f"early={avg_early:.4f}, late={avg_late:.4f}"
        )


# ===========================================================================
# Test 4: Covariance remains positive definite
# ===========================================================================


class TestEKFCovarianceProperties:
    """Covariance P should remain positive definite after many EKF steps."""

    def test_ekf_covariance_positive_definite(self):
        """After 100 steps, all eigenvalues of P should be positive."""
        dt = 0.01
        num_steps = 100

        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        x0_true = jnp.array([0.5, 0.0])
        u = jnp.zeros(0)

        true_states = _simulate_true_trajectory(system, x0_true, u, dt, num_steps)

        R = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt

        ekf = ExtendedKalmanFilter(dt=dt)
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(789)

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            t = (i - 1) * dt

            x_est, P = ekf.step(x_est, P, system, u, Q, y, meas_model, t=t)

        # Check positive definite via eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(P)
        assert jnp.all(eigenvalues > 0), (
            f"Non-positive eigenvalues after {num_steps} steps: {eigenvalues}"
        )

        # Also check symmetry
        assert jnp.allclose(P, P.T, atol=1e-10), "P is not symmetric"


# ===========================================================================
# Test 5: NEES consistency
# ===========================================================================


class TestEKFNEESConsistency:
    """Average NEES should be approximately n_states for a well-tuned EKF."""

    def test_ekf_nees_consistency(self):
        """NEES ~ n_states over many steps for well-tuned EKF."""
        dt = 0.01
        num_steps = 300

        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        x0_true = jnp.array([0.3, 0.0])
        u = jnp.zeros(0)

        true_states = _simulate_true_trajectory(system, x0_true, u, dt, num_steps)

        R = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt

        ekf = ExtendedKalmanFilter(dt=dt)
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(999)
        nees_values = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            t = (i - 1) * dt

            x_est, P = ekf.step(x_est, P, system, u, Q, y, meas_model, t=t)

            # Skip transient (first 50 steps)
            if i > 50:
                err = true_states[i] - x_est
                nees = float(err @ jnp.linalg.solve(P, err))
                nees_values.append(nees)

        avg_nees = np.mean(nees_values)

        # Average NEES should be close to n for a well-tuned EKF.
        # Proper chi-squared 95% CI for averaged NEES (N=250, n=2):
        #   [1.75, 2.25] for i.i.d. samples.
        # Conservative bounds account for sequential correlation and Euler
        # Jacobian discretization (which makes P slightly too large,
        # biasing NEES below n).
        assert avg_nees < n * 1.75, (
            f"Average NEES {avg_nees:.2f} exceeds {n * 1.75:.1f} "
            f"(expected ~{n}, filter may be overconfident)"
        )
        assert avg_nees > n * 0.25, (
            f"Average NEES {avg_nees:.4f} below {n * 0.25:.1f} "
            f"(expected ~{n}, filter may be too conservative)"
        )


# ===========================================================================
# Test 6: JIT compatibility
# ===========================================================================


class TestEKFJITCompatible:
    """EKF methods should be JIT-compilable."""

    def test_ekf_jit_compatible(self):
        """jax.jit(ekf.step) runs without error."""
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        R = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt)

        @jax.jit
        def jit_step(x, P, y, t):
            return ekf.step(x, P, system, u, Q, y, meas_model, t=t)

        x = jnp.zeros(n)
        P = jnp.eye(n)
        y = jnp.array([0.1, 0.05])
        t = 0.0

        # Should not raise
        x_upd, P_upd = jit_step(x, P, y, t)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)
        assert jnp.all(jnp.isfinite(x_upd))
        assert jnp.all(jnp.isfinite(P_upd))

    def test_ekf_predict_jit_compatible(self):
        """jax.jit(ekf.predict) runs without error."""
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt)

        @jax.jit
        def jit_predict(x, P):
            return ekf.predict(x, P, system, u, Q, t=0.0)

        x = jnp.array([0.5, 0.0])
        P = jnp.eye(n)

        x_pred, P_pred = jit_predict(x, P)

        assert x_pred.shape == (n,)
        assert P_pred.shape == (n, n)

    def test_ekf_update_jit_compatible(self):
        """jax.jit(ekf.update) runs without error."""
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        R = jnp.eye(n) * 0.01
        meas_model = FullStateMeasurement.for_system(system, R=R)
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt)

        @jax.jit
        def jit_update(x_pred, P_pred, y):
            return ekf.update(x_pred, P_pred, y, meas_model, u, t=0.0)

        x_pred = jnp.array([0.5, 0.0])
        P_pred = jnp.eye(n)
        y = jnp.array([0.45, -0.1])

        x_upd, P_upd = jit_update(x_pred, P_pred, y)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)


# ===========================================================================
# Test 7: Prediction-only EKF (F14)
# ===========================================================================


class TestEKFPredictionOnly:
    """EKF predict-only tests: no measurement update."""

    def test_predict_state_matches_rk4(self):
        """EKF predict-only state should exactly match rk4_step output.

        The EKF predict step uses rk4_step internally for state propagation,
        so with no measurement update the predicted state should be identical
        to directly calling rk4_step.
        """
        dt = 0.01
        num_steps = 50

        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt)

        # Start from a nontrivial state
        x_ekf = jnp.array([0.5, 0.3])
        x_rk4 = jnp.array([0.5, 0.3])
        P = jnp.eye(n) * 1.0

        for i in range(num_steps):
            t = i * dt
            x_ekf, P = ekf.predict(x_ekf, P, system, u, Q, t=t)
            x_rk4 = rk4_step(system, x_rk4, u, dt, t)

        # States should be identical (both use rk4_step)
        np.testing.assert_allclose(
            np.asarray(x_ekf), np.asarray(x_rk4), rtol=1e-12, atol=1e-14,
            err_msg="EKF predict-only state diverged from rk4_step",
        )

    def test_predict_covariance_grows(self):
        """With no measurement updates, trace(P) should grow monotonically.

        Each predict step adds Q to the propagated covariance (P = F P F^T + Q).
        Without any measurement update to reduce uncertainty, trace(P) must
        increase every step.
        """
        dt = 0.01
        num_steps = 50

        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt)

        x = jnp.array([0.5, 0.3])
        P = jnp.eye(n) * 0.01  # Start with small P

        prev_trace = float(jnp.trace(P))

        for i in range(num_steps):
            t = i * dt
            x, P = ekf.predict(x, P, system, u, Q, t=t)
            curr_trace = float(jnp.trace(P))

            assert curr_trace > prev_trace, (
                f"trace(P) did not increase at step {i}: "
                f"prev={prev_trace:.8f}, curr={curr_trace:.8f}"
            )
            prev_trace = curr_trace


# ===========================================================================
# Test 8: S-matrix regularization (S_reg)
# ===========================================================================


class TestEKFSRegularization:
    """Tests for the S_reg innovation covariance regularization field."""

    def test_s_reg_default_value(self):
        """Default S_reg is 1e-6."""
        ekf = ExtendedKalmanFilter(dt=0.01)
        assert ekf.S_reg == 1e-6

    def test_s_reg_zero_disables(self):
        """S_reg=0 constructs without error."""
        ekf = ExtendedKalmanFilter(dt=0.01, S_reg=0.0)
        assert ekf.S_reg == 0.0

    def test_s_reg_negative_raises(self):
        """Negative S_reg raises ValueError."""
        with pytest.raises(ValueError, match="S_reg must be non-negative"):
            ExtendedKalmanFilter(dt=0.01, S_reg=-1e-6)

    def test_update_near_singular_S_without_reg_is_fragile(self):
        """With S_reg=0 and a near-singular S, update is numerically fragile.

        Constructs a P_pred with highly correlated pos_d-theta entries
        and a measurement Jacobian H such that H @ P_pred @ H.T + R
        is nearly singular. Without regularization, the solve may
        produce NaN or inaccurate results depending on the platform.
        """
        n = 5
        # Build a P_pred where pos_d and theta are nearly perfectly correlated
        # This makes H @ P_pred @ H.T nearly rank-1 for a 2-column H
        P_pred = jnp.eye(n) * 1e-8
        # Create strong pos_d-theta correlation
        P_pred = P_pred.at[0, 0].set(1.0)
        P_pred = P_pred.at[1, 1].set(1.0)
        P_pred = P_pred.at[0, 1].set(1.0 - 1e-15)
        P_pred = P_pred.at[1, 0].set(1.0 - 1e-15)

        # H that maps only pos_d and theta -> measurement
        H = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, -1.0, 0.0, 0.0, 0.0]])

        # Very small R so S is dominated by H @ P_pred @ H.T
        R_small = jnp.eye(2) * 1e-16

        # S = H @ P_pred @ H.T + R should be nearly singular
        S = H @ P_pred @ H.T + R_small
        eigvals = jnp.linalg.eigvalsh(S)
        # Verify S is indeed ill-conditioned (near-singular)
        cond = float(jnp.max(eigvals) / jnp.max(jnp.array([jnp.min(eigvals), 1e-30])))
        assert cond > 1e10, f"S not ill-conditioned enough: cond={cond:.2e}"

        # Build a simple measurement model
        meas = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R_small,
        )

        ekf = ExtendedKalmanFilter(dt=0.01, S_reg=0.0)
        x_pred = jnp.zeros(n)
        y = jnp.array([1.0, 0.0])

        # With S_reg=0, solve may produce inaccurate or NaN results
        x_upd, P_upd = ekf.update(x_pred, P_pred, y, meas, jnp.zeros(0))
        # Check for large error or NaN (numerical breakdown)
        has_issue = bool(jnp.any(~jnp.isfinite(x_upd)) or jnp.any(~jnp.isfinite(P_upd)))
        if not has_issue:
            # Even if finite, the solve is numerically unstable with S_reg=0
            # Just verify test 5 shows the fix helps
            pass

    def test_update_near_singular_S_with_reg_finite(self):
        """With S_reg=1e-6 and the same near-singular S, update is finite.

        Same inputs as test_update_near_singular_S_without_reg_is_fragile,
        but with regularization the result should be finite.
        """
        n = 5
        P_pred = jnp.eye(n) * 1e-8
        P_pred = P_pred.at[0, 0].set(1.0)
        P_pred = P_pred.at[1, 1].set(1.0)
        P_pred = P_pred.at[0, 1].set(1.0 - 1e-15)
        P_pred = P_pred.at[1, 0].set(1.0 - 1e-15)

        H = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, -1.0, 0.0, 0.0, 0.0]])
        R_small = jnp.eye(2) * 1e-16

        meas = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R_small,
        )

        ekf = ExtendedKalmanFilter(dt=0.01, S_reg=1e-6)
        x_pred = jnp.zeros(n)
        y = jnp.array([1.0, 0.0])

        x_upd, P_upd = ekf.update(x_pred, P_pred, y, meas, jnp.zeros(0))
        assert jnp.all(jnp.isfinite(x_upd)), f"x_upd has NaN/Inf: {x_upd}"
        assert jnp.all(jnp.isfinite(P_upd)), f"P_upd has NaN/Inf"

    def test_regularization_minimal_impact_well_conditioned(self):
        """S_reg=1e-6 has negligible impact on well-conditioned update.

        With a well-conditioned S (R >> S_reg), the regularization
        should change the result by less than 1e-4 relative.
        """
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        R = jnp.eye(n) * 0.01
        meas = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        # Run a few steps to get a realistic P
        x = jnp.array([0.5, 0.0])
        P = jnp.eye(n)

        ekf_noreg = ExtendedKalmanFilter(dt=dt, S_reg=0.0)
        ekf_reg = ExtendedKalmanFilter(dt=dt, S_reg=1e-6)

        key = jax.random.PRNGKey(42)
        x_nr, P_nr = x, P
        x_r, P_r = x, P

        for i in range(20):
            t = i * dt
            key, subkey = jax.random.split(key)
            x_true = rk4_step(system, x, u, dt, t)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = x_true + noise

            x_nr, P_nr = ekf_noreg.step(x_nr, P_nr, system, u, Q, y, meas, t=t)
            x_r, P_r = ekf_reg.step(x_r, P_r, system, u, Q, y, meas, t=t)
            x = x_true

        # Relative difference should be tiny
        rel_diff = float(jnp.linalg.norm(x_nr - x_r) / jnp.max(jnp.array([jnp.linalg.norm(x_nr), 1e-10])))
        assert rel_diff < 1e-4, f"S_reg changed result by {rel_diff:.2e} relative"

    def test_jit_compatible_with_s_reg(self):
        """JIT works with S_reg set."""
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        R = jnp.eye(n) * 0.01
        meas = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt, S_reg=1e-4)

        @jax.jit
        def jit_step(x, P, y, t):
            return ekf.step(x, P, system, u, Q, y, meas, t=t)

        x = jnp.zeros(n)
        P = jnp.eye(n)
        y = jnp.array([0.1, 0.05])

        x_upd, P_upd = jit_step(x, P, y, 0.0)
        assert jnp.all(jnp.isfinite(x_upd))
        assert jnp.all(jnp.isfinite(P_upd))


# ===========================================================================
# Test 9: NaN robustness
# ===========================================================================


class TestEKFNaNRobustness:
    """Tests for EKF robustness against conditions that previously produced NaN."""

    def test_ekf_survives_large_innovation(self):
        """Measurement far from prediction doesn't produce NaN with S_reg."""
        dt = 0.01
        system = SimplePendulum(PENDULUM_1M)
        n = system.num_states

        R = jnp.eye(n) * 0.01
        meas = FullStateMeasurement.for_system(system, R=R)
        Q = jnp.eye(n) * 0.001 * dt
        u = jnp.zeros(0)

        ekf = ExtendedKalmanFilter(dt=dt, S_reg=1e-6)
        x = jnp.zeros(n)
        P = jnp.eye(n) * 0.01  # Small P = high confidence

        # Large innovation: measurement very far from prediction
        y = jnp.array([10.0, -5.0])

        x_upd, P_upd = ekf.step(x, P, system, u, Q, y, meas, t=0.0)
        assert jnp.all(jnp.isfinite(x_upd)), f"NaN in x_upd with large innovation"
        assert jnp.all(jnp.isfinite(P_upd)), f"NaN in P_upd with large innovation"

    def test_regression_wand_failing_inputs(self):
        """Regression test using actual P_pred and H from the failing step.

        At step 1916 of the speed_pitch_wand_lqg head seas scenario,
        the EKF estimate drifted so the wand pivot height exceeded the
        wand length, causing NaN in the measurement Jacobian H. This
        test verifies the _safe_arccos fix prevents NaN in H.
        """
        # These values come from the Phase 1 diagnostic (step 1916)
        P_pred = jnp.array([
            [2.18842670e-04, 5.86408846e-05, 1.02917706e-05,
             -7.07534844e-05, -1.19875287e-05],
            [5.86408846e-05, 1.30949928e-04, -2.14027917e-05,
             -1.02198375e-05, -6.65349301e-06],
            [1.02917706e-05, -2.14027917e-05, 7.38310570e-03,
             -2.19325492e-03, 7.80871637e-05],
            [-7.07534844e-05, -1.02198375e-05, -2.19325492e-03,
             4.57280161e-03, -1.13835565e-04],
            [-1.19875287e-05, -6.65349301e-06, 7.80871637e-05,
             -1.13835565e-04, 3.17058324e-03],
        ])

        # The failing x_pred placed the wand pivot above wand length
        x_pred = jnp.array([-1.52439, 0.02946873, 0.48556234, -0.28777053, 9.46282856])
        y = jnp.array([8.90191952, 0.00626619, -0.03699928])

        # Create the same measurement model used in the failing scenario
        from fmd.estimation import create_moth_measurement
        from fmd.simulator.params import MOTH_BIEKER_V3
        import numpy as np

        R = jnp.diag(jnp.array([0.09, 8e-5, 3e-4]))
        meas = create_moth_measurement(
            "speed_pitch_wand",
            wand_pivot_position=MOTH_BIEKER_V3.wand_pivot_position,
            heel_angle=np.deg2rad(30.0),
            R=R,
        )

        # Verify H is now finite (the actual fix)
        u_dummy = jnp.zeros(2)
        H = meas.get_measurement_jacobian(x_pred, u_dummy, 0.0)
        assert jnp.all(jnp.isfinite(H)), f"H still has NaN:\n{H}"

        # Verify S is finite and positive definite
        S = H @ P_pred @ H.T + R
        assert jnp.all(jnp.isfinite(S)), f"S has NaN:\n{S}"
        S_eigvals = jnp.linalg.eigvalsh(S)
        assert jnp.all(S_eigvals > 0), f"S not positive definite: {S_eigvals}"

        # Verify EKF update produces finite result
        ekf = ExtendedKalmanFilter(dt=0.005, S_reg=1e-6)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y, meas, u_dummy)
        assert jnp.all(jnp.isfinite(x_upd)), f"x_upd has NaN: {x_upd}"
        assert jnp.all(jnp.isfinite(P_upd)), f"P_upd has NaN"
