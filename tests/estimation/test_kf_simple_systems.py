"""Tests for Kalman Filter with SimplePendulum and Cartpole systems.

Validates that the existing KalmanFilter implementation correctly estimates
states of two simple nonlinear dynamic systems linearized around equilibrium.
Tests cover convergence, partial observability, covariance consistency,
linearization quality, and statistical properties (NEES, 3-sigma bounds).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import KalmanFilter, LinearMeasurementModel, FullStateMeasurement
from fmd.simulator import SimplePendulum, Cartpole, simulate, ConstantControl
from fmd.simulator import linearize, discretize_zoh
from fmd.simulator.params import PENDULUM_1M, CARTPOLE_CLASSIC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pendulum_system_matrices(dt: float = 0.01):
    """Linearize SimplePendulum at hanging equilibrium and discretize.

    Returns (system, A_d, B_d, x_eq, u_eq).

    Because SimplePendulum has *no* control input (control_names is empty),
    ``linearize`` returns B_c of shape (2, 0).  To keep the KF loop simple
    we introduce a dummy single-column B (all zeros) so that the KF predict
    step ``x_pred = A @ x + B @ u`` works with ``u = jnp.array([0.0])``.
    """
    system = SimplePendulum(PENDULUM_1M)
    x_eq = system.default_state()          # [0, 0]
    u_eq = system.default_control()        # []

    A_c, B_c = linearize(system, x_eq, u_eq)

    # Extend B_c from (2, 0) to (2, 1) with zeros so discretize_zoh works
    n = A_c.shape[0]
    B_c_ext = jnp.zeros((n, 1))
    A_d, B_d = discretize_zoh(A_c, B_c_ext, dt)

    return system, A_d, B_d, x_eq, jnp.array([0.0])


def _cartpole_system_matrices(dt: float = 0.01):
    """Linearize Cartpole at downward equilibrium (default_state) and discretize.

    Returns (system, A_d, B_d, x_eq, u_eq).
    """
    system = Cartpole(CARTPOLE_CLASSIC)
    x_eq = system.default_state()          # [0, 0, 0, 0]  (upright / theta=0)
    u_eq = system.default_control()        # [0.0]

    A_c, B_c = linearize(system, x_eq, u_eq)
    A_d, B_d = discretize_zoh(A_c, B_c, dt)

    return system, A_d, B_d, x_eq, u_eq


def _simulate_and_extract(system, x0, dt, duration, control=None):
    """Simulate system and return states array (N+1, n)."""
    result = simulate(system, x0, dt=dt, duration=duration, control=control)
    return result.states


# ===========================================================================
# TestKFWithSimplePendulum
# ===========================================================================


class TestKFWithSimplePendulum:
    """Kalman Filter convergence tests using the SimplePendulum model."""

    def test_full_state_observation_converges(self):
        """Full-state observation (H=I) converges to true state."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        # Simulate true trajectory with small perturbation
        x0_true = jnp.array([0.1, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=2.0)
        num_steps = len(true_states) - 1  # first entry is initial state

        # Measurement model: full state
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        # KF setup
        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)  # biased initial estimate
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            # Noisy measurement of true state
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)

        # Final estimation error should be small
        error = jnp.linalg.norm(x_est - true_states[-1])
        assert error < 0.1, f"Final estimation error {error:.4f} exceeds 0.1"

    def test_position_only_observation_estimates_velocity(self):
        """Measuring only theta, the KF should estimate theta_dot."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        x0_true = jnp.array([0.15, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=2.0)
        num_steps = len(true_states) - 1

        # Only observe theta (index 0)
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.01]])

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)
        vel_errors_early = []
        vel_errors_late = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(1,)) * jnp.sqrt(R[0, 0])
            y = jnp.array([true_states[i, 0]]) + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)

            vel_err = float(jnp.abs(x_est[1] - true_states[i, 1]))
            if i <= 50:
                vel_errors_early.append(vel_err)
            if i > num_steps - 50:
                vel_errors_late.append(vel_err)

        avg_early = np.mean(vel_errors_early)
        avg_late = np.mean(vel_errors_late)

        # Velocity error should decrease over time
        assert avg_late < avg_early, (
            f"Late velocity error ({avg_late:.4f}) should be less than "
            f"early error ({avg_early:.4f})"
        )

    def test_covariance_consistency_nees(self):
        """NEES should be approximately equal to num_states over many steps."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        x0_true = jnp.array([0.1, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=2.0)
        num_steps = len(true_states) - 1

        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(123)
        nees_values = []

        # Skip initial transient (first 50 steps)
        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)

            if i > 50:
                err = true_states[i] - x_est
                # NEES = err^T @ P^{-1} @ err
                nees = float(err @ jnp.linalg.solve(P, err))
                nees_values.append(nees)

        avg_nees = np.mean(nees_values)
        # Average NEES should be close to n (within generous bounds)
        assert avg_nees < n * 3.0, (
            f"Average NEES {avg_nees:.2f} exceeds {n * 3.0:.1f} "
            f"(expected ~{n})"
        )
        assert avg_nees > n * 0.1, (
            f"Average NEES {avg_nees:.4f} suspiciously low "
            f"(expected ~{n})"
        )

    def test_higher_measurement_noise_larger_error(self):
        """Higher R should give larger estimation error but still converge."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        x0_true = jnp.array([0.1, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=2.0)
        num_steps = len(true_states) - 1

        H = jnp.eye(n)

        def run_kf(R_scale, seed):
            R = jnp.eye(n) * R_scale
            kf = KalmanFilter()
            Q = jnp.eye(n) * 0.001 * dt
            x_est = jnp.zeros(n)
            P = jnp.eye(n) * 1.0

            key = jax.random.PRNGKey(seed)
            errors = []
            for i in range(1, num_steps + 1):
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(
                    jnp.diag(R)
                )
                y = true_states[i] + noise
                x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)
                errors.append(float(jnp.linalg.norm(x_est - true_states[i])))
            return np.mean(errors[-50:])

        err_low_noise = run_kf(0.001, 42)
        err_high_noise = run_kf(0.1, 42)

        # Both should converge (final error reasonable)
        assert err_high_noise < 1.0, f"High noise did not converge: {err_high_noise:.4f}"
        # Higher noise should produce larger error
        assert err_high_noise > err_low_noise, (
            f"High noise error ({err_high_noise:.4f}) should exceed "
            f"low noise error ({err_low_noise:.4f})"
        )

    def test_steady_state_covariance_reached(self):
        """Covariance P should converge to steady state after many steps."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        x0_true = jnp.array([0.1, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=2.0)
        num_steps = len(true_states) - 1

        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)
        P_traces = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)
            P_traces.append(float(jnp.trace(P)))

        # Last 50 traces should be nearly constant
        last_50 = np.array(P_traces[-50:])
        rel_range = (last_50.max() - last_50.min()) / last_50.mean()
        assert rel_range < 0.01, (
            f"Covariance trace not converged: relative range = {rel_range:.6f}"
        )


# ===========================================================================
# TestKFWithCartpole
# ===========================================================================


class TestKFWithCartpole:
    """Kalman Filter convergence tests using the Cartpole model."""

    def test_full_state_near_upright_equilibrium(self):
        """Full-state KF tracks cartpole near upright equilibrium."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _cartpole_system_matrices(dt)
        n = 4

        # Small perturbation from upright equilibrium
        x0_true = jnp.array([0.05, 0.0, 0.05, 0.0])
        true_states = _simulate_and_extract(
            system, x0_true, dt, duration=0.5,
            control=ConstantControl(u_eq),
        )
        num_steps = len(true_states) - 1

        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)

        error = jnp.linalg.norm(x_est - true_states[-1])
        assert error < 0.3, f"Final estimation error {error:.4f} exceeds 0.3"

    def test_partial_observation_position_angle(self):
        """Observing only x and theta, KF estimates x_dot and theta_dot."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _cartpole_system_matrices(dt)
        n = 4

        x0_true = jnp.array([0.05, 0.0, 0.05, 0.0])
        true_states = _simulate_and_extract(
            system, x0_true, dt, duration=0.5,
            control=ConstantControl(u_eq),
        )
        num_steps = len(true_states) - 1

        # Observe only position (x) and angle (theta) - indices 0 and 2
        H = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        R = jnp.eye(2) * 0.01

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)
        vel_errors = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(2,)) * jnp.sqrt(jnp.diag(R))
            y = H @ true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)
            # Track velocity estimation error (x_dot and theta_dot)
            vel_err = jnp.linalg.norm(
                jnp.array([x_est[1] - true_states[i, 1],
                           x_est[3] - true_states[i, 3]])
            )
            vel_errors.append(float(vel_err))

        # Velocity error in last quarter should be smaller than first quarter
        quarter = len(vel_errors) // 4
        avg_early = np.mean(vel_errors[:quarter])
        avg_late = np.mean(vel_errors[-quarter:])
        assert avg_late < avg_early, (
            f"Velocity estimation did not improve: "
            f"early={avg_early:.4f}, late={avg_late:.4f}"
        )

    def test_estimation_error_within_3sigma(self):
        """Estimation errors within 3*sqrt(P_diag) for >90% of time steps."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _cartpole_system_matrices(dt)
        n = 4

        x0_true = jnp.array([0.02, 0.0, 0.02, 0.0])
        true_states = _simulate_and_extract(
            system, x0_true, dt, duration=1.0,
            control=ConstantControl(u_eq),
        )
        num_steps = len(true_states) - 1

        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        kf = KalmanFilter()
        Q = jnp.eye(n) * 0.001 * dt
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)
        within_bounds_count = 0
        total_checked = 0

        # Skip initial transient (first 20 steps)
        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)

            if i > 20:
                total_checked += 1
                err = jnp.abs(true_states[i] - x_est)
                sigma_3 = 3.0 * jnp.sqrt(jnp.diag(P))
                if bool(jnp.all(err < sigma_3)):
                    within_bounds_count += 1

        fraction = within_bounds_count / total_checked
        assert fraction > 0.90, (
            f"Only {fraction * 100:.1f}% of estimates within 3-sigma "
            f"(expected >90%)"
        )


# ===========================================================================
# TestKFLinearizationQuality
# ===========================================================================


class TestKFLinearizationQuality:
    """Tests verifying linearization and discretization quality."""

    def test_pendulum_linearization_valid_small_angles(self):
        """Pendulum A matrix has expected structure at hanging equilibrium.

        Linearization of theta_ddot = -(g/L)*sin(theta) around theta=0 gives:
            A[0,1] = 1.0  (d(theta_dot)/d(theta_dot))
            A[1,0] = -g/L (d(theta_ddot)/d(theta))
        """
        system = SimplePendulum(PENDULUM_1M)
        x_eq = system.default_state()
        u_eq = system.default_control()

        A_c, B_c = linearize(system, x_eq, u_eq)

        g = PENDULUM_1M.g
        L = PENDULUM_1M.length

        # A[0,0] = 0, A[0,1] = 1 (theta_dot term in d(theta)/dt)
        assert float(A_c[0, 0]) == pytest.approx(0.0, abs=1e-10)
        assert float(A_c[0, 1]) == pytest.approx(1.0, abs=1e-10)

        # A[1,0] = -g/L (linearized gravity restoring torque)
        assert float(A_c[1, 0]) == pytest.approx(-g / L, abs=1e-8)
        assert float(A_c[1, 1]) == pytest.approx(0.0, abs=1e-10)

    def test_cartpole_linearization_valid_near_equilibrium(self):
        """Cartpole A and B at upright equilibrium are non-zero and finite."""
        system = Cartpole(CARTPOLE_CLASSIC)
        x_eq = system.default_state()
        u_eq = system.default_control()

        A_c, B_c = linearize(system, x_eq, u_eq)

        # A should be finite
        assert jnp.all(jnp.isfinite(A_c)), "A_c contains non-finite values"
        assert jnp.all(jnp.isfinite(B_c)), "B_c contains non-finite values"

        # A should not be all zeros (dynamics are non-trivial)
        assert float(jnp.linalg.norm(A_c)) > 0.1, "A_c is essentially zero"
        # B should not be all zeros (system is actuated)
        assert float(jnp.linalg.norm(B_c)) > 0.01, "B_c is essentially zero"

        # Expected shapes
        assert A_c.shape == (4, 4)
        assert B_c.shape == (4, 1)

    def test_discretization_preserves_stability(self):
        """Eigenvalues of A_d should be inside unit circle for stable equilibrium.

        The pendulum at theta=0 (hanging) is a stable equilibrium, so the
        continuous eigenvalues have negative real parts. After discretization
        with small dt, discrete eigenvalues should be inside the unit circle.
        """
        dt = 0.01
        _, A_d, B_d, _, _ = _pendulum_system_matrices(dt)

        eigenvalues = jnp.linalg.eigvals(A_d)
        magnitudes = jnp.abs(eigenvalues)

        assert jnp.all(magnitudes <= 1.0 + 1e-10), (
            f"Eigenvalue magnitudes {magnitudes} exceed unit circle"
        )

    def test_kf_with_zero_process_noise(self):
        """With Q=0, KF should still converge from measurements."""
        dt = 0.01
        system, A_d, B_d, x_eq, u_eq = _pendulum_system_matrices(dt)
        n = 2

        x0_true = jnp.array([0.1, 0.0])
        true_states = _simulate_and_extract(system, x0_true, dt, duration=1.0)
        num_steps = len(true_states) - 1

        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01
        Q = jnp.zeros((n, n))  # Zero process noise

        kf = KalmanFilter()
        x_est = jnp.zeros(n)
        P = jnp.eye(n) * 1.0

        key = jax.random.PRNGKey(42)
        P_traces = []

        for i in range(1, num_steps + 1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * jnp.sqrt(jnp.diag(R))
            y = true_states[i] + noise
            x_est, P = kf.step(x_est, P, A_d, B_d, Q, u_eq, y, H, R)
            P_traces.append(float(jnp.trace(P)))

        # Covariance should monotonically decrease (no Q to add uncertainty)
        P_traces = np.array(P_traces)
        assert P_traces[-1] < P_traces[0], (
            f"Covariance did not decrease: initial={P_traces[0]:.6f}, "
            f"final={P_traces[-1]:.6f}"
        )
        # Should converge close to zero
        assert P_traces[-1] < 0.1, (
            f"Final covariance trace {P_traces[-1]:.6f} too large for Q=0"
        )
