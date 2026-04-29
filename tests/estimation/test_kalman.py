"""Tests for fmd.estimation.kalman module.

Comprehensive tests for the Kalman Filter including:
- Shape verification for predict/update
- Covariance properties (symmetric, positive definite)
- Convergence behavior
- Integration with Boat2D system
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import KalmanFilter, LinearMeasurementModel
from fmd.simulator import Boat2D, simulate, ConstantControl
from fmd.simulator.params import BOAT2D_TEST_DEFAULT


class TestKalmanFilterBasics:
    """Basic shape and validity tests for KalmanFilter."""

    def test_predict_shapes(self):
        """Predict should return correctly shaped state and covariance."""
        kf = KalmanFilter()
        n, m = 4, 2  # 4 states, 2 controls

        x = jnp.zeros(n)
        P = jnp.eye(n)
        A = jnp.eye(n)
        B = jnp.zeros((n, m))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(m)

        x_pred, P_pred = kf.predict(x, P, A, B, Q, u)

        assert x_pred.shape == (n,)
        assert P_pred.shape == (n, n)

    def test_update_shapes(self):
        """Update should return correctly shaped state and covariance."""
        kf = KalmanFilter()
        n, p = 4, 2  # 4 states, 2 measurements

        x_pred = jnp.zeros(n)
        P_pred = jnp.eye(n)
        y = jnp.zeros(p)
        H = jnp.zeros((p, n))
        H = H.at[0, 0].set(1.0)  # Observe state 0
        H = H.at[1, 2].set(1.0)  # Observe state 2
        R = jnp.eye(p) * 0.01

        x_upd, P_upd = kf.update(x_pred, P_pred, y, H, R)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)

    def test_step_shapes(self):
        """Step should return correctly shaped state and covariance."""
        kf = KalmanFilter()
        n, m, p = 4, 2, 3  # 4 states, 2 controls, 3 measurements

        x = jnp.zeros(n)
        P = jnp.eye(n)
        A = jnp.eye(n)
        B = jnp.zeros((n, m))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(m)
        y = jnp.zeros(p)
        H = jnp.zeros((p, n))
        H = H.at[0, 0].set(1.0)
        H = H.at[1, 1].set(1.0)
        H = H.at[2, 2].set(1.0)
        R = jnp.eye(p) * 0.01

        x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, H, R)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)

    def test_covariance_symmetric(self):
        """Covariance matrices should remain symmetric after update."""
        kf = KalmanFilter()
        n = 4

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        P = jnp.eye(n) * 2.0  # Start with symmetric
        A = jnp.array([
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)
        y = jnp.array([1.1, 3.2])  # Noisy measurements
        H = jnp.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]])
        R = jnp.eye(2) * 0.1

        x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, H, R)

        # Check symmetry
        assert jnp.allclose(P_upd, P_upd.T, atol=1e-10)

    def test_covariance_positive_definite(self):
        """Covariance should remain positive definite after update."""
        kf = KalmanFilter()
        n = 3

        x = jnp.array([1.0, 0.5, -0.3])
        P = jnp.eye(n)
        A = jnp.array([
            [0.95, 0.1, 0.0],
            [-0.1, 0.95, 0.0],
            [0.0, 0.0, 0.9],
        ])
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.001
        u = jnp.zeros(1)
        y = jnp.array([1.05])
        H = jnp.array([[1.0, 0.0, 0.0]])
        R = jnp.array([[0.01]])

        # Run multiple steps
        for _ in range(10):
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)

        # Check positive definite via eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(P)
        assert jnp.all(eigenvalues > 0), f"Non-positive eigenvalues: {eigenvalues}"

    def test_predict_without_control(self):
        """Predict should work correctly with zero control input."""
        kf = KalmanFilter()
        n = 2

        x = jnp.array([1.0, 2.0])
        P = jnp.eye(n)
        A = jnp.array([[1.0, 0.1],
                       [0.0, 1.0]])
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)

        x_pred, P_pred = kf.predict(x, P, A, B, Q, u)

        # Verify state propagation
        expected_x = A @ x
        assert jnp.allclose(x_pred, expected_x)

        # Verify covariance propagation
        expected_P = A @ P @ A.T + Q
        assert jnp.allclose(P_pred, expected_P)

    def test_predict_with_control(self):
        """Predict should correctly incorporate control input."""
        kf = KalmanFilter()
        n, m = 2, 1

        x = jnp.array([0.0, 0.0])
        P = jnp.eye(n)
        A = jnp.eye(n)
        B = jnp.array([[0.0], [1.0]])  # Control affects state 1
        Q = jnp.eye(n) * 0.01
        u = jnp.array([5.0])

        x_pred, P_pred = kf.predict(x, P, A, B, Q, u)

        # State should be affected by control
        expected_x = A @ x + B @ u
        assert jnp.allclose(x_pred, expected_x)
        assert jnp.isclose(x_pred[1], 5.0)  # Second state should be 5.0

    def test_update_reduces_uncertainty(self):
        """Update with measurement should reduce uncertainty (in measured states)."""
        kf = KalmanFilter()
        n = 2

        x_pred = jnp.array([0.0, 0.0])
        P_pred = jnp.eye(n) * 10.0  # High initial uncertainty
        y = jnp.array([1.0])  # Measure first state
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.1]])  # Low measurement noise

        x_upd, P_upd = kf.update(x_pred, P_pred, y, H, R)

        # State should move toward measurement
        assert abs(x_upd[0] - 1.0) < abs(x_pred[0] - 1.0)

        # Uncertainty in measured state should decrease
        assert P_upd[0, 0] < P_pred[0, 0]


class TestKalmanFilterConvergence:
    """Tests for Kalman filter convergence behavior."""

    def test_converges_to_true_state(self):
        """KF should converge to true state with repeated measurements."""
        kf = KalmanFilter()
        n = 2

        # True state we're trying to estimate
        x_true = jnp.array([5.0, -3.0])

        # Initial estimate (way off)
        x = jnp.array([0.0, 0.0])
        P = jnp.eye(n) * 100.0  # Very uncertain

        # System matrices (static system)
        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.001  # Small process noise
        u = jnp.zeros(1)

        # Full state measurement with some noise
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.1

        # Generate noisy measurements and run filter
        key = jax.random.key(42)
        for _ in range(50):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * 0.316  # sqrt(0.1)
            y = x_true + noise
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)

        # Should be close to true state
        assert jnp.allclose(x, x_true, atol=0.5), f"x={x}, x_true={x_true}"

    def test_covariance_decreases(self):
        """Covariance should decrease (or stabilize) with measurements."""
        kf = KalmanFilter()
        n = 2

        x = jnp.array([0.0, 0.0])
        P = jnp.eye(n) * 10.0

        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.1

        initial_trace = jnp.trace(P)

        # Run filter with measurements
        key = jax.random.key(123)
        for _ in range(20):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, shape=(n,))
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)

        final_trace = jnp.trace(P)

        # Covariance trace should have decreased
        assert final_trace < initial_trace, \
            f"Initial trace: {initial_trace}, Final trace: {final_trace}"

    def test_reaches_steady_state_covariance(self):
        """Covariance should reach steady state for time-invariant system."""
        kf = KalmanFilter()
        n = 2

        x = jnp.zeros(n)
        P = jnp.eye(n) * 100.0

        A = jnp.array([[1.0, 0.1],
                       [0.0, 1.0]])
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)
        H = jnp.array([[1.0, 0.0]])  # Observe position only
        R = jnp.array([[0.1]])

        P_history = []
        key = jax.random.key(456)
        for _ in range(100):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, shape=(1,))
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)
            P_history.append(jnp.trace(P))

        P_history = jnp.array(P_history)

        # Last 20 values should be approximately constant (steady state)
        last_20 = P_history[-20:]
        std_last_20 = jnp.std(last_20)
        assert std_last_20 < 0.1, f"Covariance not converged: std={std_last_20}"


class TestKalmanFilterWithBoat2D:
    """Tests for KalmanFilter with Boat2D dynamics."""

    def test_kf_estimates_boat2d_states(self):
        """KF should track Boat2D states from noisy measurements."""
        # Create boat and simulate
        boat = Boat2D(BOAT2D_TEST_DEFAULT)
        dt = 0.1

        # Simple control
        control = jnp.array([50.0, 0.0])  # Thrust only, no yaw moment

        # Simulate true trajectory
        x0_true = jnp.zeros(6)
        result = simulate(
            boat, x0_true, dt=dt, duration=5.0,
            control=ConstantControl(control)
        )
        true_states = result.states

        # Linearize for KF (approximate around trajectory)
        # For simplicity, use constant linearization at rest
        # A = I + dt * Jacobian at x=0 (approximate)
        A = jnp.eye(6)
        # Position kinematics at psi=0: x_dot = u, y_dot = v
        A = A.at[0, 3].set(dt)  # dx/du = dt
        A = A.at[1, 4].set(dt)  # dy/dv = dt
        A = A.at[2, 5].set(dt)  # dpsi/dr = dt
        # Velocity dynamics (linearized)
        drag_u = BOAT2D_TEST_DEFAULT.drag_surge / BOAT2D_TEST_DEFAULT.mass
        drag_v = BOAT2D_TEST_DEFAULT.drag_sway / BOAT2D_TEST_DEFAULT.mass
        drag_r = BOAT2D_TEST_DEFAULT.drag_yaw / BOAT2D_TEST_DEFAULT.izz
        A = A.at[3, 3].set(1 - dt * drag_u)
        A = A.at[4, 4].set(1 - dt * drag_v)
        A = A.at[5, 5].set(1 - dt * drag_r)

        # Control input matrix
        B = jnp.zeros((6, 2))
        B = B.at[3, 0].set(dt / BOAT2D_TEST_DEFAULT.mass)  # thrust -> u_dot
        B = B.at[5, 1].set(dt / BOAT2D_TEST_DEFAULT.izz)   # yaw_moment -> r_dot

        # Process and measurement noise
        Q = jnp.eye(6) * 0.01
        R = jnp.eye(6) * 0.1

        # Full state measurement
        H = jnp.eye(6)

        # Initialize filter
        kf = KalmanFilter()
        x_est = jnp.zeros(6)
        P = jnp.eye(6) * 10.0  # Uncertain initial state

        # Run filter on noisy measurements
        key = jax.random.key(789)
        estimates = [x_est]

        for i in range(1, len(true_states)):
            # Generate noisy measurement
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(6,)) * 0.316
            y = true_states[i] + noise

            # Filter step
            x_est, P = kf.step(x_est, P, A, B, Q, control, y, H, R)
            estimates.append(x_est)

        estimates = jnp.stack(estimates)

        # Estimates should track true states reasonably well
        # Compare last half of trajectory (after initial convergence)
        n_skip = len(true_states) // 2
        position_error = jnp.mean(jnp.abs(estimates[n_skip:, :3] - true_states[n_skip:, :3]))
        velocity_error = jnp.mean(jnp.abs(estimates[n_skip:, 3:] - true_states[n_skip:, 3:]))

        assert position_error < 1.0, f"Position error too high: {position_error}"
        assert velocity_error < 0.5, f"Velocity error too high: {velocity_error}"

    def test_kf_with_partial_measurements(self):
        """KF should work with partial state measurements."""
        # Create boat
        boat = Boat2D(BOAT2D_TEST_DEFAULT)
        dt = 0.1

        # Linearized system (same as above)
        A = jnp.eye(6)
        A = A.at[0, 3].set(dt)
        A = A.at[1, 4].set(dt)
        A = A.at[2, 5].set(dt)
        drag_u = BOAT2D_TEST_DEFAULT.drag_surge / BOAT2D_TEST_DEFAULT.mass
        drag_v = BOAT2D_TEST_DEFAULT.drag_sway / BOAT2D_TEST_DEFAULT.mass
        drag_r = BOAT2D_TEST_DEFAULT.drag_yaw / BOAT2D_TEST_DEFAULT.izz
        A = A.at[3, 3].set(1 - dt * drag_u)
        A = A.at[4, 4].set(1 - dt * drag_v)
        A = A.at[5, 5].set(1 - dt * drag_r)

        B = jnp.zeros((6, 2))
        B = B.at[3, 0].set(dt / BOAT2D_TEST_DEFAULT.mass)
        B = B.at[5, 1].set(dt / BOAT2D_TEST_DEFAULT.izz)

        Q = jnp.eye(6) * 0.01

        # Only measure position (x, y, psi) - not velocities
        H = jnp.zeros((3, 6))
        H = H.at[0, 0].set(1.0)  # x
        H = H.at[1, 1].set(1.0)  # y
        H = H.at[2, 2].set(1.0)  # psi
        R = jnp.eye(3) * 0.01

        kf = KalmanFilter()
        x = jnp.zeros(6)
        P = jnp.eye(6)
        control = jnp.zeros(2)

        # Should work without errors
        y = jnp.array([0.1, 0.2, 0.05])  # Position measurements only
        x_upd, P_upd = kf.step(x, P, A, B, Q, control, y, H, R)

        assert x_upd.shape == (6,)
        assert P_upd.shape == (6, 6)
        assert jnp.all(jnp.isfinite(x_upd))
        assert jnp.all(jnp.isfinite(P_upd))


class TestKalmanFilterJIT:
    """Tests for JIT compatibility of KalmanFilter methods."""

    def test_predict_jit(self):
        """Predict should be JIT-compilable."""
        kf = KalmanFilter()
        n, m = 3, 2

        @jax.jit
        def jit_predict(x, P, u):
            A = jnp.eye(n)
            B = jnp.zeros((n, m))
            Q = jnp.eye(n) * 0.01
            return kf.predict(x, P, A, B, Q, u)

        x = jnp.zeros(n)
        P = jnp.eye(n)
        u = jnp.zeros(m)

        # Should not raise
        x_pred, P_pred = jit_predict(x, P, u)

        assert x_pred.shape == (n,)
        assert P_pred.shape == (n, n)

    def test_update_jit(self):
        """Update should be JIT-compilable."""
        kf = KalmanFilter()
        n, p = 3, 2

        @jax.jit
        def jit_update(x_pred, P_pred, y):
            H = jnp.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
            R = jnp.eye(p) * 0.01
            return kf.update(x_pred, P_pred, y, H, R)

        x_pred = jnp.zeros(n)
        P_pred = jnp.eye(n)
        y = jnp.zeros(p)

        # Should not raise
        x_upd, P_upd = jit_update(x_pred, P_pred, y)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)

    def test_step_jit(self):
        """Step should be JIT-compilable."""
        kf = KalmanFilter()
        n, m, p = 3, 1, 2

        @jax.jit
        def jit_step(x, P, u, y):
            A = jnp.eye(n)
            B = jnp.zeros((n, m))
            Q = jnp.eye(n) * 0.01
            H = jnp.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
            R = jnp.eye(p) * 0.01
            return kf.step(x, P, A, B, Q, u, y, H, R)

        x = jnp.zeros(n)
        P = jnp.eye(n)
        u = jnp.zeros(m)
        y = jnp.zeros(p)

        # Should not raise
        x_upd, P_upd = jit_step(x, P, u, y)

        assert x_upd.shape == (n,)
        assert P_upd.shape == (n, n)

    def test_vmap_over_initial_states(self):
        """Should be vmap-compatible over initial states."""
        kf = KalmanFilter()
        n = 2

        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.1

        def single_step(x, P, y):
            u = jnp.zeros(1)
            return kf.step(x, P, A, B, Q, u, y, H, R)

        # Batch of initial states and measurements
        batch_size = 5
        x_batch = jnp.zeros((batch_size, n))
        P_batch = jnp.tile(jnp.eye(n), (batch_size, 1, 1))
        y_batch = jnp.ones((batch_size, n))

        # vmap over batch
        x_upd_batch, P_upd_batch = jax.vmap(single_step)(x_batch, P_batch, y_batch)

        assert x_upd_batch.shape == (batch_size, n)
        assert P_upd_batch.shape == (batch_size, n, n)


class TestKalmanFilterNumericalStability:
    """Tests for numerical stability of KalmanFilter."""

    def test_joseph_form_preserves_symmetry(self):
        """Joseph form should preserve covariance symmetry under numerical stress."""
        kf = KalmanFilter()
        n = 4

        # Start with slightly asymmetric P (due to numerical errors)
        P = jnp.eye(n)
        P = P.at[0, 1].set(0.1)
        P = P.at[1, 0].set(0.1 + 1e-10)  # Tiny asymmetry

        x = jnp.zeros(n)
        A = jnp.eye(n) * 0.99
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.001
        u = jnp.zeros(1)
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.01

        key = jax.random.key(999)
        for _ in range(50):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, shape=(n,))
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)

        # Check symmetry is maintained
        assert jnp.allclose(P, P.T, atol=1e-10)

    def test_large_initial_covariance(self):
        """Should handle large initial covariance without numerical issues."""
        kf = KalmanFilter()
        n = 3

        x = jnp.zeros(n)
        P = jnp.eye(n) * 1e6  # Very large initial uncertainty

        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.1

        key = jax.random.key(111)
        for _ in range(20):
            key, subkey = jax.random.split(key)
            y = jax.random.normal(subkey, shape=(n,))
            x, P = kf.step(x, P, A, B, Q, u, y, H, R)

        # Should remain finite and well-conditioned
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(P))
        eigenvalues = jnp.linalg.eigvalsh(P)
        assert jnp.all(eigenvalues > 0)

    def test_small_R_high_confidence_measurements(self):
        """Should handle high-confidence measurements (small R)."""
        kf = KalmanFilter()
        n = 2

        x = jnp.zeros(n)
        P = jnp.eye(n)

        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01
        u = jnp.zeros(1)
        H = jnp.eye(n)
        R = jnp.eye(n) * 1e-6  # Very small measurement noise

        y = jnp.array([1.0, 2.0])
        x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, H, R)

        # State should jump nearly to measurement
        assert jnp.allclose(x_upd, y, atol=0.01)
        # Should remain well-conditioned
        assert jnp.all(jnp.isfinite(P_upd))


class TestKalmanFilterIntegrationWithMeasurementModel:
    """Tests for using KalmanFilter with MeasurementModel classes."""

    def test_with_linear_measurement_model(self):
        """KF should work with LinearMeasurementModel."""
        kf = KalmanFilter()
        n = 4

        # Create measurement model
        H = jnp.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]])
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("pos", "vel"),
            H=H,
            R=R,
        )

        # System matrices
        A = jnp.eye(n)
        B = jnp.zeros((n, 1))
        Q = jnp.eye(n) * 0.01

        x = jnp.zeros(n)
        P = jnp.eye(n)
        u = jnp.zeros(1)

        # Generate measurement from model
        x_true = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = model.measure(x_true, u)

        # Run KF update using model's H and R
        H_from_model = model.get_measurement_jacobian(x, u)
        R_from_model = model.R

        x_upd, P_upd = kf.step(x, P, A, B, Q, u, y, H_from_model, R_from_model)

        # State should move toward measurement
        assert jnp.allclose(x_upd[0], y[0], atol=0.5)
        assert jnp.allclose(x_upd[2], y[1], atol=0.5)
