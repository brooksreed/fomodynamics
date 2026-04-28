"""Tests for EKFEstimator and PassthroughEstimator."""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import ExtendedKalmanFilter, create_moth_measurement
from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.estimators import EKFEstimator, PassthroughEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.params import MOTH_BIEKER_V3


@pytest.fixture(scope="module")
def estimator_setup():
    """Shared setup for estimator tests."""
    lqr_result = design_moth_lqr(u_forward=10.0)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
    meas_model = create_moth_measurement("full_state")

    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
    ekf = ExtendedKalmanFilter(dt=0.005)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf)

    return moth, estimator, lqr_result, meas_model


class TestEKFEstimator:
    """Tests for EKFEstimator state carry and convergence."""

    def test_init_state(self, estimator_setup):
        """init_state returns (x0_est, P0) tuple."""
        _, estimator, lqr_result, _ = estimator_setup
        x0 = jnp.array(lqr_result.trim.state)
        P0 = jnp.eye(5) * 0.1

        state = estimator.init_state(x0, P0)
        x_est, P = state

        np.testing.assert_array_equal(x_est, x0)
        np.testing.assert_array_equal(P, P0)

    def test_estimate_returns_correct_types(self, estimator_setup):
        """estimate returns (x_est, (x_est, P), innovation)."""
        moth, estimator, lqr_result, meas_model = estimator_setup
        trim = lqr_result.trim
        x0 = jnp.array(trim.state)
        P0 = jnp.eye(5) * 0.1

        est_state = estimator.init_state(x0, P0)
        y = meas_model.measure(x0, jnp.zeros(2), 0.0)
        u_prev = jnp.array(trim.control)

        x_est, est_state_new, innovation = estimator.estimate(
            est_state, y, u_prev, moth, 0.0
        )

        assert x_est.shape == (5,)
        assert len(est_state_new) == 2
        assert est_state_new[0].shape == (5,)
        assert est_state_new[1].shape == (5, 5)
        assert innovation.shape == (5,)

    def test_state_carry_updates(self, estimator_setup):
        """State carry is updated after each estimate step."""
        moth, estimator, lqr_result, meas_model = estimator_setup
        trim = lqr_result.trim
        x0 = jnp.array(trim.state)
        P0 = jnp.eye(5) * 0.1

        est_state = estimator.init_state(x0, P0)
        y = meas_model.measure(x0 + 0.01, jnp.zeros(2), 0.0)
        u_prev = jnp.array(trim.control)

        _, est_state_new, _ = estimator.estimate(est_state, y, u_prev, moth, 0.0)

        # State should have changed
        x_new, P_new = est_state_new
        assert not np.allclose(x_new, x0, atol=1e-10)
        assert not np.allclose(P_new, P0, atol=1e-10)

    @pytest.mark.slow
    def test_multi_step_convergence(self, estimator_setup):
        """EKF converges over multiple steps with full-state measurements."""
        moth, estimator, lqr_result, meas_model = estimator_setup
        trim = lqr_result.trim
        x_true = jnp.array(trim.state)
        x0_est = x_true + jnp.array([0.05, 0.02, 0.0, 0.0, 0.0])
        P0 = jnp.eye(5) * 0.1

        est_state = estimator.init_state(x0_est, P0)
        u = jnp.array(trim.control)
        key = jax.random.PRNGKey(42)

        errors = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            y = meas_model.noisy_measure(x_true, jnp.zeros(2), subkey, float(i) * 0.005)
            x_est, est_state, _ = estimator.estimate(est_state, y, u, moth, float(i) * 0.005)
            errors.append(float(jnp.linalg.norm(x_est - x_true)))

        # Error should decrease from early to late
        early_avg = np.mean(errors[:10])
        late_avg = np.mean(errors[-10:])
        assert late_avg < early_avg, (
            f"EKF did not converge: early avg {early_avg:.4f}, late avg {late_avg:.4f}"
        )


class TestPassthroughEstimator:
    """Tests for PassthroughEstimator state carry and mapping."""

    def test_init_state(self):
        """init_state returns (x0_est, P0) tuple."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1

        state = est.init_state(x0, P0)
        x_est, P = state

        np.testing.assert_array_equal(x_est, x0)
        np.testing.assert_array_equal(P, P0)

    def test_estimate_returns_correct_types(self):
        """estimate returns (x_est, (x_est, P), innovation)."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        y = jnp.array([0.73])
        x_est, est_state_new, innovation = est.estimate(
            est_state, y, jnp.zeros(2), None, 0.0
        )

        assert x_est.shape == (5,)
        assert len(est_state_new) == 2
        assert est_state_new[0].shape == (5,)
        assert est_state_new[1].shape == (5, 5)
        assert innovation.shape == (1,)

    def test_wand_angle_at_slot_0(self):
        """Wand angle from measurement is placed at x_est[0]."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        wand_angle = 0.73
        y = jnp.array([wand_angle])
        x_est, _, _ = est.estimate(est_state, y, jnp.zeros(2), None, 0.0)

        np.testing.assert_allclose(x_est[0], wand_angle, atol=1e-10)

    def test_remaining_slots_zero(self):
        """Slots 1-4 of pseudo-state are zeros."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.ones(5)  # Start with nonzero to verify reset
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        y = jnp.array([0.73])
        x_est, _, _ = est.estimate(est_state, y, jnp.zeros(2), None, 0.0)

        np.testing.assert_allclose(x_est[1:], 0.0, atol=1e-10)

    def test_dummy_P_carried_through(self):
        """Covariance matrix P is carried unchanged."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.42
        est_state = est.init_state(x0, P0)

        y = jnp.array([0.73])
        _, est_state_new, _ = est.estimate(est_state, y, jnp.zeros(2), None, 0.0)

        np.testing.assert_array_equal(est_state_new[1], P0)

    def test_zero_innovation(self):
        """Innovation is always zero (no model-based prediction)."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        y = jnp.array([0.73])
        _, _, innovation = est.estimate(est_state, y, jnp.zeros(2), None, 0.0)

        np.testing.assert_allclose(innovation, 0.0, atol=1e-10)

    def test_multi_step_updates(self):
        """Multiple estimate calls update slot 0 correctly each time."""
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        for angle in [0.3, 0.5, 0.7, 0.9]:
            y = jnp.array([angle])
            x_est, est_state, _ = est.estimate(
                est_state, y, jnp.zeros(2), None, 0.0
            )
            np.testing.assert_allclose(x_est[0], angle, atol=1e-10)
            np.testing.assert_allclose(x_est[1:], 0.0, atol=1e-10)

    def test_speed_pitch_wand_measurement(self):
        """With 3-element measurement, still uses y[0] for wand angle.

        This tests the raw y[0] passthrough behavior. In practice,
        PassthroughEstimator is only used with wand-only (single-element)
        measurements from MechanicalWandController, so y[0] is always
        the wand angle. This test verifies the generic slicing behavior
        with a multi-element vector.
        """
        est = PassthroughEstimator(n_states=5)
        x0 = jnp.zeros(5)
        P0 = jnp.eye(5) * 0.1
        est_state = est.init_state(x0, P0)

        y = jnp.array([10.0, 0.05, 0.73])  # [speed, pitch, wand_angle]
        x_est, _, innovation = est.estimate(
            est_state, y, jnp.zeros(2), None, 0.0
        )

        # Slot 0 gets y[0] (which is speed in this case)
        np.testing.assert_allclose(x_est[0], 10.0, atol=1e-10)
        assert innovation.shape == (3,)
