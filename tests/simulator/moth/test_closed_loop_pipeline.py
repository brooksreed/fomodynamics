"""Tests for closed-loop pipeline infrastructure.

Tests the generic pipeline with the Moth system, result shapes,
and sense-before-dynamics ordering.
"""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import create_moth_measurement, ExtendedKalmanFilter
from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.closed_loop_pipeline import (
    ClosedLoopResult,
    simulate_closed_loop,
)
from fmd.simulator.controllers import LQRController
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.sensors import MeasurementSensor


@pytest.fixture(scope="module")
def pipeline_setup():
    """Shared setup for pipeline tests: LQR design + pipeline components."""
    lqr_result = design_moth_lqr(u_forward=10.0)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))

    meas_model = create_moth_measurement("full_state")
    sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)

    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
    ekf = ExtendedKalmanFilter(dt=0.005)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf, num_controls=2)

    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    controller = LQRController(
        K=jnp.array(lqr_result.K),
        x_trim=jnp.array(lqr_result.trim.state),
        u_trim=jnp.array(lqr_result.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    P0 = jnp.eye(5) * 0.1

    return moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model


class TestClosedLoopResult:
    """Tests for ClosedLoopResult shapes and contents."""

    def test_result_shapes(self, pipeline_setup):
        """Result arrays have correct shapes."""
        moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model = pipeline_setup
        trim = lqr_result.trim
        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.03)
        x0_est = jnp.array(trim.state)

        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_true,
            x0_est=x0_est,
            P0=P0,
            dt=0.005,
            duration=1.0,
            rng_key=jax.random.PRNGKey(42),
            params=MOTH_BIEKER_V3,
            measurement_model=meas_model,
            trim_state=jnp.array(trim.state),
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
        )

        n_steps = int(1.0 / 0.005)
        assert isinstance(result, ClosedLoopResult)
        assert result.times.shape == (n_steps,)
        assert result.true_states.shape == (n_steps + 1, 5)
        assert result.est_states.shape == (n_steps + 1, 5)
        assert result.controls.shape == (n_steps, 2)
        assert result.covariance_traces.shape == (n_steps + 1,)
        assert result.covariance_diagonals.shape == (n_steps + 1, 5)
        assert result.estimation_errors.shape == (n_steps + 1, 5)
        assert result.measurements_clean.shape == (n_steps, 5)
        assert result.measurements_noisy.shape == (n_steps, 5)
        assert result.innovations.shape == (n_steps, 5)
        assert result.params is MOTH_BIEKER_V3
        assert result.force_log is not None
        assert result.trim_state is not None
        assert result.trim_control is not None

    def test_result_type(self, pipeline_setup):
        """Result is a ClosedLoopResult instance."""
        moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model = pipeline_setup
        trim = lqr_result.trim
        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.03)
        x0_est = jnp.array(trim.state)

        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_true,
            x0_est=x0_est,
            P0=P0,
            dt=0.005,
            duration=0.5,
            rng_key=jax.random.PRNGKey(42),
            u_trim=jnp.array(trim.control),
        )

        assert isinstance(result, ClosedLoopResult)


@pytest.mark.slow
class TestClosedLoopEquivalence:
    """Test that simulate_closed_loop stabilizes the Moth system."""

    def test_stabilizes_from_perturbation(self, pipeline_setup):
        """Pipeline stabilizes Moth from heave perturbation."""
        moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model = pipeline_setup
        trim = lqr_result.trim
        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.03)
        x0_est = jnp.array(trim.state)

        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_true,
            x0_est=x0_est,
            P0=P0,
            dt=0.005,
            duration=5.0,
            rng_key=jax.random.PRNGKey(42),
            params=MOTH_BIEKER_V3,
            measurement_model=meas_model,
            trim_state=jnp.array(trim.state),
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
        )

        trim_state = np.array(trim.state)
        final_error = np.abs(result.true_states[-1] - trim_state)
        # Looser tolerance due to one-dt shift in ordering
        assert final_error[0] < 0.08, f"pos_d error {final_error[0]:.4f} > 0.08"
        assert final_error[1] < 0.03, f"theta error {final_error[1]:.4f} > 0.03"

    def test_w_true_shape_validation(self, pipeline_setup):
        """W_true with wrong shape raises ValueError."""
        moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model = pipeline_setup
        trim = lqr_result.trim
        x0_true = jnp.array(trim.state)
        x0_est = jnp.array(trim.state)

        with pytest.raises(ValueError, match="W_true must be"):
            simulate_closed_loop(
                system=moth,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=x0_true,
                x0_est=x0_est,
                P0=P0,
                dt=0.005,
                duration=0.1,
                rng_key=jax.random.PRNGKey(42),
                W_true=jnp.eye(3),
                u_trim=jnp.array(trim.control),
            )

class TestClosedLoopMeasurementModelNone:
    """Test that measurement_model=None path works correctly."""

    def test_no_measurement_model(self, pipeline_setup):
        """simulate_closed_loop without measurement_model sets metadata to None."""
        moth, sensor, estimator, controller, lqr_result, P0, Q_ekf, meas_model = pipeline_setup
        trim = lqr_result.trim
        x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.03)
        x0_est = jnp.array(trim.state)

        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0_true,
            x0_est=x0_est,
            P0=P0,
            dt=0.005,
            duration=0.5,
            rng_key=jax.random.PRNGKey(42),
            u_trim=jnp.array(trim.control),
            # measurement_model intentionally omitted
        )

        assert result.measurement_output_names is None
        assert result.measurement_state_index_map is None
        # Simulation should still produce valid results
        assert isinstance(result, ClosedLoopResult)
        n_steps = int(0.5 / 0.005)
        assert result.times.shape == (n_steps,)
        assert result.true_states.shape == (n_steps + 1, 5)
        assert result.controls.shape == (n_steps, 2)
