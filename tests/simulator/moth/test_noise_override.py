"""Tests for measurement_noise_override in the closed-loop pipeline.

Verifies that passing measurement_noise_override produces the expected
y_noisy = y_clean + override, and that two configs with the same
override see identical measurement noise.
"""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import create_moth_measurement, ExtendedKalmanFilter
from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
from fmd.simulator.controllers import LQRController
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.sensors import MeasurementSensor


@pytest.fixture(scope="module")
def lqr_design():
    """Shared LQR design for tests."""
    return design_moth_lqr(u_forward=10.0)


@pytest.fixture(scope="module")
def pipeline_components(lqr_design):
    """Build sensor/estimator/controller for full_state measurement."""
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
    meas = create_moth_measurement("full_state", num_states=moth.num_states)

    sensor = MeasurementSensor(measurement_model=meas, num_controls=2)

    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
    ekf = ExtendedKalmanFilter(dt=0.005)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)

    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    controller = LQRController(
        K=jnp.array(lqr_design.K),
        x_trim=jnp.array(lqr_design.trim.state),
        u_trim=jnp.array(lqr_design.trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    P0 = jnp.eye(5) * 0.1
    return moth, sensor, estimator, controller, meas, P0


class TestMeasurementNoiseOverride:
    """Tests for measurement_noise_override parameter."""

    def test_override_produces_correct_noise(self, lqr_design, pipeline_components):
        """y_noisy - y_clean should equal the override noise."""
        moth, sensor, estimator, controller, meas, P0 = pipeline_components
        trim = lqr_design.trim

        dt = 0.005
        duration = 0.5
        n_steps = int(round(duration / dt))
        n_meas = meas.num_outputs

        # Generate deterministic noise override
        rng = np.random.default_rng(123)
        noise_override = rng.normal(0, 0.01, (n_steps, n_meas))
        noise_jax = jnp.array(noise_override)

        x0 = jnp.array(trim.state)
        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0,
            x0_est=x0,
            P0=P0,
            dt=dt,
            duration=duration,
            rng_key=jax.random.PRNGKey(42),
            measurement_model=meas,
            trim_state=x0,
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
            measurement_noise_override=noise_jax,
        )

        # Check that y_noisy - y_clean == override
        actual_noise = np.asarray(result.measurements_noisy) - np.asarray(result.measurements_clean)
        np.testing.assert_allclose(actual_noise, noise_override, atol=1e-6)

    def test_two_configs_same_noise(self, lqr_design, pipeline_components):
        """Two runs with same override see identical measurement noise."""
        moth, sensor, estimator, controller, meas, P0 = pipeline_components
        trim = lqr_design.trim

        dt = 0.005
        duration = 0.5
        n_steps = int(round(duration / dt))
        n_meas = meas.num_outputs

        # Shared noise
        rng = np.random.default_rng(456)
        noise_override = jnp.array(rng.normal(0, 0.01, (n_steps, n_meas)))

        x0 = jnp.array(trim.state)
        kwargs = dict(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0,
            x0_est=x0,
            P0=P0,
            dt=dt,
            duration=duration,
            measurement_model=meas,
            trim_state=x0,
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
            measurement_noise_override=noise_override,
        )

        # Run with different RNG keys (should not matter with override)
        result_a = simulate_closed_loop(rng_key=jax.random.PRNGKey(1), **kwargs)
        result_b = simulate_closed_loop(rng_key=jax.random.PRNGKey(999), **kwargs)

        noise_a = np.asarray(result_a.measurements_noisy) - np.asarray(result_a.measurements_clean)
        noise_b = np.asarray(result_b.measurements_noisy) - np.asarray(result_b.measurements_clean)

        np.testing.assert_allclose(noise_a, noise_b, atol=1e-6)

    def test_backward_compatible_without_override(self, lqr_design, pipeline_components):
        """Without override, sensor noise is random (different per seed)."""
        moth, sensor, estimator, controller, meas, P0 = pipeline_components
        trim = lqr_design.trim
        x0 = jnp.array(trim.state)

        kwargs = dict(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=x0,
            x0_est=x0,
            P0=P0,
            dt=0.005,
            duration=0.5,
            measurement_model=meas,
            trim_state=x0,
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
        )

        result_a = simulate_closed_loop(rng_key=jax.random.PRNGKey(1), **kwargs)
        result_b = simulate_closed_loop(rng_key=jax.random.PRNGKey(999), **kwargs)

        noise_a = np.asarray(result_a.measurements_noisy) - np.asarray(result_a.measurements_clean)
        noise_b = np.asarray(result_b.measurements_noisy) - np.asarray(result_b.measurements_clean)

        # With different seeds and no override, noise should differ
        assert not np.allclose(noise_a, noise_b, atol=1e-4)
