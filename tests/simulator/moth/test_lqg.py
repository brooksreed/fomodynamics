"""Tests for Moth closed-loop output-feedback controller (LQR + EKF).

Tests closed-loop stability with 3 sensor configurations:
- full_state: all 5 states observed directly
- speed_pitch_height: SOG, pitch, ride height (3 measurements)
- speed_pitch_rate_height: SOG, pitch, pitch rate, ride height (4 measurements)
"""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import create_moth_measurement, ExtendedKalmanFilter
from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.moth_3d import MAIN_FLAP_MIN, MAIN_FLAP_MAX
from fmd.simulator.closed_loop_pipeline import ClosedLoopResult, simulate_closed_loop
from fmd.simulator.controllers import LQRController
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.sensors import MeasurementSensor


@pytest.fixture(scope="module")
def lqg_setup():
    """Shared LQR design + moth instance for all LQG tests."""
    result = design_moth_lqr(u_forward=10.0)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))

    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
    P0 = jnp.eye(5) * 0.1

    return moth, result, Q_ekf, P0


def _run_closed_loop(lqg_setup, variant, duration=5.0, seed=42, W_true=None):
    """Helper to run closed-loop simulation with a given measurement variant."""
    moth, lqr_result, Q_ekf, P0 = lqg_setup
    trim = lqr_result.trim

    if variant == "full_state":
        meas_model = create_moth_measurement("full_state")
    else:
        meas_model = create_moth_measurement(
            variant,
            bowsprit_position=MOTH_BIEKER_V3.bowsprit_position,
        )

    sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)
    ekf = ExtendedKalmanFilter(dt=0.005)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf, num_controls=2)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    controller = LQRController(
        K=jnp.array(lqr_result.K),
        x_trim=jnp.array(trim.state),
        u_trim=jnp.array(trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    x0_true = jnp.array(trim.state).at[0].set(trim.state[0] + 0.03)
    x0_est = jnp.array(trim.state)

    return simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=0.005,
        duration=duration,
        rng_key=jax.random.PRNGKey(seed),
        params=MOTH_BIEKER_V3,
        measurement_model=meas_model,
        trim_state=jnp.array(trim.state),
        trim_control=jnp.array(trim.control),
        u_trim=jnp.array(trim.control),
        W_true=W_true,
    )


@pytest.mark.slow
class TestLQGFullState:
    """Full-state measurement LQG tests."""

    def test_lqg_full_state_stabilizes(self, lqg_setup, artifact_saver):
        """Full-state LQG with +0.03m heave perturbation converges to trim."""
        result = _run_closed_loop(lqg_setup, "full_state")

        artifact_saver.save("test_lqg_full_state_stabilizes", {
            "times": np.array(result.times),
            "true_states": np.array(result.true_states),
            "est_states": np.array(result.est_states),
            "controls": np.array(result.controls),
            "covariance_traces": np.array(result.covariance_traces),
            "estimation_errors": np.array(result.estimation_errors),
        }, metadata={"trim_state": np.array(lqg_setup[1].trim.state)})
        trim_state = np.array(lqg_setup[1].trim.state)

        # Final true state should be close to trim.
        # At 10 m/s, nonlinear equilibrium drift (~29mm/10s) causes a steady-state
        # pos_d offset that the linear LQR cannot fully eliminate.
        final_error = np.abs(result.true_states[-1] - trim_state)
        assert final_error[0] < 0.08, f"pos_d error {final_error[0]:.4f} > 0.08"
        assert final_error[1] < 0.03, f"theta error {final_error[1]:.4f} > 0.03"

        # Estimation error should be small.
        # Heave rate (w, index 2) is the least observable state — it accounts
        # for ~87% of EKF covariance trace — so it gets a looser tolerance.
        final_est_error = np.abs(result.estimation_errors[-1])
        per_state_tol = np.array([0.05, 0.05, 0.15, 0.05, 0.05])
        assert np.all(final_est_error < per_state_tol), (
            f"Estimation error too large: {final_est_error} vs tol {per_state_tol}"
        )

    def test_lqg_estimation_error_converges(self, lqg_setup):
        """Average estimation error norm decreases from early to late window."""
        result = _run_closed_loop(lqg_setup, "full_state")

        # Compare average of early window vs late window to smooth out noise
        err_norms = np.array([np.linalg.norm(e) for e in result.estimation_errors])
        early_avg = np.mean(err_norms[5:20])
        late_avg = np.mean(err_norms[-20:])
        assert late_avg < early_avg, (
            f"Late avg error {late_avg:.4f} >= early avg error {early_avg:.4f}"
        )


@pytest.mark.slow
class TestLQGVakaros:
    """Vakaros (3-measurement) LQG tests."""

    def test_lqg_vakaros_stabilizes(self, lqg_setup):
        """Vakaros sensor suite stabilizes with looser tolerances."""
        result = _run_closed_loop(lqg_setup, "speed_pitch_height")
        trim_state = np.array(lqg_setup[1].trim.state)

        final_error = np.abs(result.true_states[-1] - trim_state)
        # Looser tolerances for partial observation + equilibrium drift at 10 m/s
        assert final_error[0] < 0.08, f"pos_d error {final_error[0]:.4f} > 0.08"
        assert final_error[1] < 0.03, f"theta error {final_error[1]:.4f} > 0.03"


@pytest.mark.slow
class TestLQGArduPilotBase:
    """ArduPilot base (4-measurement) LQG tests."""

    def test_lqg_ardupilot_base_stabilizes(self, lqg_setup):
        """ArduPilot base (4 sensors) should be at least as good as Vakaros (3 sensors)."""
        result = _run_closed_loop(lqg_setup, "speed_pitch_rate_height")
        trim_state = np.array(lqg_setup[1].trim.state)

        final_error = np.abs(result.true_states[-1] - trim_state)
        # ArduPilot base with 4 partial measurements has more estimation noise
        # compounded with equilibrium drift at 10 m/s
        assert final_error[0] < 0.10, f"pos_d error {final_error[0]:.4f} > 0.10"
        assert final_error[1] < 0.03, f"theta error {final_error[1]:.4f} > 0.03"


@pytest.mark.slow
class TestLQGControlBounds:
    """Control saturation tests."""

    def test_lqg_control_within_bounds(self, lqg_setup):
        """All controls stay within Moth actuator limits (params-driven)."""
        result = _run_closed_loop(lqg_setup, "full_state")

        # Bounds: main flap from module constants, rudder from params
        rudder_min = float(MOTH_BIEKER_V3.rudder_elevator_min)
        rudder_max = float(MOTH_BIEKER_V3.rudder_elevator_max)

        assert np.all(result.controls[:, 0] >= MAIN_FLAP_MIN - 1e-10), (
            f"Main flap below min: {result.controls[:, 0].min()}"
        )
        assert np.all(result.controls[:, 0] <= MAIN_FLAP_MAX + 1e-10), (
            f"Main flap above max: {result.controls[:, 0].max()}"
        )
        assert np.all(result.controls[:, 1] >= rudder_min - 1e-10), (
            f"Rudder elevator below min: {result.controls[:, 1].min()}"
        )
        assert np.all(result.controls[:, 1] <= rudder_max + 1e-10), (
            f"Rudder elevator above max: {result.controls[:, 1].max()}"
        )


@pytest.mark.slow
class TestClosedLoopResultData:
    """Tests for ClosedLoopResult data structure."""

    def test_params_none_backward_compat(self, lqg_setup):
        """simulate_closed_loop works when params=None (no force extraction)."""
        moth, lqr_result, Q_ekf, P0 = lqg_setup
        trim = lqr_result.trim
        meas_model = create_moth_measurement("full_state")

        sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)
        ekf = ExtendedKalmanFilter(dt=0.005)
        estimator = EKFEstimator(ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf, num_controls=2)
        u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
        controller = LQRController(
            K=jnp.array(lqr_result.K),
            x_trim=jnp.array(trim.state),
            u_trim=jnp.array(trim.control),
            u_min=u_min,
            u_max=u_max,
        )

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
            rng_key=jax.random.PRNGKey(99),
            measurement_model=meas_model,
            trim_state=jnp.array(trim.state),
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
            # params intentionally omitted (defaults to None)
        )

        assert result.params is None
        assert result.force_log is None
        assert result.heel_angle == pytest.approx(np.deg2rad(30.0))
        # Measurement/innovation arrays should still be populated
        n_steps = int(0.5 / 0.005)
        assert result.measurements_clean.shape == (n_steps, 5)
        assert result.measurements_noisy.shape == (n_steps, 5)
        assert result.innovations.shape == (n_steps, 5)

    def test_result_shapes(self, lqg_setup):
        """Verify result array shapes are consistent."""
        result = _run_closed_loop(lqg_setup, "full_state", duration=1.0)
        n_steps = int(1.0 / 0.005)

        assert result.times.shape == (n_steps,)
        assert result.true_states.shape == (n_steps + 1, 5)
        assert result.est_states.shape == (n_steps + 1, 5)
        assert result.controls.shape == (n_steps, 2)
        assert result.covariance_traces.shape == (n_steps + 1,)
        assert result.covariance_diagonals.shape == (n_steps + 1, 5)
        assert result.estimation_errors.shape == (n_steps + 1, 5)
        assert isinstance(result, ClosedLoopResult)

        # New fields from params-enriched ClosedLoopResult
        assert result.params is MOTH_BIEKER_V3
        assert result.force_log is not None
        assert result.force_log.main_foil_force.shape == (n_steps, 3)
        assert result.heel_angle >= 0.0
        assert result.measurements_clean.shape[0] == n_steps
        assert result.measurements_noisy.shape[0] == n_steps
        assert result.innovations.shape[0] == n_steps
        # full_state measurement model has 5 outputs
        assert result.measurements_clean.shape[1] == 5
        assert result.innovations.shape[1] == 5

        # Trim state and control
        assert result.trim_state.shape == (5,)
        assert result.trim_control.shape == (2,)
        # Verify trim values match the trim point used for LQR design
        np.testing.assert_array_equal(result.trim_state, np.array(lqg_setup[1].trim.state))
        np.testing.assert_array_equal(result.trim_control, np.array(lqg_setup[1].trim.control))


@pytest.mark.slow
class TestLQGProcessNoise:
    """Tests for W_true (true process noise injection)."""

    def test_w_true_injects_noise(self, lqg_setup):
        """W_true=None vs diagonal W_true produces different true state trajectories."""
        # Run without process noise
        result_clean = _run_closed_loop(lqg_setup, "full_state", duration=1.0, seed=42)

        # Run with process noise (same seed for measurement noise)
        W_true = jnp.diag(jnp.array([1e-6, 1e-6, 1e-5, 1e-5, 1e-6]))
        result_noisy = _run_closed_loop(
            lqg_setup, "full_state", duration=1.0, seed=42, W_true=W_true
        )

        # True state trajectories must differ
        diff = np.abs(result_noisy.true_states - result_clean.true_states)
        assert np.max(diff) > 1e-6, (
            f"W_true had no effect: max state diff = {np.max(diff):.2e}"
        )

    def test_w_true_shape_validation(self, lqg_setup):
        """W_true with wrong shape raises ValueError."""
        moth, lqr_result, Q_ekf, P0 = lqg_setup
        trim = lqr_result.trim
        meas_model = create_moth_measurement("full_state")
        sensor = MeasurementSensor(measurement_model=meas_model, num_controls=2)
        ekf = ExtendedKalmanFilter(dt=0.005)
        estimator = EKFEstimator(ekf=ekf, measurement_model=meas_model, Q_ekf=Q_ekf, num_controls=2)
        u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
        controller = LQRController(
            K=jnp.array(lqr_result.K),
            x_trim=jnp.array(trim.state),
            u_trim=jnp.array(trim.control),
            u_min=u_min,
            u_max=u_max,
        )

        with pytest.raises(ValueError, match="W_true must be"):
            simulate_closed_loop(
                system=moth,
                sensor=sensor,
                estimator=estimator,
                controller=controller,
                x0_true=jnp.array(trim.state),
                x0_est=jnp.array(trim.state),
                P0=P0,
                dt=0.005,
                duration=0.1,
                rng_key=jax.random.PRNGKey(42),
                W_true=jnp.eye(3),
                u_trim=jnp.array(trim.control),
            )
