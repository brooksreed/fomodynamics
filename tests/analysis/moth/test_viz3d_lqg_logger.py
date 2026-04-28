"""Smoke tests for Moth LQG Rerun .rrd writer."""

from fmd.simulator import _config  # noqa: F401

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import attrs

# Skip entire module if rerun is not installed
rerun = pytest.importorskip("rerun", reason="rerun-sdk not installed")

from fmd.estimation import create_moth_measurement, ExtendedKalmanFilter
from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.moth_forces_extract import MothForceLog
from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
from fmd.simulator.controllers import LQRController
from fmd.simulator.estimators import EKFEstimator
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.sensors import MeasurementSensor
from fmd.analysis.viz3d import write_moth_lqg_rrd


@pytest.fixture(scope="module")
def lqg_data():
    """Run a short closed-loop simulation for testing."""
    lqr = design_moth_lqr(u_forward=10.0)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
    meas = create_moth_measurement(
        "speed_pitch_height",
        bowsprit_position=MOTH_BIEKER_V3.bowsprit_position,
    )

    trim = lqr.trim
    x0_true = jnp.array(trim.state).at[0].add(0.03)
    x0_est = jnp.array(trim.state)
    P0 = jnp.eye(5) * 0.1
    Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))

    sensor = MeasurementSensor(measurement_model=meas, num_controls=2)
    ekf = ExtendedKalmanFilter(dt=0.01)
    estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)
    u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
    controller = LQRController(
        K=jnp.array(lqr.K),
        x_trim=jnp.array(trim.state),
        u_trim=jnp.array(trim.control),
        u_min=u_min,
        u_max=u_max,
    )

    result = simulate_closed_loop(
        system=moth,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
        x0_true=x0_true,
        x0_est=x0_est,
        P0=P0,
        dt=0.01,
        duration=0.1,
        rng_key=jax.random.PRNGKey(42),
        params=MOTH_BIEKER_V3,
        measurement_model=meas,
        trim_state=jnp.array(trim.state),
        trim_control=jnp.array(trim.control),
        u_trim=jnp.array(trim.control),
    )

    return MOTH_BIEKER_V3, result, result.force_log


@pytest.mark.slow
class TestWriteMothLqgRrd:
    def test_creates_rrd_file(self, lqg_data, tmp_path):
        params, result, forces = lqg_data
        output = tmp_path / "test.rrd"
        returned_path = write_moth_lqg_rrd(params, result, output, forces=forces)
        assert returned_path == output
        assert output.exists()

    def test_rrd_file_nonempty(self, lqg_data, tmp_path):
        params, result, forces = lqg_data
        output = tmp_path / "test.rrd"
        write_moth_lqg_rrd(params, result, output, forces=forces)
        assert output.stat().st_size > 0

    def test_without_forces(self, lqg_data, tmp_path):
        """Test that forces are optional."""
        params, result, _ = lqg_data
        output = tmp_path / "no_forces.rrd"
        write_moth_lqg_rrd(params, result, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_zero_duration_result_supported(self, tmp_path):
        """Logger handles valid zero-step closed-loop results."""
        lqr = design_moth_lqr(u_forward=10.0)
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        meas = create_moth_measurement(
            "speed_pitch_height",
            bowsprit_position=MOTH_BIEKER_V3.bowsprit_position,
        )
        trim = lqr.trim

        sensor = MeasurementSensor(measurement_model=meas, num_controls=2)
        ekf = ExtendedKalmanFilter(dt=0.01)
        Q_ekf = jnp.diag(jnp.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-4]))
        estimator = EKFEstimator(ekf=ekf, measurement_model=meas, Q_ekf=Q_ekf, num_controls=2)
        u_min, u_max = moth.control_lower_bounds, moth.control_upper_bounds
        controller = LQRController(
            K=jnp.array(lqr.K),
            x_trim=jnp.array(trim.state),
            u_trim=jnp.array(trim.control),
            u_min=u_min,
            u_max=u_max,
        )

        result = simulate_closed_loop(
            system=moth,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
            x0_true=jnp.array(trim.state),
            x0_est=jnp.array(trim.state),
            P0=jnp.eye(5) * 0.1,
            dt=0.01,
            duration=0.0,
            rng_key=jax.random.PRNGKey(0),
            params=MOTH_BIEKER_V3,
            measurement_model=meas,
            trim_state=jnp.array(trim.state),
            trim_control=jnp.array(trim.control),
            u_trim=jnp.array(trim.control),
        )
        output = tmp_path / "zero_duration.rrd"
        returned_path = write_moth_lqg_rrd(MOTH_BIEKER_V3, result, output)
        assert returned_path == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_force_length_mismatch_raises_value_error(self, lqg_data, tmp_path):
        """Mismatched force arrays fail with a clear error."""
        params, result, forces = lqg_data
        bad_forces: MothForceLog = replace(
            forces,
            main_foil_force=forces.main_foil_force[:-1],
        )
        output = tmp_path / "bad_forces.rrd"
        with pytest.raises(
            ValueError, match=r"forces\.main_foil_force must have length"
        ):
            write_moth_lqg_rrd(params, result, output, forces=bad_forces)

    def test_preserves_zero_heel_from_result(self, lqg_data, tmp_path, monkeypatch):
        """A zero heel in result is valid and must not fall back to 30 deg."""
        params, result, _ = lqg_data
        result_zero_heel = replace(result, heel_angle=0.0)
        output = tmp_path / "heel_zero.rrd"

        recorded_heel = []

        def _quat_spy(theta, heel):
            recorded_heel.append(float(heel))
            return np.array([0.0, 0.0, 0.0, 1.0])

        import fmd.analysis.viz3d.moth_lqg_logger as logger_mod
        monkeypatch.setattr(logger_mod, "moth_3dof_to_rerun_quat", _quat_spy)

        write_moth_lqg_rrd(params, result_zero_heel, output)
        assert output.exists()
        assert recorded_heel, "Expected quaternion conversion to be called"
        assert all(abs(h) < 1e-12 for h in recorded_heel)

    def test_prefers_result_params_over_passed_params(self, lqg_data, tmp_path, monkeypatch):
        """Logger should use result.params (simulation source of truth)."""
        params, result, _ = lqg_data
        output = tmp_path / "result_params_preferred.rrd"
        wrong_params = attrs.evolve(params, hull_length=params.hull_length + 0.5)

        captured = {"params": None}
        import fmd.analysis.viz3d.moth_lqg_logger as logger_mod
        original_build = logger_mod.build_moth_wireframe

        def _build_spy(p):
            captured["params"] = p
            return original_build(p)

        monkeypatch.setattr(logger_mod, "build_moth_wireframe", _build_spy)

        write_moth_lqg_rrd(wrong_params, result, output)
        assert output.exists()
        assert captured["params"] is result.params
