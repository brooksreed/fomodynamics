"""Integration tests for wand configuration comparison.

Tests the 4 sensor/estimator/controller configurations across calm
and wave conditions. Verifies stability, EKF convergence, wand
saturation handling, cross-config ordering, and metrics computation.
"""

from fmd.simulator import _config  # noqa: F401

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator.closed_loop_pipeline import ClosedLoopResult
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_scenarios import (
    create_baseline_config,
    create_speed_pitch_wand_config,
    create_wand_only_config,
    create_mechanical_wand_config,
    run_wand_scenario,
)
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.params.presets import WAVE_SF_BAY_LIGHT


@pytest.fixture(scope="module")
def lqr_design():
    """Shared LQR design at 10 m/s for all tests."""
    return design_moth_lqr(u_forward=10.0)


@pytest.fixture(scope="module")
def trim_state(lqr_design):
    """Trim state array."""
    return np.array(lqr_design.trim.state)


@pytest.fixture(scope="module")
def perturbation():
    """Standard perturbation."""
    return (0.05, np.radians(-2.0), 0.0, 0.0, 0.0)


class TestAllConfigsSmoke:
    """All 4 configs complete without NaN in calm water (1s smoke test)."""

    @pytest.fixture(scope="class")
    def calm_results(self, lqr_design, perturbation):
        """Run all 4 configs for 1s in calm water."""
        configs = {
            "baseline": create_baseline_config,
            "speed_pitch_wand": create_speed_pitch_wand_config,
            "wand_only": create_wand_only_config,
            "mechanical_wand": create_mechanical_wand_config,
        }
        results = {}
        for name, factory in configs.items():
            s, e, c = factory(lqr_design)
            result = run_wand_scenario(
                config_name=name,
                sensor=s,
                estimator=e,
                controller=c,
                lqr=lqr_design,
                duration=1.0,
                perturbation=perturbation,
            )
            results[name] = result
        return results

    def test_baseline_no_nan(self, calm_results):
        """Baseline completes without NaN."""
        r = calm_results["baseline"]
        assert not np.any(np.isnan(r.true_states))
        assert not np.any(np.isnan(r.controls))

    def test_speed_pitch_wand_no_nan(self, calm_results):
        """Speed+pitch+wand completes without NaN."""
        r = calm_results["speed_pitch_wand"]
        assert not np.any(np.isnan(r.true_states))
        assert not np.any(np.isnan(r.controls))

    def test_wand_only_no_nan(self, calm_results):
        """Wand-only completes without NaN."""
        r = calm_results["wand_only"]
        assert not np.any(np.isnan(r.true_states))
        assert not np.any(np.isnan(r.controls))

    def test_mechanical_wand_no_nan(self, calm_results):
        """Mechanical wand completes without NaN."""
        r = calm_results["mechanical_wand"]
        assert not np.any(np.isnan(r.true_states))
        assert not np.any(np.isnan(r.controls))

    def test_baseline_pos_d_within_bounds(self, calm_results, trim_state):
        """Baseline final pos_d within 0.5m of trim."""
        r = calm_results["baseline"]
        final_err = abs(r.true_states[-1, 0] - trim_state[0])
        assert final_err < 0.5, f"pos_d error {final_err:.4f} > 0.5m"

    def test_speed_pitch_wand_pos_d_within_bounds(self, calm_results, trim_state):
        """Speed+pitch+wand final pos_d within 0.5m of trim."""
        r = calm_results["speed_pitch_wand"]
        final_err = abs(r.true_states[-1, 0] - trim_state[0])
        assert final_err < 0.5, f"pos_d error {final_err:.4f} > 0.5m"

    def test_wand_only_pos_d_within_bounds(self, calm_results, trim_state):
        """Wand-only final pos_d within 0.5m of trim."""
        r = calm_results["wand_only"]
        final_err = abs(r.true_states[-1, 0] - trim_state[0])
        assert final_err < 0.5, f"pos_d error {final_err:.4f} > 0.5m"

    def test_mechanical_wand_pos_d_within_bounds(self, calm_results, trim_state):
        """Mechanical wand final pos_d within 0.5m of trim."""
        r = calm_results["mechanical_wand"]
        final_err = abs(r.true_states[-1, 0] - trim_state[0])
        assert final_err < 0.5, f"pos_d error {final_err:.4f} > 0.5m"

    def test_result_types(self, calm_results):
        """All results are ClosedLoopResult instances."""
        for name, r in calm_results.items():
            assert isinstance(r, ClosedLoopResult), f"{name} is not ClosedLoopResult"


@pytest.mark.slow
class TestMechanicalWandStability:
    """Mechanical wand maintains bounded flight for 10s calm water."""

    @pytest.fixture(scope="class")
    def mech_result(self, lqr_design, perturbation):
        """Run mechanical wand for 10s in calm water."""
        s, e, c = create_mechanical_wand_config(lqr_design)
        return run_wand_scenario(
            config_name="mechanical_wand_stability",
            sensor=s,
            estimator=e,
            controller=c,
            lqr=lqr_design,
            duration=10.0,
            perturbation=perturbation,
        )

    def test_no_nan(self, mech_result):
        """No NaN in trajectory."""
        assert not np.any(np.isnan(mech_result.true_states))
        assert not np.any(np.isnan(mech_result.controls))

    def test_pos_d_bounded_after_settling(self, mech_result, trim_state):
        """pos_d within 0.15m of trim for t > 2s."""
        dt = 0.005
        idx_2s = int(2.0 / dt)
        true_states = mech_result.true_states[idx_2s:]
        pos_d_err = np.abs(true_states[:, 0] - trim_state[0])
        max_err = np.max(pos_d_err)
        assert max_err < 0.15, f"max pos_d error {max_err:.4f} > 0.15m for t > 2s"

    def test_pitch_bounded_after_settling(self, mech_result, trim_state):
        """pitch within 5 deg of trim for t > 2s."""
        dt = 0.005
        idx_2s = int(2.0 / dt)
        true_states = mech_result.true_states[idx_2s:]
        theta_err = np.abs(true_states[:, 1] - trim_state[1])
        max_err_deg = np.degrees(np.max(theta_err))
        assert max_err_deg < 5.0, f"max pitch error {max_err_deg:.2f} deg > 5.0 deg for t > 2s"

    def test_no_excessive_control_saturation(self, mech_result):
        """Control not saturated >50% of time."""
        from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
        u_min = np.array(moth.control_lower_bounds)
        u_max = np.array(moth.control_upper_bounds)

        controls = mech_result.controls
        tol = 0.001  # ~0.06 deg tolerance for being "at" a bound
        at_min = np.any(controls < u_min + tol, axis=1)
        at_max = np.any(controls > u_max - tol, axis=1)
        sat_frac = float(np.mean(at_min | at_max))
        assert sat_frac < 0.5, f"Control saturated {sat_frac*100:.1f}% of time (> 50%)"


@pytest.mark.slow
class TestWandOnlyEKFConvergence:
    """wand_only_lqg EKF convergence test."""

    @pytest.fixture(scope="class")
    def wand_only_result(self, lqr_design, perturbation):
        """Run wand-only for 5s in calm water."""
        s, e, c = create_wand_only_config(lqr_design)
        return run_wand_scenario(
            config_name="wand_only_convergence",
            sensor=s,
            estimator=e,
            controller=c,
            lqr=lqr_design,
            duration=5.0,
            perturbation=perturbation,
        )

    def test_no_nan(self, wand_only_result):
        """No NaN in trajectory."""
        assert not np.any(np.isnan(wand_only_result.true_states))

    def test_ekf_convergence(self, wand_only_result):
        """Wand-observable states (pos_d, theta, w, q) stay well-estimated.

        Forward speed u (index 4) is **not observable** from a wand-only
        sensor: the wand measures geometry (pivot height -> pos_d, theta) and
        carries no forward-speed information. u is pure dead reckoning off the
        process model (observing it would need GPS / a paddle-log; even
        wand + IMU only dead-reckons u), so this test does not assert on u.

        The check is therefore an **absolute** bound on the observable states,
        not a convergence *ratio*: those states settle within ~0.5 s to a few
        cm / tenths of a degree, then drift mildly as the unobserved u estimate
        contaminates them through the process-model coupling — a ratio test is
        the wrong shape (fast settle -> small denominator).

        The §4.6 hull-frame wand fix shifts the wand measurement Jacobian's
        theta-column by -1, which changed the (already marginal) u dead-reckoning
        so that the full-state error is now u-dominated; the observable states
        remain small and the true trajectory stays stable and physical. The
        wand-only u behavior is revisited when the wand study is re-run in
        Phase 2 (C2). See docs/private/plans/physics_fixes C1.D retro.
        """
        obs_idx = [0, 1, 2, 3]  # pos_d, theta, w, q (u=4 unobservable)
        est_errors = np.asarray(wand_only_result.estimation_errors)[:, obs_idx]

        # Observable states settle fast; require they stay small for the whole
        # run (not a convergence ratio — see docstring).
        max_error = float(np.max(np.linalg.norm(est_errors, axis=1)))
        error_end = float(np.linalg.norm(est_errors[-1]))
        assert error_end < 0.1, (
            f"Wand-observable-state error too large at t=end: {error_end:.4f} "
            f"(pos_d,theta,w,q); u is intentionally excluded (unobservable)."
        )
        assert max_error < 0.15, (
            f"Wand-observable-state error spiked during run: max={max_error:.4f}"
        )


class TestWandSaturationEdgeCase:
    """Wand saturation: start boat low (wand nearly horizontal)."""

    def test_saturation_no_crash(self, lqr_design):
        """System doesn't crash when wand is nearly horizontal."""
        # Start boat very low (large negative perturbation in pos_d
        # makes the boat much deeper = wand nearly horizontal)
        low_perturbation = (0.3, 0.0, 0.0, 0.0, 0.0)

        configs = {
            "wand_only": create_wand_only_config,
            "mechanical_wand": create_mechanical_wand_config,
        }
        for name, factory in configs.items():
            s, e, c = factory(lqr_design)
            result = run_wand_scenario(
                config_name=f"{name}_saturation",
                sensor=s,
                estimator=e,
                controller=c,
                lqr=lqr_design,
                duration=1.0,
                perturbation=low_perturbation,
            )
            # Should complete without crashing
            assert isinstance(result, ClosedLoopResult), f"{name} failed"
            # No NaN (graceful degradation, not crash)
            assert not np.any(np.isnan(result.true_states)), f"{name} has NaN states"
            assert not np.any(np.isnan(result.controls)), f"{name} has NaN controls"


@pytest.mark.slow
class TestWaveIntegration:
    """At least one wand config runs with head seas for 5s without NaN."""

    def test_speed_pitch_wand_head_seas(self, lqr_design, perturbation):
        """Speed+pitch+wand runs 5s in head seas without NaN."""
        wave_params = attrs.evolve(WAVE_SF_BAY_LIGHT, mean_direction=np.pi)

        s, e, c = create_speed_pitch_wand_config(lqr_design)
        result = run_wand_scenario(
            config_name="spw_head_seas",
            sensor=s,
            estimator=e,
            controller=c,
            lqr=lqr_design,
            duration=5.0,
            perturbation=perturbation,
            wave_params=wave_params,
        )

        assert not np.any(np.isnan(result.true_states)), "NaN in true_states"
        assert not np.any(np.isnan(result.controls)), "NaN in controls"


@pytest.mark.slow
class TestEKFNaNRegression:
    """Regression tests for EKF NaN fix in head seas.

    The speed_pitch_wand_lqg config previously diverged at t~9.6s in
    WAVE_SF_BAY_LIGHT head seas due to a wand angle Jacobian singularity
    (arccos gradient NaN at clip boundary). These tests verify the fix
    (_safe_arccos in moth_wand.py + S_reg in EKF) holds for 10s runs.
    """

    def test_speed_pitch_wand_head_seas_10s(self, lqr_design, perturbation):
        """Speed+pitch+wand survives 10s head seas (previously NaN at 9.6s)."""
        wave_params = attrs.evolve(WAVE_SF_BAY_LIGHT, mean_direction=np.pi)

        s, e, c = create_speed_pitch_wand_config(lqr_design)
        result = run_wand_scenario(
            config_name="spw_head_seas_10s",
            sensor=s,
            estimator=e,
            controller=c,
            lqr=lqr_design,
            duration=10.0,
            perturbation=perturbation,
            wave_params=wave_params,
        )

        assert not np.any(np.isnan(result.true_states)), "NaN in true_states"
        assert not np.any(np.isnan(result.est_states)), "NaN in est_states"
        assert not np.any(np.isnan(result.controls)), "NaN in controls"

    def test_all_wand_configs_head_seas_10s(self, lqr_design, perturbation):
        """All 4 wand configs survive 10s head seas without NaN."""
        wave_params = attrs.evolve(WAVE_SF_BAY_LIGHT, mean_direction=np.pi)

        configs = {
            "baseline": create_baseline_config,
            "speed_pitch_wand": create_speed_pitch_wand_config,
            "wand_only": create_wand_only_config,
            "mechanical_wand": create_mechanical_wand_config,
        }

        for name, factory in configs.items():
            s, e, c = factory(lqr_design)
            result = run_wand_scenario(
                config_name=f"{name}_head_seas_10s",
                sensor=s,
                estimator=e,
                controller=c,
                lqr=lqr_design,
                duration=10.0,
                perturbation=perturbation,
                wave_params=wave_params,
            )
            assert not np.any(np.isnan(result.true_states)), f"{name}: NaN in true_states"
            assert not np.any(np.isnan(result.controls)), f"{name}: NaN in controls"


@pytest.mark.slow
class TestCrossConfigOrdering:
    """Baseline has lower RMS tracking error than wand_only_lqg."""

    @pytest.fixture(scope="class")
    def comparison_results(self, lqr_design, perturbation):
        """Run baseline and wand_only for 5s in calm water."""
        results = {}
        for name, factory in [
            ("baseline", create_baseline_config),
            ("wand_only", create_wand_only_config),
        ]:
            s, e, c = factory(lqr_design)
            result = run_wand_scenario(
                config_name=f"{name}_ordering",
                sensor=s,
                estimator=e,
                controller=c,
                lqr=lqr_design,
                duration=5.0,
                perturbation=perturbation,
            )
            results[name] = result
        return results

    def test_baseline_better_rms_pos_d(self, comparison_results, trim_state):
        """Baseline has lower RMS pos_d error than wand_only."""
        baseline_err = comparison_results["baseline"].true_states[1:, 0] - trim_state[0]
        wand_err = comparison_results["wand_only"].true_states[1:, 0] - trim_state[0]

        rms_baseline = float(np.sqrt(np.mean(baseline_err ** 2)))
        rms_wand = float(np.sqrt(np.mean(wand_err ** 2)))

        assert rms_baseline < rms_wand, (
            f"Expected baseline ({rms_baseline:.4f}) < wand_only ({rms_wand:.4f})"
        )


@pytest.mark.slow
class TestMetricsComputation:
    """Metrics computed and reported for all configs."""

    @pytest.fixture(scope="class")
    def metrics_results(self, lqr_design, perturbation):
        """Run all 4 configs for 5s and compute metrics."""
        from fmd.simulator.moth_metrics import compute_metrics

        configs = {
            "baseline": create_baseline_config,
            "speed_pitch_wand": create_speed_pitch_wand_config,
            "wand_only": create_wand_only_config,
            "mechanical_wand": create_mechanical_wand_config,
        }
        results = {}
        for name, factory in configs.items():
            s, e, c = factory(lqr_design)
            result = run_wand_scenario(
                config_name=name,
                sensor=s,
                estimator=e,
                controller=c,
                lqr=lqr_design,
                duration=5.0,
                perturbation=perturbation,
            )
            trim_state = np.array(lqr_design.trim.state)
            metrics = compute_metrics(result, trim_state)
            results[name] = metrics
        return results

    def test_rms_pos_d_computed(self, metrics_results):
        """RMS pos_d is computed and finite for all configs."""
        for name, m in metrics_results.items():
            assert np.isfinite(m["rms_pos_d"]), f"{name}: rms_pos_d not finite"
            assert m["rms_pos_d"] >= 0, f"{name}: rms_pos_d negative"

    def test_rms_theta_computed(self, metrics_results):
        """RMS theta is computed and finite for all configs."""
        for name, m in metrics_results.items():
            assert np.isfinite(m["rms_theta"]), f"{name}: rms_theta not finite"
            assert m["rms_theta"] >= 0, f"{name}: rms_theta negative"

    def test_control_effort_computed(self, metrics_results):
        """Control effort is computed and finite for all configs."""
        for name, m in metrics_results.items():
            assert np.isfinite(m["control_effort"]), f"{name}: control_effort not finite"

    def test_settling_time_computed(self, metrics_results):
        """Settling time is computed and finite for all configs."""
        for name, m in metrics_results.items():
            assert np.isfinite(m["settling_time"]), f"{name}: settling_time not finite"
            assert m["settling_time"] >= 0, f"{name}: settling_time negative"

    def test_breach_fraction_computed(self, metrics_results):
        """Breach fraction is computed and in [0, 1] for all configs."""
        for name, m in metrics_results.items():
            assert 0.0 <= m["breach_fraction"] <= 1.0, (
                f"{name}: breach_fraction {m['breach_fraction']} not in [0, 1]"
            )

    def test_no_nan_in_any_config(self, metrics_results):
        """No config reports NaN."""
        for name, m in metrics_results.items():
            assert not m["has_nan"], f"{name} has NaN"

    def test_baseline_reasonable_rms(self, metrics_results):
        """Baseline RMS theta < 0.1 rad and RMS pos_d < 0.15 m."""
        m = metrics_results["baseline"]
        assert m["rms_theta"] < 0.1, f"Baseline rms_theta {m['rms_theta']:.4f} > 0.1 rad"
        assert m["rms_pos_d"] < 0.15, f"Baseline rms_pos_d {m['rms_pos_d']:.4f} > 0.15 m"

    def test_mechanical_wand_within_2x_baseline(self, metrics_results):
        """Mechanical wand RMS < 2x baseline (with tuned pullrod_offset)."""
        baseline = metrics_results["baseline"]
        mech = metrics_results["mechanical_wand"]
        # The tuned pullrod_offset should keep mechanical wand close to baseline.
        # Allow 2x + 0.02m margin for transient effects.
        limit = 2 * baseline["rms_pos_d"] + 0.02
        assert mech["rms_pos_d"] < limit, (
            f"Mechanical rms_pos_d {mech['rms_pos_d']:.4f} > "
            f"2x baseline {baseline['rms_pos_d']:.4f} + 0.02 = {limit:.4f}"
        )


# ---------------------------------------------------------------------------
# TestTrimAtSetpoint (C2.C2 — per-setpoint calibration / Option D)
# ---------------------------------------------------------------------------

from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.components.moth_wand import (
    DEFAULT_WAND_LENGTH,
    wand_angle_from_state,
)
from fmd.simulator.moth_scenarios import create_pid_wand_config
from fmd.simulator.trim_casadi import find_moth_trim

_HEEL_30 = np.deg2rad(30.0)


@pytest.fixture(scope="module")
def deeper_target():
    """The wand_vs_pid_waves 'pid_deeper' setpoint: tip 30 cm below surface."""
    return float(compute_tip_at_surface_pos_d() + 0.30)


@pytest.fixture(scope="module")
def deeper_trim(deeper_target):
    """Pinned trim solved at the deeper setpoint (u=10, 30 deg heel)."""
    return find_moth_trim(
        MOTH_BIEKER_V3, u_forward=10.0,
        target_pos_d=deeper_target, heel_angle=_HEEL_30,
    )


@pytest.fixture(scope="module")
def pid_deeper_controller(lqr_design, deeper_target, deeper_trim):
    """PID config at the deeper setpoint, calibrated at its own trim."""
    _, _, controller = create_pid_wand_config(
        lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30, dt=0.005,
        target_pos_d=deeper_target, setpoint_trim=deeper_trim,
    )
    return controller


def _own_trim_wand_angle(pos_d, theta):
    return float(
        wand_angle_from_state(
            pos_d=jnp.array(pos_d), theta=jnp.array(theta),
            wand_pivot_position=jnp.asarray(MOTH_BIEKER_V3.wand_pivot_position),
            wand_length=DEFAULT_WAND_LENGTH, heel_angle=_HEEL_30,
        )
    )


class TestTrimAtSetpoint:
    """Per-setpoint calibration locks (C2.C2 / Option D).

    Each controller must be calibrated at its OWN pinned trim, its
    setpoint an exact calm equilibrium by construction. These are the
    round-trip trim-identity tests the roadmap's C2.C2 brief requires.
    """

    def test_pid_deeper_calibrated_at_own_trim(
        self, pid_deeper_controller, deeper_trim
    ):
        """theta_ref / flap_trim / elevator_trim come from the deeper pinned
        trim, not the natural trim (the pre-C2.C2 foreign-calibration bug)."""
        c = pid_deeper_controller
        np.testing.assert_allclose(
            float(c.theta_ref), float(deeper_trim.state[1]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.flap_trim), float(deeper_trim.control[0]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.elevator_trim), float(deeper_trim.control[1]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.pos_d_target), float(deeper_trim.state[0]), atol=1e-12
        )

    def test_round_trip_identity_natural(self, lqr_design):
        """estimate_pos_d(wand_angle_from_state(setpoint, own theta)) ==
        setpoint at the natural trim."""
        _, _, c = create_pid_wand_config(
            lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30, dt=0.005,
        )
        wa = _own_trim_wand_angle(lqr_design.trim.state[0], float(c.theta_ref))
        np.testing.assert_allclose(
            float(c.estimate_pos_d(jnp.array(wa))),
            float(c.pos_d_target), atol=1e-9,
        )

    def test_round_trip_identity_deeper(
        self, pid_deeper_controller, deeper_target
    ):
        """Same identity at the deeper setpoint — the Option D lock."""
        c = pid_deeper_controller
        wa = _own_trim_wand_angle(deeper_target, float(c.theta_ref))
        np.testing.assert_allclose(
            float(c.estimate_pos_d(jnp.array(wa))), deeper_target, atol=1e-9,
        )

    def test_pid_zero_error_identity_at_own_trim(
        self, pid_deeper_controller, deeper_trim, deeper_target
    ):
        """At its own trim the PID outputs exactly its own trim control
        (height_err = 0 -> flap = own flap_trim, elevator = own trim):
        the setpoint trim is a calm equilibrium by construction."""
        c = pid_deeper_controller
        wa = _own_trim_wand_angle(deeper_target, float(c.theta_ref))
        u, _ = c.control(jnp.zeros(5).at[0].set(wa), 0.0, c.init_controller_state())
        np.testing.assert_allclose(
            float(u[0]), float(deeper_trim.control[0]), atol=1e-9
        )
        np.testing.assert_allclose(
            float(u[1]), float(deeper_trim.control[1]), atol=1e-9
        )

    def test_pid_natural_calibration_unchanged(self, lqr_design):
        """target_pos_d=None keeps the natural-trim calibration (regression:
        pid_natural and mechanical rows must not move due to C2.C2)."""
        _, _, c = create_pid_wand_config(
            lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30, dt=0.005,
        )
        np.testing.assert_allclose(
            float(c.theta_ref), float(lqr_design.trim.state[1]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.flap_trim), float(lqr_design.trim.control[0]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.elevator_trim), float(lqr_design.trim.control[1]), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c.pos_d_target), float(lqr_design.trim.state[0]), atol=1e-12
        )

    def test_internal_solve_matches_passthrough(
        self, lqr_design, deeper_target, pid_deeper_controller
    ):
        """Omitting setpoint_trim solves the same pinned trim internally
        (bit-consistent with the passthrough path)."""
        _, _, c_internal = create_pid_wand_config(
            lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30, dt=0.005,
            target_pos_d=deeper_target,
        )
        c = pid_deeper_controller
        np.testing.assert_allclose(
            float(c_internal.theta_ref), float(c.theta_ref), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c_internal.flap_trim), float(c.flap_trim), atol=1e-12
        )
        np.testing.assert_allclose(
            float(c_internal.wand_angle_offset), float(c.wand_angle_offset),
            atol=1e-12,
        )

    def test_setpoint_trim_mismatch_raises(self, lqr_design, deeper_target):
        """A setpoint_trim pinned elsewhere than target_pos_d fails loud."""
        with pytest.raises(ValueError, match="does not match"):
            create_pid_wand_config(
                lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30,
                dt=0.005, target_pos_d=deeper_target,
                setpoint_trim=lqr_design.trim,  # pinned at natural, not target
            )

    def test_setpoint_trim_without_target_raises(self, lqr_design, deeper_trim):
        """setpoint_trim without target_pos_d is a caller error."""
        with pytest.raises(ValueError, match="without target_pos_d"):
            create_pid_wand_config(
                lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30,
                dt=0.005, setpoint_trim=deeper_trim,
            )

    def test_mechanical_static_identity(self, lqr_design):
        """Auto-tuned linkage outputs exactly the trim flap at the trim wand
        angle — the natural trim is an equilibrium of the passive loop."""
        _, _, mech = create_mechanical_wand_config(
            lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30,
        )
        wa = _own_trim_wand_angle(
            lqr_design.trim.state[0], lqr_design.trim.state[1]
        )
        np.testing.assert_allclose(
            float(mech.linkage.compute(jnp.array(wa))),
            float(lqr_design.trim.control[0]), atol=1e-9,
        )

    def test_mechanical_explicit_offset_respected(self, lqr_design):
        """linkage_overrides={'pullrod_offset': ...} bypasses the auto-tune."""
        _, _, mech = create_mechanical_wand_config(
            lqr_design, params=MOTH_BIEKER_V3, heel_angle=_HEEL_30,
            linkage_overrides={"pullrod_offset": 0.005},
        )
        np.testing.assert_allclose(mech.linkage.pullrod_offset, 0.005, atol=0.0)
