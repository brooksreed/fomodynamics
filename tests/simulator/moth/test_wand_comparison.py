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
        """Estimation error norm at t=5s < 50% of error at t=0.5s."""
        dt = 0.005
        idx_05s = int(0.5 / dt)
        est_errors = wand_only_result.estimation_errors

        error_05s = float(np.linalg.norm(est_errors[idx_05s]))
        error_end = float(np.linalg.norm(est_errors[-1]))

        # If error at 0.5s is very small (no real convergence to test),
        # just check that final error is also small
        if error_05s < 1e-6:
            assert error_end < 1e-3, (
                f"Final error {error_end:.6f} unexpectedly large despite small initial error"
            )
        else:
            ratio = error_end / error_05s
            assert ratio < 0.5, (
                f"EKF not converging: error_end/error_0.5s = {ratio:.3f} "
                f"(end={error_end:.6f}, 0.5s={error_05s:.6f})"
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
