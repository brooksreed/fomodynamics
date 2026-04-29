"""Tests for Moth LQG scenario framework.

Tests the ScenarioConfig and run_scenario() runner
with both named scenarios (baseline, surface_breach) and custom configs.
"""

from fmd.simulator import _config  # noqa: F401

import attrs
import numpy as np
import pytest

from fmd.simulator.closed_loop_pipeline import ClosedLoopResult
from fmd.simulator.moth_scenarios import (
    ScenarioConfig,
    run_scenario,
    compute_tip_at_surface_pos_d,
    SCENARIOS,
    BASELINE,
    SURFACE_BREACH,
)
from fmd.simulator.params import MOTH_BIEKER_V3


@pytest.mark.slow
class TestBaselineScenario:
    """Tests for the baseline scenario."""

    def test_baseline_scenario_runs(self):
        """Baseline scenario runs and converges within 2s."""
        config = attrs.evolve(BASELINE, name="baseline_short", duration=2.0)
        result = run_scenario(config)

        assert isinstance(result, ClosedLoopResult)
        n_steps = int(2.0 / config.dt)
        assert result.times.shape == (n_steps,)
        assert result.true_states.shape == (n_steps + 1, 5)
        assert result.controls.shape == (n_steps, 2)
        assert result.params is MOTH_BIEKER_V3

        # Check convergence: final state close to trim
        trim_state = result.trim_state
        final_error = np.abs(result.true_states[-1] - trim_state)
        assert final_error[0] < 0.05, f"pos_d error {final_error[0]:.4f} > 0.05"
        assert final_error[1] < 0.05, f"theta error {final_error[1]:.4f} > 0.05"


@pytest.mark.slow
class TestSurfaceBreachScenario:
    """Tests for the surface breach scenario.

    The trim point is set so the leeward foil tip is at the water surface.
    The controller regulates around this operating point, and measurement
    noise causes the tip to oscillate above and below the surface.
    """

    def test_surface_breach_scenario_runs(self):
        """Surface breach scenario runs and stays near trim."""
        config = attrs.evolve(SURFACE_BREACH, name="surface_breach_short", duration=2.0)
        result = run_scenario(config)

        assert isinstance(result, ClosedLoopResult)
        n_steps = int(2.0 / config.dt)
        assert result.true_states.shape == (n_steps + 1, 5)

        # Trim theta should be ~0 (target_theta=0.0)
        trim_state = result.trim_state
        assert abs(trim_state[1]) < 0.01, (
            f"Trim theta = {np.degrees(trim_state[1]):.2f} deg, expected ~0"
        )

        # Trim pos_d should match target (tip at surface)
        expected_pos_d = compute_tip_at_surface_pos_d(config.params, config.heel_angle)
        np.testing.assert_allclose(
            trim_state[0], expected_pos_d, atol=1e-6,
            err_msg="Trim pos_d doesn't match tip-at-surface target",
        )

        # Compute leeward tip depth throughout the trajectory.
        # The leeward tip rises due to heel: tip = foil_depth - (span/2)*sin(heel).
        from fmd.simulator.components.moth_forces import compute_foil_ned_depth
        params = config.params
        heel_angle = config.heel_angle
        pos_d = result.true_states[:, 0]
        theta = result.true_states[:, 1]
        total_mass = params.hull_mass + params.sailor_mass
        cg_offset = params.sailor_mass * params.sailor_position / total_mass
        foil_pos = params.main_foil_position - cg_offset
        foil_depth = np.array([
            float(compute_foil_ned_depth(pos_d[i], foil_pos[0], foil_pos[2], theta[i], heel_angle))
            for i in range(len(pos_d))
        ])
        leeward_tip_depth = foil_depth - (params.main_foil_span / 2.0) * np.sin(heel_angle)

        # Tip should be near the surface (within ~5cm of waterline on average)
        mean_tip_depth = np.mean(leeward_tip_depth)
        assert abs(mean_tip_depth) < 0.05, (
            f"Mean leeward tip depth {mean_tip_depth:.3f}m — expected near 0"
        )

        # Controller should keep the boat near trim (not diverge)
        final_error = np.abs(result.true_states[-1] - trim_state)
        assert final_error[0] < 0.10, f"pos_d error {final_error[0]:.4f} > 0.10"
        assert final_error[1] < 0.10, f"theta error {final_error[1]:.4f} > 0.10"


@pytest.mark.slow
class TestCustomConfig:
    """Tests for custom scenario configurations (requires simulation)."""

    def test_custom_config_propagates(self):
        """Custom config fields are used by run_scenario."""
        config = ScenarioConfig(
            name="custom_test",
            u_forward=8.0,
            duration=0.5,
            measurement_variant="full_state",
            perturbation=(0.02, 0.0, 0.0, 0.0, 0.0),
            seed=123,
        )
        result = run_scenario(config)

        assert isinstance(result, ClosedLoopResult)
        n_steps = int(0.5 / config.dt)
        assert result.times.shape == (n_steps,)
        # full_state measurement: 5 outputs
        assert result.measurements_clean.shape == (n_steps, 5)


class TestScenarioConfigUnit:
    """Unit tests for ScenarioConfig (no simulation needed)."""

    def test_scenarios_dict_matches_constants(self):
        """SCENARIOS dict entries match the module-level constants."""
        assert SCENARIOS["baseline"] is BASELINE
        assert SCENARIOS["surface_breach"] is SURFACE_BREACH

    def test_config_is_frozen(self):
        """ScenarioConfig is immutable."""
        with pytest.raises(AttributeError):
            BASELINE.name = "modified"

    def test_baseline_has_perturbation(self):
        """Baseline scenario has non-None perturbation."""
        assert BASELINE.perturbation is not None
        assert len(BASELINE.perturbation) == 5

    def test_surface_breach_has_target_theta(self):
        """Surface breach scenario uses target_theta=0."""
        assert SURFACE_BREACH.target_theta == 0.0

    def test_surface_breach_has_target_pos_d(self):
        """Surface breach scenario uses target_pos_d for tip at surface."""
        assert SURFACE_BREACH.target_pos_d is not None
        # pos_d should be negative (CG above water) and match geometry
        assert SURFACE_BREACH.target_pos_d < 0
        expected = compute_tip_at_surface_pos_d()
        np.testing.assert_allclose(
            SURFACE_BREACH.target_pos_d, expected, atol=1e-10,
        )

    def test_surface_breach_no_perturbation(self):
        """Surface breach scenario has no perturbation (starts at trim)."""
        assert SURFACE_BREACH.perturbation is None
