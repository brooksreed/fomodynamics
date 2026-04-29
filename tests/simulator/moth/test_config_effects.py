"""Physics verification tests for Moth 3DOF configuration effects.

Tests that configuration changes (sailor position, flap angle, elevator angle)
result in expected equilibrium and response changes based on physics.

Expected Physics:
- Sailor forward -> nose-down trim (theta decreases)
- Sailor aft -> nose-up trim (theta increases)
- Flap up (+) -> increased lift on forward foil -> nose-up moment (but nose-down trim)
- Flap down (-) -> decreased lift on forward foil -> nose-down moment (but nose-up trim)
- Elevator up (+) -> increased lift on aft rudder -> nose-down moment
- Elevator down (-) -> decreased lift on aft rudder -> nose-up moment

Note on Sailor Position:
    The Moth3D model computes system CG from sailor position and adjusts all
    component moment arms accordingly. Moving the sailor shifts the CG,
    which changes the moment arms of all force components relative to the CG.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from fmd.simulator.moth_validation import PerturbationConfig, ConfigurationVariation, run_configuration_comparison
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.moth_3d import Moth3D, ConstantSchedule, POS_D, THETA, W, Q, U
from fmd.simulator.trim_casadi import find_moth_trim



# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def moth():
    """Create baseline Moth3D instance."""
    return Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))


@pytest.fixture(scope="module")
def baseline_trim():
    """Deterministic pinned trim at 10 m/s."""
    result = find_moth_trim(
        MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
    )
    assert result.success, f"Baseline trim failed: residual={result.residual:.2e}"
    assert result.residual < 0.05
    return result


# =============================================================================
# Sailor Position Effects
# =============================================================================


class TestSailorPositionEffects:
    """Tests for sailor position effects on trim.

    Sailor position shifts the system CG, which changes all component moment
    arms. This produces two testable effects:

    1. Instantaneous moment: At a fixed state/control, moving the sailor
       changes the pitch acceleration (q_dot). Sailor forward -> nose-down.
    2. Trim control change: To maintain the same depth/pitch, different
       controls are needed when the sailor moves.
    """

    @pytest.mark.parametrize("offset_m", [0.05, 0.1, 0.15, 0.2])
    def test_sailor_forward_produces_nose_down_moment(self, offset_m, baseline_trim):
        """Sailor forward should produce nose-down moment at baseline trim."""
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        new_pos = MOTH_BIEKER_V3.sailor_position.copy()
        new_pos[0] += offset_m
        modified = attrs.evolve(MOTH_BIEKER_V3, sailor_position=new_pos)
        moth = Moth3D(modified, u_forward=ConstantSchedule(10.0))
        deriv = moth.forward_dynamics(bl_state, bl_ctrl)
        q_dot = float(deriv[3])

        assert q_dot < -0.1, (
            f"Sailor +{offset_m*100:.0f}cm should produce nose-down moment, "
            f"but q_dot = {np.degrees(q_dot):.2f} deg/s^2"
        )

    @pytest.mark.parametrize("offset_m", [-0.05, -0.1, -0.15, -0.2])
    def test_sailor_aft_produces_nose_up_moment(self, offset_m, baseline_trim):
        """Sailor aft should produce nose-up moment at baseline trim."""
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        new_pos = MOTH_BIEKER_V3.sailor_position.copy()
        new_pos[0] += offset_m
        modified = attrs.evolve(MOTH_BIEKER_V3, sailor_position=new_pos)
        moth = Moth3D(modified, u_forward=ConstantSchedule(10.0))
        deriv = moth.forward_dynamics(bl_state, bl_ctrl)
        q_dot = float(deriv[3])

        assert q_dot > 0.1, (
            f"Sailor {offset_m*100:.0f}cm should produce nose-up moment, "
            f"but q_dot = {np.degrees(q_dot):.2f} deg/s^2"
        )

    def test_sailor_position_sensitivity_range(self, baseline_trim):
        """10cm sailor offset should produce 0.1-100 rad/s^2 pitch acceleration.

        With the new inertia (Iyy~119 vs old 8), the same CG shift produces
        less angular acceleration. The range is relaxed accordingly.
        """
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        new_pos = MOTH_BIEKER_V3.sailor_position.copy()
        new_pos[0] += 0.1  # 10cm forward
        modified = attrs.evolve(MOTH_BIEKER_V3, sailor_position=new_pos)
        moth = Moth3D(modified, u_forward=ConstantSchedule(10.0))
        deriv = moth.forward_dynamics(bl_state, bl_ctrl)
        q_dot = abs(float(deriv[3]))

        assert q_dot > 0.1, f"10cm sailor offset q_dot={q_dot:.2f} too small (expect >0.1 rad/s^2)"
        assert q_dot < 100.0, f"10cm sailor offset q_dot={q_dot:.2f} too large (expect <100 rad/s^2)"

    def test_sailor_effect_comparable_to_flap(self, moth, baseline_trim):
        """10cm sailor offset should produce q_dot comparable to ~0.25 deg flap change.

        At 10 m/s, lift forces scale with u^2, so flap effectiveness per degree
        is higher than at 6 m/s. The sailor CG shift effect scales more weakly,
        so we compare against a smaller flap deflection.
        """
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        # Sailor 10cm forward
        new_pos = MOTH_BIEKER_V3.sailor_position.copy()
        new_pos[0] += 0.1
        modified = attrs.evolve(MOTH_BIEKER_V3, sailor_position=new_pos)
        moth_sailor = Moth3D(modified, u_forward=ConstantSchedule(10.0))
        q_dot_sailor = abs(float(moth_sailor.forward_dynamics(bl_state, bl_ctrl)[3]))

        # 0.25 deg flap change
        flap_ctrl = bl_ctrl.at[0].set(bl_ctrl[0] + jnp.radians(0.25))
        q_dot_flap = abs(float(moth.forward_dynamics(bl_state, flap_ctrl)[3]))

        assert q_dot_sailor > q_dot_flap, (
            f"10cm sailor q_dot={q_dot_sailor:.2f} should exceed 0.25 deg flap q_dot={q_dot_flap:.2f}"
        )

    def test_sailor_moment_monotonic(self, baseline_trim):
        """Sailor position effect on pitch moment should be monotonic."""
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        offsets = [-0.2, -0.1, 0, 0.1, 0.2]
        q_dots = []
        for off in offsets:
            new_pos = MOTH_BIEKER_V3.sailor_position.copy()
            new_pos[0] += off
            modified = attrs.evolve(MOTH_BIEKER_V3, sailor_position=new_pos)
            moth_off = Moth3D(modified, u_forward=ConstantSchedule(10.0))
            deriv = moth_off.forward_dynamics(bl_state, bl_ctrl)
            q_dots.append(float(deriv[3]))

        # q_dot should monotonically decrease as sailor moves forward
        for i in range(len(q_dots) - 1):
            assert q_dots[i] > q_dots[i + 1], (
                f"q_dot not monotonically decreasing: "
                f"q_dot[{i}]={np.degrees(q_dots[i]):.3f} deg/s^2 vs "
                f"q_dot[{i+1}]={np.degrees(q_dots[i+1]):.3f} deg/s^2"
            )


# =============================================================================
# Flap Effects (derivative-based)
# =============================================================================


class TestFlapEffects:
    """Tests for main flap angle effects using instantaneous derivatives.

    Physics: The main foil is forward of CG (position_x = +0.55m). Changing flap
    angle changes lift on the main foil. Since the foil is forward of CG:
    - More lift (+flap) creates a nose-UP moment (forward lift arm)
    - The w_dot (heave accel) is the primary direct effect

    For trim-based tests, when the optimizer re-trims with a fixed flap offset,
    the theta adjusts to compensate. +flap means more lift, so the optimizer
    finds a lower theta to reduce AoA and balance forces.
    """

    @pytest.mark.parametrize("flap_deg", [1.0, 2.0, 3.0, 5.0])
    def test_flap_up_increases_lift(self, moth, baseline_trim, flap_deg):
        """Positive flap produces upward heave acceleration (w_dot < 0).

        This is the direct physics test: +flap -> more AoA -> more lift
        -> w_dot < 0 (upward force exceeds weight at trim state).
        """
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)
        ctrl_pert = ctrl.at[0].add(jnp.deg2rad(flap_deg))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        w_dot = float(deriv[W])

        assert w_dot < -0.1, (
            f"Flap +{flap_deg} deg should produce upward accel, "
            f"but w_dot={w_dot:.3f}"
        )

    @pytest.mark.parametrize("flap_deg", [-1.0, -2.0, -3.0, -5.0])
    def test_flap_down_decreases_lift(self, moth, baseline_trim, flap_deg):
        """Negative flap produces downward heave acceleration (w_dot > 0)."""
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)
        ctrl_pert = ctrl.at[0].add(jnp.deg2rad(flap_deg))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        w_dot = float(deriv[W])

        assert w_dot > 0.1, (
            f"Flap {flap_deg} deg should produce downward accel, "
            f"but w_dot={w_dot:.3f}"
        )

    def test_flap_w_dot_monotonic(self, moth, baseline_trim):
        """Heave acceleration should increase monotonically with flap angle."""
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        offsets_deg = [-5.0, -2.0, 0, 2.0, 5.0]
        w_dots = []
        for off in offsets_deg:
            c = ctrl.at[0].add(jnp.deg2rad(off))
            deriv = moth.forward_dynamics(state, c, 0.0)
            w_dots.append(float(deriv[W]))

        # w_dot should monotonically decrease (more negative = more upward)
        # as flap increases
        for i in range(len(w_dots) - 1):
            assert w_dots[i] > w_dots[i + 1], (
                f"w_dot not monotonically decreasing: "
                f"w_dot({offsets_deg[i]:.0f})={w_dots[i]:.3f} vs "
                f"w_dot({offsets_deg[i+1]:.0f})={w_dots[i+1]:.3f}"
            )

    def test_flap_magnitude_proportionality(self, moth, baseline_trim):
        """Flap effect on w_dot should scale approximately linearly.

        2 deg flap should produce ~2x the w_dot change of 1 deg.
        """
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        deriv_0 = moth.forward_dynamics(state, ctrl, 0.0)
        deriv_1 = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(1.0)), 0.0)
        deriv_2 = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(2.0)), 0.0)

        delta_1 = float(deriv_1[W]) - float(deriv_0[W])
        delta_2 = float(deriv_2[W]) - float(deriv_0[W])

        ratio = delta_2 / delta_1
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2x scaling, got {ratio:.2f}x "
            f"(delta_1={delta_1:.3f}, delta_2={delta_2:.3f})"
        )

    @pytest.mark.parametrize("flap_deg", [2.0, 3.0, 5.0])
    def test_flap_up_nose_down_trim(self, flap_deg):
        """Flap up (positive) should result in nose-down trim (theta decreases).

        When re-trimming with a fixed flap offset, the boat settles at a
        lower theta because the increased lift from the flap is compensated
        by reducing the geometric AoA.

        Note: +1 deg is excluded — at small offsets (<~1.5 deg), the
        pinned-flap solver converges to a different equilibrium branch
        (theta~1.0 deg) than the free baseline (theta~0.82 deg). This is
        a multi-equilibrium bifurcation in the nonlinear dynamics, not a
        cross-term coupling effect. See investigation report:
        docs/plans/surge_ned_thrust/investigations/flap_crossover_2026-03-17.md
        """
        configs = [
            ConfigurationVariation(name="baseline"),
            ConfigurationVariation(flap_offset_deg=flap_deg, name=f"flap_+{flap_deg}"),
        ]
        result = run_configuration_comparison(configs, run_simulation=False)

        baseline = result.configuration_results[0]
        flap_up = result.configuration_results[1]

        delta_theta = flap_up.trim_state[1] - baseline.trim_state[1]

        assert delta_theta < 0, (
            f"Flap +{flap_deg} deg should produce nose-down trim, "
            f"but delta_theta = {np.degrees(delta_theta):.3f} deg"
        )

    def test_flap_effect_monotonic(self):
        """Flap effect on trim pitch should be monotonic among converged configs."""
        offsets_deg = [-5.0, -2.0, 0, 2.0, 5.0]
        configs = [
            ConfigurationVariation(flap_offset_deg=off, name=f"f_{off}")
            for off in offsets_deg
        ]
        result = run_configuration_comparison(configs, run_simulation=False)

        # Filter to well-converged configs (asymmetric bounds can cause saturation)
        # CasADi with fixed_controls achieves tight residuals
        converged = [
            (off, r.trim_state[1])
            for off, r in zip(offsets_deg, result.configuration_results)
            if r.trim_residual < 0.05
        ]
        assert len(converged) >= 3, f"Too few converged configs: {len(converged)}"

        # Theta should monotonically decrease as flap increases
        for i in range(len(converged) - 1):
            off_i, theta_i = converged[i]
            off_j, theta_j = converged[i + 1]
            assert theta_i >= theta_j, (
                f"Theta not monotonically decreasing: "
                f"flap={off_i} deg theta={np.degrees(theta_i):.3f} deg vs "
                f"flap={off_j} deg theta={np.degrees(theta_j):.3f} deg"
            )


# =============================================================================
# Elevator Effects (derivative-based)
# =============================================================================


class TestElevatorEffects:
    """Tests for rudder elevator angle effects.

    Physics: The rudder is aft of CG (position_x = -1.755m). Changing elevator
    angle changes lift on the aft rudder. Since the rudder is aft of CG:
    - More lift (+elevator) creates a nose-DOWN moment (aft lift arm)
    - This is the primary effect on q_dot
    """

    @pytest.mark.parametrize("elev_deg", [0.5, 1.0, 1.5, 2.0])
    def test_elevator_up_nose_down_accel(self, moth, baseline_trim, elev_deg):
        """Positive elevator produces nose-down pitch acceleration (q_dot < 0).

        Direct derivative check: +elevator -> more lift at tail -> nose-down.
        """
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)
        ctrl_pert = ctrl.at[1].add(jnp.deg2rad(elev_deg))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        q_dot = float(deriv[Q])

        assert q_dot < -0.1, (
            f"Elevator +{elev_deg} deg should produce nose-down, "
            f"but q_dot={q_dot:.3f}"
        )

    @pytest.mark.parametrize("elev_deg", [-0.5, -1.0, -1.5, -2.0])
    def test_elevator_down_nose_up_accel(self, moth, baseline_trim, elev_deg):
        """Negative elevator produces nose-up pitch acceleration (q_dot > 0)."""
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)
        ctrl_pert = ctrl.at[1].add(jnp.deg2rad(elev_deg))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        q_dot = float(deriv[Q])

        assert q_dot > 0.1, (
            f"Elevator {elev_deg} deg should produce nose-up, "
            f"but q_dot={q_dot:.3f}"
        )

    def test_elevator_q_dot_monotonic(self, moth, baseline_trim):
        """Pitch acceleration should decrease monotonically with elevator angle."""
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        offsets_deg = [-2.0, -1.0, 0, 1.0, 2.0]
        q_dots = []
        for off in offsets_deg:
            c = ctrl.at[1].add(jnp.deg2rad(off))
            deriv = moth.forward_dynamics(state, c, 0.0)
            q_dots.append(float(deriv[Q]))

        # q_dot should monotonically decrease as elevator increases
        for i in range(len(q_dots) - 1):
            assert q_dots[i] > q_dots[i + 1], (
                f"q_dot not monotonically decreasing: "
                f"q_dot({offsets_deg[i]:.0f})={q_dots[i]:.3f} vs "
                f"q_dot({offsets_deg[i+1]:.0f})={q_dots[i+1]:.3f}"
            )

    def test_elevator_magnitude_proportionality(self, moth, baseline_trim):
        """Elevator effect on q_dot should scale approximately linearly."""
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        deriv_0 = moth.forward_dynamics(state, ctrl, 0.0)
        deriv_1 = moth.forward_dynamics(state, ctrl.at[1].add(jnp.deg2rad(1.0)), 0.0)
        deriv_2 = moth.forward_dynamics(state, ctrl.at[1].add(jnp.deg2rad(2.0)), 0.0)

        delta_1 = float(deriv_1[Q]) - float(deriv_0[Q])
        delta_2 = float(deriv_2[Q]) - float(deriv_0[Q])

        ratio = delta_2 / delta_1
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2x scaling, got {ratio:.2f}x "
            f"(delta_1={delta_1:.3f}, delta_2={delta_2:.3f})"
        )

    @pytest.mark.parametrize("elev_deg", [0.5, 1.0, 1.5, 2.0])
    def test_elevator_up_nose_down_trim(self, elev_deg):
        """Elevator up should produce nose-down trim via configuration comparison."""
        configs = [
            ConfigurationVariation(name="baseline"),
            ConfigurationVariation(elevator_offset_deg=elev_deg, name=f"elev_+{elev_deg}"),
        ]
        result = run_configuration_comparison(configs, run_simulation=False)

        baseline = result.configuration_results[0]
        elev_up = result.configuration_results[1]

        delta_theta = elev_up.trim_state[1] - baseline.trim_state[1]

        assert delta_theta < 0, (
            f"Elevator +{elev_deg} deg should produce nose-down trim, "
            f"but delta_theta = {np.degrees(delta_theta):.3f} deg"
        )

    def test_elevator_effect_monotonic(self):
        """Elevator effect on trim pitch should be monotonic among converged configs."""
        offsets_deg = [-2.0, -1.0, 0, 1.0, 2.0]
        configs = [
            ConfigurationVariation(elevator_offset_deg=off, name=f"e_{off}")
            for off in offsets_deg
        ]
        result = run_configuration_comparison(configs, run_simulation=False)

        # Filter to well-converged configs
        # CasADi with fixed_controls achieves tight residuals
        converged = [
            (off, r.trim_state[1])
            for off, r in zip(offsets_deg, result.configuration_results)
            if r.trim_residual < 0.05
        ]
        assert len(converged) >= 3, f"Too few converged configs: {len(converged)}"

        for i in range(len(converged) - 1):
            off_i, theta_i = converged[i]
            off_j, theta_j = converged[i + 1]
            assert theta_i >= theta_j, (
                f"Theta not monotonically decreasing: "
                f"elev={off_i} deg theta={np.degrees(theta_i):.3f} deg vs "
                f"elev={off_j} deg theta={np.degrees(theta_j):.3f} deg"
            )


# =============================================================================
# Combined Configuration Effects
# =============================================================================


class TestCombinedEffects:
    """Tests for combined configuration changes."""

    def test_sailor_and_flap_combined_moment(self, baseline_trim):
        """Sailor forward + flap down both produce nose-down pitch moment.

        Combined effect should be stronger than either alone.
        """
        import attrs

        bl_state = jnp.array(baseline_trim.state)
        bl_ctrl = jnp.array(baseline_trim.control)

        base_moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))

        # Sailor forward only
        fwd_pos = MOTH_BIEKER_V3.sailor_position.copy()
        fwd_pos[0] += 0.1
        p_fwd = attrs.evolve(MOTH_BIEKER_V3, sailor_position=fwd_pos)
        moth_fwd = Moth3D(p_fwd, u_forward=ConstantSchedule(10.0))
        q_dot_sailor = float(moth_fwd.forward_dynamics(bl_state, bl_ctrl)[3])

        # Flap down only (reduce lift, reducing nose-up moment)
        flap_ctrl = bl_ctrl.at[0].set(bl_ctrl[0] + jnp.radians(-2.0))
        q_dot_flap = float(base_moth.forward_dynamics(bl_state, flap_ctrl)[3])

        # Combined: sailor forward + flap down
        moth_combined = Moth3D(p_fwd, u_forward=ConstantSchedule(10.0))
        q_dot_combined = float(moth_combined.forward_dynamics(bl_state, flap_ctrl)[3])

        # All three should be nose-down (negative q_dot)
        assert q_dot_sailor < 0, "Sailor forward should produce nose-down"
        assert q_dot_flap < 0, "Flap down should produce nose-down"
        # Combined should be more nose-down than either alone
        assert q_dot_combined < q_dot_sailor, "Combined should be more nose-down than sailor alone"
        assert q_dot_combined < q_dot_flap, "Combined should be more nose-down than flap alone"

    def test_reinforcing_flap_elevator_combo(self, moth, baseline_trim):
        """Flap up + elevator up: both produce w_dot < 0 (upward lift).

        Derivative check at trim state.
        """
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        # Flap up alone
        ctrl_flap = ctrl.at[0].add(jnp.deg2rad(2.0))
        w_dot_flap = float(moth.forward_dynamics(state, ctrl_flap, 0.0)[W])

        # Elevator up alone
        ctrl_elev = ctrl.at[1].add(jnp.deg2rad(1.0))
        w_dot_elev = float(moth.forward_dynamics(state, ctrl_elev, 0.0)[W])

        # Both should produce upward heave accel
        assert w_dot_flap < 0, f"Flap up should produce upward accel, got w_dot={w_dot_flap:.3f}"
        assert w_dot_elev < 0, f"Elev up should produce upward accel, got w_dot={w_dot_elev:.3f}"

    def test_opposing_flap_elevator_partial_cancel(self, moth, baseline_trim):
        """Flap up + elevator down: opposing effects should partially cancel.

        Flap up increases total lift (w_dot more negative).
        Elevator down decreases total lift (w_dot more positive).
        Combined w_dot should be between the two individual effects.
        """
        state = jnp.array(baseline_trim.state)
        ctrl = jnp.array(baseline_trim.control)

        # Flap up
        ctrl_flap = ctrl.at[0].add(jnp.deg2rad(2.0))
        w_dot_flap = float(moth.forward_dynamics(state, ctrl_flap, 0.0)[W])

        # Elevator down
        ctrl_elev = ctrl.at[1].add(jnp.deg2rad(-1.0))
        w_dot_elev = float(moth.forward_dynamics(state, ctrl_elev, 0.0)[W])

        # Combined
        ctrl_both = ctrl.at[0].add(jnp.deg2rad(2.0)).at[1].add(jnp.deg2rad(-1.0))
        w_dot_both = float(moth.forward_dynamics(state, ctrl_both, 0.0)[W])

        # w_dot_both should be between flap-only and elev-only
        lower = min(w_dot_flap, w_dot_elev)
        upper = max(w_dot_flap, w_dot_elev)
        assert lower < w_dot_both < upper, (
            f"Combined w_dot={w_dot_both:.3f} not between "
            f"flap={w_dot_flap:.3f} and elev={w_dot_elev:.3f}"
        )


# =============================================================================
# Trim Solver Quality Tests
# =============================================================================


class TestTrimQuality:
    """Tests for trim solver quality across configurations."""

    def test_all_configurations_converge(self):
        """All default configurations should produce valid trim.

        Configurations with BOTH flap and elevator constrained may not
        have valid equilibria because there's no control freedom.

        CasADi with fixed_controls achieves tight residuals for feasible
        configs. Large elevator offsets (|offset| >= 2 deg) may saturate
        the flap, making the problem infeasible.
        """
        from fmd.simulator.moth_validation import default_configuration_variations

        configs = default_configuration_variations()
        result = run_configuration_comparison(configs, run_simulation=False)

        for r in result.configuration_results:
            cfg = r.config
            if cfg.flap_offset_deg == 0 and cfg.elevator_offset_deg == 0:
                assert r.trim_success, f"{r.config.name}: baseline trim failed"
                assert r.trim_residual < 0.05, (
                    f"{r.config.name}: baseline residual {r.trim_residual:.2e} too large"
                )
            elif "extreme" in cfg.name.lower():
                pass  # May be outside trim envelope
            elif cfg.flap_offset_deg != 0 and cfg.elevator_offset_deg != 0:
                pass  # Combined offsets may be outside trim envelope
            elif cfg.flap_offset_deg != 0 and cfg.elevator_offset_deg == 0:
                assert r.trim_success, f"{r.config.name}: trim solver failed"
                assert r.trim_residual < 0.05
            elif cfg.flap_offset_deg == 0 and abs(cfg.elevator_offset_deg) < 2.0:
                # Small elevator offsets should converge
                assert r.trim_success, f"{r.config.name}: trim solver failed"
                assert r.trim_residual < 0.05
            # Large elevator offsets (±2°) may saturate the flap

    def test_extreme_configurations_converge(self):
        """Extreme but valid configurations should still find trim.

        CasADi with fixed_controls handles extreme configs well.
        """
        extreme_configs = [
            ConfigurationVariation(flap_offset_deg=8.0, name="flap +8 deg"),
            ConfigurationVariation(flap_offset_deg=-8.0, name="flap -8 deg"),
            ConfigurationVariation(elevator_offset_deg=3.0, name="elev +3 deg"),
            ConfigurationVariation(elevator_offset_deg=-3.0, name="elev -3 deg"),
        ]
        result = run_configuration_comparison(extreme_configs, run_simulation=False)

        converged_count = 0
        for r in result.configuration_results:
            if r.trim_success and r.trim_residual < 0.05:
                converged_count += 1

        assert converged_count >= len(extreme_configs) // 2, (
            f"Only {converged_count}/{len(extreme_configs)} extreme configs found valid trim"
        )


# =============================================================================
# Nominal Trim Regression Tests
# =============================================================================


class TestNominalTrimRegression:
    """Regression tests for nominal trim with CG-corrected model."""

    def test_nominal_trim_converges(self):
        """Baseline trim should converge with low residual."""
        result = run_configuration_comparison(
            [ConfigurationVariation(name="baseline")],
            u_forward=10.0,
            run_simulation=False,
        )
        bl = result.configuration_results[0]
        assert bl.trim_success, "Baseline trim should converge"
        assert bl.trim_residual < 0.02, f"Residual {bl.trim_residual:.2e} too large"

    def test_nominal_trim_physically_reasonable(self):
        """Baseline trim state should be physically reasonable.

        At 10 m/s: pos_d ~ -1.3m, theta ~ 0.3 deg.
        """
        result = run_configuration_comparison(
            [ConfigurationVariation(name="baseline")],
            u_forward=10.0,
            run_simulation=False,
        )
        bl = result.configuration_results[0]
        depth = bl.trim_state[0]
        theta_deg = np.degrees(bl.trim_state[1])

        # 10 m/s trim: pos_d near -1.3, small positive theta
        assert -1.5 < depth < -1.1, f"Trim pos_d {depth:.3f}m outside expected range"
        assert -1.0 < theta_deg < 3.0, f"Trim pitch {theta_deg:.2f} deg outside expected range"
