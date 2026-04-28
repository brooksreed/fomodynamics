"""Open-loop test cases for Moth 3DOF model.

Tests trim equilibrium, impulse responses (via instantaneous derivatives),
and convention regression for the Moth longitudinal dynamics.

Tests validate that:
1. Trim equilibrium produces near-zero derivatives and minimal short-term drift.
2. Flap impulses produce the correct heave response direction and magnitude.
3. Elevator impulses produce the correct pitch response direction and magnitude.
4. Force sign conventions are consistent with the FRD frame.

Note on open-loop stability:
    The Moth 3DOF at trim is open-loop UNSTABLE (dominant eigenvalue ~+0.5-0.8).
    This is physically realistic: a foiling moth requires active control to
    maintain equilibrium. Impulse tests use instantaneous derivative checks
    (no simulation) to avoid instability concerns entirely.

Geometry and moment directions:
    Main foil: position_x=+0.55m (forward of CG), position_z=+1.82m (below CG).
        +flap -> increased AoA -> more lift (Fz < 0 in body frame).
        Because foil is FORWARD of CG, more lift -> nose-UP moment (q_dot > 0).
    Rudder:    position_x=-1.755m (aft of CG), position_z=+1.77m (below CG).
        +elevator -> increased AoA -> more lift on tail.
        Because rudder is AFT of CG, more lift -> nose-DOWN moment (q_dot < 0).

    IMPORTANT: The instantaneous moment direction differs from the re-trimmed
    behavior. When the trim solver re-balances the system after a flap increase,
    theta DECREASES (nose-down trim) because the solver reduces AoA elsewhere to
    re-achieve moment balance. So: +flap -> nose-UP moment (instantaneous) but
    nose-DOWN trim (re-trimmed). Tests in this file check instantaneous physics.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from fmd.simulator.moth_3d import Moth3D, ConstantSchedule, POS_D, THETA, W, Q, U
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.integrator import simulate

from fmd.simulator.control import ConstantControl
from fmd.simulator.components.moth_forces import (
    create_moth_components,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def moth():
    """Create a Moth3D instance for testing."""
    return Moth3D(MOTH_BIEKER_V3)


@pytest.fixture(scope="module")
def trim():
    """Find trim at 10 m/s with pinned targets for deterministic baseline.

    Pins both theta and pos_d so the trim point is fully determined
    (no multiple-solution ambiguity). Values chosen near the 10 m/s
    trim point: pos_d=-1.3m, theta=0.005 rad (~0.3 deg).
    """
    result = find_moth_trim(
        MOTH_BIEKER_V3, u_forward=10.0, target_theta=0.005, target_pos_d=-1.3,
    )
    assert result.success, f"Trim solver failed: residual={result.residual:.2e}"
    assert result.residual < 0.05, f"Trim residual too large: {result.residual:.2e}"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simulation parameters for short-duration tests.
_DT = 0.001          # 1 ms integration step
_EQ_DURATION = 0.02  # 20 ms for equilibrium (stays very close to trim)


# ---------------------------------------------------------------------------
# TestStaticEquilibrium
# ---------------------------------------------------------------------------

class TestStaticEquilibrium:
    """Verify that the trim point is a true equilibrium.

    The Moth is open-loop unstable, so we test over a very short window
    (20 ms) where the inherent instability has not yet amplified
    the tiny trim residual into a noticeable deviation.
    """

    def test_trim_equilibrium_bounded(self, moth, trim):
        """Simulate 20 ms from trim with trim controls; all states stay near trim."""
        x0 = jnp.array(trim.state)
        ctrl = ConstantControl(jnp.array(trim.control))

        result = simulate(moth, x0, dt=_DT, duration=_EQ_DURATION, control=ctrl)
        states = np.array(result.states)

        # No NaN or Inf
        assert not np.any(np.isnan(states)), "NaN detected in states"
        assert not np.any(np.isinf(states)), "Inf detected in states"

        # Maximum deviation from trim in each state channel
        trim_state = np.array(trim.state)
        deviation = np.abs(states - trim_state[None, :])
        max_dev = np.max(deviation, axis=0)

        # Tolerances per state over 20 ms: very tight
        # pos_d (m), theta (rad), w (m/s), q (rad/s), u (m/s)
        # u tolerance slightly relaxed with surge_enabled=True: small
        # residual thrust imbalance causes ~1.2e-3 m/s drift over 20 ms
        tol = np.array([1e-5, 1e-4, 1e-3, 1e-2, 2e-3])
        for i, name in enumerate(["pos_d", "theta", "w", "q", "u"]):
            assert max_dev[i] < tol[i], (
                f"{name} max deviation {max_dev[i]:.6e} exceeds tolerance {tol[i]}"
            )

    def test_trim_equilibrium_drift(self, moth, trim):
        """Quantify numerical drift over 20 ms at trim.

        Even with the open-loop instability, the drift over 20 ms should
        be extremely small since the trim residual is < 1e-4 and the
        instability has barely started to grow.
        """
        x0 = jnp.array(trim.state)
        ctrl = ConstantControl(jnp.array(trim.control))

        result = simulate(moth, x0, dt=_DT, duration=_EQ_DURATION, control=ctrl)
        states = np.array(result.states)

        drift = states[-1] - states[0]

        # Drift per state should be very small over 20 ms
        # u tolerance slightly relaxed: surge_enabled=True causes small drift
        drift_tol = np.array([1e-5, 1e-4, 1e-3, 1e-2, 2e-3])
        for i, name in enumerate(["pos_d", "theta", "w", "q", "u"]):
            assert abs(drift[i]) < drift_tol[i], (
                f"{name} drift {drift[i]:.2e} exceeds tolerance {drift_tol[i]:.2e}"
            )


# ---------------------------------------------------------------------------
# TestFlapImpulse
# ---------------------------------------------------------------------------

class TestFlapImpulse:
    """Verify heave response to main foil flap perturbations.

    Uses instantaneous derivative checks (no simulation needed).
    Validates both direction and magnitude of the response.

    Physics: +flap increases effective AoA on the main foil. Since
    lift ~ CL(AoA) * q_dyn * area, more AoA -> more upward lift ->
    w_dot < 0 (upward in NED). The foil is forward of CG, so more
    lift also creates a nose-up pitching moment, but this class only
    tests the heave (w_dot) channel.
    """

    def test_positive_flap_produces_upward_acceleration(self, moth, trim):
        """Positive flap increases lift -> upward heave accel (w_dot < 0).

        +flap -> increased AoA -> more lift (Fz more negative in body frame)
        -> w_dot < 0 (upward acceleration in NED).
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)
        ctrl_pert = ctrl.at[0].add(jnp.deg2rad(5.0))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        w_dot = float(deriv[W])

        assert w_dot < -0.5, (
            f"Expected significant upward accel from +5 deg flap, got w_dot={w_dot:.3f}"
        )

    def test_negative_flap_produces_downward_acceleration(self, moth, trim):
        """Negative flap decreases lift -> downward heave accel (w_dot > 0).

        -flap -> decreased AoA -> less lift -> w_dot > 0 (downward).
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)
        ctrl_pert = ctrl.at[0].add(jnp.deg2rad(-5.0))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        w_dot = float(deriv[W])

        assert w_dot > 0.5, (
            f"Expected significant downward accel from -5 deg flap, got w_dot={w_dot:.3f}"
        )

    def test_flap_magnitude_scaling(self, moth, trim):
        """5 deg flap produces ~5x the heave acceleration of 1 deg.

        Lift is approximately linear in flap angle for small deflections,
        so the w_dot ratio should be near 5.
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)

        deriv_1 = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(1.0)), 0.0)
        deriv_5 = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(5.0)), 0.0)

        w_dot_1 = float(deriv_1[W])
        w_dot_5 = float(deriv_5[W])

        # Both should be negative (upward), take ratio of magnitudes
        ratio = abs(w_dot_5) / abs(w_dot_1)
        assert 3.0 < ratio < 7.0, (
            f"Expected ~5x scaling, got {ratio:.1f}x "
            f"(w_dot_1={w_dot_1:.3f}, w_dot_5={w_dot_5:.3f})"
        )

    def test_flap_antisymmetric_response(self, moth, trim):
        """Equal positive and negative flap produce opposite-sign w_dot."""
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)

        deriv_pos = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(3.0)), 0.0)
        deriv_neg = moth.forward_dynamics(state, ctrl.at[0].add(jnp.deg2rad(-3.0)), 0.0)

        w_dot_pos = float(deriv_pos[W])
        w_dot_neg = float(deriv_neg[W])

        assert w_dot_pos * w_dot_neg < 0, (
            f"Expected opposite signs: w_dot(+3)={w_dot_pos:.3f}, w_dot(-3)={w_dot_neg:.3f}"
        )


# ---------------------------------------------------------------------------
# TestElevatorImpulse
# ---------------------------------------------------------------------------

class TestElevatorImpulse:
    """Verify pitch response to rudder elevator perturbations.

    Uses instantaneous derivative checks (no simulation needed).
    """

    def test_positive_elevator_produces_nose_down(self, moth, trim):
        """Positive elevator -> nose-down pitch acceleration (q_dot < 0).

        Rudder is aft (position_x < 0). +elevator -> more lift at tail
        -> nose-down moment -> q_dot < 0.
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)
        ctrl_pert = ctrl.at[1].add(jnp.deg2rad(3.0))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        q_dot = float(deriv[Q])

        assert q_dot < -0.5, (
            f"Expected significant nose-down accel from +3 deg elevator, "
            f"got q_dot={q_dot:.3f}"
        )

    def test_negative_elevator_produces_nose_up(self, moth, trim):
        """Negative elevator -> nose-up pitch acceleration (q_dot > 0).

        -elevator -> less lift at tail -> nose-up moment -> q_dot > 0.
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)
        ctrl_pert = ctrl.at[1].add(jnp.deg2rad(-3.0))

        deriv = moth.forward_dynamics(state, ctrl_pert, 0.0)
        q_dot = float(deriv[Q])

        assert q_dot > 0.5, (
            f"Expected significant nose-up accel from -3 deg elevator, "
            f"got q_dot={q_dot:.3f}"
        )

    def test_elevator_magnitude_scaling(self, moth, trim):
        """3 deg elevator produces ~3x the pitch acceleration of 1 deg."""
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)

        deriv_1 = moth.forward_dynamics(state, ctrl.at[1].add(jnp.deg2rad(1.0)), 0.0)
        deriv_3 = moth.forward_dynamics(state, ctrl.at[1].add(jnp.deg2rad(3.0)), 0.0)

        q_dot_1 = float(deriv_1[Q])
        q_dot_3 = float(deriv_3[Q])

        ratio = abs(q_dot_3) / abs(q_dot_1)
        assert 2.0 < ratio < 4.5, (
            f"Expected ~3x scaling, got {ratio:.1f}x "
            f"(q_dot_1={q_dot_1:.3f}, q_dot_3={q_dot_3:.3f})"
        )

    def test_combined_flap_elevator_pitch_dominance(self, moth, trim):
        """Flap+2 and elev+1 combined: net moment is nose-down (elevator dominates).

        The main foil is forward of CG (position_x=+0.55m) and the rudder is
        aft (position_x=-1.755m). Both generate upward lift when deflected
        positively, but the moments are opposite:
        - +flap -> nose-UP moment (forward foil increases lift)
        - +elev -> nose-DOWN moment (aft rudder increases lift)

        With the current geometry, the rudder moment arm is 3.2x longer than
        the main foil's. Even at +2 deg flap / +1 deg elevator, the rudder's
        nose-down moment dominates because its longer arm outweighs the main
        foil's larger area (area ratio 1.66, arm ratio 3.19).
        """
        state = jnp.array(trim.state)
        ctrl = jnp.array(trim.control)

        ctrl_both = ctrl.at[0].add(jnp.deg2rad(2.0)).at[1].add(jnp.deg2rad(1.0))
        deriv_both = moth.forward_dynamics(state, ctrl_both, 0.0)
        q_dot_both = float(deriv_both[Q])

        assert q_dot_both < 0, (
            f"Combined flap+2/elev+1 should be net nose-down, got q_dot={q_dot_both:.3f}"
        )


# ---------------------------------------------------------------------------
# TestConventionRegression
# ---------------------------------------------------------------------------

class TestConventionRegression:
    """Instantaneous force/moment sign convention checks.

    These are NOT full simulations -- they call component compute_moth()
    directly and check the sign of the returned forces/moments.
    """

    def test_positive_flap_increases_lift(self, trim):
        """At trim, positive flap produces more negative Fz (more upward lift)."""
        foil, *_ = create_moth_components(MOTH_BIEKER_V3)

        state = jnp.array(trim.state)
        u_forward = 10.0

        # Baseline: trim control
        ctrl_base = jnp.array(trim.control)
        force_base, _ = foil.compute_moth(state, ctrl_base, u_forward, t=0.0)

        # Perturbed: +5 deg flap
        ctrl_pos = ctrl_base.at[0].add(jnp.deg2rad(5.0))
        force_pos, _ = foil.compute_moth(state, ctrl_pos, u_forward, t=0.0)

        # Positive flap should make Fz MORE negative (more upward lift in FRD)
        assert float(force_pos[2]) < float(force_base[2]), (
            f"Expected Fz to decrease (more lift) with +flap. "
            f"Fz_base={float(force_base[2]):.4f}, Fz_pos={float(force_pos[2]):.4f}"
        )

    def test_aft_rudder_upward_force_produces_nose_down_moment(self, trim):
        """Aft rudder with upward lift produces nose-down moment (My < 0).

        The rudder is at position_x = -1.755 (aft of CG in FRD).
        Positive elevator angle -> positive AoA -> positive CL -> positive lift
        -> Fz < 0 (upward in FRD).
        M_y = -position_x * Fz = -(-1.755) * (negative) = negative -> nose-down.
        """
        _, rudder, *_ = create_moth_components(MOTH_BIEKER_V3)

        state = jnp.array(trim.state)
        u_forward = 10.0

        # Apply positive elevator to ensure upward lift from rudder
        ctrl = jnp.array(trim.control)
        ctrl = ctrl.at[1].add(jnp.deg2rad(3.0))

        force_r, moment_r = rudder.compute_moth(state, ctrl, u_forward, t=0.0)

        # Rudder position_x should be negative (aft)
        assert rudder.position_x < 0.0, (
            f"Expected rudder aft of CG (position_x < 0), got {rudder.position_x}"
        )

        # With positive elevator angle and the trim state,
        # the rudder should produce upward lift (Fz < 0)
        assert float(force_r[2]) < 0.0, (
            f"Expected upward rudder lift (Fz < 0) with +elevator, got Fz={float(force_r[2]):.4f}"
        )

        # Nose-down moment: My < 0
        assert float(moment_r[1]) < 0.0, (
            f"Expected nose-down moment (My < 0) from aft rudder with upward lift, "
            f"got My={float(moment_r[1]):.4f}"
        )


# ---------------------------------------------------------------------------
# TestOpenLoopFromTrimAtCalibrated Speeds (Phase 1.5 validation)
# ---------------------------------------------------------------------------

class TestOpenLoopFromTrim:
    """Open-loop validation at calibrated speeds.

    For each calibrated speed, run a 5s open-loop sim from trim state
    and verify states stay bounded (no divergence to inf/NaN). The moth
    is open-loop unstable so states will drift, but they should not
    diverge to infinity within 5s.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("speed", [10.0, 12.0, 15.0])
    def test_open_loop_5s_bounded(self, speed):
        """Open-loop sim from trim stays finite and bounded over 5s.

        8 m/s excluded: CasADi trim solver has convergence issues at
        this speed with NED sail thrust.
        """
        speed_moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(speed))
        result_trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=speed)
        assert result_trim.success or result_trim.residual < 0.1, (
            f"Trim failed at {speed} m/s: residual={result_trim.residual:.2e}"
        )

        x0 = jnp.array(result_trim.state)
        ctrl = ConstantControl(jnp.array(result_trim.control))
        sim = simulate(speed_moth, x0, dt=0.001, duration=5.0, control=ctrl)

        states = np.array(sim.states)
        assert not np.any(np.isnan(states)), f"NaN in states at {speed} m/s"
        assert not np.any(np.isinf(states)), f"Inf in states at {speed} m/s"

        # States should remain bounded (not diverge wildly)
        # pos_d should stay within [-3, 1] m (trim is near -1.3m)
        # theta should stay within [-90, 90] deg
        assert np.all(states[:, POS_D] > -3.0) and np.all(states[:, POS_D] < 1.0), (
            f"pos_d unbounded at {speed} m/s: range [{states[:,POS_D].min():.2f}, {states[:,POS_D].max():.2f}]"
        )
        assert np.all(np.abs(states[:, THETA]) < np.pi / 2), (
            f"theta unbounded at {speed} m/s: range [{np.degrees(states[:,THETA].min()):.1f}, {np.degrees(states[:,THETA].max()):.1f}] deg"
        )

    @pytest.mark.slow
    def test_small_pitch_perturbation_bounded(self, moth):
        """0.5 deg pitch perturbation from 10 m/s trim stays bounded over 5s."""
        result_trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result_trim.success

        x0 = jnp.array(result_trim.state)
        x0 = x0.at[THETA].add(np.radians(0.5))
        ctrl = ConstantControl(jnp.array(result_trim.control))
        sim = simulate(moth, x0, dt=0.001, duration=5.0, control=ctrl)

        states = np.array(sim.states)
        assert not np.any(np.isnan(states)), "NaN in states"
        assert not np.any(np.isinf(states)), "Inf in states"
        assert np.all(np.abs(states[:, POS_D]) < 2.0), "pos_d unbounded"
        assert np.all(np.abs(states[:, THETA]) < np.pi / 2), "theta unbounded"
