"""Tests for Moth 3DOF longitudinal dynamics model."""

import pytest
import warnings
import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx
from jax import Array

from fmd.simulator.moth_3d import (
    Moth3D,
    MothGeometry,
    ConstantSchedule,
    ConstantArraySchedule,
    POS_D, THETA, W, Q, U,
    MAIN_FLAP, RUDDER_ELEVATOR,
    MAIN_FLAP_MIN, MAIN_FLAP_MAX,
    RUDDER_ELEVATOR_MIN, RUDDER_ELEVATOR_MAX,
    _compute_cg_offset,
)
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator import simulate
from fmd.simulator.integrator import (
    result_with_meta,
    compute_aux_trajectory,
    SimulationResult,
)


class TestMoth3DBasics:
    """Basic construction and metadata tests."""

    def test_create_from_params(self):
        """Moth3D can be created from MothParams."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth is not None
        assert moth.total_mass == MOTH_BIEKER_V3.total_mass
        assert moth.iyy == pytest.approx(MOTH_BIEKER_V3.composite_pitch_inertia)
        assert moth.g == MOTH_BIEKER_V3.g

    def test_num_states_is_5(self):
        """Moth3D has 5 states."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.num_states == 5

    def test_num_controls_is_2(self):
        """Moth3D has 2 controls."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.num_controls == 2

    def test_state_names(self):
        """State names are correct."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.state_names == ("pos_d", "theta", "w", "q", "u")

    def test_control_names(self):
        """Control names are correct."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.control_names == ("main_flap_angle", "rudder_elevator_angle")

    def test_default_state(self):
        """Default state is CG above water, level, with nominal speed."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        assert state.shape == (5,)
        assert float(state[POS_D]) == pytest.approx(-1.3)
        assert float(state[THETA]) == pytest.approx(0.0)
        assert float(state[W]) == pytest.approx(0.0)
        assert float(state[Q]) == pytest.approx(0.0)
        assert float(state[U]) == pytest.approx(10.0)

    def test_has_force_components(self):
        """Moth3D has force component fields after construction."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.main_foil is not None
        assert moth.rudder is not None
        assert moth.sail is not None
        assert moth.hull_drag is not None


class TestMoth3DForwardSpeed:
    """Tests for forward speed schedule."""

    def test_default_speed_is_10(self):
        """Default forward speed is 10 m/s."""
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.get_forward_speed(0.0) == pytest.approx(10.0)
        assert moth.get_forward_speed(10.0) == pytest.approx(10.0)

    def test_custom_constant_speed(self):
        """Custom constant forward speed."""
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(8.0))
        assert moth.get_forward_speed(0.0) == pytest.approx(8.0)
        assert moth.get_forward_speed(5.0) == pytest.approx(8.0)

    def test_time_varying_speed(self):
        """Time-varying forward speed."""
        class LinearSchedule(eqx.Module):
            base: float
            rate: float
            def __call__(self, t):
                return self.base + self.rate * t

        # Linear ramp from 4 to 8 m/s over 10 seconds
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=LinearSchedule(4.0, 0.4))
        assert moth.get_forward_speed(0.0) == pytest.approx(4.0)
        assert moth.get_forward_speed(5.0) == pytest.approx(6.0)
        assert moth.get_forward_speed(10.0) == pytest.approx(8.0)


class TestMoth3DControl:
    """Tests for control interface."""

    def test_control_bounds_main_flap(self):
        """Main flap bounds are -10 deg to +15 deg."""
        moth = Moth3D(MOTH_BIEKER_V3)
        bounds = moth.control_bounds[MAIN_FLAP]
        assert bounds[0] == pytest.approx(np.deg2rad(-10), rel=0.01)
        assert bounds[1] == pytest.approx(np.deg2rad(15), rel=0.01)

    def test_control_bounds_elevator(self):
        """Rudder elevator bounds are -3 deg to +6 deg."""
        moth = Moth3D(MOTH_BIEKER_V3)
        bounds = moth.control_bounds[RUDDER_ELEVATOR]
        assert bounds[0] == pytest.approx(np.deg2rad(-3), rel=0.01)
        assert bounds[1] == pytest.approx(np.deg2rad(6), rel=0.01)

    def test_control_lower_bounds(self):
        """Control lower bounds property."""
        moth = Moth3D(MOTH_BIEKER_V3)
        lb = moth.control_lower_bounds
        assert lb.shape == (2,)
        assert float(lb[MAIN_FLAP]) == pytest.approx(MAIN_FLAP_MIN)
        assert float(lb[RUDDER_ELEVATOR]) == pytest.approx(RUDDER_ELEVATOR_MIN)

    def test_control_upper_bounds(self):
        """Control upper bounds property."""
        moth = Moth3D(MOTH_BIEKER_V3)
        ub = moth.control_upper_bounds
        assert ub.shape == (2,)
        assert float(ub[MAIN_FLAP]) == pytest.approx(MAIN_FLAP_MAX)
        assert float(ub[RUDDER_ELEVATOR]) == pytest.approx(RUDDER_ELEVATOR_MAX)

    def test_default_control_zeros(self):
        """Default control is zeros."""
        moth = Moth3D(MOTH_BIEKER_V3)
        ctrl = moth.default_control()
        assert ctrl.shape == (2,)
        assert float(ctrl[MAIN_FLAP]) == pytest.approx(0.0)
        assert float(ctrl[RUDDER_ELEVATOR]) == pytest.approx(0.0)

    def test_default_control_within_bounds(self):
        """Default control is within bounds."""
        moth = Moth3D(MOTH_BIEKER_V3)
        ctrl = moth.default_control()
        lb = moth.control_lower_bounds
        ub = moth.control_upper_bounds
        assert jnp.all(ctrl >= lb)
        assert jnp.all(ctrl <= ub)


class TestMoth3DDynamics:
    """Tests for forward dynamics with real force model."""

    def test_forward_dynamics_returns_shape_5(self):
        """Forward dynamics returns correct shape."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-0.15, 0.1, 0.2, 0.05, 10.0])
        control = jnp.array([0.1, 0.05])
        deriv = moth.forward_dynamics(state, control, t=1.0)
        assert deriv.shape == (5,)

    def test_dynamics_are_nonzero(self):
        """Dynamics produce nonzero derivatives (not placeholder)."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])
        deriv = moth.forward_dynamics(state, control)
        # With nonzero state and control, dynamics should be nonzero
        assert not jnp.allclose(deriv, jnp.zeros(5))

    def test_kinematics_pos_d_dot(self):
        """pos_d_dot = -u*sin(theta) + w*cos(theta)."""
        moth = Moth3D(MOTH_BIEKER_V3)
        theta = 0.1
        w = 0.5
        u_fwd = 10.0  # default speed
        state = jnp.array([-1.3, theta, w, 0.0, u_fwd])
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        expected_pos_d_dot = -u_fwd * jnp.sin(theta) + w * jnp.cos(theta)
        assert float(deriv[POS_D]) == pytest.approx(float(expected_pos_d_dot), rel=1e-6)

    def test_kinematics_theta_dot_equals_q(self):
        """theta_dot = q (kinematic relationship)."""
        moth = Moth3D(MOTH_BIEKER_V3)
        q_val = 0.3
        state = jnp.array([-1.3, 0.0, 0.0, q_val, 10.0])
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        assert float(deriv[THETA]) == pytest.approx(q_val, rel=1e-6)

    def test_nose_up_pitch_reduces_depth(self):
        """Nose-up pitch (theta > 0) with forward speed reduces depth (rises).

        pos_d_dot = -u*sin(theta) + w*cos(theta)
        With theta > 0 and w=0: pos_d_dot = -u*sin(theta) < 0 (depth decreases = rises)
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.1, 0.0, 0.0, 10.0])  # Nose-up, no heave velocity
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        # pos_d_dot should be negative (rising)
        assert float(deriv[POS_D]) < 0.0

    def test_positive_w_increases_depth(self):
        """Positive w (downward body velocity) increases depth.

        pos_d_dot = -u*sin(theta) + w*cos(theta)
        With theta=0 and w > 0: pos_d_dot = w > 0 (depth increases)
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.0, 1.0, 0.0, 10.0])  # Level with downward w
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        assert float(deriv[POS_D]) > 0.0

    def test_jit_compatible(self):
        """Forward dynamics is JIT-compatible."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.0, 0.0, 0.0, 10.0])
        control = jnp.array([0.0, 0.0])

        @jax.jit
        def jit_dynamics(s, c):
            return moth.forward_dynamics(s, c)

        deriv = jit_dynamics(state, control)
        assert deriv.shape == (5,)
        assert jnp.all(jnp.isfinite(deriv))

    def test_vmap_compatible(self):
        """Forward dynamics can be vmapped."""
        moth = Moth3D(MOTH_BIEKER_V3)
        states = jnp.array([
            [-1.3, 0.0, 0.0, 0.0, 10.0],
            [-1.2, 0.1, 0.2, 0.1, 10.0],
            [-1.1, -0.1, -0.2, -0.1, 10.0],
        ])
        control = moth.default_control()

        vmapped_dynamics = jax.vmap(
            lambda s: moth.forward_dynamics(s, control)
        )
        derivs = vmapped_dynamics(states)
        assert derivs.shape == (3, 5)
        assert jnp.all(jnp.isfinite(derivs))

    def test_all_derivatives_finite(self):
        """All derivative components are finite for various states."""
        moth = Moth3D(MOTH_BIEKER_V3)
        test_states = [
            jnp.array([-1.3, 0.0, 0.0, 0.0, 10.0]),
            jnp.array([-0.10, 0.1, 0.5, 0.2, 10.0]),
            jnp.array([-0.50, -0.05, -0.3, -0.1, 10.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 10.0]),  # At surface
        ]
        control = jnp.array([0.05, 0.02])
        for state in test_states:
            deriv = moth.forward_dynamics(state, control)
            assert jnp.all(jnp.isfinite(deriv)), f"Non-finite deriv for state={state}"


class TestMoth3DPhysics:
    """Tests for physics behavior of the integrated model."""

    def test_gravity_produces_downward_force_at_theta_zero(self):
        """At theta=0, gravity produces m*g downward force.

        When foil is above water (foil_depth < 0), foil lift is zero,
        so w_dot is dominated by gravity: w_dot ~ g.
        With effective foil position_z ~ 1.94 (after CG offset), need pos_d < -1.94.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        # Foil near surface: pos_d=-2.0, foil_depth = -2.0 + 1.94 ~ -0.06
        state = jnp.array([-2.0, 0.0, 0.0, 0.0, 10.0])
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        # w_dot should be positive (downward) and close to g
        assert float(deriv[W]) > 0.0
        # Should be close to g (gravity dominates, no foil lift)
        assert float(deriv[W]) == pytest.approx(moth.g, rel=0.1)

    def test_foil_produces_lift_at_foiling_depth(self):
        """At foiling depth (pos_d=-1.3), foil produces significant lift."""
        moth = Moth3D(MOTH_BIEKER_V3)
        # At foiling depth with slight nose-up pitch
        state = jnp.array([-1.3, 0.03, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.0])  # Some flap angle
        deriv = moth.forward_dynamics(state, control)
        # w_dot should be significantly different from pure gravity
        # (foil lift partially counteracts gravity)
        pure_gravity = moth.g
        assert abs(float(deriv[W]) - pure_gravity) > 1.0

    def test_no_foil_lift_when_foil_above_water(self):
        """When foil is above water (foil_depth < 0), main foil produces no lift.

        With effective position_z ~ 1.94 (after CG offset), foil_depth = pos_d + 1.94.
        For foil above water, need pos_d < -1.94.
        We compare foil-above-water vs foil-submerged.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        # Foil near surface: pos_d=-2.0, foil_depth = -2.0 + 1.94 ~ -0.06
        state_above = jnp.array([-2.0, 0.05, 0.0, 0.0, 10.0])
        # Foil submerged: pos_d=-1.3, foil well below surface
        state_depth = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.0])

        d_above = moth.forward_dynamics(state_above, control)
        d_depth = moth.forward_dynamics(state_depth, control)

        # Above water, w_dot should be more positive (less upward lift)
        # than at depth (where foil counteracts gravity)
        assert float(d_above[W]) > float(d_depth[W])

    def test_sail_moment_is_nose_down(self):
        """Sail moment is nose-down (effective ce_position_z ~ -2.0).

        The sail CE is above the CG (z ~ -2.0 in body frame after CG offset),
        so the forward thrust creates a nose-down pitching moment.
        With foils near/above water, q_dot should be dominated by the sail moment.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        # Foil near surface: foil_depth = -2.0 + 1.94 ~ -0.06 (minimal lift)
        state = jnp.array([-2.0, 0.0, 0.0, 0.0, 10.0])
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        # Sail moment ~ -2.0 * thrust (nose-down)
        # q_dot = total_My / Iyy
        # The sail moment dominates here (foil near surface has minimal contribution)
        assert float(deriv[Q]) < 0.0  # Nose-down pitch acceleration

    def test_elevator_deflection_changes_q_dot(self):
        """Elevator deflection changes pitch acceleration."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.0, 0.0, 0.0, 10.0])
        ctrl_zero = jnp.array([0.0, 0.0])
        ctrl_elev = jnp.array([0.0, 0.05])

        d_zero = moth.forward_dynamics(state, ctrl_zero)
        d_elev = moth.forward_dynamics(state, ctrl_elev)

        # Elevator deflection should change q_dot
        assert float(d_zero[Q]) != pytest.approx(float(d_elev[Q]), abs=0.01)

    def test_flap_deflection_changes_w_dot(self):
        """Flap deflection changes vertical acceleration."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.0, 0.0, 0.0, 10.0])
        ctrl_zero = jnp.array([0.0, 0.0])
        ctrl_flap = jnp.array([0.1, 0.0])

        d_zero = moth.forward_dynamics(state, ctrl_zero)
        d_flap = moth.forward_dynamics(state, ctrl_flap)

        # Flap deflection should change w_dot (more lift at depth)
        assert float(d_zero[W]) != pytest.approx(float(d_flap[W]), abs=0.01)

    def test_stable_short_simulation(self):
        """Short simulation produces finite derivatives.

        Note: The untrimmed default state diverges because the model
        is not at equilibrium. The trim finder (WI-3) will provide
        proper equilibrium initial conditions. For now, we verify
        that the dynamics produce finite values at the initial state
        and for a few initial timesteps.
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        initial = moth.default_state()
        # Verify derivatives are finite at initial state
        deriv = moth.forward_dynamics(initial, moth.default_control())
        assert jnp.all(jnp.isfinite(deriv)), "Derivatives should be finite at default state"
        # Short simulation (0.01s) should produce finite states
        result = simulate(moth, initial, dt=0.001, duration=0.01)
        assert jnp.all(jnp.isfinite(result.states)), "Short simulation should be finite"


class TestMoth3DSimulation:
    """Tests for simulation infrastructure."""

    def test_simulate_runs(self):
        """Simulation runs without error."""
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        assert result.times.shape[0] == 201
        assert result.states.shape == (201, 5)
        assert result.controls.shape == (201, 2)

    def test_theta_wrapping(self):
        """Theta should be wrapped to [-pi, pi]."""
        moth = Moth3D(MOTH_BIEKER_V3)
        # Start with theta just under 2*pi
        state = jnp.array([-1.3, 6.0, 0.0, 0.0, 10.0])
        wrapped = moth.post_step(state)
        # 6.0 rad should wrap to approximately -0.28 rad
        assert float(wrapped[THETA]) == pytest.approx(6.0 - 2 * np.pi, abs=0.01)
        assert -np.pi <= float(wrapped[THETA]) <= np.pi

    def test_theta_wrapping_negative(self):
        """Theta wrapping works for negative angles."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, -4.0, 0.0, 0.0, 10.0])
        wrapped = moth.post_step(state)
        # -4.0 rad should wrap to approximately 2.28 rad
        expected = -4.0 + 2 * np.pi
        assert float(wrapped[THETA]) == pytest.approx(expected, abs=0.01)
        assert -np.pi <= float(wrapped[THETA]) <= np.pi

    def test_post_step_preserves_other_states(self):
        """post_step only modifies theta."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-0.15, 6.0, 0.3, 0.1, 10.0])
        wrapped = moth.post_step(state)
        assert float(wrapped[POS_D]) == pytest.approx(-0.15)
        assert float(wrapped[W]) == pytest.approx(0.3)
        assert float(wrapped[Q]) == pytest.approx(0.1)
        assert float(wrapped[U]) == pytest.approx(10.0)


class TestMoth3DJacobian:
    """Tests for Jacobian computation."""

    def test_state_jacobian_shape(self):
        """State Jacobian has correct shape."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()
        A = moth.get_state_jacobian(state, control)
        assert A.shape == (5, 5)

    def test_control_jacobian_shape(self):
        """Control Jacobian has correct shape."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()
        B = moth.get_control_jacobian(state, control)
        assert B.shape == (5, 2)

    def test_state_jacobian_nonzero(self):
        """State Jacobian is nonzero (real dynamics have state coupling)."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.05, 0.1, 0.05, 10.0])
        control = jnp.array([0.05, 0.02])
        A = moth.get_state_jacobian(state, control)
        # Should not be all zeros
        assert float(jnp.max(jnp.abs(A))) > 1e-6

    def test_control_jacobian_nonzero(self):
        """Control Jacobian is nonzero (controls affect dynamics)."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.05, 0.1, 0.05, 10.0])
        control = jnp.array([0.05, 0.02])
        B = moth.get_control_jacobian(state, control)
        # Should not be all zeros
        assert float(jnp.max(jnp.abs(B))) > 1e-6


class TestMoth3DEquinoxModule:
    """Tests for Equinox module properties."""

    def test_is_pytree(self):
        """Moth3D is a valid JAX PyTree."""
        moth = Moth3D(MOTH_BIEKER_V3)
        leaves, treedef = jax.tree_util.tree_flatten(moth)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed is not None

    def test_pytree_roundtrip_preserves_dynamics(self):
        """PyTree round-trip preserves dynamics behavior."""
        moth = Moth3D(MOTH_BIEKER_V3)
        leaves, treedef = jax.tree_util.tree_flatten(moth)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        state = jnp.array([-1.3, 0.05, 0.1, 0.05, 10.0])
        control = jnp.array([0.05, 0.02])

        deriv1 = moth.forward_dynamics(state, control)
        deriv2 = reconstructed.forward_dynamics(state, control)
        assert jnp.allclose(deriv1, deriv2)


class TestMoth3DAddedMass:
    """Tests for added mass effects on dynamics."""

    def test_added_mass_reduces_w_dot(self):
        """Higher added mass should reduce w_dot for same force."""
        import attrs

        # Full damping (default)
        moth_full = Moth3D(MOTH_BIEKER_V3)

        # No added mass
        params_no_am = attrs.evolve(
            MOTH_BIEKER_V3,
            added_mass_heave=0.0,
            added_inertia_pitch=0.0,
        )
        moth_no_am = Moth3D(params_no_am)

        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        deriv_full = moth_full.forward_dynamics(state, control)
        deriv_no_am = moth_no_am.forward_dynamics(state, control)

        # With added mass, |w_dot| should be smaller
        assert abs(float(deriv_full[W])) < abs(float(deriv_no_am[W]))

    def test_added_inertia_reduces_q_dot(self):
        """Higher added inertia should reduce q_dot for same moment."""
        import attrs

        # Full damping (default)
        moth_full = Moth3D(MOTH_BIEKER_V3)

        # No added mass
        params_no_am = attrs.evolve(
            MOTH_BIEKER_V3,
            added_mass_heave=0.0,
            added_inertia_pitch=0.0,
        )
        moth_no_am = Moth3D(params_no_am)

        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        deriv_full = moth_full.forward_dynamics(state, control)
        deriv_no_am = moth_no_am.forward_dynamics(state, control)

        # With added inertia, |q_dot| should be smaller
        assert abs(float(deriv_full[Q])) < abs(float(deriv_no_am[Q]))

    def test_added_mass_extracted_from_params(self):
        """Moth3D should extract added mass from params."""
        import attrs

        custom_params = attrs.evolve(
            MOTH_BIEKER_V3,
            added_mass_heave=20.0,
            added_inertia_pitch=15.0,
        )
        moth = Moth3D(custom_params)

        assert moth.added_mass_heave == 20.0
        assert moth.added_inertia_pitch == 15.0


class TestMoth3DSailorSchedule:
    """Tests for time-varying sailor position schedule (Wave 4F)."""

    def test_constant_schedule_matches_default(self):
        """Explicit constant schedule produces same forward_dynamics as default.

        When sailor_position_schedule returns the same constant position
        as params.sailor_position, results must be numerically identical.
        """
        moth_default = Moth3D(MOTH_BIEKER_V3)
        sp = MOTH_BIEKER_V3.sailor_position
        _sp = (float(sp[0]), float(sp[1]), float(sp[2]))
        moth_sched = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(_sp)),
        )

        state = jnp.array([-1.3, 0.05, 0.1, 0.02, 10.0])
        control = jnp.array([0.05, 0.02])

        d_default = moth_default.forward_dynamics(state, control, t=1.0)
        d_sched = moth_sched.forward_dynamics(state, control, t=1.0)
        assert jnp.allclose(d_default, d_sched, atol=1e-10), (
            f"default={d_default}, sched={d_sched}"
        )

    def test_step_change_moment_direction(self):
        """Sailor moving forward produces nose-down q_dot shift.

        Sailor forward shifts system CG forward, increasing the moment arm
        of the main foil (which produces nose-up moment), but the CG shift
        also moves the reference point. The net effect of forward CG shift
        is to produce a more nose-down (or less nose-up) pitch tendency.
        """
        sp = MOTH_BIEKER_V3.sailor_position
        sp_nominal = (float(sp[0]), float(sp[1]), float(sp[2]))
        sp_forward = (float(sp[0]) + 0.3, float(sp[1]), float(sp[2]))

        moth_nominal = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(sp_nominal)),
        )
        moth_forward = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(sp_forward)),
        )

        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        d_nominal = moth_nominal.forward_dynamics(state, control)
        d_forward = moth_forward.forward_dynamics(state, control)

        # Forward CG should produce more nose-down (smaller q_dot)
        assert float(d_forward[Q]) < float(d_nominal[Q])

    def test_linear_ramp_smooth_dynamics(self):
        """Linear ramp schedule produces finite, smooth derivatives."""
        sp = MOTH_BIEKER_V3.sailor_position
        x0, y0, z0 = float(sp[0]), float(sp[1]), float(sp[2])

        class RampPositionSchedule(eqx.Module):
            x0: float
            y0: float
            z0: float
            rate: float
            def __call__(self, t):
                dx = self.rate * t
                return jnp.array([self.x0 + dx, self.y0, self.z0])

        moth = Moth3D(MOTH_BIEKER_V3, sailor_position_schedule=RampPositionSchedule(x0, y0, z0, 0.02))
        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        derivs = []
        for t in [0.0, 1.0, 2.0, 5.0, 10.0]:
            d = moth.forward_dynamics(state, control, t=t)
            assert jnp.all(jnp.isfinite(d)), f"Non-finite deriv at t={t}"
            derivs.append(d)

        # q_dot should monotonically decrease (more nose-down as sailor moves fwd)
        q_dots = [float(d[Q]) for d in derivs]
        for i in range(1, len(q_dots)):
            assert q_dots[i] <= q_dots[i - 1] + 1e-8, (
                f"q_dot not monotonically decreasing: t={[0,1,2,5,10][i]}, "
                f"q_dot={q_dots[i]:.6f} > prev={q_dots[i-1]:.6f}"
            )

    def test_composite_inertia_changes(self):
        """Different sailor positions produce different q_dot magnitudes."""
        sp = MOTH_BIEKER_V3.sailor_position
        sp_near = (float(sp[0]) * 0.5, float(sp[1]), float(sp[2]) * 0.5)
        sp_far = (float(sp[0]) * 2.0, float(sp[1]), float(sp[2]) * 2.0)

        moth_near = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(sp_near)),
        )
        moth_far = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(sp_far)),
        )

        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        d_near = moth_near.forward_dynamics(state, control)
        d_far = moth_far.forward_dynamics(state, control)

        # Sailor further from hull CG = larger composite inertia = smaller |q_dot|
        # (Moments also change due to different CG offset, but inertia effect dominates)
        assert float(d_near[Q]) != pytest.approx(float(d_far[Q]), abs=1e-6)

    def test_jit_compatible(self):
        """Sailor position schedule works under jax.jit."""
        sp = MOTH_BIEKER_V3.sailor_position
        _sp = (float(sp[0]), float(sp[1]), float(sp[2]))
        moth = Moth3D(
            MOTH_BIEKER_V3,
            sailor_position_schedule=ConstantArraySchedule(jnp.array(_sp)),
        )

        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        @jax.jit
        def jit_dynamics(s, c, t):
            return moth.forward_dynamics(s, c, t)

        deriv = jit_dynamics(state, control, 1.0)
        assert deriv.shape == (5,)
        assert jnp.all(jnp.isfinite(deriv))

    def test_kinematic_correction_zero_for_constant(self):
        """Constant schedule produces zero kinematic correction.

        pos_d_dot should match the original formula exactly:
        pos_d_dot = -u*sin(theta) + w*cos(theta)
        """
        moth = Moth3D(MOTH_BIEKER_V3)
        theta = 0.1
        w = 0.5
        u_fwd = 10.0  # default speed
        state = jnp.array([-1.3, theta, w, 0.0, u_fwd])
        control = jnp.zeros(2)
        deriv = moth.forward_dynamics(state, control)
        expected_pos_d_dot = -u_fwd * jnp.sin(theta) + w * jnp.cos(theta)
        assert float(deriv[POS_D]) == pytest.approx(float(expected_pos_d_dot), rel=1e-6)

    def test_full_simulation_with_schedule(self):
        """simulate() completes without error using a sailor schedule."""
        sp = MOTH_BIEKER_V3.sailor_position
        x0, y0, z0 = float(sp[0]), float(sp[1]), float(sp[2])

        class OscillatingPositionSchedule(eqx.Module):
            x0: float
            y0: float
            z0: float
            amplitude: float
            frequency: float
            def __call__(self, t):
                dx = self.amplitude * jnp.sin(self.frequency * t)
                return jnp.array([self.x0 + dx, self.y0, self.z0])

        moth = Moth3D(MOTH_BIEKER_V3, sailor_position_schedule=OscillatingPositionSchedule(x0, y0, z0, 0.05, 0.5))
        result = simulate(moth, moth.default_state(), dt=0.005, duration=2.0)
        n = result.times.shape[0]
        assert n >= 401  # 2.0/0.005 = 400 steps + initial; ceil may add 1
        assert result.states.shape == (n, 5)
        assert jnp.all(jnp.isfinite(result.states[:10]))


class TestMoth3DSurgeDynamics:
    """Tests for surge dynamics (u_dot equation)."""

    def test_frozen_surge_zero_u_dot(self):
        """With surge_enabled=False, u_dot is always zero."""
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=False)
        state = jnp.array([-1.3, 0.1, 0.2, 0.05, 10.0])
        control = jnp.array([0.05, 0.02])
        deriv = moth.forward_dynamics(state, control)
        assert float(deriv[U]) == pytest.approx(0.0)

    def test_active_surge_nonzero_u_dot(self):
        """With surge_enabled=True and nonzero theta, u_dot is nonzero.

        gravity_fx = -m*g*sin(theta), so any nonzero theta produces surge force.
        """
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        state = jnp.array([-1.3, 0.1, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])
        deriv = moth.forward_dynamics(state, control)
        assert float(deriv[U]) != pytest.approx(0.0, abs=1e-6)

    def test_coriolis_w_dot_sign(self):
        """Coriolis term +q*u is present in w_dot equation.

        w_dot = total_fz / m_eff + q * u
        Verify by computing total_fz from components independently,
        then checking w_dot matches the formula including Coriolis.
        """
        from fmd.simulator.moth_3d import _compute_cg_offset

        moth = Moth3D(MOTH_BIEKER_V3)
        q_val = 0.1
        state = jnp.array([-1.3, 0.0, 0.0, q_val, 10.0])
        control = jnp.zeros(2)
        u_safe = jnp.maximum(state[U], 0.1)

        # Compute total_fz from components (same as forward_dynamics)
        r_sailor = moth.sailor_position_schedule(0.0)
        cg = _compute_cg_offset(r_sailor, moth.sailor_mass, moth.total_mass)
        f_foil, _ = moth.main_foil.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_rudder, _ = moth.rudder.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_sail, _ = moth.sail.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_hull, _ = moth.hull_drag.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_main_strut, _ = moth.main_strut.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_rudder_strut, _ = moth.rudder_strut.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        gravity_fz = moth.total_mass * moth.g * jnp.cos(state[THETA])
        total_fz = (f_foil[2] + f_rudder[2] + f_sail[2] + f_hull[2]
                    + f_main_strut[2] + f_rudder_strut[2] + gravity_fz)

        m_eff_heave = moth.total_mass + moth.added_mass_heave
        expected_w_dot = float(total_fz) / m_eff_heave + q_val * float(u_safe)

        d = moth.forward_dynamics(state, control)
        actual_w_dot = float(d[W])

        # w_dot should match total_fz/m_eff + q*u exactly
        np.testing.assert_allclose(actual_w_dot, expected_w_dot, atol=1e-10)
        # Coriolis contribution q*u = 1.0 should make w_dot more positive
        # than force-only part
        force_only = float(total_fz) / m_eff_heave
        assert actual_w_dot > force_only

    def test_coriolis_u_dot_sign(self):
        """Coriolis term -q*w is present in u_dot.

        u_dot = total_fx / m_eff_surge - q * w
        Verify by computing total_fx from components independently,
        then checking u_dot matches the formula including Coriolis.
        """
        from fmd.simulator.moth_3d import _compute_cg_offset

        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        q_val, w_val = 0.1, 2.0
        state = jnp.array([-1.3, 0.0, w_val, q_val, 10.0])
        control = jnp.zeros(2)
        u_safe = jnp.maximum(state[U], 0.1)

        # Compute total_fx from components (same as forward_dynamics)
        r_sailor = moth.sailor_position_schedule(0.0)
        cg = _compute_cg_offset(r_sailor, moth.sailor_mass, moth.total_mass)
        f_foil, _ = moth.main_foil.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_rudder, _ = moth.rudder.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_sail, _ = moth.sail.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_hull, _ = moth.hull_drag.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_main_strut, _ = moth.main_strut.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        f_rudder_strut, _ = moth.rudder_strut.compute_moth(state, control, u_safe, 0.0, cg_offset=cg)
        gravity_fx = -moth.total_mass * moth.g * jnp.sin(state[THETA])
        total_fx = (f_foil[0] + f_rudder[0] + f_sail[0] + f_hull[0]
                    + f_main_strut[0] + f_rudder_strut[0] + gravity_fx)

        m_eff_surge = moth.total_mass + moth.added_mass_surge
        expected_u_dot = float(total_fx) / m_eff_surge - q_val * w_val

        d = moth.forward_dynamics(state, control)
        actual_u_dot = float(d[U])

        # u_dot should match total_fx/m_eff - q*w exactly
        np.testing.assert_allclose(actual_u_dot, expected_u_dot, atol=1e-10)
        # Coriolis contribution -q*w = -0.2 should make u_dot more negative
        # than force-only part
        force_only = float(total_fx) / m_eff_surge
        assert actual_u_dot < force_only

    def test_gravity_fx_negative_when_pitched_up(self):
        """With theta > 0, gravity_fx = -m*g*sin(theta) < 0, decelerating surge.

        Increasing theta should make u_dot more negative (or less positive).
        """
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        # Small pitch
        state_small = jnp.array([-1.3, 0.02, 0.0, 0.0, 10.0])
        # Larger pitch
        state_large = jnp.array([-1.3, 0.10, 0.0, 0.0, 10.0])
        control = jnp.zeros(2)

        d_small = moth.forward_dynamics(state_small, control)
        d_large = moth.forward_dynamics(state_large, control)

        # Larger theta -> more negative gravity_fx -> smaller u_dot
        assert float(d_large[U]) < float(d_small[U])

    def test_surge_jit_compatible(self):
        """surge_enabled=True model can be jit-compiled."""
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        state = jnp.array([-1.3, 0.05, 0.0, 0.0, 10.0])
        control = jnp.array([0.05, 0.02])

        @jax.jit
        def jit_dynamics(s, c):
            return moth.forward_dynamics(s, c)

        deriv = jit_dynamics(state, control)
        assert deriv.shape == (5,)
        assert jnp.all(jnp.isfinite(deriv))

    def test_trim_converges_with_surge(self):
        """CasADi trim solver converges (always has surge as free variable)."""
        from fmd.simulator.trim_casadi import find_moth_trim

        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result.residual < 0.05, f"Trim did not converge: residual={result.residual}"

    def test_calibrated_thrust_balances_at_10ms(self):
        """Calibrated sail thrust lookup gives u ~ 10.0 at surge-enabled trim.

        MOTH_BIEKER_V3 has a calibrated thrust lookup table where the
        surge-enabled foiling trim equilibrium is near the target speed.
        Buoyancy is disabled during verification because the buoyancy
        terms (though zero at foiling height) can cause the optimizer
        to converge to different local minima.
        """
        import attrs
        from fmd.simulator.trim_casadi import find_moth_trim

        p = attrs.evolve(MOTH_BIEKER_V3, hull_buoyancy_coeff=0.0)
        moth = Moth3D(p, surge_enabled=True)
        result = find_moth_trim(p, u_forward=10.0)
        trim = result
        u = float(trim.state[U])
        assert abs(u - 10.0) < 0.5, f"trim u={u:.4f}, expected ~10.0"


    def test_trim_converges_surge_disabled(self):
        """Trim at 10 m/s with surge_enabled=False succeeds.

        The CasADi solver always treats surge as a free variable
        regardless of the JAX model's surge_enabled flag.
        """
        from fmd.simulator.trim_casadi import find_moth_trim

        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result.success, f"Trim failed: residual={result.residual:.2e}"
        assert result.residual < 0.05

        # Verify the trim state works with surge_enabled=False model
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=False)
        deriv = moth.forward_dynamics(jnp.array(result.state), jnp.array(result.control))
        assert float(deriv[U]) == pytest.approx(0.0), "u_dot should be zero with surge disabled"

    def test_surge_disabled_u_constant_during_sim(self):
        """With surge_enabled=False, state[U] stays constant through simulation.

        Simulate 1s with a perturbed theta. Even though theta perturbation
        would produce gravity_fx in the surge-enabled model, u should remain
        constant at 10 m/s throughout.
        """
        from fmd.simulator.trim_casadi import find_moth_trim
        from fmd.simulator.integrator import simulate, ConstantControl

        result = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert result.success

        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=False,
                      u_forward=ConstantSchedule(10.0))

        # Perturb theta by 2 degrees
        perturbed_state = result.state.copy()
        perturbed_state[THETA] += np.radians(2.0)
        perturbed_state[W] = perturbed_state[U] * np.tan(perturbed_state[THETA])

        sim = simulate(
            moth,
            jnp.array(perturbed_state),
            dt=0.005,
            duration=1.0,
            control=ConstantControl(jnp.array(result.control)),
        )
        states = np.array(sim.states)
        u_values = states[:, U]

        # U should be exactly constant (= initial value) throughout
        np.testing.assert_allclose(
            u_values, perturbed_state[U], atol=1e-10,
            err_msg="state[U] should remain constant with surge_enabled=False"
        )

    def test_lqr_k_u_column_zero_surge_disabled(self):
        """LQR gain matrix K has zero column for surge state.

        Since surge is not controllable by flap/elevator, the gain matrix
        should have K[:, U_idx] = [0, 0] regardless of surge_enabled.
        """
        from fmd.simulator.moth_lqr import design_moth_lqr

        lqr_result = design_moth_lqr(MOTH_BIEKER_V3, u_forward=10.0)
        K = lqr_result.K

        # K is 2x5 (2 controls, 5 states: pos_d, theta, w, q, u)
        # The U column (index 4) should be zero
        k_u_col = K[:, U]
        np.testing.assert_allclose(
            k_u_col, 0.0, atol=1e-10,
            err_msg="K[:, U] should be zero — surge is not controllable by flap/elevator"
        )


class TestMothGeometry:
    """Tests for get_geometry() method."""

    def test_returns_moth_geometry(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        geom = moth.get_geometry()
        assert isinstance(geom, MothGeometry)

    def test_cg_offset_nonzero(self):
        """Sailor mass produces a nonzero CG offset."""
        moth = Moth3D(MOTH_BIEKER_V3)
        geom = moth.get_geometry()
        assert np.any(geom.cg_offset != 0.0)

    def test_adjusted_positions_differ_from_raw(self):
        """CG-adjusted foil positions differ from raw hull-CG positions."""
        moth = Moth3D(MOTH_BIEKER_V3)
        geom = moth.get_geometry()
        assert not np.allclose(geom.main_foil_position, geom.main_foil_raw_position)
        assert not np.allclose(geom.rudder_position, geom.rudder_raw_position)

    def test_adjusted_positions_match_dynamics(self):
        """get_geometry positions match what forward_dynamics uses."""
        moth = Moth3D(MOTH_BIEKER_V3)
        geom = moth.get_geometry(t=0.0)
        # Manually compute what forward_dynamics uses
        r_sailor = moth.sailor_position_schedule(0.0)
        cg = _compute_cg_offset(r_sailor, moth.sailor_mass, moth.total_mass)
        expected_main_x = moth.main_foil.position_x - float(cg[0])
        expected_main_z = moth.main_foil.position_z - float(cg[2])
        assert geom.main_foil_position[0] == pytest.approx(expected_main_x, abs=1e-10)
        assert geom.main_foil_position[2] == pytest.approx(expected_main_z, abs=1e-10)

    def test_composite_iyy_matches_params(self):
        """Composite Iyy matches the nominal value from params."""
        moth = Moth3D(MOTH_BIEKER_V3)
        geom = moth.get_geometry()
        assert geom.composite_iyy == pytest.approx(
            MOTH_BIEKER_V3.composite_pitch_inertia, rel=1e-6)

    def test_time_varying_geometry(self):
        """Geometry changes with time-varying sailor schedule."""
        sp = MOTH_BIEKER_V3.sailor_position
        x0, y0, z0 = float(sp[0]), float(sp[1]), float(sp[2])

        class RampPositionSchedule(eqx.Module):
            x0: float
            y0: float
            z0: float
            rate: float
            def __call__(self, t):
                return jnp.array([self.x0 + self.rate * t, self.y0, self.z0])

        moth = Moth3D(MOTH_BIEKER_V3, sailor_position_schedule=RampPositionSchedule(x0, y0, z0, 0.1))
        g0 = moth.get_geometry(t=0.0)
        g5 = moth.get_geometry(t=5.0)
        # CG offset should differ
        assert not np.allclose(g0.cg_offset, g5.cg_offset)
        # Main foil adjusted position should differ
        assert not np.allclose(g0.main_foil_position, g5.main_foil_position)


class TestMothComputeAux:
    """Tests for compute_aux method and aux_names."""

    def test_aux_names_length(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.num_aux == 32
        assert len(moth.aux_names) == 32

    def test_aux_names_correct(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        assert moth.aux_names[0] == "main_df"
        assert moth.aux_names[1] == "rudder_df"
        assert moth.aux_names[4] == "pos_d_dot"
        assert moth.aux_names[11] == "hull_buoyancy"
        assert moth.aux_names[12] == "cg_offset_x"
        assert moth.aux_names[13] == "cg_offset_z"
        assert moth.aux_names[14] == "main_alpha_geo"
        assert moth.aux_names[15] == "main_alpha_eff"
        assert moth.aux_names[16] == "rudder_alpha_geo"
        assert moth.aux_names[17] == "rudder_alpha_eff"
        assert moth.aux_names[18] == "total_fx"
        assert moth.aux_names[19] == "total_fz"
        assert moth.aux_names[20] == "total_my"
        assert moth.aux_names[21] == "u_fwd"

    def test_compute_aux_shape(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()
        aux = moth.compute_aux(state, control)
        assert aux.shape == (32,)

    def test_compute_aux_finite(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()
        aux = moth.compute_aux(state, control)
        assert jnp.all(jnp.isfinite(aux))

    def test_depth_factor_near_one_at_trim(self):
        """At trim, foils are well submerged so depth factor ~ 1."""
        from fmd.simulator.trim_casadi import find_moth_trim

        moth = Moth3D(MOTH_BIEKER_V3)
        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        aux = moth.compute_aux(trim.state, trim.control)
        main_df = float(aux[0])
        rudder_df = float(aux[1])
        assert main_df == pytest.approx(1.0, abs=0.01)
        assert rudder_df == pytest.approx(1.0, abs=0.01)

    def test_depth_factor_low_when_above_water(self):
        """When boat is well above water, depth factors should be low."""
        moth = Moth3D(MOTH_BIEKER_V3)
        # pos_d = -2.0 -> foils well above water
        state = jnp.array([-2.0, 0.0, 0.0, 0.0, 10.0])
        control = jnp.zeros(2)
        aux = moth.compute_aux(state, control)
        main_df = float(aux[0])
        assert main_df < 0.1

    def test_pos_d_dot_matches_dynamics(self):
        """pos_d_dot from aux matches the derivative from forward_dynamics."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = jnp.array([-1.3, 0.05, 0.1, 0.02, 10.0])
        control = jnp.array([0.05, 0.02])
        deriv = moth.forward_dynamics(state, control)
        aux = moth.compute_aux(state, control)
        assert float(aux[4]) == pytest.approx(float(deriv[POS_D]), rel=1e-10)

    def test_cg_offset_matches_geometry(self):
        """cg_offset from aux matches get_geometry."""
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()
        geom = moth.get_geometry(t=0.0)
        aux = moth.compute_aux(state, control, t=0.0)
        assert float(aux[12]) == pytest.approx(geom.cg_offset[0], abs=1e-10)
        assert float(aux[13]) == pytest.approx(geom.cg_offset[2], abs=1e-10)

    def test_jit_compatible(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()
        control = moth.default_control()

        @jax.jit
        def jit_aux(s, c):
            return moth.compute_aux(s, c)

        aux = jit_aux(state, control)
        assert aux.shape == (32,)
        assert jnp.all(jnp.isfinite(aux))

    def test_compute_aux_with_wave_env(self):
        """Verify aux depth factors account for wave elevation."""
        from fmd.simulator.params import WAVE_REGULAR_1M
        from fmd.simulator import Environment
        moth = Moth3D(MOTH_BIEKER_V3)
        env = Environment.with_waves(WAVE_REGULAR_1M)
        state = moth.default_state()
        control = jnp.zeros(moth.num_controls)
        # At t where wave elevation is nonzero, aux should differ from calm
        aux_calm = moth.compute_aux(state, control, t=0.0)
        aux_wave = moth.compute_aux(state, control, t=0.5, env=env)
        # Depth factors should differ (wave changes effective depth)
        assert aux_calm[0] != aux_wave[0]  # main_df differs

    def test_compute_aux_with_surge_enabled(self):
        """Verify aux works with surge dynamics enabled."""
        moth = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        state = moth.default_state()
        control = jnp.zeros(moth.num_controls)
        aux = moth.compute_aux(state, control, t=0.0)
        assert aux.shape == (moth.num_aux,)
        assert jnp.all(jnp.isfinite(aux))
        # u_fwd should come from state[U], not schedule
        u_fwd_idx = moth.aux_names.index("u_fwd")
        assert float(aux[u_fwd_idx]) == pytest.approx(float(state[U]), rel=1e-10)


class TestComputeAuxTrajectory:
    """Tests for compute_aux_trajectory integration."""

    def test_returns_dict_with_correct_keys(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        aux_dict = compute_aux_trajectory(moth, result)
        assert set(aux_dict.keys()) == set(moth.aux_names)

    def test_correct_shapes(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        aux_dict = compute_aux_trajectory(moth, result)
        n_steps = result.times.shape[0]
        for name in moth.aux_names:
            assert aux_dict[name].shape == (n_steps,), f"{name} shape mismatch"

    def test_empty_for_no_aux_system(self):
        """Systems with num_aux=0 return empty dict."""
        from fmd.simulator import SimplePendulum
        from fmd.simulator.params import PENDULUM_1M
        pend = SimplePendulum(PENDULUM_1M)
        result = simulate(pend, jnp.array([0.5, 0.0]), dt=0.01, duration=0.5)
        aux_dict = compute_aux_trajectory(pend, result)
        assert aux_dict == {}


class TestResultWithMetaAux:
    """Tests for result_with_meta aux integration and merge behavior."""

    def test_auto_populates_aux(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        rich = result_with_meta(moth, result)
        assert "main_df" in rich.outputs
        assert "rudder_df" in rich.outputs
        assert "pos_d_dot" in rich.outputs
        assert rich.outputs["main_df"].shape[0] == result.times.shape[0]

    def test_no_aux_system_empty_outputs(self):
        """Systems without aux produce no auto-populated outputs."""
        from fmd.simulator import SimplePendulum
        from fmd.simulator.params import PENDULUM_1M
        pend = SimplePendulum(PENDULUM_1M)
        result = simulate(pend, jnp.array([0.5, 0.0]), dt=0.01, duration=0.5)
        rich = result_with_meta(pend, result)
        assert rich.outputs == {}

    def test_user_override_with_warning(self):
        """User-provided output overrides aux with warning."""
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        custom = np.ones(result.times.shape[0]) * 42.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rich = result_with_meta(moth, result, outputs={"main_df": custom})
            assert len(w) == 1
            assert "overrides" in str(w[0].message)
        assert np.allclose(rich.outputs["main_df"], 42.0)

    def test_user_new_key_no_warning(self):
        """User-provided output with new key does not warn."""
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        custom = np.ones(result.times.shape[0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rich = result_with_meta(moth, result, outputs={"custom_key": custom})
            assert len(w) == 0
        assert "custom_key" in rich.outputs

    def test_all_18_aux_quantities(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate(moth, moth.default_state(), dt=0.005, duration=1.0)
        rich = result_with_meta(moth, result)
        for name in moth.aux_names:
            assert name in rich.outputs, f"Missing aux output: {name}"

    def test_compatible_with_simulate_noisy(self):
        """result_with_meta works with simulate_noisy output."""
        from fmd.simulator.integrator import simulate_noisy
        moth = Moth3D(MOTH_BIEKER_V3)
        result = simulate_noisy(
            moth, moth.default_state(), dt=0.005, duration=1.0,
            prng_key=jax.random.key(42))
        rich = result_with_meta(moth, result)
        assert "main_df" in rich.outputs

    def test_compatible_with_simulate_symplectic(self):
        """result_with_meta works with simulate_symplectic.

        Moth3D doesn't support symplectic, so test with a pendulum
        (num_aux=0) to verify the code path works.
        """
        from fmd.simulator import SimplePendulum, simulate_symplectic
        from fmd.simulator.params import PENDULUM_1M
        pend = SimplePendulum(PENDULUM_1M)
        result = simulate_symplectic(pend, jnp.array([0.5, 0.0]), dt=0.01, duration=0.5)
        rich = result_with_meta(pend, result)
        assert rich.outputs == {}
