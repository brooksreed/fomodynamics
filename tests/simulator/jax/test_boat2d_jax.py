"""Tests for JAX Boat2D implementation.

These tests verify:
1. Correct physics (Coriolis coupling, steady-state solutions)
2. JIT compilation works
3. Autodiff works through simulation
4. Numerical correctness against golden values
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import Boat2D as Boat2DJax, simulate, ConstantControl
from fmd.simulator.params import Boat2DParams, BOAT2D_TEST_DEFAULT

from .conftest import (
    ANALYTICAL_RTOL,
    TRAJ_RTOL,
    TRAJ_ATOL,
    DERIV_RTOL,
    DERIV_ATOL,
)
from .golden_values import (
    BOAT2D_DERIV_AT_REST,
    BOAT2D_DERIV_WITH_VELOCITY,
    BOAT2D_DERIV_WITH_CONTROL,
    BOAT2D_TRAJ_STRAIGHT_FINAL,
)


class TestBoat2DJaxBasics:
    """Basic tests for Boat2DJax."""

    def test_create_from_params(self):
        """Can create boat from params object."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        assert boat.mass == 100.0
        assert boat.izz == 50.0
        assert boat.drag_surge == 10.0
        assert boat.drag_sway == 20.0
        assert boat.drag_yaw == 5.0

    def test_create_from_values(self):
        """Can create boat with from_values."""
        boat = Boat2DJax.from_values(
            mass=150.0, izz=75.0, drag_surge=15.0, drag_sway=30.0, drag_yaw=7.5
        )
        assert boat.mass == 150.0
        assert boat.izz == 75.0

    def test_state_names(self):
        """State names are correct."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        assert boat.state_names == ("x", "y", "psi", "u", "v", "r")

    def test_control_names(self):
        """Control names are correct."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        assert boat.control_names == ("thrust", "yaw_moment")

    def test_num_states(self):
        """num_states is 6."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        assert boat.num_states == 6

    def test_num_controls(self):
        """num_controls is 2."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        assert boat.num_controls == 2

    def test_default_state(self):
        """Default state is zeros (at origin, at rest)."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = boat.default_state()
        np.testing.assert_allclose(state, jnp.zeros(6), atol=1e-15)

    def test_default_control(self):
        """Default control is zeros."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        control = boat.default_control()
        np.testing.assert_allclose(control, jnp.zeros(2), atol=1e-15)


class TestBoat2DJaxDerivatives:
    """Tests for state derivative computation."""

    def test_derivative_at_rest(self):
        """Zero derivative when at rest with no control."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = jnp.zeros(6)
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        np.testing.assert_allclose(deriv, jnp.zeros(6), atol=1e-15)

    def test_derivative_with_surge_only(self):
        """Derivative correct for pure surge motion."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        # Heading north (psi=0), moving forward at u=5 m/s
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        # x_dot = u * cos(0) = 5
        # y_dot = u * sin(0) = 0
        # u_dot = -drag_surge/mass * u = -10/100 * 5 = -0.5
        assert deriv[0] == pytest.approx(5.0)  # x_dot
        assert deriv[1] == pytest.approx(0.0)  # y_dot
        assert deriv[3] == pytest.approx(-0.5)  # u_dot (drag only)

    def test_derivative_with_thrust(self):
        """Derivative correct with thrust applied."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = jnp.zeros(6)
        control = jnp.array([100.0, 0.0])  # 100 N thrust

        deriv = boat.forward_dynamics(state, control)

        # u_dot = thrust/mass = 100/100 = 1.0
        assert deriv[3] == pytest.approx(1.0)

    def test_coriolis_coupling(self):
        """Coriolis terms couple surge/sway/yaw correctly."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        # Moving forward and turning
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 1.0])  # u=5, r=1
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        # u_dot includes +r*v = 1*0 = 0 (no sway)
        # v_dot includes -r*u = -1*5 = -5
        # Check v_dot has Coriolis contribution
        expected_v_dot = -(boat.drag_sway / boat.mass) * 0.0 - 1.0 * 5.0
        assert deriv[4] == pytest.approx(expected_v_dot)

    def test_heading_affects_position_derivative(self):
        """Heading angle transforms body velocity to NED correctly."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        # Heading east (psi = pi/2), moving forward at u=10 m/s
        state = jnp.array([0.0, 0.0, jnp.pi / 2, 10.0, 0.0, 0.0])
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        # x_dot = u * cos(pi/2) = 0
        # y_dot = u * sin(pi/2) = 10
        np.testing.assert_allclose(deriv[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(deriv[1], 10.0, atol=1e-10)


class TestBoat2DJaxSteadyState:
    """Tests for analytical steady-state solutions."""

    def test_steady_state_surge(self):
        """Steady-state surge velocity is thrust/drag."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        thrust = 50.0

        u_ss = boat.steady_state_surge(thrust)

        # u_ss = thrust / drag_surge = 50 / 10 = 5
        assert u_ss == pytest.approx(5.0)

    def test_steady_state_yaw_rate(self):
        """Steady-state yaw rate is moment/drag."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        yaw_moment = 10.0

        r_ss = boat.steady_state_yaw_rate(yaw_moment)

        # r_ss = yaw_moment / drag_yaw = 10 / 5 = 2
        assert r_ss == pytest.approx(2.0)

    def test_surge_time_constant(self):
        """Surge time constant is mass/drag."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)

        tau = boat.surge_time_constant()

        # tau = mass / drag_surge = 100 / 10 = 10
        assert tau == pytest.approx(10.0)

    def test_yaw_time_constant(self):
        """Yaw time constant is inertia/drag."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)

        tau_r = boat.yaw_time_constant()

        # tau_r = izz / drag_yaw = 50 / 5 = 10
        assert tau_r == pytest.approx(10.0)

    def test_steady_turn_sway(self):
        """Steady-state sway in a turn."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        u = 5.0
        r = 0.1

        v_ss = boat.steady_turn_sway(u, r)

        # v_ss = -mass * r * u / drag_sway = -100 * 0.1 * 5 / 20 = -2.5
        assert v_ss == pytest.approx(-2.5)


class TestBoat2DJaxSimulation:
    """Tests for boat simulation."""

    def test_simulate_returns_result(self):
        """simulate() returns JaxSimulationResult."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        initial = boat.default_state()
        control = ConstantControl(jnp.array([50.0, 0.0]))

        result = simulate(boat, initial, dt=0.01, duration=1.0, control=control)

        assert result.times.shape[0] > 0
        assert result.states.shape[0] == result.times.shape[0]
        assert result.states.shape[1] == 6
        assert result.controls.shape[1] == 2

    def test_surge_accelerates_with_thrust(self):
        """Boat accelerates in surge with thrust."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        initial = boat.default_state()
        control = ConstantControl(jnp.array([100.0, 0.0]))

        result = simulate(boat, initial, dt=0.01, duration=2.0, control=control)

        # Surge velocity should increase
        final_u = result.states[-1, 3]
        assert final_u > 0

    def test_approaches_steady_state(self):
        """Surge velocity approaches steady-state value."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        thrust = 50.0
        initial = boat.default_state()
        control = ConstantControl(jnp.array([thrust, 0.0]))

        # Simulate for many time constants
        tau = boat.surge_time_constant()
        result = simulate(boat, initial, dt=0.01, duration=10 * tau, control=control)

        # Should approach steady-state
        expected_u_ss = boat.steady_state_surge(thrust)
        final_u = result.states[-1, 3]

        np.testing.assert_allclose(final_u, expected_u_ss, rtol=0.01)

    def test_heading_wrapping(self):
        """Heading stays within [-pi, pi] after wrapping."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        initial = boat.default_state()
        # Apply constant yaw moment to spin
        control = ConstantControl(jnp.array([0.0, 50.0]))

        # Simulate long enough to complete several rotations
        result = simulate(boat, initial, dt=0.01, duration=30.0, control=control)

        # All headings should be in [-pi, pi]
        psi = result.states[:, 2]
        assert jnp.all(psi >= -jnp.pi)
        assert jnp.all(psi <= jnp.pi)


class TestBoat2DJaxGoldenValues:
    """Verify Boat2DJax against golden values for numerical regression."""

    def test_derivative_at_rest_matches_golden(self):
        """Derivative at rest matches golden value."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = jnp.zeros(6)
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        np.testing.assert_allclose(
            deriv, BOAT2D_DERIV_AT_REST, rtol=DERIV_RTOL, atol=DERIV_ATOL
        )

    def test_derivative_with_velocity_matches_golden(self):
        """Derivative with velocity matches golden value."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 1.0, 0.5])
        control = jnp.zeros(2)

        deriv = boat.forward_dynamics(state, control)

        np.testing.assert_allclose(
            deriv, BOAT2D_DERIV_WITH_VELOCITY, rtol=DERIV_RTOL, atol=DERIV_ATOL
        )

    def test_derivative_with_control_matches_golden(self):
        """Derivative with control matches golden value."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        state = jnp.array([10.0, 20.0, np.pi / 4, 3.0, -0.5, 0.2])
        control = jnp.array([100.0, 10.0])

        deriv = boat.forward_dynamics(state, control)

        np.testing.assert_allclose(
            deriv, BOAT2D_DERIV_WITH_CONTROL, rtol=DERIV_RTOL, atol=DERIV_ATOL
        )

    def test_trajectory_matches_golden(self):
        """Straight-line trajectory final state matches golden value."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        initial = jnp.zeros(6)
        control = ConstantControl(jnp.array([50.0, 0.0]))

        result = simulate(boat, initial, dt=0.01, duration=5.0, control=control)

        np.testing.assert_allclose(
            result.states[-1],
            BOAT2D_TRAJ_STRAIGHT_FINAL,
            rtol=TRAJ_RTOL,
            atol=TRAJ_ATOL,
        )


class TestBoat2DJaxJIT:
    """Tests for JIT compilation."""

    def test_derivative_jit(self):
        """forward_dynamics can be JIT compiled."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)

        @jax.jit
        def compute_deriv(b, s, c):
            return b.forward_dynamics(s, c)

        state = jnp.array([0.0, 0.0, 0.0, 5.0, 1.0, 0.1])
        control = jnp.array([50.0, 5.0])
        result = compute_deriv(boat, state, control)

        assert jnp.all(jnp.isfinite(result))

    def test_simulate_jit(self):
        """simulate() can be JIT compiled."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        control = ConstantControl(jnp.array([50.0, 5.0]))

        @jax.jit
        def run_sim(b, s0):
            return simulate(b, s0, dt=0.01, duration=1.0, control=control)

        initial = jnp.zeros(6)
        result = run_sim(boat, initial)

        assert jnp.all(jnp.isfinite(result.states))


class TestBoat2DJaxGrad:
    """Tests for autodiff through Boat2D simulation."""

    def test_grad_wrt_initial_state(self):
        """Can compute gradient w.r.t. initial state."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)
        control = ConstantControl(jnp.array([50.0, 0.0]))

        def loss(initial):
            result = simulate(boat, initial, dt=0.01, duration=1.0, control=control)
            # Final position
            return result.states[-1, 0] ** 2 + result.states[-1, 1] ** 2

        initial = jnp.zeros(6)
        grad = jax.grad(loss)(initial)

        assert grad.shape == (6,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_control(self):
        """Can compute gradient w.r.t. control values."""
        boat = Boat2DJax(BOAT2D_TEST_DEFAULT)

        def loss(control_value):
            control = ConstantControl(control_value)
            initial = jnp.zeros(6)
            result = simulate(boat, initial, dt=0.01, duration=1.0, control=control)
            # Final surge velocity
            return result.states[-1, 3]

        control_val = jnp.array([50.0, 0.0])
        grad = jax.grad(loss)(control_val)

        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))
        # More thrust should increase final velocity
        assert grad[0] > 0

    def test_grad_wrt_mass(self):
        """Can compute gradient w.r.t. mass parameter."""

        def loss(mass):
            boat = Boat2DJax.from_values(
                mass=mass, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
            )
            control = ConstantControl(jnp.array([50.0, 0.0]))
            initial = jnp.zeros(6)
            result = simulate(boat, initial, dt=0.01, duration=1.0, control=control)
            return result.states[-1, 3]  # Final surge velocity

        grad = jax.grad(loss)(100.0)
        assert jnp.isfinite(grad)
        # More mass should decrease acceleration, so negative gradient
        assert grad < 0

    def test_grad_wrt_drag(self):
        """Can compute gradient w.r.t. drag parameter."""

        def loss(drag_surge):
            boat = Boat2DJax.from_values(
                mass=100.0, izz=50.0, drag_surge=drag_surge, drag_sway=20.0, drag_yaw=5.0
            )
            control = ConstantControl(jnp.array([50.0, 0.0]))
            initial = jnp.zeros(6)
            result = simulate(boat, initial, dt=0.01, duration=2.0, control=control)
            return result.states[-1, 3]

        grad = jax.grad(loss)(10.0)
        assert jnp.isfinite(grad)
        # More drag should decrease final velocity
        assert grad < 0
