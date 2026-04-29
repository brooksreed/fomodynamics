"""Tests for JAX Planar Quadrotor (2D) implementation.

These tests verify:
1. Correct physics (hover equilibrium, freefall, thrust response)
2. Energy conservation and power balance
3. JIT compilation works
4. Autodiff works through simulation
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl, ZeroControl
from fmd.simulator.params import (
    PlanarQuadrotorParams,
    PLANAR_QUAD_TEST_DEFAULT,
    PLANAR_QUAD_CRAZYFLIE,
    PLANAR_QUAD_HEAVY,
)
from fmd.simulator.planar_quadrotor import X, Z, THETA, X_DOT, Z_DOT, THETA_DOT, T1, T2

from .conftest import ANALYTICAL_RTOL, TRAJ_RTOL, DERIV_RTOL


class TestPlanarQuadrotorJaxBasics:
    """Basic tests for PlanarQuadrotorJax construction and attributes."""

    def test_create_from_params(self):
        """Can create planar quadrotor from params object."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.mass == 1.0
        assert quad.arm_length == 0.25
        assert quad.inertia_pitch == 0.01
        assert quad.g == pytest.approx(9.80665)

    def test_create_custom_params(self):
        """Can create planar quadrotor with custom params."""
        params = PlanarQuadrotorParams(
            mass=2.0,
            arm_length=0.3,
            inertia_pitch=0.02,
            g=10.0,
        )
        quad = PlanarQuadrotor(params)
        assert quad.mass == 2.0
        assert quad.arm_length == 0.3
        assert quad.inertia_pitch == 0.02
        assert quad.g == 10.0

    def test_create_from_values(self):
        """Can create planar quadrotor using from_values (JAX-traceable)."""
        quad = PlanarQuadrotor.from_values(
            mass=1.5,
            arm_length=0.2,
            inertia_pitch=0.015,
        )
        assert quad.mass == 1.5
        assert quad.arm_length == 0.2
        assert quad.inertia_pitch == 0.015
        assert quad.g == pytest.approx(9.80665)  # Default

    def test_state_names(self):
        """State names are correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.state_names == ("x", "z", "theta", "x_dot", "z_dot", "theta_dot")

    def test_control_names(self):
        """Control names are correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.control_names == ("T1", "T2")

    def test_num_states(self):
        """num_states is 6."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.num_states == 6

    def test_num_controls(self):
        """num_controls is 2."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.num_controls == 2

    def test_default_state(self):
        """Default state is zeros (hover at origin)."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        np.testing.assert_allclose(state, jnp.zeros(6), atol=1e-15)

    def test_default_control(self):
        """Default control is hover thrust."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        control = quad.default_control()
        T_hover = quad.hover_thrust_per_rotor()
        np.testing.assert_allclose(control, jnp.array([T_hover, T_hover]), rtol=1e-14)

    def test_hover_control(self):
        """Hover control is equal thrust per rotor."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        control = quad.hover_control()
        T_hover = quad.hover_thrust_per_rotor()
        assert control[T1] == pytest.approx(T_hover)
        assert control[T2] == pytest.approx(T_hover)
        # Total thrust equals weight
        assert control[T1] + control[T2] == pytest.approx(quad.mass * quad.g)

    def test_hover_thrust_per_rotor(self):
        """Hover thrust per rotor is mg/2."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        expected = (quad.mass * quad.g) / 2.0
        assert quad.hover_thrust_per_rotor() == pytest.approx(expected)

    def test_hover_thrust_total(self):
        """Hover thrust total is mg."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        expected = quad.mass * quad.g
        assert quad.hover_thrust_total() == pytest.approx(expected)

    def test_different_presets(self):
        """Different presets have expected values."""
        test = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        crazyflie = PlanarQuadrotor(PLANAR_QUAD_CRAZYFLIE)
        heavy = PlanarQuadrotor(PLANAR_QUAD_HEAVY)

        assert test.mass == 1.0
        assert crazyflie.mass == 0.030
        assert heavy.mass == 2.0


class TestPlanarQuadrotorJaxDerivatives:
    """Tests for state derivative computation."""

    def test_derivative_at_hover(self):
        """Zero derivative at hover equilibrium."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        control = quad.hover_control()

        deriv = quad.forward_dynamics(state, control)

        # Should be zero (hover is equilibrium)
        np.testing.assert_allclose(deriv, jnp.zeros(6), atol=1e-12)

    def test_derivative_freefall(self):
        """Derivative correct in freefall (zero thrust)."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        control = jnp.zeros(2)  # No thrust

        deriv = quad.forward_dynamics(state, control)

        # Only z_ddot should be -g (freefall)
        expected = jnp.array([0.0, 0.0, 0.0, 0.0, -quad.g, 0.0])
        np.testing.assert_allclose(deriv, expected, atol=1e-12)

    def test_derivative_excess_thrust(self):
        """Derivative correct with excess thrust (climb)."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        T_hover = quad.hover_thrust_per_rotor()
        # 20% excess thrust
        control = jnp.array([1.2 * T_hover, 1.2 * T_hover])

        deriv = quad.forward_dynamics(state, control)

        # Should accelerate upward (z_ddot > 0)
        assert deriv[Z_DOT] > 0
        # No horizontal or rotational acceleration (symmetric)
        assert deriv[X_DOT] == pytest.approx(0.0, abs=1e-12)
        assert deriv[THETA_DOT] == pytest.approx(0.0, abs=1e-12)

    def test_derivative_differential_thrust(self):
        """Derivative correct with differential thrust (roll/pitch moment)."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        T_hover = quad.hover_thrust_per_rotor()
        # More thrust on T1 (right) -> positive moment -> positive theta_ddot
        control = jnp.array([T_hover + 0.5, T_hover - 0.5])

        deriv = quad.forward_dynamics(state, control)

        # Should have positive angular acceleration
        expected_moment = 1.0 * quad.arm_length  # Delta T = 1.0 N
        expected_theta_ddot = expected_moment / quad.inertia_pitch
        assert deriv[THETA_DOT] == pytest.approx(expected_theta_ddot, rel=1e-10)
        # Total thrust equals hover, so z_ddot should be ~0
        assert deriv[Z_DOT] == pytest.approx(0.0, abs=1e-10)

    def test_derivative_tilted(self):
        """Derivative correct when tilted."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        theta = 0.1  # Small tilt
        state = jnp.array([0.0, 0.0, theta, 0.0, 0.0, 0.0])
        control = quad.hover_control()

        deriv = quad.forward_dynamics(state, control)

        # Thrust has horizontal component when tilted
        T_total = control[T1] + control[T2]
        expected_x_ddot = -(T_total / quad.mass) * jnp.sin(theta)
        expected_z_ddot = (T_total / quad.mass) * jnp.cos(theta) - quad.g

        assert deriv[X_DOT] == pytest.approx(expected_x_ddot, rel=1e-10)
        assert deriv[Z_DOT] == pytest.approx(expected_z_ddot, rel=1e-10)

    def test_derivative_with_velocity(self):
        """Derivative correct with initial velocity."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = jnp.array([1.0, 2.0, 0.0, 3.0, 4.0, 0.5])
        control = quad.hover_control()

        deriv = quad.forward_dynamics(state, control)

        # Position derivatives should equal velocities
        assert deriv[X] == pytest.approx(state[X_DOT], rel=1e-14)
        assert deriv[Z] == pytest.approx(state[Z_DOT], rel=1e-14)
        assert deriv[THETA] == pytest.approx(state[THETA_DOT], rel=1e-14)


class TestPlanarQuadrotorJaxPhysics:
    """Tests for physical correctness."""

    def test_hover_equilibrium_is_stable(self):
        """Hover equilibrium maintains position."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        control = ConstantControl(quad.hover_control())

        result = simulate(quad, state0, dt=0.001, duration=5.0, control=control)

        # Should stay at origin
        final = result.states[-1]
        np.testing.assert_allclose(final[:3], jnp.zeros(3), atol=1e-8)
        np.testing.assert_allclose(final[3:], jnp.zeros(3), atol=1e-8)

    def test_freefall_acceleration(self):
        """Freefall has correct acceleration."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        control = ConstantControl(jnp.zeros(2))  # Zero thrust for freefall

        result = simulate(quad, state0, dt=0.0001, duration=0.5, control=control)

        # z = z0 - 0.5*g*t^2
        t = result.times[-1]
        expected_z = -0.5 * quad.g * t**2
        np.testing.assert_allclose(result.states[-1, Z], expected_z, rtol=1e-4)

        # z_dot = -g*t
        expected_z_dot = -quad.g * t
        np.testing.assert_allclose(result.states[-1, Z_DOT], expected_z_dot, rtol=1e-4)

    def test_excess_thrust_climbs(self):
        """Excess thrust causes climb."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        T_hover = quad.hover_thrust_per_rotor()
        # 50% excess thrust
        control = ConstantControl(jnp.array([1.5 * T_hover, 1.5 * T_hover]))

        result = simulate(quad, state0, dt=0.001, duration=1.0, control=control)

        # Should have climbed
        assert result.states[-1, Z] > 0
        assert result.states[-1, Z_DOT] > 0

    def test_deficit_thrust_falls(self):
        """Deficit thrust causes fall."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        T_hover = quad.hover_thrust_per_rotor()
        # 50% of hover thrust
        control = ConstantControl(jnp.array([0.5 * T_hover, 0.5 * T_hover]))

        result = simulate(quad, state0, dt=0.001, duration=1.0, control=control)

        # Should have fallen
        assert result.states[-1, Z] < 0
        assert result.states[-1, Z_DOT] < 0

    def test_differential_thrust_rotates(self):
        """Differential thrust causes rotation."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        T_hover = quad.hover_thrust_per_rotor()
        # More thrust on T1 -> positive moment
        control = ConstantControl(jnp.array([T_hover + 1.0, T_hover - 1.0]))

        result = simulate(quad, state0, dt=0.001, duration=0.5, control=control)

        # Should have rotated positive
        assert result.states[-1, THETA] > 0
        assert result.states[-1, THETA_DOT] > 0

    def test_tilted_moves_horizontally(self):
        """Tilted quadrotor with hover thrust moves horizontally."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        theta = 0.2  # Tilt
        state0 = jnp.array([0.0, 0.0, theta, 0.0, 0.0, 0.0])
        control = ConstantControl(quad.hover_control())

        result = simulate(quad, state0, dt=0.001, duration=1.0, control=control)

        # Should have moved in negative x direction (thrust has negative x component)
        assert result.states[-1, X] < 0
        # Should have fallen (cos(theta) < 1, so vertical thrust < weight)
        assert result.states[-1, Z] < 0

    def test_energy_freefall(self):
        """Energy is conserved in freefall."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        # Start with some height
        state0 = jnp.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

        result = simulate(quad, state0, dt=0.0001, duration=1.0)

        # Compute energy at each time step
        energies = jax.vmap(quad.energy)(result.states)
        initial_energy = energies[0]

        # Energy should be constant (within numerical tolerance)
        np.testing.assert_allclose(
            energies, initial_energy * jnp.ones_like(energies), rtol=1e-4
        )

    def test_power_balance(self):
        """Power delivered by thrust equals rate of energy change."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        # Test with upward velocity and hover thrust
        state = jnp.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])  # Moving up
        control = quad.hover_control()

        # Power from thrust = F dot v
        # At theta=0: F = (0, T_total), v = (0, 2)
        P_thrust = quad.power_thrust(state, control)
        T_total = control[T1] + control[T2]
        expected_P_thrust = T_total * state[Z_DOT]  # Thrust aligned with velocity

        assert P_thrust == pytest.approx(expected_P_thrust, rel=1e-10)

        # Power from gravity = -m*g*z_dot (negative when going up)
        P_gravity = quad.power_gravity(state)
        expected_P_gravity = -quad.mass * quad.g * state[Z_DOT]

        assert P_gravity == pytest.approx(expected_P_gravity, rel=1e-10)

        # At hover: P_thrust - m*g*z_dot should balance for constant KE
        # (net vertical force is zero, so no KE change from vertical motion)


class TestPlanarQuadrotorJaxSimulation:
    """Tests for planar quadrotor simulation."""

    def test_simulate_returns_result(self):
        """simulate() returns result with correct shape."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        control = ConstantControl(quad.hover_control())

        result = simulate(quad, state0, dt=0.01, duration=1.0, control=control)

        assert result.times.shape[0] > 0
        assert result.states.shape[0] == result.times.shape[0]
        assert result.states.shape[1] == 6

    def test_simulate_initial_state_preserved(self):
        """Initial state is in result."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = jnp.array([1.0, 2.0, 0.1, 0.5, -0.5, 0.2])
        control = ConstantControl(quad.hover_control())

        result = simulate(quad, state0, dt=0.01, duration=1.0, control=control)

        np.testing.assert_allclose(result.states[0], state0, rtol=1e-14)

    def test_simulate_times_correct(self):
        """Time array is correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = quad.default_state()
        control = ConstantControl(quad.hover_control())

        result = simulate(quad, state0, dt=0.01, duration=1.0, control=control)

        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(1.0)

    def test_create_state_helper(self):
        """create_state helper works correctly."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        state = quad.create_state(x=1.0, z=2.0, theta=0.1, x_dot=0.5)
        expected = jnp.array([1.0, 2.0, 0.1, 0.5, 0.0, 0.0])

        np.testing.assert_allclose(state, expected, atol=1e-15)

    def test_create_control_helper(self):
        """create_control helper works correctly."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        # Default to hover
        control = quad.create_control()
        T_hover = quad.hover_thrust_per_rotor()
        np.testing.assert_allclose(control, jnp.array([T_hover, T_hover]), rtol=1e-14)

        # Custom values
        control = quad.create_control(T1_val=5.0, T2_val=6.0)
        np.testing.assert_allclose(control, jnp.array([5.0, 6.0]), atol=1e-15)


class TestPlanarQuadrotorJaxUtilities:
    """Tests for utility methods."""

    def test_speed(self):
        """Speed calculation is correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = jnp.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0])

        speed = quad.speed(state)

        assert speed == pytest.approx(5.0, rel=1e-14)

    def test_flight_path_angle(self):
        """Flight path angle calculation is correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        # Level flight
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        assert quad.flight_path_angle(state) == pytest.approx(0.0, abs=1e-14)

        # Vertical climb
        state = jnp.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0])
        assert quad.flight_path_angle(state) == pytest.approx(jnp.pi / 2, rel=1e-14)

        # 45 degree climb
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 5.0, 0.0])
        assert quad.flight_path_angle(state) == pytest.approx(jnp.pi / 4, rel=1e-14)

    def test_angle_of_attack(self):
        """Angle of attack calculation is correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        # Nose pointing along velocity vector
        state = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # theta=0, gamma=0
        assert quad.angle_of_attack(state) == pytest.approx(0.0, abs=1e-14)

        # Nose up, level flight -> positive alpha
        state = jnp.array([0.0, 0.0, 0.2, 5.0, 0.0, 0.0])  # theta=0.2, gamma=0
        assert quad.angle_of_attack(state) == pytest.approx(0.2, rel=1e-14)


class TestPlanarQuadrotorJaxJIT:
    """Tests for JIT compilation."""

    def test_derivative_jit(self):
        """forward_dynamics can be JIT compiled."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        @jax.jit
        def compute_deriv(p, s, c):
            return p.forward_dynamics(s, c)

        state = jnp.array([0.0, 0.0, 0.1, 1.0, 2.0, 0.1])
        control = quad.hover_control()
        result = compute_deriv(quad, state, control)

        assert jnp.all(jnp.isfinite(result))

    def test_simulate_jit(self):
        """simulate() can be JIT compiled."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        @jax.jit
        def run_sim(p, s0, ctrl):
            return simulate(p, s0, dt=0.01, duration=1.0, control=ConstantControl(ctrl))

        state0 = quad.default_state()
        control = quad.hover_control()
        result = run_sim(quad, state0, control)

        assert jnp.all(jnp.isfinite(result.states))

    def test_energy_jit(self):
        """energy() can be JIT compiled."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        @jax.jit
        def compute_energy(p, s):
            return p.energy(s)

        state = jnp.array([1.0, 2.0, 0.1, 1.0, 2.0, 0.5])
        result = compute_energy(quad, state)

        assert jnp.isfinite(result)

    def test_jit_no_retrace(self):
        """JIT doesn't retrace with same shapes."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        @jax.jit
        def run_sim(p, s0, ctrl):
            return simulate(p, s0, dt=0.01, duration=1.0, control=ConstantControl(ctrl))

        # First call compiles
        result1 = run_sim(quad, quad.default_state(), quad.hover_control())

        # Second call should use cached compilation
        state2 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result2 = run_sim(quad, state2, quad.hover_control())

        # Both should produce valid results
        assert jnp.all(jnp.isfinite(result1.states))
        assert jnp.all(jnp.isfinite(result2.states))


class TestPlanarQuadrotorJaxGrad:
    """Tests for autodiff through planar quadrotor simulation."""

    def test_grad_wrt_initial_position(self):
        """Can compute gradient w.r.t. initial position."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        def loss(z0):
            state0 = jnp.array([0.0, z0, 0.0, 0.0, 0.0, 0.0])
            result = simulate(quad, state0, dt=0.01, duration=0.5,
                            control=ConstantControl(quad.hover_control()))
            return result.states[-1, Z] ** 2

        z0 = 1.0
        grad = jax.grad(loss)(z0)

        assert jnp.isfinite(grad)

    def test_grad_wrt_initial_state(self):
        """Can compute gradient w.r.t. full initial state."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        def loss(initial_state):
            result = simulate(quad, initial_state, dt=0.01, duration=0.5,
                            control=ConstantControl(quad.hover_control()))
            return jnp.sum(result.states[-1] ** 2)

        initial = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (6,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_control(self):
        """Can compute gradient w.r.t. control."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        def loss(thrust_scale):
            control = thrust_scale * quad.hover_control()
            state0 = quad.default_state()
            result = simulate(quad, state0, dt=0.01, duration=0.5,
                            control=ConstantControl(control))
            # Minimize final altitude deviation from 1.0
            return (result.states[-1, Z] - 1.0) ** 2

        thrust_scale = 1.0
        grad = jax.grad(loss)(thrust_scale)

        assert jnp.isfinite(grad)

    def test_grad_wrt_mass(self):
        """Can compute gradient w.r.t. mass parameter."""

        def loss(mass):
            quad = PlanarQuadrotor.from_values(
                mass=mass,
                arm_length=0.25,
                inertia_pitch=0.01,
            )
            # Use thrust that would be hover for 1kg
            control = ConstantControl(jnp.array([4.90, 4.90]))
            state0 = quad.default_state()
            result = simulate(quad, state0, dt=0.01, duration=0.5, control=control)
            return result.states[-1, Z] ** 2

        mass = 1.0
        grad = jax.grad(loss)(mass)

        assert jnp.isfinite(grad)

    def test_grad_through_energy(self):
        """Can compute gradient through energy function."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        def loss(initial_state):
            result = simulate(quad, initial_state, dt=0.01, duration=0.5,
                            control=ConstantControl(quad.hover_control()))
            # Sum of energy at all timesteps
            return jnp.sum(jax.vmap(quad.energy)(result.states))

        initial = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (6,)
        assert jnp.all(jnp.isfinite(grad))


class TestPlanarQuadrotorJaxVmap:
    """Tests for vectorized simulation."""

    def test_vmap_over_initial_states(self):
        """Can vmap over initial states."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        def simulate_single(initial):
            return simulate(quad, initial, dt=0.01, duration=0.5,
                          control=ConstantControl(quad.hover_control())).states[-1]

        # Batch of initial heights
        initials = jnp.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        ])

        final_states = jax.vmap(simulate_single)(initials)

        assert final_states.shape == (3, 6)
        assert jnp.all(jnp.isfinite(final_states))

    def test_vmap_energy_computation(self):
        """Can vmap energy computation."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state0 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        result = simulate(quad, state0, dt=0.01, duration=1.0,
                         control=ConstantControl(quad.hover_control()))
        energies = jax.vmap(quad.energy)(result.states)

        assert energies.shape == (result.states.shape[0],)
        assert jnp.all(jnp.isfinite(energies))


class TestPlanarQuadrotorJaxGoldenValues:
    """Regression tests with golden values."""

    def test_hover_equilibrium_golden(self):
        """Hover equilibrium derivative is zero."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        control = quad.hover_control()

        deriv = quad.forward_dynamics(state, control)

        np.testing.assert_allclose(deriv, jnp.zeros(6), atol=1e-12)

    def test_freefall_golden(self):
        """Freefall derivative is correct."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        state = quad.default_state()
        control = jnp.zeros(2)

        deriv = quad.forward_dynamics(state, control)

        expected = jnp.array([0.0, 0.0, 0.0, 0.0, -9.80665, 0.0])
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)

    def test_energy_golden(self):
        """Energy calculation matches golden values."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

        # At z=0, zero velocity
        state = jnp.zeros(6)
        energy = quad.energy(state)
        assert energy == pytest.approx(0.0, abs=1e-15)

        # At z=1m, zero velocity
        state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        energy = quad.energy(state)
        expected = quad.mass * quad.g * 1.0  # m*g*h
        assert energy == pytest.approx(expected, rel=1e-10)

        # With velocity
        state = jnp.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
        energy = quad.energy(state)
        expected_KE = 0.5 * quad.mass * (3.0**2 + 4.0**2)  # 0.5*m*v^2
        assert energy == pytest.approx(expected_KE, rel=1e-10)
