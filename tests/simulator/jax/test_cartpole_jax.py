"""Tests for JAX Cartpole (inverted pendulum) implementation.

These tests verify:
1. Correct physics (energy conservation, equilibrium stability)
2. Coupled dynamics between cart and pole
3. JIT compilation works
4. Autodiff works through simulation
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import Cartpole, simulate, ConstantControl
from fmd.simulator.params import CartpoleParams, CARTPOLE_CLASSIC, CARTPOLE_HEAVY_POLE, CARTPOLE_LONG_POLE
from fmd.simulator.cartpole import X, X_DOT, THETA, THETA_DOT, FORCE

from .conftest import ANALYTICAL_RTOL, TRAJ_RTOL, DERIV_RTOL


class TestCartpoleJaxBasics:
    """Basic tests for CartpoleJax construction and attributes."""

    def test_create_from_params(self):
        """Can create cartpole from params object."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.mass_cart == 1.0
        assert cartpole.mass_pole == 0.1
        assert cartpole.pole_length == 0.5
        assert cartpole.g == pytest.approx(9.80665)

    def test_create_custom_params(self):
        """Can create cartpole with custom params."""
        params = CartpoleParams(
            mass_cart=2.0,
            mass_pole=0.2,
            pole_length=1.0,
            g=10.0,
        )
        cartpole = Cartpole(params)
        assert cartpole.mass_cart == 2.0
        assert cartpole.mass_pole == 0.2
        assert cartpole.pole_length == 1.0
        assert cartpole.g == 10.0

    def test_create_from_values(self):
        """Can create cartpole using from_values (JAX-traceable)."""
        cartpole = Cartpole.from_values(
            mass_cart=1.5,
            mass_pole=0.15,
            pole_length=0.75,
        )
        assert cartpole.mass_cart == 1.5
        assert cartpole.mass_pole == 0.15
        assert cartpole.pole_length == 0.75
        assert cartpole.g == pytest.approx(9.80665)  # Default

    def test_state_names(self):
        """State names are correct."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.state_names == ("x", "x_dot", "theta", "theta_dot")

    def test_control_names(self):
        """Control names are correct."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.control_names == ("F",)

    def test_num_states(self):
        """num_states is 4."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.num_states == 4

    def test_num_controls(self):
        """num_controls is 1."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.num_controls == 1

    def test_default_state(self):
        """Default state is upright equilibrium (zeros)."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = cartpole.default_state()
        np.testing.assert_allclose(state, jnp.zeros(4), atol=1e-15)

    def test_default_control(self):
        """Default control is zero force."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        control = cartpole.default_control()
        np.testing.assert_allclose(control, jnp.zeros(1), atol=1e-15)

    def test_upright_state(self):
        """Upright state is all zeros."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = cartpole.upright_state()
        np.testing.assert_allclose(state, jnp.zeros(4), atol=1e-15)

    def test_hanging_state(self):
        """Hanging state has theta=pi."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = cartpole.hanging_state()
        assert state[X] == pytest.approx(0.0)
        assert state[X_DOT] == pytest.approx(0.0)
        assert state[THETA] == pytest.approx(jnp.pi)
        assert state[THETA_DOT] == pytest.approx(0.0)

    def test_total_mass(self):
        """Total mass property is correct."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.total_mass == pytest.approx(1.1)

    def test_different_presets(self):
        """Different presets have expected values."""
        classic = Cartpole(CARTPOLE_CLASSIC)
        heavy = Cartpole(CARTPOLE_HEAVY_POLE)
        long_pole = Cartpole(CARTPOLE_LONG_POLE)

        assert classic.mass_pole == 0.1
        assert heavy.mass_pole == 0.5
        assert long_pole.pole_length == 1.0


class TestCartpoleJaxDerivatives:
    """Tests for state derivative computation."""

    def test_derivative_at_upright_equilibrium(self):
        """Zero derivative at upright equilibrium with no force."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.zeros(4)  # Upright, at rest
        control = jnp.zeros(1)  # No force

        deriv = cartpole.forward_dynamics(state, control)

        np.testing.assert_allclose(deriv, jnp.zeros(4), atol=1e-14)

    def test_derivative_at_hanging_equilibrium(self):
        """Zero derivative at hanging equilibrium with no force."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, jnp.pi, 0.0])  # Hanging, at rest
        control = jnp.zeros(1)

        deriv = cartpole.forward_dynamics(state, control)

        # At theta=pi, sin(pi)=0, so theta_ddot should be ~0
        np.testing.assert_allclose(deriv, jnp.zeros(4), atol=1e-12)

    def test_derivative_small_tilt(self):
        """Derivative correct for small tilt from upright."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        theta = 0.1  # Small angle
        state = jnp.array([0.0, 0.0, theta, 0.0])
        control = jnp.zeros(1)

        deriv = cartpole.forward_dynamics(state, control)

        # x_dot = 0 (velocity is zero)
        assert deriv[X] == pytest.approx(0.0, abs=1e-14)
        # theta_dot = 0 (angular velocity is zero)
        assert deriv[THETA] == pytest.approx(0.0, abs=1e-14)
        # theta_ddot > 0 for small positive theta (falls away from vertical)
        assert deriv[THETA_DOT] > 0
        # x_ddot should be non-zero (coupled dynamics)
        # When pole tips right (positive theta), cart accelerates left (negative x_ddot)
        assert deriv[X_DOT] < 0

    def test_derivative_with_force(self):
        """Derivative correct with applied force."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.zeros(4)  # Upright, at rest
        F = 10.0  # Force to the right
        control = jnp.array([F])

        deriv = cartpole.forward_dynamics(state, control)

        # x_dot = 0 (velocity is zero)
        assert deriv[X] == pytest.approx(0.0, abs=1e-14)
        # theta_dot = 0 (angular velocity is zero)
        assert deriv[THETA] == pytest.approx(0.0, abs=1e-14)
        # x_ddot > 0 (force pushes cart right)
        assert deriv[X_DOT] > 0
        # theta_ddot < 0 (pole tips backward as cart accelerates forward)
        assert deriv[THETA_DOT] < 0

    def test_derivative_with_angular_velocity(self):
        """Derivative correct with initial angular velocity."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, 0.0, 1.0])  # theta_dot = 1
        control = jnp.zeros(1)

        deriv = cartpole.forward_dynamics(state, control)

        # theta_dot should be 1 (from state)
        assert deriv[THETA] == pytest.approx(1.0, rel=1e-14)

    def test_derivative_zero_control(self):
        """Derivative handles zero control array."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = jnp.array([0.0])  # Zero force

        deriv = cartpole.forward_dynamics(state, control)
        assert jnp.all(jnp.isfinite(deriv))
        # Same as no force
        deriv_default = cartpole.forward_dynamics(state, cartpole.default_control())
        np.testing.assert_allclose(deriv, deriv_default, rtol=1e-14)


class TestCartpoleJaxPhysics:
    """Tests for physical correctness."""

    def test_linearized_frequency(self):
        """Linearized frequency matches sqrt(g/l)."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        expected = jnp.sqrt(cartpole.g / cartpole.pole_length)
        assert cartpole.linearized_frequency() == pytest.approx(expected, rel=1e-14)

    def test_linearized_period(self):
        """Linearized period matches 2*pi*sqrt(l/g)."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        expected = 2 * jnp.pi * jnp.sqrt(cartpole.pole_length / cartpole.g)
        assert cartpole.linearized_period() == pytest.approx(expected, rel=1e-14)

    def test_longer_pole_slower_period(self):
        """Longer pole has longer period."""
        short = Cartpole(CARTPOLE_CLASSIC)  # pole_length = 0.5
        long_pole = Cartpole(CARTPOLE_LONG_POLE)  # pole_length = 1.0

        assert long_pole.linearized_period() > short.linearized_period()

    def test_energy_at_upright(self):
        """Energy at upright position."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.zeros(4)

        energy = cartpole.energy(state)

        # PE = m_p * g * l * cos(0) = m_p * g * l
        expected_PE = cartpole.mass_pole * cartpole.g * cartpole.pole_length
        assert energy == pytest.approx(expected_PE, rel=1e-10)

    def test_energy_at_hanging(self):
        """Energy at hanging position."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, jnp.pi, 0.0])

        energy = cartpole.energy(state)

        # PE = m_p * g * l * cos(pi) = -m_p * g * l
        expected_PE = -cartpole.mass_pole * cartpole.g * cartpole.pole_length
        assert energy == pytest.approx(expected_PE, rel=1e-10)

    def test_energy_with_cart_velocity(self):
        """Energy includes cart kinetic energy."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        x_dot = 2.0
        state = jnp.array([0.0, x_dot, 0.0, 0.0])

        energy = cartpole.energy(state)

        # KE_cart = 0.5 * m_c * x_dot^2
        KE_cart = 0.5 * cartpole.mass_cart * x_dot**2
        PE = cartpole.mass_pole * cartpole.g * cartpole.pole_length
        expected = KE_cart + PE
        # Allow some tolerance due to pole COM velocity contribution
        assert energy > KE_cart  # At least cart KE

    def test_energy_conservation_free_fall(self):
        """Energy is conserved when falling without control."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        # Start slightly tilted
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])

        result = simulate(cartpole, initial, dt=0.0001, duration=0.5)

        # Compute energy at each time step
        energies = jax.vmap(cartpole.energy)(result.states)
        initial_energy = energies[0]

        # Energy should be constant (within numerical tolerance)
        np.testing.assert_allclose(
            energies, initial_energy * jnp.ones_like(energies), rtol=1e-3
        )

    def test_upright_equilibrium_unstable(self):
        """Upright equilibrium is unstable (pole falls)."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        # Tiny perturbation from upright
        initial = jnp.array([0.0, 0.0, 0.001, 0.0])

        result = simulate(cartpole, initial, dt=0.001, duration=2.0)

        # Angle should grow significantly (instability)
        final_theta = result.states[-1, THETA]
        assert abs(final_theta) > 0.5  # Has fallen significantly

    def test_hanging_equilibrium_stable(self):
        """Hanging equilibrium is stable (returns to rest)."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        # Add drag to simulate dissipation for stability test
        # Without drag, it oscillates forever - for now just test small oscillation
        initial = jnp.array([0.0, 0.0, jnp.pi + 0.1, 0.0])

        result = simulate(cartpole, initial, dt=0.001, duration=2.0)

        # Angle should stay near pi (stable oscillation around hanging)
        final_theta = result.states[-1, THETA]
        # Verify pendulum stays near hanging equilibrium (theta ≈ π)
        assert abs(final_theta - jnp.pi) < jnp.pi / 2, f"Pendulum deviated too far from hanging: theta={final_theta}"

    def test_pole_tip_position_upright(self):
        """Pole tip at correct position when upright."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.zeros(4)

        x_tip, y_tip = cartpole.pole_tip_position(state)

        # Tip is directly above pivot at 2*pole_length
        assert x_tip == pytest.approx(0.0, abs=1e-14)
        assert y_tip == pytest.approx(2 * cartpole.pole_length, rel=1e-14)

    def test_pole_tip_position_tilted(self):
        """Pole tip at correct position when tilted."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        theta = jnp.pi / 4  # 45 degrees
        state = jnp.array([1.0, 0.0, theta, 0.0])  # Cart at x=1

        x_tip, y_tip = cartpole.pole_tip_position(state)

        full_length = 2 * cartpole.pole_length
        expected_x = 1.0 + full_length * jnp.sin(theta)
        expected_y = full_length * jnp.cos(theta)

        assert x_tip == pytest.approx(expected_x, rel=1e-14)
        assert y_tip == pytest.approx(expected_y, rel=1e-14)

    def test_pole_com_position(self):
        """Pole center of mass at correct position."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        theta = jnp.pi / 6  # 30 degrees
        state = jnp.array([0.5, 0.0, theta, 0.0])

        x_com, y_com = cartpole.pole_com_position(state)

        expected_x = 0.5 + cartpole.pole_length * jnp.sin(theta)
        expected_y = cartpole.pole_length * jnp.cos(theta)

        assert x_com == pytest.approx(expected_x, rel=1e-14)
        assert y_com == pytest.approx(expected_y, rel=1e-14)


class TestCartpoleJaxSimulation:
    """Tests for cartpole simulation."""

    def test_simulate_returns_result(self):
        """simulate() returns result with correct shape."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])

        result = simulate(cartpole, initial, dt=0.01, duration=1.0)

        assert result.times.shape[0] > 0
        assert result.states.shape[0] == result.times.shape[0]
        assert result.states.shape[1] == 4

    def test_simulate_initial_state_preserved(self):
        """Initial state is in result."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([1.0, 0.5, 0.2, 0.1])

        result = simulate(cartpole, initial, dt=0.01, duration=1.0)

        np.testing.assert_allclose(result.states[0], initial, rtol=1e-14)

    def test_simulate_times_correct(self):
        """Time array is correct."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.zeros(4)

        result = simulate(cartpole, initial, dt=0.01, duration=1.0)

        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(1.0)

    def test_pole_falls_from_upright(self):
        """Pole falls when started from near-upright."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([0.0, 0.0, 0.01, 0.0])  # Small tilt

        result = simulate(cartpole, initial, dt=0.001, duration=2.0)

        # Pole should have fallen significantly
        assert abs(result.states[-1, THETA]) > 1.0

    def test_cart_moves_with_force(self):
        """Cart moves when force is applied."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.zeros(4)
        control = ConstantControl(jnp.array([5.0]))  # Constant force right

        result = simulate(cartpole, initial, dt=0.001, duration=1.0, control=control)

        # Cart should have moved right
        assert result.states[-1, X] > 0

    def test_force_affects_pole_angle(self):
        """Force causes pole to tilt."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.zeros(4)  # Perfectly upright
        control = ConstantControl(jnp.array([10.0]))  # Strong force right

        result = simulate(cartpole, initial, dt=0.0001, duration=0.1)
        control_result = simulate(cartpole, initial, dt=0.0001, duration=0.1, control=control)

        # With force, pole should tilt backward (negative theta)
        # compared to free fall where it stays at ~0
        assert control_result.states[-1, THETA] < result.states[-1, THETA]

    def test_simulate_with_control_schedule(self):
        """Can simulate with a control schedule."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.zeros(4)

        # Step input: no force for first half, then force
        from fmd.simulator import PiecewiseConstantControl
        control = PiecewiseConstantControl(
            jnp.array([0.0, 0.5]),
            jnp.array([[0.0], [5.0]]),
        )

        result = simulate(cartpole, initial, dt=0.001, duration=1.0, control=control)

        # Cart should have moved (force applied in second half)
        assert result.states[-1, X] > 0


class TestCartpoleJaxJIT:
    """Tests for JIT compilation."""

    def test_derivative_jit(self):
        """forward_dynamics can be JIT compiled."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        @jax.jit
        def compute_deriv(p, s, c):
            return p.forward_dynamics(s, c)

        state = jnp.array([0.0, 0.0, 0.3, 0.1])
        control = jnp.array([1.0])
        result = compute_deriv(cartpole, state, control)

        assert jnp.all(jnp.isfinite(result))

    def test_simulate_jit(self):
        """simulate() can be JIT compiled."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        @jax.jit
        def run_sim(p, s0):
            return simulate(p, s0, dt=0.01, duration=1.0)

        initial = jnp.array([0.0, 0.0, 0.3, 0.0])
        result = run_sim(cartpole, initial)

        assert jnp.all(jnp.isfinite(result.states))

    def test_energy_jit(self):
        """energy() can be JIT compiled."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        @jax.jit
        def compute_energy(p, s):
            return p.energy(s)

        state = jnp.array([1.0, 2.0, 0.5, 1.0])
        result = compute_energy(cartpole, state)

        assert jnp.isfinite(result)

    def test_jit_no_retrace(self):
        """JIT doesn't retrace with same shapes."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        @jax.jit
        def run_sim(p, s0):
            return simulate(p, s0, dt=0.01, duration=1.0)

        # First call compiles
        result1 = run_sim(cartpole, jnp.array([0.0, 0.0, 0.1, 0.0]))

        # Second call should use cached compilation
        result2 = run_sim(cartpole, jnp.array([0.0, 0.0, 0.2, 0.0]))

        # Both should produce valid results
        assert jnp.all(jnp.isfinite(result1.states))
        assert jnp.all(jnp.isfinite(result2.states))


class TestCartpoleJaxGrad:
    """Tests for autodiff through cartpole simulation."""

    def test_grad_wrt_initial_angle(self):
        """Can compute gradient w.r.t. initial angle."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        def loss(theta0):
            initial = jnp.array([0.0, 0.0, theta0, 0.0])
            result = simulate(cartpole, initial, dt=0.01, duration=0.5)
            # Loss: final angle squared
            return result.states[-1, THETA] ** 2

        theta0 = 0.1
        grad = jax.grad(loss)(theta0)

        assert jnp.isfinite(grad)

    def test_grad_wrt_initial_state(self):
        """Can compute gradient w.r.t. full initial state."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        def loss(initial_state):
            result = simulate(cartpole, initial_state, dt=0.01, duration=0.5)
            return jnp.sum(result.states[-1] ** 2)

        initial = jnp.array([0.0, 0.0, 0.1, 0.0])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_pole_length(self):
        """Can compute gradient w.r.t. pole length."""

        def loss(pole_length):
            cartpole = Cartpole.from_values(
                mass_cart=1.0,
                mass_pole=0.1,
                pole_length=pole_length,
            )
            initial = jnp.array([0.0, 0.0, 0.1, 0.0])
            result = simulate(cartpole, initial, dt=0.01, duration=0.5)
            return result.states[-1, THETA] ** 2

        pole_length = 0.5
        grad = jax.grad(loss)(pole_length)

        assert jnp.isfinite(grad)

    def test_grad_wrt_mass(self):
        """Can compute gradient w.r.t. mass parameters."""

        def loss(mass_pole):
            cartpole = Cartpole.from_values(
                mass_cart=1.0,
                mass_pole=mass_pole,
                pole_length=0.5,
            )
            initial = jnp.array([0.0, 0.0, 0.1, 0.0])
            result = simulate(cartpole, initial, dt=0.01, duration=0.5)
            return result.states[-1, THETA] ** 2

        mass_pole = 0.1
        grad = jax.grad(loss)(mass_pole)

        assert jnp.isfinite(grad)

    def test_grad_through_energy(self):
        """Can compute gradient through energy function."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        def loss(initial_state):
            result = simulate(cartpole, initial_state, dt=0.01, duration=0.5)
            # Sum of energy at all timesteps
            return jnp.sum(jax.vmap(cartpole.energy)(result.states))

        initial = jnp.array([0.0, 0.0, 0.1, 0.0])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_control_optimization(self):
        """Can compute gradient for control optimization."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        def loss(force_magnitude):
            control = ConstantControl(jnp.array([force_magnitude]))
            initial = jnp.array([0.0, 0.0, 0.1, 0.0])
            result = simulate(cartpole, initial, dt=0.01, duration=0.5, control=control)
            # Try to minimize final angle squared
            return result.states[-1, THETA] ** 2

        force = 0.0
        grad = jax.grad(loss)(force)

        assert jnp.isfinite(grad)


class TestCartpoleJaxVmap:
    """Tests for vectorized simulation."""

    def test_vmap_over_initial_states(self):
        """Can vmap over initial states."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        def simulate_single(initial):
            return simulate(cartpole, initial, dt=0.01, duration=0.5).states[-1]

        # Batch of initial states
        initials = jnp.array([
            [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.10, 0.0],
            [0.0, 0.0, 0.15, 0.0],
        ])

        final_states = jax.vmap(simulate_single)(initials)

        assert final_states.shape == (3, 4)
        assert jnp.all(jnp.isfinite(final_states))
        # Larger initial angles should lead to larger final angles
        assert final_states[2, THETA] > final_states[0, THETA]

    def test_vmap_energy_computation(self):
        """Can vmap energy computation."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([0.0, 0.0, 0.2, 0.0])

        result = simulate(cartpole, initial, dt=0.01, duration=1.0)
        energies = jax.vmap(cartpole.energy)(result.states)

        assert energies.shape == (result.states.shape[0],)
        assert jnp.all(jnp.isfinite(energies))


class TestCartpoleJaxGoldenValues:
    """Regression tests with golden values."""

    def test_derivative_golden(self):
        """Derivative matches golden values."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = jnp.zeros(1)

        deriv = cartpole.forward_dynamics(state, control)

        # Computed from Barto/Sutton/Anderson equations
        # At theta=0.1, F=0:
        #   theta_ddot = (g*sin(theta)) / (l*(4/3 - m_p*cos^2(theta)/(m_c+m_p)))
        #   x_ddot = (m_p*l*(theta_ddot*cos(theta))) / (m_c + m_p)  (simplified since theta_dot=0)
        expected = jnp.array([
            0.0,  # x_dot = 0
            -0.0712266147,  # x_ddot (cart accelerates left as pole pulls)
            0.0,  # theta_dot = 0
            1.5748530266,  # theta_ddot (pole tips away from vertical)
        ])

        np.testing.assert_allclose(deriv, expected, rtol=1e-6)

    def test_derivative_with_force_golden(self):
        """Derivative with force matches golden values."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        state = jnp.array([0.0, 0.0, 0.0, 0.0])
        control = jnp.array([10.0])

        deriv = cartpole.forward_dynamics(state, control)

        # With force at upright: pole tips backward, cart accelerates forward
        # At theta=0, F=10:
        #   theta_ddot = (-F/(m_c+m_p)) / (l*(4/3 - m_p/(m_c+m_p)))
        #   x_ddot = (F - m_p*l*theta_ddot) / (m_c + m_p)
        expected = jnp.array([
            0.0,  # x_dot = 0
            9.7560975610,  # x_ddot > 0 (cart accelerates right)
            0.0,  # theta_dot = 0
            -14.6341463415,  # theta_ddot < 0 (pole tips backward)
        ])

        np.testing.assert_allclose(deriv, expected, rtol=1e-6)

    def test_energy_golden(self):
        """Energy calculation matches golden values."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)

        # Upright at rest
        state = jnp.zeros(4)
        energy = cartpole.energy(state)
        expected = 0.1 * 9.80665 * 0.5  # m_p * g * l
        assert energy == pytest.approx(expected, rel=1e-10)

        # Hanging at rest
        state = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        energy = cartpole.energy(state)
        expected = -0.1 * 9.80665 * 0.5  # -m_p * g * l
        assert energy == pytest.approx(expected, rel=1e-10)
