"""Tests for JAX SimplePendulum implementation.

These tests verify:
1. Correct physics (period, energy conservation)
2. JIT compilation works
3. Autodiff works through simulation
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import SimplePendulum as SimplePendulumJax, simulate
from fmd.simulator.params import SimplePendulumParams, PENDULUM_1M, PENDULUM_2M

from .conftest import ANALYTICAL_RTOL, TRAJ_RTOL, TRAJ_ATOL


class TestSimplePendulumJaxBasics:
    """Basic tests for SimplePendulumJax."""

    def test_create_from_params(self):
        """Can create pendulum from params object."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        assert pendulum.length == 1.0
        assert pendulum.g == pytest.approx(9.80665)

    def test_create_custom_params(self):
        """Can create pendulum with custom params."""
        params = SimplePendulumParams(length=2.5, g=10.0)
        pendulum = SimplePendulumJax(params)
        assert pendulum.length == 2.5
        assert pendulum.g == 10.0

    def test_state_names(self):
        """State names are correct."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        assert pendulum.state_names == ("theta", "theta_dot")

    def test_control_names_empty(self):
        """Control names are empty (no control inputs)."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        assert pendulum.control_names == ()

    def test_num_states(self):
        """num_states is 2."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        assert pendulum.num_states == 2

    def test_num_controls(self):
        """num_controls is 0."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        assert pendulum.num_controls == 0

    def test_default_state(self):
        """Default state is zeros (hanging at rest)."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        state = pendulum.default_state()
        np.testing.assert_allclose(state, jnp.zeros(2), atol=1e-15)

    def test_default_control(self):
        """Default control is empty array."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        control = pendulum.default_control()
        assert control.shape == (0,)


class TestSimplePendulumJaxDerivatives:
    """Tests for state derivative computation."""

    def test_derivative_at_rest(self):
        """Zero derivative when hanging at rest."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        state = jnp.array([0.0, 0.0])  # theta=0, theta_dot=0
        control = jnp.array([])

        deriv = pendulum.forward_dynamics(state, control)

        np.testing.assert_allclose(deriv, jnp.zeros(2), atol=1e-15)

    def test_derivative_small_angle(self):
        """Derivative correct for small displacement."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        theta = 0.1  # Small angle
        state = jnp.array([theta, 0.0])
        control = jnp.array([])

        deriv = pendulum.forward_dynamics(state, control)

        # theta_dot = 0, theta_ddot = -(g/L)*sin(theta) ≈ -(g/L)*theta
        expected_theta_ddot = -(pendulum.g / pendulum.length) * jnp.sin(theta)
        np.testing.assert_allclose(deriv[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(deriv[1], expected_theta_ddot, rtol=1e-14)

    def test_derivative_with_velocity(self):
        """Derivative correct with angular velocity."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        theta = 0.5
        theta_dot = 1.0
        state = jnp.array([theta, theta_dot])
        control = jnp.array([])

        deriv = pendulum.forward_dynamics(state, control)

        expected_theta_dot = theta_dot
        expected_theta_ddot = -(pendulum.g / pendulum.length) * jnp.sin(theta)

        np.testing.assert_allclose(deriv[0], expected_theta_dot, rtol=1e-14)
        np.testing.assert_allclose(deriv[1], expected_theta_ddot, rtol=1e-14)


class TestSimplePendulumJaxSimulation:
    """Tests for pendulum simulation."""

    def test_simulate_returns_result(self):
        """simulate() returns JaxSimulationResult."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.1, 0.0])

        result = simulate(pendulum, initial, dt=0.01, duration=1.0)

        assert result.times.shape[0] > 0
        assert result.states.shape[0] == result.times.shape[0]
        assert result.states.shape[1] == 2

    def test_simulate_initial_state_preserved(self):
        """Initial state is in result."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.5, 0.1])

        result = simulate(pendulum, initial, dt=0.01, duration=1.0)

        np.testing.assert_allclose(result.states[0], initial, rtol=1e-14)

    def test_simulate_times_correct(self):
        """Time array is correct."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.1, 0.0])

        result = simulate(pendulum, initial, dt=0.01, duration=1.0)

        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(1.0)

    def test_small_angle_period(self):
        """Period matches analytical for small angles."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        theta0 = 0.05  # Small angle (~3 degrees)
        initial = jnp.array([theta0, 0.0])

        # Simulate for ~3 periods
        expected_period = float(pendulum.period_small_angle())
        result = simulate(pendulum, initial, dt=0.001, duration=3.5 * expected_period)

        # Find zero crossings (theta going from positive to negative)
        # These occur once per period (at the end of each oscillation)
        theta = np.array(result.states[:, 0])
        times = np.array(result.times)
        zero_crossings = []
        for i in range(1, len(theta)):
            if theta[i - 1] > 0 and theta[i] <= 0:
                # Linear interpolation to find crossing time
                t = times[i - 1] + (times[i] - times[i - 1]) * (
                    theta[i - 1] / (theta[i - 1] - theta[i])
                )
                zero_crossings.append(t)

        # Should have at least 3 crossings for 3 periods
        assert len(zero_crossings) >= 3, f"Only found {len(zero_crossings)} crossings"

        # Period is time between consecutive same-direction crossings
        # (positive-to-negative crossings happen once per period)
        measured_period = zero_crossings[1] - zero_crossings[0]

        # Relax tolerance for simulation numerical error
        # The small angle formula is approximate, and integration introduces error
        np.testing.assert_allclose(
            measured_period, expected_period, rtol=1e-3
        )


class TestSimplePendulumJaxEnergy:
    """Tests for energy conservation."""

    def test_energy_at_rest(self):
        """Energy is zero when hanging at rest."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        state = jnp.array([0.0, 0.0])

        energy = pendulum.energy(state)

        assert energy == pytest.approx(0.0, abs=1e-15)

    def test_energy_displaced(self):
        """Energy is positive when displaced."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        state = jnp.array([0.5, 0.0])  # Displaced, no velocity

        energy = pendulum.energy(state)

        # E = g*L*(1 - cos(theta))
        expected = pendulum.g * pendulum.length * (1 - jnp.cos(0.5))
        np.testing.assert_allclose(energy, expected, rtol=1e-14)

    def test_energy_conservation(self):
        """Energy is conserved during simulation."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # Released from rest

        result = simulate(pendulum, initial, dt=0.001, duration=5.0)

        # Compute energy at each time step
        energies = jax.vmap(pendulum.energy)(result.states)
        initial_energy = energies[0]

        # Energy should be constant (within numerical tolerance)
        np.testing.assert_allclose(
            energies, initial_energy * jnp.ones_like(energies), rtol=1e-4
        )


class TestSimplePendulumJaxJIT:
    """Tests for JIT compilation."""

    def test_derivative_jit(self):
        """forward_dynamics can be JIT compiled."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        @jax.jit
        def compute_deriv(p, s):
            return p.forward_dynamics(s, jnp.array([]))

        state = jnp.array([0.3, 0.1])
        result = compute_deriv(pendulum, state)

        assert jnp.all(jnp.isfinite(result))

    def test_simulate_jit(self):
        """simulate() can be JIT compiled."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        @jax.jit
        def run_sim(p, s0):
            return simulate(p, s0, dt=0.01, duration=1.0)

        initial = jnp.array([0.3, 0.0])
        result = run_sim(pendulum, initial)

        assert jnp.all(jnp.isfinite(result.states))

    def test_jit_no_retrace(self):
        """JIT doesn't retrace with same shapes."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        @jax.jit
        def run_sim(p, s0):
            return simulate(p, s0, dt=0.01, duration=1.0)

        # First call compiles
        result1 = run_sim(pendulum, jnp.array([0.1, 0.0]))

        # Second call should use cached compilation
        result2 = run_sim(pendulum, jnp.array([0.2, 0.0]))

        # Both should produce valid results
        assert jnp.all(jnp.isfinite(result1.states))
        assert jnp.all(jnp.isfinite(result2.states))


class TestSimplePendulumJaxGrad:
    """Tests for autodiff through pendulum simulation."""

    def test_grad_wrt_initial_angle(self):
        """Can compute gradient w.r.t. initial angle."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(theta0):
            initial = jnp.array([theta0, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            # Loss: final angle squared
            return result.states[-1, 0] ** 2

        theta0 = 0.3
        grad = jax.grad(loss)(theta0)

        assert jnp.isfinite(grad)

    def test_grad_wrt_length(self):
        """Can compute gradient w.r.t. pendulum length."""

        def loss(length):
            # Use from_values to avoid attrs validation during tracing
            pendulum = SimplePendulumJax.from_values(length=length)
            initial = jnp.array([0.3, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0] ** 2

        length = 1.0
        grad = jax.grad(loss)(length)

        assert jnp.isfinite(grad)

    def test_grad_simulation_loss(self):
        """Gradient through entire simulation trajectory."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def trajectory_loss(initial_state):
            result = simulate(pendulum, initial_state, dt=0.01, duration=2.0)
            # Track maximum displacement
            return jnp.max(jnp.abs(result.states[:, 0]))

        initial = jnp.array([0.3, 0.0])
        grad = jax.grad(trajectory_loss)(initial)

        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))


class TestSimplePendulumJaxCartesian:
    """Tests for Cartesian position computation."""

    def test_cartesian_at_rest(self):
        """Cartesian position correct when hanging at rest."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        state = jnp.array([0.0, 0.0])

        x, y = pendulum.cartesian_position(state)

        assert x == pytest.approx(0.0, abs=1e-15)
        assert y == pytest.approx(-1.0)  # -L when hanging

    def test_cartesian_displaced(self):
        """Cartesian position correct when displaced."""
        pendulum = SimplePendulumJax(PENDULUM_1M)
        theta = jnp.pi / 6  # 30 degrees
        state = jnp.array([theta, 0.0])

        x, y = pendulum.cartesian_position(state)

        expected_x = pendulum.length * jnp.sin(theta)
        expected_y = -pendulum.length * jnp.cos(theta)

        np.testing.assert_allclose(x, expected_x, rtol=1e-14)
        np.testing.assert_allclose(y, expected_y, rtol=1e-14)
