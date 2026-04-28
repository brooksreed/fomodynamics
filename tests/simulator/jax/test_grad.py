"""Gradient smoke tests for JAX implementations.

These tests verify that jax.grad works through the simulation
infrastructure. They don't verify gradient correctness (that would
require finite differences), just that gradients can be computed
and are finite.
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import SimplePendulum as SimplePendulumJax, simulate
from fmd.simulator.params import PENDULUM_1M


class TestPendulumGradientSmoke:
    """Gradient smoke tests for SimplePendulumJax."""

    def test_grad_final_angle_wrt_initial_angle(self):
        """Gradient of final angle w.r.t. initial angle exists."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(theta0):
            initial = jnp.array([theta0, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0]

        grad = jax.grad(loss)(0.3)
        assert jnp.isfinite(grad)

    def test_grad_final_angle_wrt_initial_velocity(self):
        """Gradient of final angle w.r.t. initial velocity exists."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(theta_dot0):
            initial = jnp.array([0.3, theta_dot0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0]

        grad = jax.grad(loss)(0.0)
        assert jnp.isfinite(grad)

    def test_grad_trajectory_sum_wrt_initial(self):
        """Gradient of trajectory integral w.r.t. initial state."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(initial):
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            # Sum of squared angles over trajectory
            return jnp.sum(result.states[:, 0] ** 2)

        initial = jnp.array([0.3, 0.0])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_length(self):
        """Gradient w.r.t. pendulum length exists."""

        def loss(length):
            # Use from_values to avoid attrs validation during tracing
            pendulum = SimplePendulumJax.from_values(length=length)
            initial = jnp.array([0.3, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0] ** 2

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_grad_wrt_gravity(self):
        """Gradient w.r.t. gravity exists."""

        def loss(g):
            # Use from_values to avoid attrs validation during tracing
            pendulum = SimplePendulumJax.from_values(length=1.0, g=g)
            initial = jnp.array([0.3, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0] ** 2

        grad = jax.grad(loss)(9.81)
        assert jnp.isfinite(grad)

    def test_grad_max_angle_wrt_initial_energy(self):
        """Gradient of max angle w.r.t. initial conditions."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(initial):
            result = simulate(pendulum, initial, dt=0.01, duration=2.0)
            # Max absolute angle
            return jnp.max(jnp.abs(result.states[:, 0]))

        initial = jnp.array([0.3, 0.5])
        grad = jax.grad(loss)(initial)

        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_energy_conservation_loss(self):
        """Gradient of energy conservation loss."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def energy_loss(initial):
            result = simulate(pendulum, initial, dt=0.01, duration=2.0)
            # Compute energy at each timestep
            energies = jax.vmap(pendulum.energy)(result.states)
            # Loss: variance of energy (should be zero for perfect conservation)
            return jnp.var(energies)

        initial = jnp.array([0.5, 0.0])
        grad = jax.grad(energy_loss)(initial)

        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))


class TestGradientNumericalCheck:
    """Numerical gradient checks using finite differences."""

    def test_gradient_numerical_check_angle(self):
        """Verify gradient is approximately correct via finite differences."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(theta0):
            initial = jnp.array([theta0, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=0.5)
            return result.states[-1, 0]

        theta0 = 0.3
        eps = 1e-5

        # Autodiff gradient
        auto_grad = jax.grad(loss)(theta0)

        # Finite difference gradient
        fd_grad = (loss(theta0 + eps) - loss(theta0 - eps)) / (2 * eps)

        # Should be close (not exact due to numerical differentiation)
        np.testing.assert_allclose(auto_grad, fd_grad, rtol=1e-3)

    def test_gradient_numerical_check_length(self):
        """Verify gradient w.r.t. length is approximately correct."""

        def loss(length):
            # Use from_values to avoid attrs validation during tracing
            pendulum = SimplePendulumJax.from_values(length=length)
            initial = jnp.array([0.3, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=0.5)
            return result.states[-1, 0]

        length = 1.0
        eps = 1e-5

        auto_grad = jax.grad(loss)(length)
        fd_grad = (loss(length + eps) - loss(length - eps)) / (2 * eps)

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=1e-3)


class TestHigherOrderGradients:
    """Tests for higher-order derivatives."""

    def test_hessian_exists(self):
        """Hessian (second derivative) can be computed."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def loss(theta0):
            initial = jnp.array([theta0, 0.0])
            result = simulate(pendulum, initial, dt=0.01, duration=0.5)
            return result.states[-1, 0] ** 2

        # Second derivative
        hessian = jax.grad(jax.grad(loss))(0.3)
        assert jnp.isfinite(hessian)

    def test_jacobian_trajectory(self):
        """Jacobian of trajectory w.r.t. initial state."""
        pendulum = SimplePendulumJax(PENDULUM_1M)

        def trajectory(initial):
            result = simulate(pendulum, initial, dt=0.1, duration=1.0)
            return result.states[-1]  # Final state

        initial = jnp.array([0.3, 0.0])
        jac = jax.jacobian(trajectory)(initial)

        assert jac.shape == (2, 2)  # 2 outputs x 2 inputs
        assert jnp.all(jnp.isfinite(jac))
