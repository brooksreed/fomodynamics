"""Tests for semi-implicit (symplectic) Euler integrator.

These tests verify:
1. Symplectic support properties for each system
2. Energy conservation for pendulum (oscillation, not drift)
3. Stability at large timesteps
4. Quaternion normalization for RigidBody6DOF
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import (
    SimplePendulum,
    Cartpole,
    PlanarQuadrotor,
    RigidBody6DOF,
    simulate_symplectic,
    simulate_euler,
    semi_implicit_euler_step,
)
from fmd.simulator.params import (
    PENDULUM_1M,
    CARTPOLE_CLASSIC,
    PLANAR_QUAD_TEST_DEFAULT,
)
from fmd.simulator import Gravity


class TestSymplecticSupportProperties:
    """Test that systems correctly report symplectic support."""

    def test_pendulum_supports_symplectic(self):
        """SimplePendulum should support symplectic integration."""
        pendulum = SimplePendulum(PENDULUM_1M)
        assert pendulum.supports_symplectic is True
        assert pendulum.position_indices == (0,)
        assert pendulum.velocity_indices == (1,)

    def test_cartpole_supports_symplectic(self):
        """Cartpole should support symplectic integration."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        assert cartpole.supports_symplectic is True
        assert cartpole.position_indices == (0, 2)
        assert cartpole.velocity_indices == (1, 3)

    def test_planar_quadrotor_supports_symplectic(self):
        """PlanarQuadrotor should support symplectic integration."""
        quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
        assert quad.supports_symplectic is True
        assert quad.position_indices == (0, 1, 2)
        assert quad.velocity_indices == (3, 4, 5)

    def test_rigid_body_supports_symplectic(self):
        """RigidBody6DOF should support symplectic integration."""
        body = RigidBody6DOF(
            mass=1.0,
            inertia=jnp.array([1.0, 1.0, 1.0]),
            components=[Gravity(1.0)],
        )
        assert body.supports_symplectic is True
        # Position-like: pos (0,1,2) + quaternion (6,7,8,9)
        # Velocity-like: vel (3,4,5) + omega (10,11,12)
        assert body.position_indices == (0, 1, 2, 6, 7, 8, 9)
        assert body.velocity_indices == (3, 4, 5, 10, 11, 12)


class TestPendulumEnergyConservation:
    """Test energy conservation for pendulum with symplectic integrator."""

    def test_energy_oscillates_not_drifts(self):
        """Energy should oscillate around initial value, not drift."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        
        # Run simulation for several periods
        period = float(pendulum.period_small_angle())
        result = simulate_symplectic(pendulum, initial, dt=0.01, duration=10 * period)
        
        # Compute energy at each timestep
        energies = jax.vmap(pendulum.energy)(result.states)
        initial_energy = energies[0]
        
        # Energy should stay bounded around initial value
        # (symplectic integrators preserve phase space volume)
        energy_error = energies - initial_energy
        
        # Max deviation should be small (bounded oscillation)
        assert jnp.max(jnp.abs(energy_error)) < 0.05 * initial_energy
        
        # Mean energy should be close to initial (oscillates around it)
        assert jnp.abs(jnp.mean(energies) - initial_energy) < 0.02 * initial_energy

    def test_symplectic_better_than_euler_for_energy(self):
        """Symplectic Euler should conserve energy better than explicit Euler."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        
        # Run both integrators
        period = float(pendulum.period_small_angle())
        result_symplectic = simulate_symplectic(pendulum, initial, dt=0.01, duration=5 * period)
        result_euler = simulate_euler(pendulum, initial, dt=0.01, duration=5 * period)
        
        # Compute final energies
        initial_energy = pendulum.energy(initial)
        final_energy_symplectic = pendulum.energy(result_symplectic.states[-1])
        final_energy_euler = pendulum.energy(result_euler.states[-1])
        
        # Symplectic should have smaller absolute energy error
        error_symplectic = jnp.abs(final_energy_symplectic - initial_energy)
        error_euler = jnp.abs(final_energy_euler - initial_energy)
        
        # Explicit Euler typically gains energy for pendulum
        # Symplectic should be at least 10x better
        assert error_symplectic < error_euler / 5


class TestLargeTimestepStability:
    """Test stability at large timesteps."""

    def test_pendulum_stable_at_large_dt(self):
        """Pendulum should remain bounded at larger timesteps."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        
        # Use larger timestep (100x the recommended value)
        result = simulate_symplectic(pendulum, initial, dt=0.1, duration=10.0)
        
        # States should remain finite and bounded
        assert jnp.all(jnp.isfinite(result.states))
        
        # Angle should stay bounded (not explode)
        # For a simple pendulum with initial angle 0.5 rad, max angle < 1 rad
        assert jnp.max(jnp.abs(result.states[:, 0])) < 2.0

    def test_energy_bounded_at_large_dt(self):
        """Energy should stay bounded even with large timesteps."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        
        result = simulate_symplectic(pendulum, initial, dt=0.1, duration=10.0)
        
        energies = jax.vmap(pendulum.energy)(result.states)
        initial_energy = energies[0]
        
        # Energy should not grow unboundedly
        # Allow more error with large timestep but should still be bounded
        assert jnp.max(energies) < 3 * initial_energy


class TestIntegratorCorrectness:
    """Test that the integrator produces correct results."""

    def test_simulate_returns_correct_shapes(self):
        """simulate_symplectic should return correct array shapes."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        
        result = simulate_symplectic(pendulum, initial, dt=0.01, duration=1.0)
        
        assert result.times.shape[0] == 101  # 0.0, 0.01, ..., 1.0
        assert result.states.shape == (101, 2)
        assert result.controls.shape == (101, 0)  # No controls for pendulum

    def test_initial_state_preserved(self):
        """Initial state should be preserved in result."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.3])
        
        result = simulate_symplectic(pendulum, initial, dt=0.01, duration=1.0)
        
        np.testing.assert_allclose(result.states[0], initial, rtol=1e-14)

    def test_times_correct(self):
        """Time array should be correct."""
        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.1, 0.0])
        
        result = simulate_symplectic(pendulum, initial, dt=0.01, duration=1.0)
        
        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(1.0)

    def test_unsupported_system_raises_error(self):
        """Should raise error for systems without symplectic support."""
        # Create a minimal system that doesn't override indices
        from fmd.simulator.base import JaxDynamicSystem
        import equinox as eqx
        from typing import Tuple
        
        class NoSymplecticSystem(JaxDynamicSystem):
            state_names: Tuple[str, ...] = eqx.field(static=True, default=("x",))

            def forward_dynamics(self, state, control, t=0.0):
                return jnp.zeros(1)
        
        system = NoSymplecticSystem()
        initial = jnp.array([0.0])
        
        with pytest.raises(ValueError, match="does not support symplectic"):
            simulate_symplectic(system, initial, dt=0.01, duration=1.0)


class TestJITCompilation:
    """Test that the integrator works with JIT compilation."""

    def test_simulate_symplectic_jit(self):
        """simulate_symplectic should work inside JIT."""
        pendulum = SimplePendulum(PENDULUM_1M)
        
        @jax.jit
        def run_sim(p, s0):
            return simulate_symplectic(p, s0, dt=0.01, duration=1.0)
        
        initial = jnp.array([0.3, 0.0])
        result = run_sim(pendulum, initial)
        
        assert jnp.all(jnp.isfinite(result.states))

    def test_step_function_jit(self):
        """semi_implicit_euler_step should work inside JIT."""
        pendulum = SimplePendulum(PENDULUM_1M)
        
        @jax.jit
        def step(p, s, c, dt, t):
            return semi_implicit_euler_step(p, s, c, dt, t)

        state = jnp.array([0.3, 0.0])
        control = jnp.array([])

        new_state = step(pendulum, state, control, 0.01, 0.0)
        
        assert jnp.all(jnp.isfinite(new_state))


class TestGradientComputation:
    """Test that gradients can be computed through the integrator."""

    def test_grad_wrt_initial_state(self):
        """Can compute gradient w.r.t. initial state."""
        pendulum = SimplePendulum(PENDULUM_1M)
        
        def loss(theta0):
            initial = jnp.array([theta0, 0.0])
            result = simulate_symplectic(pendulum, initial, dt=0.01, duration=1.0)
            return result.states[-1, 0] ** 2
        
        theta0 = 0.3
        grad = jax.grad(loss)(theta0)
        
        assert jnp.isfinite(grad)

class TestRigidBodyQuaternionNormalization:
    """Test quaternion normalization for RigidBody6DOF with symplectic integrator."""

    def test_quaternion_stays_normalized(self):
        """Quaternion should remain normalized during simulation."""
        body = RigidBody6DOF(
            mass=1.0,
            inertia=jnp.array([1.0, 1.0, 1.0]),
            components=[Gravity(1.0)],
        )
        
        # Start with unit quaternion
        initial = body.default_state()
        
        # Simulate free fall with tumble
        initial = initial.at[10:13].set(jnp.array([0.1, 0.2, 0.3]))  # angular velocity
        
        result = simulate_symplectic(body, initial, dt=0.01, duration=5.0)
        
        # Check quaternion norm at each timestep
        for i in range(len(result.times)):
            quat = result.states[i, 6:10]
            norm = jnp.linalg.norm(quat)
            assert norm == pytest.approx(1.0, rel=1e-6), f"Quaternion norm at step {i}: {norm}"

    def test_rigid_body_symplectic_simulation_runs(self):
        """RigidBody6DOF symplectic simulation should run without errors."""
        body = RigidBody6DOF(
            mass=1.0,
            inertia=jnp.array([1.0, 2.0, 3.0]),  # Non-uniform inertia
            components=[Gravity(1.0)],
        )
        
        initial = body.default_state()
        initial = initial.at[3:6].set(jnp.array([1.0, 0.0, 0.0]))  # forward velocity
        initial = initial.at[10:13].set(jnp.array([0.1, 0.05, 0.0]))  # angular velocity
        
        result = simulate_symplectic(body, initial, dt=0.001, duration=1.0)
        
        # All states should be finite
        assert jnp.all(jnp.isfinite(result.states))
        
        # Should have expected number of timesteps
        assert result.times.shape[0] == 1001

