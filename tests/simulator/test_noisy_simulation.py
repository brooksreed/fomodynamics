"""Tests for simulate_noisy() - stochastic simulation with process noise.

This module tests the process noise integration in the simulator:
- Basic functionality (noise adds variance, different keys give different results)
- Backward compatibility (no noise matches deterministic simulate())
- JIT compatibility
- System-specific tests (Boat2D)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator import SimplePendulum, Boat2D, simulate, simulate_noisy
from fmd.simulator.noise import GaussianNoise, ZeroNoise
from fmd.simulator.params import PENDULUM_1M, BOAT2D_TEST_DEFAULT


class TestSimulateNoisyBasics:
    """Basic tests for simulate_noisy() functionality."""

    def test_no_noise_matches_deterministic(self):
        """With process_noise=None, simulate_noisy matches simulate exactly."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # 0.5 rad initial angle
        dt = 0.01
        duration = 1.0

        # Deterministic simulation
        result_det = simulate(system, initial, dt=dt, duration=duration)

        # Noisy simulation with no noise (should be identical)
        result_noisy = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=None, prng_key=None
        )

        np.testing.assert_allclose(
            np.asarray(result_det.times),
            np.asarray(result_noisy.times),
            rtol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(result_det.states),
            np.asarray(result_noisy.states),
            rtol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(result_det.controls),
            np.asarray(result_noisy.controls),
            rtol=1e-14
        )

    def test_requires_prng_key_with_noise(self):
        """ValueError raised when process_noise provided but prng_key is None."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        noise = GaussianNoise.isotropic(dim=2, variance=0.01)

        with pytest.raises(ValueError, match="prng_key is required"):
            simulate_noisy(
                system, initial, dt=0.01, duration=1.0,
                process_noise=noise, prng_key=None
            )

    def test_noise_adds_variance(self):
        """Process noise adds variance to the trajectory."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        dt = 0.01
        duration = 1.0

        # Deterministic simulation
        result_det = simulate(system, initial, dt=dt, duration=duration)

        # Noisy simulation with significant noise
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)
        key = jax.random.key(42)
        result_noisy = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key
        )

        # Trajectories should differ
        diff = np.asarray(result_noisy.states) - np.asarray(result_det.states)
        assert np.any(np.abs(diff) > 1e-6), "Noisy trajectory should differ from deterministic"

    def test_different_keys_different_trajectories(self):
        """Different PRNG keys produce different trajectories."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        dt = 0.01
        duration = 1.0
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)

        key1 = jax.random.key(42)
        key2 = jax.random.key(123)

        result1 = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key1
        )
        result2 = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key2
        )

        # Trajectories should be different
        diff = np.asarray(result1.states) - np.asarray(result2.states)
        assert np.any(np.abs(diff) > 1e-6), "Different keys should produce different trajectories"

    def test_same_key_reproducible(self):
        """Same PRNG key produces identical trajectories."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        dt = 0.01
        duration = 1.0
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)

        key = jax.random.key(42)

        result1 = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key
        )
        result2 = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key
        )

        np.testing.assert_allclose(
            np.asarray(result1.states),
            np.asarray(result2.states),
            rtol=1e-14
        )


class TestSimulateNoisyWithBoat2D:
    """Test noisy simulation with Boat2D model."""

    def test_boat2d_noisy_simulation(self):
        """Boat2D can be simulated with process noise."""
        system = Boat2D(BOAT2D_TEST_DEFAULT)
        initial = system.default_state()
        dt = 0.01
        duration = 2.0

        # Create noise matching Boat2D state dimension (6 states)
        noise = GaussianNoise.isotropic(dim=6, variance=0.0001)
        key = jax.random.key(42)

        result = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=noise, prng_key=key
        )

        # Verify result structure
        assert result.times.shape[0] == result.states.shape[0]
        assert result.states.shape[1] == 6  # Boat2D has 6 states

        # Verify simulation ran without errors and trajectory is reasonable
        assert not np.any(np.isnan(result.states))
        assert not np.any(np.isinf(result.states))


class TestSimulateNoisyJIT:
    """Test JIT compatibility of simulate_noisy."""

    def test_jit_compatible(self):
        """simulate_noisy can be JIT-compiled."""
        system = SimplePendulum(PENDULUM_1M)
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)

        # Define a function that uses simulate_noisy
        def run_simulation(initial, key):
            return simulate_noisy(
                system, initial, dt=0.01, duration=0.5,
                process_noise=noise, prng_key=key
            )

        # JIT compile
        run_simulation_jit = jax.jit(run_simulation)

        initial = jnp.array([0.5, 0.0])
        key = jax.random.key(42)

        # Run JIT-compiled version
        result = run_simulation_jit(initial, key)

        # Verify it produces valid output
        assert result.times.shape[0] > 1
        assert not np.any(np.isnan(result.states))

        # Run again to verify caching works
        result2 = run_simulation_jit(initial, key)
        np.testing.assert_allclose(
            np.asarray(result.states),
            np.asarray(result2.states),
            rtol=1e-14
        )


class TestZeroNoiseEquivalence:
    """Test that ZeroNoise produces deterministic results."""

    def test_zero_noise_matches_deterministic(self):
        """ZeroNoise should produce same results as deterministic simulate."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        dt = 0.01
        duration = 1.0

        # Deterministic simulation
        result_det = simulate(system, initial, dt=dt, duration=duration)

        # Simulation with ZeroNoise
        zero_noise = ZeroNoise(dim=2)
        key = jax.random.key(42)  # Key is ignored by ZeroNoise

        result_zero = simulate_noisy(
            system, initial, dt=dt, duration=duration,
            process_noise=zero_noise, prng_key=key
        )

        np.testing.assert_allclose(
            np.asarray(result_det.states),
            np.asarray(result_zero.states),
            rtol=1e-14
        )


class TestSimulateNoisyEdgeCases:
    """Edge case tests for simulate_noisy."""

    def test_short_duration(self):
        """Handle very short simulations."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)
        key = jax.random.key(42)

        result = simulate_noisy(
            system, initial, dt=0.01, duration=0.02,
            process_noise=noise, prng_key=key
        )

        # Should have a few time steps
        assert result.times.shape[0] >= 2

    def test_large_noise_still_runs(self):
        """Large noise should not cause simulation to fail."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])

        # Large noise (may produce non-physical results but shouldn't crash)
        noise = GaussianNoise.isotropic(dim=2, variance=1.0)
        key = jax.random.key(42)

        result = simulate_noisy(
            system, initial, dt=0.01, duration=0.5,
            process_noise=noise, prng_key=key
        )

        # Should complete without NaN (post_step doesn't blow up pendulum)
        # Note: For systems with post_step normalization, large noise may be tolerable
        assert result.times.shape[0] > 1

    def test_non_divisible_duration(self):
        """Duration not divisible by dt should work correctly."""
        system = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)
        key = jax.random.key(42)

        result = simulate_noisy(
            system, initial, dt=0.3, duration=1.0,
            process_noise=noise, prng_key=key
        )

        # Should end exactly at duration
        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(1.0)
