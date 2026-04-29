"""Noise models for simulation and estimation.

This package provides noise models for:
- Process noise (state transition uncertainty)
- Measurement noise (sensor uncertainty)
- Environmental disturbances (future)

All noise models use JAX PRNG for reproducible randomness and
are JIT-compatible for use in compiled simulation loops.

Key Pattern - PRNG Key Management:
    import jax

    key = jax.random.key(0)          # Create initial key
    key, subkey = jax.random.split(key)  # Split for each use
    sample = noise_model.sample(subkey, shape)  # subkey is consumed
    # Continue using 'key' for next operations

Classes:
    NoiseModel: Abstract base class for all noise models
    ZeroNoise: Always returns zero (for deterministic runs)
    GaussianNoise: Multivariate Gaussian noise
    ScalarGaussianNoise: Convenience class for 1D Gaussian

Example:
    from fmd.simulator.noise import GaussianNoise
    import jax
    import jax.numpy as jnp

    # Create process noise model (4D state)
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.01, 0.01]))
    process_noise = GaussianNoise(mean=jnp.zeros(4), cov=Q)

    # Generate noise samples
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    w = process_noise.sample(subkey, (4,))
    print(f"Process noise sample: {w}")

    # Or use convenience factory
    noise = GaussianNoise.isotropic(dim=4, variance=0.01)
"""

from fmd.simulator.noise.base import (
    NoiseModel,
    ZeroNoise,
)
from fmd.simulator.noise.gaussian import (
    GaussianNoise,
    ScalarGaussianNoise,
)

__all__ = [
    # Base classes
    "NoiseModel",
    "ZeroNoise",
    # Gaussian noise
    "GaussianNoise",
    "ScalarGaussianNoise",
]
