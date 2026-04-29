"""Base classes for noise models.

Noise models are JAX-compatible modules that generate random samples
for process noise, measurement noise, and disturbances.

Key Design Decisions:
    - All randomness via JAX PRNGKey (no numpy.random)
    - Pure functions (no internal state mutation)
    - JIT-compatible (no Python control flow in sample())
    - Key splitting handled by caller

PRNG Key Management Pattern:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    sample = noise_model.sample(subkey, (dim,))
    # Use 'key' for next operation, 'subkey' was consumed

Example:
    from fmd.simulator.noise import GaussianNoise
    import jax
    import jax.numpy as jnp

    noise = GaussianNoise.isotropic(dim=4, variance=0.01)
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    sample = noise.sample(subkey, (4,))
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class NoiseModel(eqx.Module):
    """Abstract base class for noise models.

    Noise models generate random samples from a specified distribution.
    All implementations must be JAX-compatible for use in JIT-compiled
    simulation loops.

    PRNG Key Management:
        The caller is responsible for splitting keys. Each call to
        sample() consumes one key and returns deterministic results
        for the same key. This enables reproducible simulations.

    Subclasses must implement:
        - sample(key, shape) -> Array
        - dim property

    Example:
        key = jax.random.key(0)
        key, subkey = jax.random.split(key)
        sample = noise_model.sample(subkey, (noise_model.dim,))
    """

    @abstractmethod
    def sample(self, key: jax.random.PRNGKey, shape: tuple[int, ...]) -> Array:
        """Generate a noise sample.

        Args:
            key: JAX PRNG key for random generation. The key is consumed
                 by this call; caller should split before calling.
            shape: Output shape. For multivariate distributions,
                   this is typically (dim,) for a single sample.

        Returns:
            Noise sample array with the requested shape
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the noise vector.

        For scalar noise, returns 1.
        For multivariate noise, returns the vector dimension.
        """
        pass

    def sample_trajectory(
        self,
        key: jax.random.PRNGKey,
        num_steps: int,
    ) -> Array:
        """Generate noise samples for a trajectory.

        Convenience method for generating a sequence of independent
        noise samples, e.g., for process noise over multiple timesteps.

        Args:
            key: JAX PRNG key (will be split internally)
            num_steps: Number of time steps

        Returns:
            Array of shape (num_steps, dim) with independent noise samples
        """
        keys = jax.random.split(key, num_steps)
        return jax.vmap(lambda k: self.sample(k, (self.dim,)))(keys)


class ZeroNoise(NoiseModel):
    """Noise model that always returns zero.

    Useful for:
        - Deterministic simulation (no process/measurement noise)
        - Testing without noise
        - Placeholder before adding real noise
        - A/B testing noise vs no-noise scenarios

    Attributes:
        _dim: Dimension of the zero noise vector

    Example:
        noise = ZeroNoise(dim=4)
        sample = noise.sample(key, (4,))  # Returns jnp.zeros(4)
    """

    _dim: int

    def __init__(self, dim: int):
        """Create zero noise model.

        Args:
            dim: Dimension of the noise vector

        Raises:
            ValueError: If dim <= 0
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    def sample(self, key: jax.random.PRNGKey, shape: tuple[int, ...]) -> Array:
        """Return zeros (ignores the key).

        Args:
            key: JAX PRNG key (ignored)
            shape: Output shape

        Returns:
            Array of zeros with the requested shape
        """
        return jnp.zeros(shape)

    @property
    def dim(self) -> int:
        """Dimension of the zero noise."""
        return self._dim
