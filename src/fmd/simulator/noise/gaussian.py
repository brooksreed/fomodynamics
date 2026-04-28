"""Gaussian noise model.

Provides multivariate Gaussian noise for process and measurement
noise in estimation and simulation.

The implementation uses Cholesky decomposition for efficient sampling:
    x = mean + L @ z  where z ~ N(0, I) and L @ L^T = cov

Example:
    from fmd.simulator.noise import GaussianNoise
    import jax
    import jax.numpy as jnp

    # Create 4D process noise
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.01, 0.01]))
    noise = GaussianNoise(mean=jnp.zeros(4), cov=Q)

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    sample = noise.sample(subkey, (4,))
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from fmd.simulator.noise.base import NoiseModel


class GaussianNoise(NoiseModel):
    """Multivariate Gaussian noise model.

    Generates samples from N(mean, cov) using JAX random.

    The covariance matrix is validated at construction and the
    Cholesky decomposition is precomputed for efficient sampling.

    Attributes:
        mean: Mean vector of shape (dim,)
        cov: Covariance matrix of shape (dim, dim), must be symmetric PSD
        _chol: Precomputed Cholesky decomposition (lower triangular)

    Sampling Method:
        Uses the transformation x = mean + L @ z where:
        - z ~ N(0, I) is standard normal
        - L is the Cholesky factor (L @ L^T = cov)

    Example:
        # 3D Gaussian with diagonal covariance
        noise = GaussianNoise(
            mean=jnp.zeros(3),
            cov=jnp.diag(jnp.array([0.1, 0.2, 0.1])),
        )

        key = jax.random.key(42)
        key, subkey = jax.random.split(key)
        sample = noise.sample(subkey, (3,))
    """

    mean: Array
    cov: Array
    _chol: Array  # Cholesky decomposition for efficient sampling

    def __init__(self, mean: Array, cov: Array):
        """Create Gaussian noise model.

        Args:
            mean: Mean vector, shape (dim,)
            cov: Covariance matrix, shape (dim, dim). Must be symmetric
                 and positive semi-definite.

        Raises:
            ValueError: If shapes are inconsistent or cov is not valid PSD
        """
        mean = jnp.asarray(mean)
        cov = jnp.asarray(cov)

        # Validate shapes
        if mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {mean.shape}")
        if cov.ndim != 2:
            raise ValueError(f"cov must be 2D, got shape {cov.shape}")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square, got shape {cov.shape}")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(
                f"mean dim ({mean.shape[0]}) must match cov dim ({cov.shape[0]})"
            )

        # Validate symmetry (use numpy for validation - this is construction time)
        cov_np = np.asarray(cov)
        if not np.allclose(cov_np, cov_np.T, rtol=1e-10, atol=1e-10):
            raise ValueError("cov must be symmetric")

        # Validate positive semi-definite via eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_np)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"cov must be positive semi-definite, "
                f"got min eigenvalue {eigenvalues.min()}"
            )

        # Compute Cholesky decomposition
        # Add small regularization for numerical stability with near-singular cov
        cov_reg = cov + jnp.eye(cov.shape[0]) * 1e-10
        chol = jnp.linalg.cholesky(cov_reg)

        self.mean = mean
        self.cov = cov
        self._chol = chol

    def sample(self, key: jax.random.PRNGKey, shape: tuple[int, ...]) -> Array:
        """Generate Gaussian noise sample.

        Uses the Cholesky decomposition for efficient sampling:
            x = mean + L @ z  where z ~ N(0, I) and L @ L^T = cov

        Args:
            key: JAX PRNG key
            shape: Requested output shape. Should be (dim,) for a single
                   sample or (batch, dim) for batched samples.

        Returns:
            Gaussian noise sample(s)
        """
        # Generate standard normal samples
        z = jax.random.normal(key, shape)

        # Transform: x = mean + L @ z
        if z.ndim == 1:
            # Single sample: (dim,)
            return self.mean + self._chol @ z
        else:
            # Batched: (batch, dim) @ (dim, dim).T -> (batch, dim)
            # Using z @ L.T is equivalent to L @ z for each row
            return self.mean + (z @ self._chol.T)

    @property
    def dim(self) -> int:
        """Dimension of the Gaussian distribution."""
        return self.mean.shape[0]

    @classmethod
    def from_diagonal(cls, mean: Array, variances: Array) -> GaussianNoise:
        """Create Gaussian with diagonal covariance (independent components).

        Convenience constructor for the common case of independent noise
        in each dimension.

        Args:
            mean: Mean vector of shape (dim,)
            variances: Variance for each dimension (diagonal of cov)

        Returns:
            GaussianNoise with diagonal covariance matrix

        Example:
            noise = GaussianNoise.from_diagonal(
                mean=jnp.zeros(4),
                variances=jnp.array([0.1, 0.1, 0.01, 0.01]),
            )
        """
        variances = jnp.asarray(variances)
        if jnp.any(variances < 0):
            raise ValueError("variances must be non-negative")
        cov = jnp.diag(variances)
        return cls(mean=jnp.asarray(mean), cov=cov)

    @classmethod
    def from_std(cls, mean: Array, std: Array) -> GaussianNoise:
        """Create Gaussian from standard deviations.

        Args:
            mean: Mean vector of shape (dim,)
            std: Standard deviation for each dimension

        Returns:
            GaussianNoise with diagonal covariance (std^2)

        Example:
            noise = GaussianNoise.from_std(
                mean=jnp.zeros(2),
                std=jnp.array([0.1, 0.05]),  # sqrt of variance
            )
        """
        std = jnp.asarray(std)
        if jnp.any(std < 0):
            raise ValueError("std must be non-negative")
        return cls.from_diagonal(mean=mean, variances=std**2)

    @classmethod
    def isotropic(cls, dim: int, variance: float) -> GaussianNoise:
        """Create isotropic (spherical) Gaussian.

        Same variance in all dimensions, zero mean. Useful for simple
        noise models where all states have similar uncertainty.

        Args:
            dim: Dimension of the noise
            variance: Scalar variance for all dimensions

        Returns:
            GaussianNoise with cov = variance * I

        Example:
            # 4D isotropic noise
            noise = GaussianNoise.isotropic(dim=4, variance=0.01)
        """
        if variance < 0:
            raise ValueError(f"variance must be non-negative, got {variance}")
        return cls(
            mean=jnp.zeros(dim),
            cov=jnp.eye(dim) * variance,
        )


class ScalarGaussianNoise(GaussianNoise):
    """Convenience class for 1D Gaussian noise.

    Simplifies the common case of scalar (1D) Gaussian noise.

    Example:
        noise = ScalarGaussianNoise(mean=0.0, variance=0.01)
        key = jax.random.key(0)
        key, subkey = jax.random.split(key)
        sample = noise.sample_scalar(subkey)  # Returns float
    """

    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        """Create scalar Gaussian noise.

        Args:
            mean: Scalar mean (default 0.0)
            variance: Scalar variance (default 1.0)

        Raises:
            ValueError: If variance <= 0
        """
        if variance <= 0:
            raise ValueError(f"variance must be positive, got {variance}")
        super().__init__(
            mean=jnp.array([mean]),
            cov=jnp.array([[variance]]),
        )

    def sample_scalar(self, key: jax.random.PRNGKey) -> float:
        """Sample a single scalar value.

        Convenience method that returns a Python float instead of
        a 1-element array.

        Args:
            key: JAX PRNG key

        Returns:
            Single scalar noise sample
        """
        return float(self.sample(key, (1,))[0])
