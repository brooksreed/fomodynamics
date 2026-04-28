"""Tests for fmd.simulator.noise package.

Tests cover:
- GaussianNoise: sampling, statistical properties, factories, validation
- ZeroNoise: always returns zeros
- ScalarGaussianNoise: 1D convenience class
- JAX compatibility: JIT, vmap, reproducibility
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator.noise import (
    NoiseModel,
    ZeroNoise,
    GaussianNoise,
    ScalarGaussianNoise,
)


class TestGaussianNoise:
    """Tests for GaussianNoise."""

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)
        key = jax.random.key(0)

        sample = noise.sample(key, (4,))
        assert sample.shape == (4,)

    def test_sample_shape_batched(self):
        """Test batched sampling."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)
        key = jax.random.key(0)

        samples = noise.sample(key, (10, 3))
        assert samples.shape == (10, 3)

    def test_mean_convergence(self):
        """Sample mean should converge to specified mean."""
        mean = jnp.array([1.0, -2.0, 0.5])
        cov = jnp.eye(3) * 0.1
        noise = GaussianNoise(mean=mean, cov=cov)

        # Generate many samples
        key = jax.random.key(42)
        keys = jax.random.split(key, 10000)
        samples = jax.vmap(lambda k: noise.sample(k, (3,)))(keys)

        sample_mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(sample_mean, mean, atol=0.05)

    def test_cov_convergence(self):
        """Sample covariance should converge to specified covariance."""
        mean = jnp.zeros(2)
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        noise = GaussianNoise(mean=mean, cov=cov)

        # Generate many samples
        key = jax.random.key(123)
        keys = jax.random.split(key, 10000)
        samples = jax.vmap(lambda k: noise.sample(k, (2,)))(keys)

        sample_cov = jnp.cov(samples.T)
        assert jnp.allclose(sample_cov, cov, atol=0.1)

    def test_different_keys_different_samples(self):
        """Different keys should produce different samples."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)

        key1 = jax.random.key(0)
        key2 = jax.random.key(1)

        sample1 = noise.sample(key1, (4,))
        sample2 = noise.sample(key2, (4,))

        assert not jnp.allclose(sample1, sample2)

    def test_same_key_reproducible(self):
        """Same key should produce identical sample."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)
        key = jax.random.key(42)

        sample1 = noise.sample(key, (4,))
        sample2 = noise.sample(key, (4,))

        assert jnp.allclose(sample1, sample2)

    def test_jit_compatible(self):
        """sample should be JIT-compilable."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)

        @jax.jit
        def sample_noise(key):
            return noise.sample(key, (3,))

        key = jax.random.key(0)
        sample = sample_noise(key)
        assert sample.shape == (3,)

    def test_vmap_over_keys(self):
        """sample should be vmap-able over batch of keys."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)
        key = jax.random.key(0)
        keys = jax.random.split(key, 100)

        samples = jax.vmap(lambda k: noise.sample(k, (4,)))(keys)
        assert samples.shape == (100, 4)

        # Samples should be different (verify randomness)
        assert not jnp.allclose(samples[0], samples[1])

    def test_dim_property(self):
        """dim property should return correct dimension."""
        noise = GaussianNoise.isotropic(dim=5, variance=1.0)
        assert noise.dim == 5

        noise2 = GaussianNoise(
            mean=jnp.zeros(3), cov=jnp.eye(3) * 0.1
        )
        assert noise2.dim == 3


class TestGaussianNoiseFactories:
    """Tests for GaussianNoise factory methods."""

    def test_from_diagonal(self):
        """from_diagonal creates correct diagonal covariance."""
        mean = jnp.array([1.0, 2.0, 3.0])
        variances = jnp.array([0.1, 0.2, 0.3])
        noise = GaussianNoise.from_diagonal(mean=mean, variances=variances)

        expected_cov = jnp.diag(variances)
        assert jnp.allclose(noise.mean, mean)
        assert jnp.allclose(noise.cov, expected_cov)
        assert noise.dim == 3

    def test_from_std(self):
        """from_std creates correct covariance from std."""
        mean = jnp.array([0.0, 0.0])
        std = jnp.array([0.1, 0.2])
        noise = GaussianNoise.from_std(mean=mean, std=std)

        expected_cov = jnp.diag(std**2)
        assert jnp.allclose(noise.cov, expected_cov)

    def test_isotropic(self):
        """isotropic creates spherical covariance."""
        noise = GaussianNoise.isotropic(dim=4, variance=0.5)

        expected_mean = jnp.zeros(4)
        expected_cov = jnp.eye(4) * 0.5

        assert jnp.allclose(noise.mean, expected_mean)
        assert jnp.allclose(noise.cov, expected_cov)
        assert noise.dim == 4

    def test_isotropic_zero_variance(self):
        """isotropic with zero variance should work (degenerate case)."""
        noise = GaussianNoise.isotropic(dim=3, variance=0.0)
        key = jax.random.key(0)

        # Samples should be near zero (note: 1e-10 regularization in Cholesky
        # means samples are sqrt(1e-10) * z ~ 1e-5 scale)
        sample = noise.sample(key, (3,))
        assert jnp.allclose(sample, jnp.zeros(3), atol=1e-4)


class TestGaussianNoiseValidation:
    """Tests for GaussianNoise validation."""

    def test_invalid_cov_shape_non_square(self):
        """Non-square cov should raise ValueError."""
        with pytest.raises(ValueError, match="must be square"):
            GaussianNoise(mean=jnp.zeros(3), cov=jnp.ones((3, 4)))

    def test_invalid_cov_shape_wrong_dim(self):
        """Cov with wrong dimension should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D"):
            GaussianNoise(mean=jnp.zeros(3), cov=jnp.ones(3))

    def test_invalid_mean_shape(self):
        """Non-1D mean should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1D"):
            GaussianNoise(mean=jnp.zeros((3, 1)), cov=jnp.eye(3))

    def test_asymmetric_cov(self):
        """Asymmetric cov should raise ValueError."""
        cov = jnp.array([[1.0, 0.5], [0.3, 1.0]])  # Not symmetric
        with pytest.raises(ValueError, match="must be symmetric"):
            GaussianNoise(mean=jnp.zeros(2), cov=cov)

    def test_non_psd_cov(self):
        """Non-PSD cov should raise ValueError."""
        # Negative definite matrix
        cov = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            GaussianNoise(mean=jnp.zeros(2), cov=cov)

    def test_mismatched_dims(self):
        """Mismatched mean/cov dims should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            GaussianNoise(mean=jnp.zeros(3), cov=jnp.eye(4))

    def test_negative_variance_from_diagonal(self):
        """Negative variance in from_diagonal should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            GaussianNoise.from_diagonal(
                mean=jnp.zeros(3), variances=jnp.array([0.1, -0.1, 0.1])
            )

    def test_negative_std_from_std(self):
        """Negative std in from_std should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            GaussianNoise.from_std(
                mean=jnp.zeros(2), std=jnp.array([0.1, -0.1])
            )

    def test_negative_variance_isotropic(self):
        """Negative variance in isotropic should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            GaussianNoise.isotropic(dim=3, variance=-0.1)


class TestZeroNoise:
    """Tests for ZeroNoise."""

    def test_returns_zeros(self):
        """ZeroNoise should always return zeros."""
        noise = ZeroNoise(dim=5)
        key = jax.random.key(123)

        sample = noise.sample(key, (5,))
        assert jnp.allclose(sample, jnp.zeros(5))

    def test_returns_zeros_batched(self):
        """ZeroNoise should return zeros for batched shapes."""
        noise = ZeroNoise(dim=3)
        key = jax.random.key(456)

        samples = noise.sample(key, (10, 3))
        assert jnp.allclose(samples, jnp.zeros((10, 3)))

    def test_invalid_dim_zero(self):
        """dim=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ZeroNoise(dim=0)

    def test_invalid_dim_negative(self):
        """dim<0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ZeroNoise(dim=-1)

    def test_dim_property(self):
        """dim property should return correct dimension."""
        noise = ZeroNoise(dim=7)
        assert noise.dim == 7

    def test_jit_compatible(self):
        """ZeroNoise should be JIT-compatible."""
        noise = ZeroNoise(dim=4)

        @jax.jit
        def sample_zero(key):
            return noise.sample(key, (4,))

        key = jax.random.key(0)
        sample = sample_zero(key)
        assert jnp.allclose(sample, jnp.zeros(4))


class TestScalarGaussianNoise:
    """Tests for ScalarGaussianNoise."""

    def test_basic_creation(self):
        """ScalarGaussianNoise should create 1D Gaussian."""
        noise = ScalarGaussianNoise(mean=1.0, variance=0.1)
        assert noise.dim == 1
        assert jnp.allclose(noise.mean, jnp.array([1.0]))
        assert jnp.allclose(noise.cov, jnp.array([[0.1]]))

    def test_default_values(self):
        """Default mean=0, variance=1."""
        noise = ScalarGaussianNoise()
        assert jnp.allclose(noise.mean, jnp.array([0.0]))
        assert jnp.allclose(noise.cov, jnp.array([[1.0]]))

    def test_sample_scalar(self):
        """sample_scalar should return a float."""
        noise = ScalarGaussianNoise(mean=0.0, variance=1.0)
        key = jax.random.key(0)

        sample = noise.sample_scalar(key)
        assert isinstance(sample, float)

    def test_sample_returns_array(self):
        """Regular sample should return array."""
        noise = ScalarGaussianNoise(mean=0.0, variance=1.0)
        key = jax.random.key(0)

        sample = noise.sample(key, (1,))
        assert sample.shape == (1,)

    def test_invalid_variance_zero(self):
        """variance=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ScalarGaussianNoise(variance=0.0)

    def test_invalid_variance_negative(self):
        """variance<0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ScalarGaussianNoise(variance=-1.0)

    def test_mean_convergence(self):
        """Scalar samples should have correct mean."""
        noise = ScalarGaussianNoise(mean=5.0, variance=0.1)
        key = jax.random.key(42)
        keys = jax.random.split(key, 10000)

        samples = jnp.array(
            [noise.sample_scalar(k) for k in keys]
        )
        assert jnp.abs(jnp.mean(samples) - 5.0) < 0.05


class TestSampleTrajectory:
    """Tests for sample_trajectory method."""

    def test_shape(self):
        """Trajectory should have shape (num_steps, dim)."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)
        key = jax.random.key(0)

        trajectory = noise.sample_trajectory(key, num_steps=100)
        assert trajectory.shape == (100, 4)

    def test_samples_independent(self):
        """Trajectory samples should be independent."""
        noise = GaussianNoise.isotropic(dim=2, variance=1.0)
        key = jax.random.key(42)

        trajectory = noise.sample_trajectory(key, num_steps=1000)

        # Check autocorrelation is near zero
        # (samples at different times should be uncorrelated)
        corr = jnp.corrcoef(trajectory[:-1, 0], trajectory[1:, 0])[0, 1]
        assert jnp.abs(corr) < 0.1  # Should be near zero

    def test_reproducible(self):
        """Same key should produce same trajectory."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)
        key = jax.random.key(123)

        traj1 = noise.sample_trajectory(key, num_steps=50)
        traj2 = noise.sample_trajectory(key, num_steps=50)

        assert jnp.allclose(traj1, traj2)

    def test_different_keys_different_trajectories(self):
        """Different keys should produce different trajectories."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)

        traj1 = noise.sample_trajectory(jax.random.key(0), num_steps=50)
        traj2 = noise.sample_trajectory(jax.random.key(1), num_steps=50)

        assert not jnp.allclose(traj1, traj2)

    def test_zero_noise_trajectory(self):
        """ZeroNoise trajectory should be all zeros."""
        noise = ZeroNoise(dim=4)
        key = jax.random.key(0)

        trajectory = noise.sample_trajectory(key, num_steps=100)
        assert jnp.allclose(trajectory, jnp.zeros((100, 4)))

    def test_jit_compatible(self):
        """sample_trajectory should be JIT-compatible."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)

        @jax.jit
        def gen_trajectory(key):
            return noise.sample_trajectory(key, num_steps=50)

        key = jax.random.key(0)
        trajectory = gen_trajectory(key)
        assert trajectory.shape == (50, 3)


class TestBatchedSampling:
    """Tests for batched sampling behavior."""

    def test_batched_shape(self):
        """Batched shape (n, dim) should work correctly."""
        noise = GaussianNoise.isotropic(dim=4, variance=1.0)
        key = jax.random.key(0)

        samples = noise.sample(key, (100, 4))
        assert samples.shape == (100, 4)

    def test_batched_mean(self):
        """Batched samples should have correct mean."""
        mean = jnp.array([1.0, -1.0])
        noise = GaussianNoise(mean=mean, cov=jnp.eye(2) * 0.01)
        key = jax.random.key(42)

        samples = noise.sample(key, (10000, 2))
        sample_mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(sample_mean, mean, atol=0.05)

    def test_batched_vs_vmap(self):
        """Batched sampling should give different samples per row."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)
        key = jax.random.key(0)

        samples = noise.sample(key, (100, 3))

        # Each row should be different (same key, but different samples)
        # This is because batched sampling generates all at once
        assert samples.shape == (100, 3)
        # Variance should be non-zero
        assert jnp.std(samples) > 0.5


class TestNoiseModelABC:
    """Tests for NoiseModel abstract base class."""

    def test_is_equinox_module(self):
        """NoiseModel subclasses should be Equinox modules."""
        import equinox as eqx

        noise = GaussianNoise.isotropic(dim=3, variance=1.0)
        assert isinstance(noise, eqx.Module)

        zero = ZeroNoise(dim=3)
        assert isinstance(zero, eqx.Module)

    def test_gaussian_is_noise_model(self):
        """GaussianNoise should be a NoiseModel."""
        noise = GaussianNoise.isotropic(dim=3, variance=1.0)
        assert isinstance(noise, NoiseModel)

    def test_zero_is_noise_model(self):
        """ZeroNoise should be a NoiseModel."""
        noise = ZeroNoise(dim=3)
        assert isinstance(noise, NoiseModel)

    def test_scalar_is_noise_model(self):
        """ScalarGaussianNoise should be a NoiseModel."""
        noise = ScalarGaussianNoise(mean=0.0, variance=1.0)
        assert isinstance(noise, NoiseModel)


class TestCholesky:
    """Tests for Cholesky decomposition in GaussianNoise."""

    def test_cholesky_stored(self):
        """Cholesky decomposition should be stored."""
        cov = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        noise = GaussianNoise(mean=jnp.zeros(2), cov=cov)

        # Cholesky should be lower triangular
        assert jnp.allclose(noise._chol, jnp.tril(noise._chol))

        # L @ L^T should equal cov (approximately, due to regularization)
        reconstructed = noise._chol @ noise._chol.T
        assert jnp.allclose(reconstructed, cov, atol=1e-8)

    def test_diagonal_cholesky(self):
        """Diagonal cov should have diagonal Cholesky."""
        variances = jnp.array([1.0, 4.0, 9.0])
        noise = GaussianNoise.from_diagonal(mean=jnp.zeros(3), variances=variances)

        expected_chol = jnp.diag(jnp.sqrt(variances))
        assert jnp.allclose(noise._chol, expected_chol, atol=1e-8)
