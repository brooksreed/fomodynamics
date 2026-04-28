"""Pytest configuration for JAX simulator tests.

This module:
1. Uses centralized JAX configuration from fmd.core.jax_config
2. Provides fixtures for common test setup
3. Skips tests if JAX is not installed

Usage:
    Tests in this directory automatically get float64 enabled.
    Use the jax_available marker to skip tests when JAX is missing.
"""

import pytest
import numpy as np

# Try to import JAX, skip all tests in this directory if unavailable
try:
    # Use centralized config (enables float64, configures GPU)
    from fmd.core.jax_config import configure_jax  # noqa: F401

    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


def pytest_collection_modifyitems(config, items):
    """Skip JAX tests if JAX is not available."""
    if HAS_JAX:
        return

    skip_jax = pytest.mark.skip(reason="JAX not installed")
    for item in items:
        # Skip all tests in the jax test directory
        if "simulator/jax" in str(item.fspath):
            item.add_marker(skip_jax)


@pytest.fixture(autouse=True)
def verify_float64():
    """Verify float64 is enabled for all tests.

    This fixture runs automatically before each test to ensure
    JAX is configured correctly. Fails fast if float64 is not enabled.
    """
    if not HAS_JAX:
        pytest.skip("JAX not installed")

    arr = jnp.array(1.0)
    assert arr.dtype == jnp.float64, (
        f"float64 not enabled! Got {arr.dtype}. "
        "Ensure fmd.core.jax_config is imported before any jax.numpy operations."
    )


# Tolerance constants for tests
# For float64 with structurally identical formulas:
DERIV_RTOL = 1e-12  # Derivative computation (pure math)
DERIV_ATOL = 1e-14

# For trajectory comparison (accumulates numerical differences):
TRAJ_RTOL = 1e-10  # Very tight for fixed-step RK4
TRAJ_ATOL = 1e-12

# For analytical property verification:
ANALYTICAL_RTOL = 1e-6  # Period, steady-state, etc.


@pytest.fixture
def identity_state():
    """Create a rigid body state with identity quaternion.

    Returns 13-element state: [pos(3), vel(3), quat(4), omega(3)]
    """
    if not HAS_JAX:
        pytest.skip("JAX not installed")

    state = jnp.zeros(13)
    state = state.at[6].set(1.0)  # qw = 1 (identity quaternion)
    return state


@pytest.fixture
def random_unit_quaternion():
    """Generate a random unit quaternion for testing."""
    if not HAS_JAX:
        pytest.skip("JAX not installed")

    # Use numpy for random generation (JAX random requires key management)
    rng = np.random.default_rng(42)
    q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    return jnp.array(q)
