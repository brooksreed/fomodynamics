"""Pytest fixtures for constraint tests."""

import pytest
import jax.numpy as jnp

# Ensure float64 is enabled
from fmd.simulator import _config  # noqa: F401


@pytest.fixture
def simple_state():
    """13-element state vector (RigidBody6DOF layout).

    Layout:
        [0-2]: pos_n, pos_e, pos_d (NED position)
        [3-5]: vel_u, vel_v, vel_w (body velocity)
        [6-9]: qw, qx, qy, qz (quaternion, identity)
        [10-12]: omega_p, omega_q, omega_r (angular velocity)
    """
    state = jnp.zeros(13)
    state = state.at[6].set(1.0)  # Identity quaternion qw=1
    return state


@pytest.fixture
def simple_control():
    """4-element control vector (e.g., quadrotor: thrust + 3 torques)."""
    return jnp.zeros(4)


@pytest.fixture
def state_3d():
    """Simple 3D state for physical constraint tests."""
    return jnp.array([0.0, 0.0, 0.0])


@pytest.fixture
def control_2d():
    """Simple 2D control for bound constraint tests."""
    return jnp.array([0.0, 0.0])


@pytest.fixture
def planar_state():
    """6-element planar vehicle state.

    Layout:
        [0-1]: x, y (position)
        [2]: psi (heading)
        [3-4]: vx, vy (velocity)
        [5]: omega (angular velocity)
    """
    return jnp.zeros(6)


@pytest.fixture
def planar_control():
    """2-element planar control (e.g., thrust, steering)."""
    return jnp.zeros(2)
