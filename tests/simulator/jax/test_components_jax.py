"""Tests for JAX force and moment components.

These tests verify that the JAX components:
1. Produce correct physics (gravity direction, magnitude)
2. Match golden values for numerical regression
3. Work with JAX JIT compilation
4. Work with JAX autodiff
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator.components import JaxForceElement, JaxGravity
from fmd.simulator import euler_to_quat, identity_quat
from fmd.simulator.params.base import STANDARD_GRAVITY

from .conftest import DERIV_RTOL, DERIV_ATOL
from .golden_values import (
    GRAVITY_FORCE_LEVEL,
    GRAVITY_MOMENT_LEVEL,
    GRAVITY_FORCE_PITCHED_45,
    GRAVITY_MOMENT_PITCHED_45,
    GRAVITY_FORCE_ROLLED_90,
    GRAVITY_MOMENT_ROLLED_90,
)


def create_rigid_body_state(
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    quaternion=None,
    omega=(0.0, 0.0, 0.0),
):
    """Create a 13-element rigid body state vector.

    Args:
        position: (N, E, D) position in NED frame
        velocity: (u, v, w) body velocity
        quaternion: [qw, qx, qy, qz] or None for identity
        omega: (p, q, r) angular velocity in body frame

    Returns:
        JAX array of shape (13,)
    """
    if quaternion is None:
        quaternion = [1.0, 0.0, 0.0, 0.0]

    state = jnp.array([
        position[0], position[1], position[2],  # pos_n, pos_e, pos_d
        velocity[0], velocity[1], velocity[2],  # vel_u, vel_v, vel_w
        quaternion[0], quaternion[1], quaternion[2], quaternion[3],  # quat
        omega[0], omega[1], omega[2],  # omega_p, omega_q, omega_r
    ])
    return state


class TestJaxForceElementInterface:
    """Tests for the JaxForceElement base class interface."""

    def test_zero_force_moment(self):
        """zero_force_moment returns zero vectors."""
        force, moment = JaxForceElement.zero_force_moment()

        np.testing.assert_allclose(force, jnp.zeros(3), atol=1e-15)
        np.testing.assert_allclose(moment, jnp.zeros(3), atol=1e-15)

    def test_zero_force_moment_shapes(self):
        """zero_force_moment returns correct shapes."""
        force, moment = JaxForceElement.zero_force_moment()

        assert force.shape == (3,)
        assert moment.shape == (3,)


class TestJaxGravityBasics:
    """Basic tests for JaxGravity component."""

    def test_default_gravity_value(self):
        """Default g value is STANDARD_GRAVITY."""
        gravity = JaxGravity(mass=1.0)
        assert gravity.g == STANDARD_GRAVITY

    def test_custom_gravity_value(self):
        """Custom g value can be specified."""
        gravity = JaxGravity(mass=1.0, g=10.0)
        assert gravity.g == 10.0

    def test_mass_stored(self):
        """Mass is stored correctly."""
        gravity = JaxGravity(mass=5.0)
        assert gravity.mass == 5.0

    def test_output_shapes(self):
        """compute() returns correct shapes."""
        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state()
        control = jnp.array([])

        force, moment = gravity.compute(state, control)

        assert force.shape == (3,)
        assert moment.shape == (3,)

    def test_no_moment(self):
        """Gravity produces no moment about CoM."""
        gravity = JaxGravity(mass=10.0)
        state = create_rigid_body_state()
        control = jnp.array([])

        _, moment = gravity.compute(state, control)

        np.testing.assert_allclose(moment, jnp.zeros(3), atol=1e-15)


class TestJaxGravityPhysics:
    """Physics correctness tests for JaxGravity."""

    def test_gravity_magnitude_identity_orientation(self):
        """Gravity magnitude equals m*g with identity quaternion."""
        mass = 5.0
        g = 9.80665
        gravity = JaxGravity(mass=mass, g=g)
        state = create_rigid_body_state()  # Identity quaternion
        control = jnp.array([])

        force, _ = gravity.compute(state, control)

        expected_magnitude = mass * g
        actual_magnitude = jnp.linalg.norm(force)
        np.testing.assert_allclose(actual_magnitude, expected_magnitude, rtol=DERIV_RTOL)

    def test_gravity_direction_identity_orientation(self):
        """Gravity points +Z in body frame with identity quaternion.

        With identity quaternion, body frame = NED frame.
        Gravity should point +D (down), which is +Z.
        """
        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state()  # Identity quaternion
        control = jnp.array([])

        force, _ = gravity.compute(state, control)

        # Should be [0, 0, +g]
        assert force[0] == pytest.approx(0.0, abs=1e-14)
        assert force[1] == pytest.approx(0.0, abs=1e-14)
        assert force[2] > 0  # Points down (+D in NED)

    def test_gravity_pitched_up(self):
        """When pitched up, gravity has -X component in body frame.

        If vehicle is pitched up (nose up), gravity in body frame
        should have a component toward the tail (-X body direction).
        A negative X component means the force points toward the tail.
        """
        pitch = jnp.pi / 4  # 45 degrees nose up
        euler = jnp.array([0.0, pitch, 0.0])
        q = euler_to_quat(euler)

        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state(quaternion=q)
        control = jnp.array([])

        force, _ = gravity.compute(state, control)

        # Pitched up: gravity has -X component (toward tail direction)
        assert force[0] < 0, "Gravity should have -X component (toward tail) when pitched up"
        # Still has +Z component (not fully inverted)
        assert force[2] > 0, "Gravity should still have +Z component at 45 deg pitch"

    def test_gravity_rolled_right(self):
        """When rolled right, gravity has +Y component in body frame.

        If right wing is down, gravity in body frame has component
        toward right wing (+Y body).
        """
        roll = jnp.pi / 4  # 45 degrees right wing down
        euler = jnp.array([roll, 0.0, 0.0])
        q = euler_to_quat(euler)

        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state(quaternion=q)
        control = jnp.array([])

        force, _ = gravity.compute(state, control)

        # Rolled right: gravity pulls toward right wing (+Y body)
        assert force[1] > 0, "Gravity should have +Y component when rolled right"

    def test_gravity_inverted(self):
        """When inverted (180 deg roll), gravity points -Z in body frame."""
        roll = jnp.pi  # Inverted
        euler = jnp.array([roll, 0.0, 0.0])
        q = euler_to_quat(euler)

        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state(quaternion=q)
        control = jnp.array([])

        force, _ = gravity.compute(state, control)

        # Inverted: gravity points up in body frame (-Z)
        assert force[2] < 0, "Gravity should point -Z when inverted"
        np.testing.assert_allclose(force[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(force[1], 0.0, atol=1e-10)


class TestJaxGravityGoldenValues:
    """Tests comparing JaxGravity to golden values for numerical regression."""

    def test_matches_golden_level_attitude(self):
        """JaxGravity matches golden value at level attitude (mass=1.0)."""
        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state()  # Identity quaternion

        force, moment = gravity.compute(state, jnp.array([]))

        np.testing.assert_allclose(force, GRAVITY_FORCE_LEVEL, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        np.testing.assert_allclose(moment, GRAVITY_MOMENT_LEVEL, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_golden_pitched_45(self):
        """JaxGravity matches golden value at 45 degree pitch (mass=1.0)."""
        gravity = JaxGravity(mass=1.0)
        pitch = np.radians(45)
        euler = jnp.array([0.0, pitch, 0.0])
        q = euler_to_quat(euler)
        state = create_rigid_body_state(quaternion=q)

        force, moment = gravity.compute(state, jnp.array([]))

        np.testing.assert_allclose(force, GRAVITY_FORCE_PITCHED_45, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        np.testing.assert_allclose(moment, GRAVITY_MOMENT_PITCHED_45, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_golden_rolled_90(self):
        """JaxGravity matches golden value at 90 degree roll (mass=1.0)."""
        gravity = JaxGravity(mass=1.0)
        roll = np.radians(90)
        euler = jnp.array([roll, 0.0, 0.0])
        q = euler_to_quat(euler)
        state = create_rigid_body_state(quaternion=q)

        force, moment = gravity.compute(state, jnp.array([]))

        np.testing.assert_allclose(force, GRAVITY_FORCE_ROLLED_90, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        np.testing.assert_allclose(moment, GRAVITY_MOMENT_ROLLED_90, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestJaxGravityJIT:
    """Tests for JIT compilation of JaxGravity."""

    def test_jit_compiles(self):
        """JaxGravity.compute can be JIT compiled."""
        gravity = JaxGravity(mass=1.0)
        state = create_rigid_body_state()
        control = jnp.array([])

        @jax.jit
        def compute_force(grav, s, c):
            return grav.compute(s, c)

        force, moment = compute_force(gravity, state, control)

        # Should produce valid results
        assert jnp.all(jnp.isfinite(force))
        assert jnp.all(jnp.isfinite(moment))

    def test_jit_matches_eager(self):
        """JIT result matches eager execution."""
        gravity = JaxGravity(mass=2.5)
        euler = jnp.array([0.1, 0.2, 0.3])
        q = euler_to_quat(euler)
        state = create_rigid_body_state(quaternion=q)
        control = jnp.array([])

        # Eager execution
        force_eager, moment_eager = gravity.compute(state, control)

        # JIT execution
        @jax.jit
        def compute_force(grav, s, c):
            return grav.compute(s, c)

        force_jit, moment_jit = compute_force(gravity, state, control)

        np.testing.assert_allclose(force_jit, force_eager, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        np.testing.assert_allclose(moment_jit, moment_eager, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestJaxGravityGrad:
    """Tests for autodiff through JaxGravity."""

    def test_grad_wrt_state(self):
        """Can compute gradient w.r.t. state."""
        gravity = JaxGravity(mass=1.0)

        def loss(state):
            force, _ = gravity.compute(state, jnp.array([]))
            return jnp.sum(force ** 2)

        state = create_rigid_body_state()
        grad = jax.grad(loss)(state)

        # Gradient should exist and be finite
        assert grad.shape == state.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_quaternion_orientation(self):
        """Gradient through orientation-dependent gravity."""
        gravity = JaxGravity(mass=1.0)

        def loss(euler_angles):
            q = euler_to_quat(euler_angles)
            state = create_rigid_body_state(quaternion=q)
            force, _ = gravity.compute(state, jnp.array([]))
            # Loss: X component of force (varies with pitch)
            return force[0]

        euler = jnp.array([0.0, 0.3, 0.0])  # Some pitch
        grad = jax.grad(loss)(euler)

        # Gradient should exist and be finite
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

        # Pitch gradient should be non-zero (changing pitch changes force[0])
        assert grad[1] != 0.0, "Pitch gradient should be non-zero"

    def test_grad_wrt_mass(self):
        """Can compute gradient w.r.t. mass parameter."""

        def loss(mass):
            gravity = JaxGravity(mass=mass)
            state = create_rigid_body_state()
            force, _ = gravity.compute(state, jnp.array([]))
            return jnp.sum(force ** 2)

        mass = 2.0
        grad = jax.grad(loss)(mass)

        # Gradient should be finite and positive (more mass = larger force = larger loss)
        assert jnp.isfinite(grad)
        assert grad > 0, "Gradient w.r.t. mass should be positive"
