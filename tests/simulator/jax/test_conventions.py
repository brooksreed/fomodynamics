"""Tests for coordinate frame and quaternion conventions.

These tests verify that the JAX implementation follows the same
conventions as the numpy implementation:
- NED coordinate frame (North-East-Down)
- Hamilton quaternion convention, scalar-first [qw, qx, qy, qz]
- rotate_vector transforms body -> NED
- rotate_vector_inverse transforms NED -> body
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import (
    quat_to_dcm,
    euler_to_quat,
    rotate_vector,
    rotate_vector_inverse,
    identity_quat,
)

from .conftest import DERIV_RTOL, DERIV_ATOL


class TestIdentityQuaternion:
    """Tests for identity quaternion convention."""

    def test_identity_values(self):
        """Identity quaternion is [1, 0, 0, 0] (scalar-first)."""
        q = identity_quat()
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-15)

    def test_identity_produces_identity_dcm(self):
        """Identity quaternion produces identity rotation matrix."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        dcm = quat_to_dcm(q)
        np.testing.assert_allclose(dcm, jnp.eye(3), atol=1e-15)


class TestYawRotation:
    """Tests for yaw rotation (rotation about Down axis in NED)."""

    def test_90_degree_yaw_rotates_north_to_east(self):
        """90 degree yaw rotation moves +X body to +Y NED (+N -> +E).

        In NED:
        - X axis = North
        - Y axis = East
        - Z axis = Down

        A 90 degree yaw (rotation about Z/Down axis) should:
        - Rotate a vector pointing North (+X body) to point East (+Y NED)
        """
        # Quaternion for 90 degree yaw
        yaw = jnp.pi / 2
        euler = jnp.array([0.0, 0.0, yaw])  # [roll, pitch, yaw]
        q = euler_to_quat(euler)

        # +X in body frame (pointing "forward" in body)
        v_body = jnp.array([1.0, 0.0, 0.0])

        # Rotate body -> NED
        v_ned = rotate_vector(q, v_body)

        # Should now point East (+Y in NED)
        expected = jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v_ned, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_180_degree_yaw_reverses_north(self):
        """180 degree yaw rotation reverses North direction."""
        yaw = jnp.pi
        euler = jnp.array([0.0, 0.0, yaw])
        q = euler_to_quat(euler)

        v_body = jnp.array([1.0, 0.0, 0.0])
        v_ned = rotate_vector(q, v_body)

        # Should point South (-X in NED)
        expected = jnp.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(v_ned, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_negative_yaw(self):
        """Negative yaw rotates in opposite direction."""
        yaw = -jnp.pi / 2
        euler = jnp.array([0.0, 0.0, yaw])
        q = euler_to_quat(euler)

        v_body = jnp.array([1.0, 0.0, 0.0])
        v_ned = rotate_vector(q, v_body)

        # Should point West (-Y in NED)
        expected = jnp.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(v_ned, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestPitchRotation:
    """Tests for pitch rotation (rotation about East axis in NED)."""

    def test_positive_pitch_nose_up(self):
        """Positive pitch raises the nose (body +X points up in NED).

        Positive pitch = rotation about +Y (East) axis.
        Body +X should tilt toward -Z (up in NED, since +Z is Down).
        """
        pitch = jnp.pi / 4  # 45 degrees nose up
        euler = jnp.array([0.0, pitch, 0.0])
        q = euler_to_quat(euler)

        # Body +X (forward)
        v_body = jnp.array([1.0, 0.0, 0.0])
        v_ned = rotate_vector(q, v_body)

        # Should have positive N component, negative D component (pointing up)
        assert v_ned[0] > 0, "Should still have some North component"
        assert v_ned[2] < 0, "Nose up means negative D (up in NED)"


class TestRollRotation:
    """Tests for roll rotation (rotation about North axis in NED)."""

    def test_positive_roll_right_wing_down(self):
        """Positive roll tilts right wing down.

        Positive roll = rotation about +X (North) axis.
        Body +Y (right wing) should tilt toward +Z (Down in NED).
        """
        roll = jnp.pi / 4  # 45 degrees right wing down
        euler = jnp.array([roll, 0.0, 0.0])
        q = euler_to_quat(euler)

        # Body +Y (right wing)
        v_body = jnp.array([0.0, 1.0, 0.0])
        v_ned = rotate_vector(q, v_body)

        # Right wing should have positive Z (down) component
        assert v_ned[2] > 0, "Right wing down means positive D (down in NED)"


class TestNEDGravityDirection:
    """Tests for gravity direction in NED frame."""

    def test_gravity_points_positive_d(self):
        """In NED, gravity points in +D direction (down).

        With identity quaternion (body = NED), a gravity vector
        of [0, 0, +g] in NED should transform to [0, 0, +g] in body.
        """
        q = identity_quat()

        # Gravity in NED frame: points down (+D)
        gravity_ned = jnp.array([0.0, 0.0, 9.80665])

        # Transform to body frame (should be unchanged with identity quat)
        gravity_body = rotate_vector_inverse(q, gravity_ned)

        expected = jnp.array([0.0, 0.0, 9.80665])
        np.testing.assert_allclose(gravity_body, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_pitched_up_gravity_in_body(self):
        """When pitched up, gravity has component along body -X.

        If vehicle is pitched up (nose up), gravity in body frame
        should have a component toward the tail (-X body direction).
        A negative X component means the force points toward the tail.
        """
        pitch = jnp.pi / 4  # 45 degrees nose up
        euler = jnp.array([0.0, pitch, 0.0])
        q = euler_to_quat(euler)

        gravity_ned = jnp.array([0.0, 0.0, 9.80665])
        gravity_body = rotate_vector_inverse(q, gravity_ned)

        # Pitched up: gravity has component toward tail (-X body direction)
        # Negative X component = force pointing in -X direction = toward tail
        assert gravity_body[0] < 0, "Gravity should have -X component (toward tail) when pitched up"


class TestRotateVectorInverse:
    """Tests for rotate_vector_inverse convention."""

    def test_inverse_reverses_rotation(self):
        """rotate_vector_inverse reverses rotate_vector."""
        yaw = jnp.pi / 3  # Arbitrary rotation
        euler = jnp.array([0.1, 0.2, yaw])
        q = euler_to_quat(euler)

        v_original = jnp.array([1.0, 2.0, 3.0])

        # Rotate body -> NED
        v_ned = rotate_vector(q, v_original)

        # Rotate NED -> body (should recover original)
        v_recovered = rotate_vector_inverse(q, v_ned)

        np.testing.assert_allclose(v_recovered, v_original, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_inverse_is_transpose(self):
        """rotate_vector_inverse uses R.T, not R."""
        euler = jnp.array([0.1, 0.2, 0.3])
        q = euler_to_quat(euler)

        v = jnp.array([1.0, 2.0, 3.0])

        # Direct computation using DCM
        dcm = quat_to_dcm(q)

        result_rotate = rotate_vector(q, v)
        result_inverse = rotate_vector_inverse(q, v)

        np.testing.assert_allclose(result_rotate, dcm @ v, rtol=DERIV_RTOL, atol=DERIV_ATOL)
        np.testing.assert_allclose(result_inverse, dcm.T @ v, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestDCMProperties:
    """Tests for DCM (rotation matrix) properties."""

    def test_dcm_columns_are_body_axes_in_ned(self):
        """DCM columns are body axes expressed in NED frame.

        DCM[i, j] = projection of body axis j onto NED axis i.
        So DCM[:, 0] is body X axis in NED coordinates.
        """
        yaw = jnp.pi / 2  # 90 degree yaw
        euler = jnp.array([0.0, 0.0, yaw])
        q = euler_to_quat(euler)

        dcm = quat_to_dcm(q)

        # Body X axis (forward) should now point East in NED
        body_x_in_ned = dcm[:, 0]
        expected = jnp.array([0.0, 1.0, 0.0])  # East
        np.testing.assert_allclose(body_x_in_ned, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_dcm_rows_are_ned_axes_in_body(self):
        """DCM rows are NED axes expressed in body frame.

        DCM[i, j] = projection of body axis j onto NED axis i.
        So DCM[0, :] = DCM.T[:, 0] tells us where NED North is in body.
        """
        yaw = jnp.pi / 2  # 90 degree yaw
        euler = jnp.array([0.0, 0.0, yaw])
        q = euler_to_quat(euler)

        dcm = quat_to_dcm(q)

        # NED North axis should be body -Y after 90 deg yaw
        ned_north_in_body = dcm.T[:, 0]  # = dcm[0, :]
        expected = jnp.array([0.0, -1.0, 0.0])  # -Y body
        np.testing.assert_allclose(ned_north_in_body, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)
