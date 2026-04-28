"""Tests for quaternion math utilities."""

import numpy as np
import pytest

from fmd.core import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_derivative,
    quat_to_dcm,
    dcm_to_quat,
    quat_to_euler,
    euler_to_quat,
    rotate_vector,
    rotate_vector_inverse,
    identity_quat,
    quaternion_distance,
)


class TestQuaternionBasics:
    """Test basic quaternion operations."""

    def test_identity_quaternion(self):
        """Identity quaternion should be [1, 0, 0, 0]."""
        q = identity_quat()
        assert np.allclose(q, [1, 0, 0, 0])

    def test_normalize(self):
        """Normalize should produce unit quaternion."""
        q = np.array([1, 1, 1, 1])
        q_norm = quat_normalize(q)
        assert np.isclose(np.linalg.norm(q_norm), 1.0)

    def test_normalize_zero_quaternion(self):
        """Zero quaternion returns [0,0,0,0] (clamp-and-divide contract)."""
        q = np.array([0, 0, 0, 0])
        q_norm = quat_normalize(q)
        assert np.allclose(q_norm, [0, 0, 0, 0])

    def test_conjugate(self):
        """Conjugate should negate vector part."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = quat_conjugate(q)
        assert np.allclose(q_conj, [0.5, -0.5, -0.5, -0.5])

    def test_multiply_identity(self):
        """Multiplying by identity should give same quaternion."""
        q = quat_normalize([1, 2, 3, 4])
        identity = identity_quat()
        result = quat_multiply(identity, q)
        assert np.allclose(result, q)

    def test_multiply_inverse(self):
        """q * q_conj should give identity (for unit quaternion)."""
        q = quat_normalize([1, 2, 3, 4])
        q_conj = quat_conjugate(q)
        result = quat_multiply(q, q_conj)
        assert np.allclose(result, identity_quat())


class TestQuaternionRotations:
    """Test quaternion-based rotations."""

    def test_dcm_identity(self):
        """Identity quaternion should give identity DCM."""
        q = identity_quat()
        dcm = quat_to_dcm(q)
        assert np.allclose(dcm, np.eye(3))

    def test_dcm_to_quat_roundtrip(self):
        """Converting DCM to quaternion and back should be consistent."""
        q_original = quat_normalize([1, 2, 3, 4])
        dcm = quat_to_dcm(q_original)
        q_recovered = dcm_to_quat(dcm)
        # Quaternion can have opposite sign and represent same rotation
        assert np.allclose(q_original, q_recovered) or np.allclose(q_original, -q_recovered)

    def test_rotate_vector_identity(self):
        """Identity rotation should not change vector."""
        q = identity_quat()
        v = np.array([1, 2, 3])
        v_rotated = rotate_vector(q, v)
        assert np.allclose(v_rotated, v)

    def test_rotate_vector_inverse(self):
        """Rotate and inverse rotate should give original."""
        q = quat_normalize([1, 2, 3, 4])
        v = np.array([1, 2, 3])
        v_rotated = rotate_vector(q, v)
        v_recovered = rotate_vector_inverse(q, v_rotated)
        assert np.allclose(v_recovered, v)

    def test_90_degree_rotation_about_z(self):
        """90-degree rotation about Z should swap X and Y."""
        # 90 degrees about Z: quat = [cos(45°), 0, 0, sin(45°)]
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

        # With the repo's NED conventions, +90 deg yaw maps +X (North) to +Y (East).
        v = np.array([1, 0, 0])
        v_rotated = rotate_vector(q, v)
        assert np.allclose(v_rotated, [0, 1, 0], atol=1e-10)


class TestEulerAngles:
    """Test Euler angle conversions."""

    def test_euler_identity(self):
        """Zero Euler angles should give identity quaternion."""
        euler = np.array([0, 0, 0])
        q = euler_to_quat(euler)
        assert np.allclose(q, identity_quat())

    def test_euler_roundtrip(self):
        """Converting to quaternion and back should give same angles."""
        euler_original = np.array([0.1, 0.2, 0.3])
        q = euler_to_quat(euler_original)
        euler_recovered = quat_to_euler(q)
        assert np.allclose(euler_original, euler_recovered)

    def test_pure_roll(self):
        """Pure roll rotation."""
        roll = np.pi / 4  # 45 degrees
        euler = np.array([roll, 0, 0])
        q = euler_to_quat(euler)
        euler_back = quat_to_euler(q)
        assert np.allclose(euler, euler_back)

    def test_pure_yaw(self):
        """Pure yaw rotation."""
        yaw = np.pi / 3  # 60 degrees
        euler = np.array([0, 0, yaw])
        q = euler_to_quat(euler)
        euler_back = quat_to_euler(q)
        assert np.allclose(euler, euler_back)

    def test_gimbal_lock_positive(self):
        """Gimbal lock at pitch = +pi/2 should not produce NaN."""
        q = euler_to_quat(np.array([0.0, np.pi / 2, 0.0]))
        euler = quat_to_euler(q)
        assert not np.any(np.isnan(euler))
        assert np.isclose(euler[1], np.pi / 2, atol=1e-10)

    def test_gimbal_lock_negative(self):
        """Gimbal lock at pitch = -pi/2 should not produce NaN."""
        q = euler_to_quat(np.array([0.0, -np.pi / 2, 0.0]))
        euler = quat_to_euler(q)
        assert not np.any(np.isnan(euler))
        assert np.isclose(euler[1], -np.pi / 2, atol=1e-10)


class TestQuaternionDerivative:
    """Test quaternion derivative computation."""

    def test_zero_angular_velocity(self):
        """Zero angular velocity should give zero derivative."""
        q = quat_normalize([1, 2, 3, 4])
        omega = np.zeros(3)
        q_dot = quat_derivative(q, omega)
        assert np.allclose(q_dot, np.zeros(4))

    def test_derivative_preserves_norm(self):
        """Quaternion derivative should be orthogonal to quaternion."""
        q = quat_normalize([1, 2, 3, 4])
        omega = np.array([0.1, 0.2, 0.3])
        q_dot = quat_derivative(q, omega)
        # q_dot should be orthogonal to q for unit quaternion
        assert np.isclose(np.dot(q, q_dot), 0, atol=1e-10)


class TestQuaternionDistance:
    """Test quaternion geodesic distance computation."""

    def test_identity_distance(self):
        """Distance between same quaternion should be zero."""
        q = identity_quat()
        assert np.isclose(quaternion_distance(q, q), 0.0)

    def test_identity_distance_arbitrary(self):
        """Distance between same arbitrary quaternion should be zero."""
        q = quat_normalize([1, 2, 3, 4])
        assert np.isclose(quaternion_distance(q, q), 0.0, atol=1e-7)

    def test_antipodal_distance(self):
        """Distance between q and -q should be zero (same rotation)."""
        q = identity_quat()
        assert np.isclose(quaternion_distance(q, -q), 0.0, atol=1e-7)

    def test_antipodal_distance_arbitrary(self):
        """Distance between arbitrary q and -q should be zero."""
        q = quat_normalize([1, 2, 3, 4])
        assert np.isclose(quaternion_distance(q, -q), 0.0, atol=1e-7)

    def test_90_degree_rotation(self):
        """90-degree rotation should give distance of pi/2."""
        q_identity = identity_quat()
        # 90 degrees about z-axis: quat = [cos(45°), 0, 0, sin(45°)]
        angle = np.pi / 2
        q_90z = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        dist = quaternion_distance(q_identity, q_90z)
        assert np.isclose(dist, np.pi/2, atol=1e-10)

    def test_180_degree_rotation(self):
        """180-degree rotation should give distance of pi."""
        q_identity = identity_quat()
        # 180 degrees about z-axis: quat = [cos(90°), 0, 0, sin(90°)] = [0, 0, 0, 1]
        q_180z = np.array([0, 0, 0, 1])
        dist = quaternion_distance(q_identity, q_180z)
        assert np.isclose(dist, np.pi, atol=1e-10)

    def test_45_degree_rotation(self):
        """45-degree rotation should give distance of pi/4."""
        q_identity = identity_quat()
        angle = np.pi / 4  # 45 degrees
        q_45x = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        dist = quaternion_distance(q_identity, q_45x)
        assert np.isclose(dist, np.pi/4, atol=1e-10)

    def test_symmetry(self):
        """Distance should be symmetric: d(q1, q2) = d(q2, q1)."""
        q1 = quat_normalize([1, 2, 3, 4])
        q2 = quat_normalize([0, 1, 0, 1])
        assert np.isclose(quaternion_distance(q1, q2), quaternion_distance(q2, q1))

    def test_numerical_stability_near_identity(self):
        """Should handle quaternions very close to each other."""
        q1 = identity_quat()
        # Very small rotation
        epsilon = 1e-8
        q2 = quat_normalize([1, epsilon, 0, 0])
        dist = quaternion_distance(q1, q2)
        assert dist >= 0  # Should not be negative
        assert dist < 0.001  # Should be very small
