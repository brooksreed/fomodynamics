"""Tests for JAX quaternion math utilities.

These tests verify that the JAX quaternion functions produce results
that match the numpy implementations within numerical tolerance.
"""

import pytest
import numpy as np

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_derivative,
    quat_to_dcm,
    quat_to_euler,
    euler_to_quat,
    rotate_vector,
    rotate_vector_inverse,
    identity_quat,
)
from fmd.core.quaternion import (
    quat_multiply as np_quat_multiply,
    quat_conjugate as np_quat_conjugate,
    quat_normalize as np_quat_normalize,
    quat_derivative as np_quat_derivative,
    quat_to_dcm as np_quat_to_dcm,
    quat_to_euler as np_quat_to_euler,
    euler_to_quat as np_euler_to_quat,
    rotate_vector as np_rotate_vector,
    rotate_vector_inverse as np_rotate_vector_inverse,
    identity_quat as np_identity_quat,
)

from .conftest import DERIV_RTOL, DERIV_ATOL


class TestQuaternionMultiply:
    """Tests for quat_multiply."""

    def test_identity_multiply(self):
        """Identity quaternion is multiplicative identity."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        p = jnp.array([0.5, 0.5, 0.5, 0.5])
        p = p / jnp.linalg.norm(p)

        result = quat_multiply(q, p)
        np.testing.assert_allclose(result, p, rtol=DERIV_RTOL, atol=DERIV_ATOL)

        result = quat_multiply(p, q)
        np.testing.assert_allclose(result, p, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self, random_unit_quaternion):
        """JAX result matches numpy implementation."""
        q1 = random_unit_quaternion
        q2 = jnp.array([0.5, 0.5, -0.5, 0.5])

        jax_result = quat_multiply(q1, q2)
        np_result = np_quat_multiply(np.array(q1), np.array(q2))

        np.testing.assert_allclose(jax_result, np_result, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_non_commutative(self):
        """Quaternion multiplication is non-commutative."""
        q1 = jnp.array([0.5, 0.5, 0.5, 0.5])
        q2 = jnp.array([0.5, -0.5, 0.5, -0.5])

        result1 = quat_multiply(q1, q2)
        result2 = quat_multiply(q2, q1)

        # Should NOT be equal (non-commutative)
        assert not jnp.allclose(result1, result2)


class TestQuaternionConjugate:
    """Tests for quat_conjugate."""

    def test_conjugate_values(self):
        """Conjugate negates vector part."""
        q = jnp.array([0.5, 0.1, 0.2, 0.3])
        result = quat_conjugate(q)

        expected = jnp.array([0.5, -0.1, -0.2, -0.3])
        np.testing.assert_allclose(result, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self, random_unit_quaternion):
        """JAX result matches numpy implementation."""
        q = random_unit_quaternion

        jax_result = quat_conjugate(q)
        np_result = np_quat_conjugate(np.array(q))

        np.testing.assert_allclose(jax_result, np_result, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestQuaternionNormalize:
    """Tests for quat_normalize."""

    def test_produces_unit_quaternion(self):
        """Result has unit norm."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = quat_normalize(q)

        norm = jnp.sqrt(jnp.sum(result ** 2))
        np.testing.assert_allclose(norm, 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self):
        """JAX result matches numpy implementation."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])

        jax_result = quat_normalize(q)
        np_result = np_quat_normalize(np.array(q))

        np.testing.assert_allclose(jax_result, np_result, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestQuaternionDerivative:
    """Tests for quat_derivative."""

    def test_zero_omega_zero_derivative(self):
        """Zero angular velocity gives zero derivative."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        omega = jnp.array([0.0, 0.0, 0.0])

        result = quat_derivative(q, omega)
        np.testing.assert_allclose(result, jnp.zeros(4), rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self, random_unit_quaternion):
        """JAX result matches numpy implementation."""
        q = random_unit_quaternion
        omega = jnp.array([0.1, 0.2, 0.3])

        jax_result = quat_derivative(q, omega)
        np_result = np_quat_derivative(np.array(q), np.array(omega))

        np.testing.assert_allclose(jax_result, np_result, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestQuaternionToDCM:
    """Tests for quat_to_dcm."""

    def test_identity_quat_identity_dcm(self):
        """Identity quaternion gives identity matrix."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        dcm = quat_to_dcm(q)

        np.testing.assert_allclose(dcm, jnp.eye(3), rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_dcm_is_orthogonal(self, random_unit_quaternion):
        """DCM is an orthogonal matrix (R @ R.T = I)."""
        dcm = quat_to_dcm(random_unit_quaternion)

        rrt = dcm @ dcm.T
        np.testing.assert_allclose(rrt, jnp.eye(3), rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_dcm_has_unit_determinant(self, random_unit_quaternion):
        """DCM has determinant +1 (proper rotation)."""
        dcm = quat_to_dcm(random_unit_quaternion)

        det = jnp.linalg.det(dcm)
        np.testing.assert_allclose(det, 1.0, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self, random_unit_quaternion):
        """JAX result matches numpy implementation."""
        q = random_unit_quaternion

        jax_dcm = quat_to_dcm(q)
        np_dcm = np_quat_to_dcm(np.array(q))

        np.testing.assert_allclose(jax_dcm, np_dcm, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestEulerConversions:
    """Tests for euler_to_quat and quat_to_euler."""

    def test_zero_euler_identity_quat(self):
        """Zero Euler angles give identity quaternion."""
        euler = jnp.array([0.0, 0.0, 0.0])
        q = euler_to_quat(euler)

        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_roundtrip(self):
        """euler -> quat -> euler roundtrip preserves values."""
        euler = jnp.array([0.1, 0.2, 0.3])
        q = euler_to_quat(euler)
        euler_back = quat_to_euler(q)

        np.testing.assert_allclose(euler_back, euler, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self):
        """JAX results match numpy implementations."""
        euler = jnp.array([0.1, -0.2, 0.5])

        jax_q = euler_to_quat(euler)
        np_q = np_euler_to_quat(np.array(euler))
        np.testing.assert_allclose(jax_q, np_q, rtol=DERIV_RTOL, atol=DERIV_ATOL)

        jax_euler = quat_to_euler(jax_q)
        np_euler = np_quat_to_euler(np.array(jax_q))
        np.testing.assert_allclose(jax_euler, np_euler, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestRotateVector:
    """Tests for rotate_vector and rotate_vector_inverse."""

    def test_identity_no_rotation(self):
        """Identity quaternion doesn't rotate vector."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        v = jnp.array([1.0, 2.0, 3.0])

        result = rotate_vector(q, v)
        np.testing.assert_allclose(result, v, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_inverse_reverses_rotation(self, random_unit_quaternion):
        """rotate_vector_inverse reverses rotate_vector."""
        q = random_unit_quaternion
        v = jnp.array([1.0, 2.0, 3.0])

        v_ned = rotate_vector(q, v)
        v_back = rotate_vector_inverse(q, v_ned)

        np.testing.assert_allclose(v_back, v, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self, random_unit_quaternion):
        """JAX results match numpy implementations."""
        q = random_unit_quaternion
        v = jnp.array([1.0, 2.0, 3.0])

        jax_result = rotate_vector(q, v)
        np_result = np_rotate_vector(np.array(q), np.array(v))
        np.testing.assert_allclose(jax_result, np_result, rtol=DERIV_RTOL, atol=DERIV_ATOL)

        jax_inv = rotate_vector_inverse(q, v)
        np_inv = np_rotate_vector_inverse(np.array(q), np.array(v))
        np.testing.assert_allclose(jax_inv, np_inv, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestIdentityQuat:
    """Tests for identity_quat."""

    def test_identity_values(self):
        """Identity quaternion is [1, 0, 0, 0]."""
        q = identity_quat()
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_matches_numpy(self):
        """JAX result matches numpy implementation."""
        jax_q = identity_quat()
        np_q = np_identity_quat()
        np.testing.assert_allclose(jax_q, np_q, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestJITCompatibility:
    """Tests that quaternion functions work with JAX JIT."""

    def test_quat_multiply_jit(self, random_unit_quaternion):
        """quat_multiply can be JIT compiled."""
        q1 = random_unit_quaternion
        q2 = jnp.array([0.5, 0.5, -0.5, 0.5])

        jit_multiply = jax.jit(quat_multiply)
        result = jit_multiply(q1, q2)

        # Should match non-JIT result
        expected = quat_multiply(q1, q2)
        np.testing.assert_allclose(result, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_quat_to_dcm_jit(self, random_unit_quaternion):
        """quat_to_dcm can be JIT compiled."""
        q = random_unit_quaternion

        jit_to_dcm = jax.jit(quat_to_dcm)
        result = jit_to_dcm(q)

        expected = quat_to_dcm(q)
        np.testing.assert_allclose(result, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)

    def test_rotate_vector_jit(self, random_unit_quaternion):
        """rotate_vector can be JIT compiled."""
        q = random_unit_quaternion
        v = jnp.array([1.0, 2.0, 3.0])

        jit_rotate = jax.jit(rotate_vector)
        result = jit_rotate(q, v)

        expected = rotate_vector(q, v)
        np.testing.assert_allclose(result, expected, rtol=DERIV_RTOL, atol=DERIV_ATOL)


class TestGradientCompatibility:
    """Tests that quaternion functions work with JAX autodiff."""

    def test_quat_normalize_grad(self):
        """Can compute gradient through quat_normalize."""
        def loss(q):
            q_norm = quat_normalize(q)
            return jnp.sum(q_norm ** 2)

        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        grad = jax.grad(loss)(q)

        # Gradient should exist and be finite
        assert jnp.all(jnp.isfinite(grad))

    def test_rotate_vector_grad(self, random_unit_quaternion):
        """Can compute gradient through rotate_vector."""
        def loss(q, v):
            v_rotated = rotate_vector(q, v)
            return jnp.sum(v_rotated ** 2)

        q = random_unit_quaternion
        v = jnp.array([1.0, 2.0, 3.0])

        grad_q, grad_v = jax.grad(loss, argnums=(0, 1))(q, v)

        # Gradients should exist and be finite
        assert jnp.all(jnp.isfinite(grad_q))
        assert jnp.all(jnp.isfinite(grad_v))
