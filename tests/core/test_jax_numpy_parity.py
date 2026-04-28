"""Tests for JAX/NumPy parity in fmd.core math functions.

These tests verify that the fmd.core math functions produce identical results
when given JAX arrays vs NumPy arrays as inputs. This is critical because:

1. fmd.core functions use np.asarray() which should work with both array types
2. Users may pass JAX arrays to core functions in mixed-backend codebases
3. Results must be numerically identical within floating point tolerance

Test strategy:
- For each function, run with np.array inputs and jnp.array inputs
- Compare results using np.testing.assert_allclose
- Use tight tolerances (rtol=1e-12, atol=1e-14) for structurally identical operations
"""

import numpy as np
import pytest

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

# Enable float64 before any computations
from fmd.core.jax_config import configure_jax  # noqa: F401

from fmd.core.quaternion import (
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
)

from fmd.core.operations import (
    wrap_angle,
    unwrap_angle,
    circular_subtract,
    circular_mean,
    angle_difference_to_vector,
)

# Tolerances for parity testing - tight since operations are structurally identical
PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-14


# ==============================================================================
# Quaternion Operations Parity Tests
# ==============================================================================


class TestQuatMultiplyParity:
    """Test quat_multiply produces identical results with NumPy and JAX arrays."""

    def test_identity_multiply(self):
        """Multiply by identity quaternion."""
        q1_np = np.array([1.0, 0.0, 0.0, 0.0])
        q2_np = np.array([0.707, 0.707, 0.0, 0.0])

        q1_jax = jnp.array(q1_np)
        q2_jax = jnp.array(q2_np)

        result_np = quat_multiply(q1_np, q2_np)
        result_jax = quat_multiply(q1_jax, q2_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_general_multiply(self):
        """Multiply two general unit quaternions."""
        q1_np = np.array([0.5, 0.5, 0.5, 0.5])
        q2_np = np.array([0.5, -0.5, 0.5, -0.5])

        q1_jax = jnp.array(q1_np)
        q2_jax = jnp.array(q2_np)

        result_np = quat_multiply(q1_np, q2_np)
        result_jax = quat_multiply(q1_jax, q2_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_mixed_inputs(self):
        """Test with one NumPy and one JAX input."""
        q1_np = np.array([0.5, 0.5, 0.5, 0.5])
        q2_jax = jnp.array([0.5, -0.5, 0.5, -0.5])

        # Both orderings
        result1 = quat_multiply(q1_np, q2_jax)
        result2 = quat_multiply(q2_jax, q1_np)

        # Compare against pure NumPy
        q2_np = np.array([0.5, -0.5, 0.5, -0.5])
        expected1 = quat_multiply(q1_np, q2_np)
        expected2 = quat_multiply(q2_np, q1_np)

        np.testing.assert_allclose(
            np.asarray(result1), expected1, rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
        np.testing.assert_allclose(
            np.asarray(result2), expected2, rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestQuatConjugateParity:
    """Test quat_conjugate produces identical results with NumPy and JAX arrays."""

    def test_conjugate(self):
        """Test quaternion conjugate."""
        q_np = np.array([0.5, 0.1, 0.2, 0.3])
        q_jax = jnp.array(q_np)

        result_np = quat_conjugate(q_np)
        result_jax = quat_conjugate(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestQuatNormalizeParity:
    """Test quat_normalize produces identical results with NumPy and JAX arrays."""

    def test_normalize_unnormalized(self):
        """Test normalizing an unnormalized quaternion."""
        q_np = np.array([1.0, 2.0, 3.0, 4.0])
        q_jax = jnp.array(q_np)

        result_np = quat_normalize(q_np)
        result_jax = quat_normalize(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_normalize_near_unit(self):
        """Test normalizing a near-unit quaternion."""
        q_np = np.array([0.999, 0.001, 0.001, 0.001])
        q_jax = jnp.array(q_np)

        result_np = quat_normalize(q_np)
        result_jax = quat_normalize(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestQuatDerivativeParity:
    """Test quat_derivative produces identical results with NumPy and JAX arrays."""

    def test_derivative_zero_omega(self):
        """Test derivative with zero angular velocity."""
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        omega_np = np.array([0.0, 0.0, 0.0])

        q_jax = jnp.array(q_np)
        omega_jax = jnp.array(omega_np)

        result_np = quat_derivative(q_np, omega_np)
        result_jax = quat_derivative(q_jax, omega_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_derivative_nonzero_omega(self):
        """Test derivative with nonzero angular velocity."""
        q_np = np.array([0.5, 0.5, 0.5, 0.5])
        omega_np = np.array([0.1, 0.2, 0.3])

        q_jax = jnp.array(q_np)
        omega_jax = jnp.array(omega_np)

        result_np = quat_derivative(q_np, omega_np)
        result_jax = quat_derivative(q_jax, omega_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestQuatToDCMParity:
    """Test quat_to_dcm produces identical results with NumPy and JAX arrays."""

    def test_dcm_identity(self):
        """Test DCM from identity quaternion."""
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        q_jax = jnp.array(q_np)

        result_np = quat_to_dcm(q_np)
        result_jax = quat_to_dcm(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_dcm_general(self):
        """Test DCM from general quaternion."""
        q_np = np.array([0.5, 0.5, 0.5, 0.5])
        q_jax = jnp.array(q_np)

        result_np = quat_to_dcm(q_np)
        result_jax = quat_to_dcm(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestDCMToQuatParity:
    """Test dcm_to_quat produces identical results with NumPy and JAX arrays."""

    def test_dcm_to_quat_identity(self):
        """Test quaternion from identity DCM."""
        dcm_np = np.eye(3)
        dcm_jax = jnp.eye(3)

        result_np = dcm_to_quat(dcm_np)
        result_jax = dcm_to_quat(dcm_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_dcm_to_quat_rotation(self):
        """Test quaternion from general rotation DCM."""
        # 45 degree rotation about z-axis
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        dcm_np = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        dcm_jax = jnp.array(dcm_np)

        result_np = dcm_to_quat(dcm_np)
        result_jax = dcm_to_quat(dcm_jax)

        # Quaternions may differ by sign (both represent same rotation)
        result_np_arr = np.asarray(result_np)
        result_jax_arr = np.asarray(result_jax)

        # Check either same or negated
        same = np.allclose(result_np_arr, result_jax_arr, rtol=PARITY_RTOL, atol=PARITY_ATOL)
        negated = np.allclose(result_np_arr, -result_jax_arr, rtol=PARITY_RTOL, atol=PARITY_ATOL)
        assert same or negated, f"Results differ: {result_np_arr} vs {result_jax_arr}"


class TestQuatToEulerParity:
    """Test quat_to_euler produces identical results with NumPy and JAX arrays."""

    def test_euler_identity(self):
        """Test Euler angles from identity quaternion."""
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        q_jax = jnp.array(q_np)

        result_np = quat_to_euler(q_np)
        result_jax = quat_to_euler(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_euler_general(self):
        """Test Euler angles from general quaternion."""
        # Create quaternion from known Euler angles
        euler = np.array([0.1, 0.2, 0.3])
        q_np = euler_to_quat(euler)
        q_jax = jnp.array(q_np)

        result_np = quat_to_euler(q_np)
        result_jax = quat_to_euler(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestEulerToQuatParity:
    """Test euler_to_quat produces identical results with NumPy and JAX arrays."""

    def test_euler_to_quat_zero(self):
        """Test quaternion from zero Euler angles."""
        euler_np = np.array([0.0, 0.0, 0.0])
        euler_jax = jnp.array(euler_np)

        result_np = euler_to_quat(euler_np)
        result_jax = euler_to_quat(euler_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_euler_to_quat_general(self):
        """Test quaternion from general Euler angles."""
        euler_np = np.array([0.1, -0.2, 0.5])
        euler_jax = jnp.array(euler_np)

        result_np = euler_to_quat(euler_np)
        result_jax = euler_to_quat(euler_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestRotateVectorParity:
    """Test rotate_vector produces identical results with NumPy and JAX arrays."""

    def test_rotate_identity(self):
        """Test rotation by identity quaternion."""
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        v_np = np.array([1.0, 2.0, 3.0])

        q_jax = jnp.array(q_np)
        v_jax = jnp.array(v_np)

        result_np = rotate_vector(q_np, v_np)
        result_jax = rotate_vector(q_jax, v_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_rotate_general(self):
        """Test general rotation."""
        q_np = np.array([0.5, 0.5, 0.5, 0.5])
        v_np = np.array([1.0, 0.0, 0.0])

        q_jax = jnp.array(q_np)
        v_jax = jnp.array(v_np)

        result_np = rotate_vector(q_np, v_np)
        result_jax = rotate_vector(q_jax, v_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestRotateVectorInverseParity:
    """Test rotate_vector_inverse produces identical results with NumPy and JAX arrays."""

    def test_rotate_inverse_identity(self):
        """Test inverse rotation by identity quaternion."""
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        v_np = np.array([1.0, 2.0, 3.0])

        q_jax = jnp.array(q_np)
        v_jax = jnp.array(v_np)

        result_np = rotate_vector_inverse(q_np, v_np)
        result_jax = rotate_vector_inverse(q_jax, v_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_rotate_inverse_general(self):
        """Test general inverse rotation."""
        q_np = np.array([0.5, 0.5, 0.5, 0.5])
        v_np = np.array([1.0, 0.0, 0.0])

        q_jax = jnp.array(q_np)
        v_jax = jnp.array(v_np)

        result_np = rotate_vector_inverse(q_np, v_np)
        result_jax = rotate_vector_inverse(q_jax, v_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


# ==============================================================================
# Circular Operations Parity Tests
# ==============================================================================


class TestWrapAngleParity:
    """Test wrap_angle produces identical results with NumPy and JAX arrays."""

    def test_wrap_scalar(self):
        """Test wrapping scalar angles."""
        angles = [2 * np.pi, -2 * np.pi, np.pi + 0.1, -np.pi - 0.1]

        for angle in angles:
            result_np = wrap_angle(np.array(angle))
            result_jax = wrap_angle(jnp.array(angle))

            np.testing.assert_allclose(
                float(result_np), float(result_jax),
                rtol=PARITY_RTOL, atol=PARITY_ATOL,
                err_msg=f"Mismatch for angle {angle}"
            )

    def test_wrap_array(self):
        """Test wrapping array of angles."""
        angles_np = np.array([0, np.pi, 2*np.pi, -np.pi, 3*np.pi])
        angles_jax = jnp.array(angles_np)

        result_np = wrap_angle(angles_np)
        result_jax = wrap_angle(angles_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_wrap_custom_range(self):
        """Test wrapping with custom range [0, 2pi)."""
        angles_np = np.array([-np.pi/2, 0, np.pi, 3*np.pi/2])
        angles_jax = jnp.array(angles_np)

        result_np = wrap_angle(angles_np, low=0, high=2*np.pi)
        result_jax = wrap_angle(angles_jax, low=0, high=2*np.pi)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestUnwrapAngleParity:
    """Test unwrap_angle produces identical results with NumPy and JAX arrays."""

    def test_unwrap_discontinuity(self):
        """Test unwrapping across discontinuity."""
        angles_np = np.array([170, 180, -170]) * np.pi / 180
        angles_jax = jnp.array(angles_np)

        result_np = unwrap_angle(angles_np)
        result_jax = unwrap_angle(angles_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_unwrap_smooth(self):
        """Test unwrapping smooth sequence."""
        angles_np = np.linspace(0, 2*np.pi, 10)
        angles_np = wrap_angle(angles_np)  # Introduce wrap
        angles_jax = jnp.array(angles_np)

        result_np = unwrap_angle(angles_np)
        result_jax = unwrap_angle(angles_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestCircularSubtractParity:
    """Test circular_subtract produces identical results with NumPy and JAX arrays."""

    def test_subtract_across_zero(self):
        """Test subtraction across 0 degree boundary."""
        a_np = np.radians(1)
        b_np = np.radians(359)

        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)

        result_np = circular_subtract(a_np, b_np)
        result_jax = circular_subtract(a_jax, b_jax)

        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_subtract_arrays(self):
        """Test subtracting arrays of angles."""
        a_np = np.radians([1, 359, 180])
        b_np = np.radians([359, 1, 180])

        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)

        result_np = circular_subtract(a_np, b_np)
        result_jax = circular_subtract(a_jax, b_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_subtract_same_value(self):
        """Test subtracting same angle gives zero."""
        a_np = np.radians(180)
        a_jax = jnp.array(a_np)

        result_np = circular_subtract(a_np, a_np)
        result_jax = circular_subtract(a_jax, a_jax)

        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestCircularMeanParity:
    """Test circular_mean produces identical results with NumPy and JAX arrays."""

    def test_mean_across_zero(self):
        """Test mean of angles across 0 degrees."""
        angles_np = np.radians([350, 10])
        angles_jax = jnp.array(angles_np)

        result_np = circular_mean(angles_np)
        result_jax = circular_mean(angles_jax)

        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_mean_same_direction(self):
        """Test mean of angles in same direction."""
        angles_np = np.radians([45, 45, 45])
        angles_jax = jnp.array(angles_np)

        result_np = circular_mean(angles_np)
        result_jax = circular_mean(angles_jax)

        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_mean_opposite_directions(self):
        """Test mean of opposite angles (undefined direction)."""
        angles_np = np.radians([0, 180])
        angles_jax = jnp.array(angles_np)

        result_np = circular_mean(angles_np)
        result_jax = circular_mean(angles_jax)

        # Both should return 0.0 by contract
        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_weighted_mean(self):
        """Test weighted circular mean."""
        angles_np = np.radians([0, 90])
        weights_np = np.array([3.0, 1.0])

        angles_jax = jnp.array(angles_np)
        weights_jax = jnp.array(weights_np)

        result_np = circular_mean(angles_np, weights_np)
        result_jax = circular_mean(angles_jax, weights_jax)

        np.testing.assert_allclose(
            float(result_np), float(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


class TestAngleDifferenceToVectorParity:
    """Test angle_difference_to_vector produces identical results with NumPy and JAX arrays."""

    def test_target_ahead(self):
        """Test when target is directly ahead."""
        heading_np = np.radians(45)
        target_np = np.radians(45)

        heading_jax = jnp.array(heading_np)
        target_jax = jnp.array(target_np)

        lat_np, lon_np = angle_difference_to_vector(heading_np, target_np)
        lat_jax, lon_jax = angle_difference_to_vector(heading_jax, target_jax)

        np.testing.assert_allclose(
            float(lat_np), float(lat_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
        np.testing.assert_allclose(
            float(lon_np), float(lon_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_target_right(self):
        """Test when target is 90 degrees to the right."""
        heading_np = np.radians(0)
        target_np = np.radians(90)

        heading_jax = jnp.array(heading_np)
        target_jax = jnp.array(target_np)

        lat_np, lon_np = angle_difference_to_vector(heading_np, target_np)
        lat_jax, lon_jax = angle_difference_to_vector(heading_jax, target_jax)

        np.testing.assert_allclose(
            float(lat_np), float(lat_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
        np.testing.assert_allclose(
            float(lon_np), float(lon_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_array_inputs(self):
        """Test with array inputs."""
        heading_np = np.radians([0, 45, 90])
        target_np = np.radians([90, 45, 0])

        heading_jax = jnp.array(heading_np)
        target_jax = jnp.array(target_np)

        lat_np, lon_np = angle_difference_to_vector(heading_np, target_np)
        lat_jax, lon_jax = angle_difference_to_vector(heading_jax, target_jax)

        np.testing.assert_allclose(
            np.asarray(lat_np), np.asarray(lat_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
        np.testing.assert_allclose(
            np.asarray(lon_np), np.asarray(lon_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )


# ==============================================================================
# Edge Cases and Stress Tests
# ==============================================================================


class TestEdgeCases:
    """Test edge cases for parity."""

    def test_very_small_quaternion(self):
        """Test quaternion operations with very small values."""
        q_np = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        q_jax = jnp.array(q_np)

        # normalize should handle near-zero gracefully
        result_np = quat_normalize(q_np)
        result_jax = quat_normalize(q_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_large_angles(self):
        """Test circular operations with large angles."""
        angles_np = np.array([100 * np.pi, -100 * np.pi, 1000 * np.pi])
        angles_jax = jnp.array(angles_np)

        result_np = wrap_angle(angles_np)
        result_jax = wrap_angle(angles_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_random_quaternions(self):
        """Test with multiple random quaternions."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            q_np = rng.standard_normal(4)
            q_np = q_np / np.linalg.norm(q_np)
            q_jax = jnp.array(q_np)

            v_np = rng.standard_normal(3)
            v_jax = jnp.array(v_np)

            # Test rotate_vector
            result_np = rotate_vector(q_np, v_np)
            result_jax = rotate_vector(q_jax, v_jax)

            np.testing.assert_allclose(
                np.asarray(result_np), np.asarray(result_jax),
                rtol=PARITY_RTOL, atol=PARITY_ATOL
            )

    def test_random_angles(self):
        """Test circular operations with random angles."""
        rng = np.random.default_rng(42)

        angles_np = rng.uniform(-10 * np.pi, 10 * np.pi, size=100)
        angles_jax = jnp.array(angles_np)

        result_np = wrap_angle(angles_np)
        result_jax = wrap_angle(angles_jax)

        np.testing.assert_allclose(
            np.asarray(result_np), np.asarray(result_jax),
            rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
