"""Quaternion math utilities for 6-DOF rigid body dynamics.

All functions are pure (no side effects) and use vectorized numpy operations.
Most functions are CasADi-compatible (no if/else on array values), except
dcm_to_quat() which uses Shepperd's method with branching.

Quaternion convention: [qw, qx, qy, qz] (scalar-first).
Hamilton convention is used for quaternion multiplication.

Note: The fmd.simulator package uses JAX (jax.numpy) for JIT compilation
and autodiff. This module remains numpy-only for use in fmd.core and
as a reference for future CasADi ports (e.g., for MPC/NMPC applications).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quat_multiply(q1: ArrayLike, q2: ArrayLike) -> NDArray:
    """Hamilton product of two quaternions.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        Product quaternion q1 ⊗ q2
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: ArrayLike) -> NDArray:
    """Quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Conjugate quaternion [qw, -qx, -qy, -qz]
    """
    q = np.asarray(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: ArrayLike) -> NDArray:
    """Normalize quaternion to unit length.

    Uses clamp-and-divide: norm is clamped to a minimum of 1e-10 before
    dividing, so a zero quaternion returns [0, 0, 0, 0] (not identity).
    This matches the JAX implementation in fmd.simulator.quaternion.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Unit quaternion. Near-zero inputs produce near-zero output.
    """
    q = np.asarray(q)
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    norm = np.maximum(norm, 1e-10)
    return q / norm


def quat_derivative(q: ArrayLike, omega: ArrayLike) -> NDArray:
    """Quaternion derivative from angular velocity.

    Computes q̇ = 0.5 * Ω ⊗ q where Ω = [0, ωx, ωy, ωz]

    Args:
        q: Current quaternion [qw, qx, qy, qz]
        omega: Angular velocity in body frame [ωx, ωy, ωz] (rad/s)

    Returns:
        Quaternion derivative [q̇w, q̇x, q̇y, q̇z]
    """
    q = np.asarray(q)
    omega = np.asarray(omega)

    # Omega as quaternion [0, ωx, ωy, ωz]
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

    # q̇ = 0.5 * omega_quat ⊗ q
    return 0.5 * quat_multiply(omega_quat, q)


def quat_to_dcm(q: ArrayLike) -> NDArray:
    """Convert quaternion to Direction Cosine Matrix (rotation matrix).

    Returns R such that v_ned = R @ v_body (body to NED transformation).

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix (DCM)
    """
    q = np.asarray(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Precompute products
    qwqw = qw * qw
    qxqx = qx * qx
    qyqy = qy * qy
    qzqz = qz * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz

    return np.array([
        [qwqw + qxqx - qyqy - qzqz, 2*(qxqy - qwqz), 2*(qxqz + qwqy)],
        [2*(qxqy + qwqz), qwqw - qxqx + qyqy - qzqz, 2*(qyqz - qwqx)],
        [2*(qxqz - qwqy), 2*(qyqz + qwqx), qwqw - qxqx - qyqy + qzqz],
    ])


def dcm_to_quat(R: ArrayLike) -> NDArray:
    """Convert Direction Cosine Matrix to quaternion.

    Uses Shepperd's method for numerical stability.
    NOT CasADi-compatible (uses Python if/else branching on array values).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [qw, qx, qy, qz]
    """
    R = np.asarray(R)

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # Shepperd's method - find largest diagonal element
    # Using smooth max approximation would be needed for CasADi
    # For now, use standard approach
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return quat_normalize(np.array([qw, qx, qy, qz]))


def quat_to_euler(q: ArrayLike) -> NDArray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX (yaw-pitch-roll) convention, NED frame.

    Gimbal Lock:
        Gimbal lock occurs when pitch approaches +/-90 degrees (pi/2 radians).
        At this singularity, roll and yaw become coupled and cannot be
        distinguished - only their sum or difference is defined. This is an
        inherent limitation of Euler angle representations.

        This implementation handles gimbal lock by:
        1. Clamping sin(pitch) to [-1, 1] to avoid NaN from arcsin
        2. Returning valid (but non-unique) roll/yaw values at the singularity

        For applications requiring robust orientation near vertical (e.g., acrobatic
        flight), consider using quaternions directly or an alternative Euler sequence.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Euler angles [roll, pitch, yaw] in radians. At gimbal lock (pitch = +/-pi/2),
        roll and yaw values are mathematically non-unique.
    """
    q = np.asarray(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Roll (rotation about x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    # Clamp to avoid numerical issues near gimbal lock
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotation about z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(euler: ArrayLike) -> NDArray:
    """Convert Euler angles to quaternion.

    Uses ZYX (yaw-pitch-roll) convention, NED frame.

    Args:
        euler: Euler angles [roll, pitch, yaw] in radians

    Returns:
        Quaternion [qw, qx, qy, qz]
    """
    euler = np.asarray(euler)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def rotate_vector(q: ArrayLike, v: ArrayLike) -> NDArray:
    """Rotate a vector by a quaternion.

    Computes v' = q ⊗ [0, v] ⊗ q* (body to NED transformation).

    Args:
        q: Quaternion [qw, qx, qy, qz]
        v: Vector [vx, vy, vz] in body frame

    Returns:
        Rotated vector in NED frame
    """
    q = np.asarray(q)
    v = np.asarray(v)

    # More efficient than full quaternion multiplication
    dcm = quat_to_dcm(q)
    return dcm @ v


def rotate_vector_inverse(q: ArrayLike, v: ArrayLike) -> NDArray:
    """Rotate a vector by the inverse of a quaternion.

    Computes v' = q* ⊗ [0, v] ⊗ q (NED to body transformation).

    Args:
        q: Quaternion [qw, qx, qy, qz]
        v: Vector [vx, vy, vz] in NED frame

    Returns:
        Rotated vector in body frame
    """
    q = np.asarray(q)
    v = np.asarray(v)

    dcm = quat_to_dcm(q)
    return dcm.T @ v


def identity_quat() -> NDArray:
    """Return the identity quaternion (no rotation).

    Returns:
        Identity quaternion [1, 0, 0, 0]
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_distance(q1: ArrayLike, q2: ArrayLike) -> float:
    """Compute geodesic distance between two quaternions.

    The geodesic distance is the angle of the shortest rotation between two
    orientations, measured on the unit quaternion sphere. This metric:
    - Is invariant to the q/-q ambiguity (same rotation, opposite quaternions)
    - Returns values in [0, pi] radians
    - Is 0 for identical rotations, pi for 180-degree difference

    Formula: distance = 2 * arccos(clamp(|q1 · q2|, 0, 1))

    Args:
        q1: First quaternion [qw, qx, qy, qz] (scalar-first, should be normalized)
        q2: Second quaternion [qw, qx, qy, qz] (scalar-first, should be normalized)

    Returns:
        Geodesic distance in radians, in range [0, pi]

    Examples:
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        >>> quaternion_distance(q, q)  # same rotation
        0.0
        >>> quaternion_distance(q, -q)  # antipodal (same rotation)
        0.0
        >>> q_90z = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        >>> np.isclose(quaternion_distance(q, q_90z), np.pi/2)  # 90 degrees
        True
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    # Dot product of quaternions
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

    # Take absolute value to handle q/-q ambiguity, clamp for numerical stability
    dot_clamped = np.clip(np.abs(dot), 0.0, 1.0)

    return 2.0 * np.arccos(dot_clamped)
