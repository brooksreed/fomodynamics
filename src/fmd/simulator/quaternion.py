"""JAX quaternion math utilities for 6-DOF rigid body dynamics.

All functions are pure JAX operations for JIT compatibility.
Quaternion convention: [qw, qx, qy, qz] (scalar-first, Hamilton).

This module mirrors blur.core.quaternion but uses JAX arrays.
The formulas are structurally identical to ensure numerical equivalence.

Conventions:
    - Coordinate frame: NED (North-East-Down)
    - Quaternion: Hamilton convention, scalar-first [qw, qx, qy, qz]
    - rotate_vector(q, v): transforms body frame -> NED frame
    - rotate_vector_inverse(q, v): transforms NED frame -> body frame
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import jax.numpy as jnp
from jax import Array


def quat_multiply(q1: Array, q2: Array) -> Array:
    """Hamilton product of two quaternions.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        Product quaternion q1 @ q2
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: Array) -> Array:
    """Quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Conjugate quaternion [qw, -qx, -qy, -qz]
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: Array) -> Array:
    """Normalize quaternion to unit length.

    Uses clamp-and-divide: norm is clamped to a minimum of 1e-10 before
    dividing, so a zero quaternion returns [0, 0, 0, 0] (not identity).
    This matches the numpy implementation in blur.core.quaternion.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Unit quaternion. Near-zero inputs produce near-zero output.
    """
    norm = jnp.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    norm = jnp.maximum(norm, 1e-10)
    return q / norm


def quat_derivative(q: Array, omega: Array) -> Array:
    """Quaternion derivative from angular velocity.

    Computes dq/dt = 0.5 * Omega @ q where Omega = [0, wx, wy, wz]

    Args:
        q: Current quaternion [qw, qx, qy, qz]
        omega: Angular velocity in body frame [wx, wy, wz] (rad/s)

    Returns:
        Quaternion derivative [dqw, dqx, dqy, dqz]
    """
    # Omega as quaternion [0, wx, wy, wz]
    omega_quat = jnp.array([0.0, omega[0], omega[1], omega[2]])

    # dq/dt = 0.5 * omega_quat @ q
    return 0.5 * quat_multiply(omega_quat, q)


def quat_to_dcm(q: Array) -> Array:
    """Convert quaternion to Direction Cosine Matrix (rotation matrix).

    Returns R such that v_ned = R @ v_body (body to NED transformation).

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix (DCM)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Precompute products (same formula as numpy version)
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

    return jnp.array([
        [qwqw + qxqx - qyqy - qzqz, 2*(qxqy - qwqz), 2*(qxqz + qwqy)],
        [2*(qxqy + qwqz), qwqw - qxqx + qyqy - qzqz, 2*(qyqz - qwqx)],
        [2*(qxqz - qwqy), 2*(qyqz + qwqx), qwqw - qxqx - qyqy + qzqz],
    ])


def quat_to_euler(q: Array) -> Array:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX (yaw-pitch-roll) convention, NED frame.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Roll (rotation about x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    # Clamp to avoid numerical issues near gimbal lock
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (rotation about z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw])


def euler_to_quat(euler: Array) -> Array:
    """Convert Euler angles to quaternion.

    Uses ZYX (yaw-pitch-roll) convention, NED frame.

    Args:
        euler: Euler angles [roll, pitch, yaw] in radians

    Returns:
        Quaternion [qw, qx, qy, qz]
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return jnp.array([qw, qx, qy, qz])


def rotate_vector(q: Array, v: Array) -> Array:
    """Rotate a vector by a quaternion (body to NED transformation).

    Computes v' = R(q) @ v where R(q) is the rotation matrix from q.

    Args:
        q: Quaternion [qw, qx, qy, qz]
        v: Vector [vx, vy, vz] in body frame

    Returns:
        Rotated vector in NED frame
    """
    dcm = quat_to_dcm(q)
    return dcm @ v


def rotate_vector_inverse(q: Array, v: Array) -> Array:
    """Rotate a vector by the inverse of a quaternion (NED to body transformation).

    Computes v' = R(q)^T @ v where R(q) is the rotation matrix from q.

    Args:
        q: Quaternion [qw, qx, qy, qz]
        v: Vector [vx, vy, vz] in NED frame

    Returns:
        Rotated vector in body frame
    """
    dcm = quat_to_dcm(q)
    return dcm.T @ v


def identity_quat() -> Array:
    """Return the identity quaternion (no rotation).

    Returns:
        Identity quaternion [1, 0, 0, 0]
    """
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def euler_to_dcm_jax(roll: Array, pitch: Array, yaw: Array) -> Array:
    """Convert Euler angles to Direction Cosine Matrix (rotation matrix).

    Uses ZYX (yaw-pitch-roll) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This produces a rotation matrix R such that:
        v_world = R @ v_body

    The ZYX convention means rotations are applied in order:
    1. First rotate about body x-axis by roll
    2. Then rotate about (new) body y-axis by pitch
    3. Finally rotate about (new) body z-axis by yaw

    Args:
        roll: Roll angle (rotation about x-axis) in radians
        pitch: Pitch angle (rotation about y-axis) in radians
        yaw: Yaw angle (rotation about z-axis) in radians

    Returns:
        3x3 rotation matrix (DCM)
    """
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # Computed element-by-element for clarity and correctness
    # Reference: Stevens & Lewis "Aircraft Control and Simulation", eq. 1.3-20
    return jnp.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ],
    ])


def dcm_to_quat_jax(R: Array) -> Array:
    """Convert Direction Cosine Matrix to quaternion using Shepperd's method.

    This is a JAX-compatible implementation of Shepperd's method for
    numerically stable DCM to quaternion conversion. Uses jax.lax.cond
    for JIT compatibility.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [qw, qx, qy, qz] (normalized)
    """
    import jax.lax as lax

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # Branch 0: trace > 0
    def branch_trace(R):
        s = 0.5 / jnp.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
        return jnp.array([qw, qx, qy, qz])

    # Branch 1: R[0,0] is largest diagonal
    def branch_x(R):
        s = 2.0 * jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
        return jnp.array([qw, qx, qy, qz])

    # Branch 2: R[1,1] is largest diagonal
    def branch_y(R):
        s = 2.0 * jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
        return jnp.array([qw, qx, qy, qz])

    # Branch 3: R[2,2] is largest diagonal
    def branch_z(R):
        s = 2.0 * jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
        return jnp.array([qw, qx, qy, qz])

    # Use nested lax.cond for JIT-compatible branching
    # Determine which branch to take based on diagonal elements
    q = lax.cond(
        trace > 0,
        branch_trace,
        lambda R: lax.cond(
            R[0, 0] > R[1, 1],
            lambda R: lax.cond(
                R[0, 0] > R[2, 2],
                branch_x,
                branch_z,
                R
            ),
            lambda R: lax.cond(
                R[1, 1] > R[2, 2],
                branch_y,
                branch_z,
                R
            ),
            R
        ),
        R
    )

    return quat_normalize(q)
