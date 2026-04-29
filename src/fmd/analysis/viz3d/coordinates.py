"""Frame coordinate conversions between fomodynamics and Rerun display.

Frame conventions:
    Body (FRD): +X forward, +Y starboard, +Z down (right-handed)
    World (NED): +X north, +Y east, +Z down (right-handed)
    Rerun display:    +X east, +Y north, +Z up (right-handed, Z-up)

Mapping:
    NED [N, E, D] -> Rerun [E, N, -D]
    FRD [x_fwd, y_stbd, z_down] -> Rerun [y_stbd, x_fwd, -z_down]

Quaternion convention mismatch:
    scalar-first (qw, qx, qy, qz)
    Rerun: xyzw format (qx, qy, qz, qw)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ned_to_rerun(ned: NDArray) -> NDArray:
    """Convert NED world positions to Rerun display coordinates.

    [N, E, D] -> [E, N, -D]

    Args:
        ned: Position(s) in NED frame. Shape (..., 3).

    Returns:
        Position(s) in Rerun frame. Same shape.
    """
    ned = np.asarray(ned)
    out = np.empty_like(ned)
    out[..., 0] = ned[..., 1]   # X_rerun = E
    out[..., 1] = ned[..., 0]   # Y_rerun = N
    out[..., 2] = -ned[..., 2]  # Z_rerun = -D (up)
    return out


def frd_to_rerun(frd: NDArray) -> NDArray:
    """Convert FRD body-frame vectors to Rerun display coordinates.

    [x_fwd, y_stbd, z_down] -> [y_stbd, x_fwd, -z_down]

    Args:
        frd: Vector(s) in FRD body frame. Shape (..., 3).

    Returns:
        Vector(s) in Rerun frame. Same shape.
    """
    frd = np.asarray(frd)
    out = np.empty_like(frd)
    out[..., 0] = frd[..., 1]   # X_rerun = y_stbd
    out[..., 1] = frd[..., 0]   # Y_rerun = x_fwd
    out[..., 2] = -frd[..., 2]  # Z_rerun = -z_down (up)
    return out


def pitch_to_rerun_quat(theta: NDArray) -> NDArray:
    """Convert Moth pitch angle to Rerun xyzw quaternion.

    In FRD body frame, pitch is rotation about +Y (starboard axis).
    Positive theta = nose-up.

    After the FRD->Rerun axis swap:
    - FRD +Y (starboard) maps to Rerun +X (east)
    - So pitch becomes rotation about Rerun +X axis

    For Moth 3DOF with only pitch, the quaternion in Rerun xyzw is:
        (sin(θ/2), 0, 0, cos(θ/2))

    But we need to account for the axis swap properly. In FRD:
    - Pitch rotation axis is [0, 1, 0] (Y_body = starboard)
    - After FRD->Rerun swap, this axis becomes [1, 0, 0] (X_rerun)

    So the Rerun quaternion (xyzw) for pitch-only rotation is:
        qx = sin(θ/2), qy = 0, qz = 0, qw = cos(θ/2)

    Args:
        theta: Pitch angle(s) in radians. Scalar or array.

    Returns:
        Quaternion(s) in Rerun xyzw format. Shape (..., 4).
    """
    theta = np.asarray(theta)
    half = theta / 2.0
    qx = np.sin(half)
    qy = np.zeros_like(half)
    qz = np.zeros_like(half)
    qw = np.cos(half)
    return np.stack([qx, qy, qz, qw], axis=-1)


def moth_3dof_to_rerun_quat(
    theta: NDArray, heel_angle: float = 0.0
) -> NDArray:
    """Convert Moth pitch + heel to Rerun xyzw quaternion.

    Builds a fomodynamics quaternion (FRD body -> NED world) (scalar-first) from pitch and heel, then
    converts via ``fmd_quat_to_rerun`` to Rerun xyzw format.

    In FRD body frame (Hamilton convention — rightmost rotation applied first):
        q_pitch = rotation about +Y (starboard) by theta
        q_heel  = rotation about +X (forward) by heel_angle
        q_total = q_pitch ⊗ q_heel  (heel applied first, then pitch)

    At ``heel_angle=0`` the output matches ``pitch_to_rerun_quat(theta)``.

    Args:
        theta: Pitch angle(s) in radians. Scalar or array.
        heel_angle: Heel (roll) angle in radians. Positive = starboard down.

    Returns:
        Quaternion(s) in Rerun xyzw format. Shape (..., 4).
    """
    theta = np.asarray(theta)
    half_theta = theta / 2.0
    half_heel = heel_angle / 2.0

    # Pitch quaternion: rotation about FRD Y  [cos(θ/2), 0, sin(θ/2), 0]
    q_pitch = np.stack([
        np.cos(half_theta),
        np.zeros_like(half_theta),
        np.sin(half_theta),
        np.zeros_like(half_theta),
    ], axis=-1)

    # Heel quaternion: rotation about FRD X  [cos(φ/2), sin(φ/2), 0, 0]
    q_heel = np.array([
        np.cos(half_heel), np.sin(half_heel), 0.0, 0.0
    ])

    # Hamilton convention: q_total = q_pitch ⊗ q_heel applies heel first, then pitch
    q_total = _quat_multiply(q_pitch, q_heel)

    return fmd_quat_to_rerun(q_total)


def _quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """Multiply two quaternions (scalar-first format).

    Args:
        q1, q2: Quaternions [w, x, y, z]. Shape (4,) or (N, 4).

    Returns:
        Product q1 ⊗ q2 in scalar-first format.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


# Frame transformation quaternion: 180° rotation about [1,1,0]/√2
# This converts between NED/FRD and Rerun's Z-up frame
# q_frame = [cos(90°), sin(90°)*axis] = [0, 1/√2, 1/√2, 0]
_SQRT2_INV = 1.0 / np.sqrt(2.0)
_Q_FRAME = np.array([0.0, _SQRT2_INV, _SQRT2_INV, 0.0])
_Q_FRAME_INV = np.array([0.0, -_SQRT2_INV, -_SQRT2_INV, 0.0])  # conjugate


def fmd_quat_to_rerun(quat: NDArray) -> NDArray:
    """Convert fomodynamics quaternion (FRD body -> NED world) to Rerun quaternion with frame transformation.

    fomodynamics quaternion (FRD body -> NED world)s represent rotation from body (FRD) to world (NED).
    Rerun uses a Z-up world frame. This function:
    1. Conjugates the quaternion by the frame transformation
       (Hamilton convention: q_rerun = q_frame ⊗ q_blur ⊗ q_frame⁻¹,
       rightmost applied first)
    2. Converts from scalar-first [qw,qx,qy,qz] to xyzw format

    The frame transformation is a 180° rotation about [1,1,0]/√2, which
    maps NED axes to Rerun axes: N→Y, E→X, D→-Z.

    Args:
        quat: Quaternion(s) in scalar-first format [qw,qx,qy,qz]. Shape (4,) or (N, 4).

    Returns:
        Quaternion(s) in Rerun xyzw format. Same shape.
    """
    quat = np.asarray(quat)

    # Apply frame transformation: q_rerun = q_frame ⊗ q_blur ⊗ q_frame^{-1}
    q_transformed = _quat_multiply(_Q_FRAME, quat)
    q_transformed = _quat_multiply(q_transformed, _Q_FRAME_INV)

    # Convert from scalar-first to xyzw
    if q_transformed.ndim == 1:
        return np.array([
            q_transformed[1], q_transformed[2],
            q_transformed[3], q_transformed[0]
        ])
    return np.column_stack([
        q_transformed[:, 1], q_transformed[:, 2],
        q_transformed[:, 3], q_transformed[:, 0]
    ])
