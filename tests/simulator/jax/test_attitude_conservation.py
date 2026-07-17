"""Attitude-kinematics conservation locks for the quaternion derivative.

These pin the *direction* of the body-frame quaternion kinematics, which the
existing quaternion tests do not: they check zero-omega, q.qdot orthogonality,
and JAX<->NumPy parity, all of which hold for BOTH the correct body-frame form
(q_dot = 1/2 q (X) omega) and the wrong world-frame form (1/2 omega (X) q). The
two agree only for single-axis motion, which is why the sign bug (physics review
3.1) survived 2148 tests.

Each test below is written to FAIL on the pre-fix (omega-left) code in the
physically wrong way: a torque-free tumble whose world-frame angular-momentum
direction wanders ~120 deg, and a generic-attitude DCM-rate mismatch.
"""
import numpy as np
import jax.numpy as jnp

from fmd.simulator import (
    RigidBody6DOF,
    create_state,
    simulate,
    rotate_vector,
    euler_to_quat,
    quat_to_dcm,
    quat_derivative,
)


def _skew(w):
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def test_torque_free_tumble_conserves_world_L_direction():
    """Torque-free asymmetric tumble: world-frame angular momentum
    L_world = R(q) (I omega) is fixed in BOTH magnitude and direction.

    Asymmetric inertia + tilted attitude + multi-axis omega is the configuration
    that discriminates the body-frame kinematics from the world-frame form. The
    pre-fix integrator wanders L_world direction ~120 deg here (physics review
    3.1 numerical proof); the correct kinematics hold it fixed to well under a
    degree over 10 s.
    """
    inertia = jnp.array([1.0, 2.0, 3.0])  # asymmetric (diagonal)
    body = RigidBody6DOF(mass=1.0, inertia=inertia, components=[])

    q0 = euler_to_quat(jnp.array([0.5, 0.4, 0.3]))  # tilted, multi-axis
    omega0 = jnp.array([0.5, 2.0, 0.3])             # multi-axis, near intermediate axis
    state0 = create_state(quaternion=q0, angular_velocity=omega0)

    result = simulate(body, state0, dt=0.001, duration=10.0)

    L = []
    for s in result.states:
        q = s[6:10]
        omega = s[10:13]
        L.append(np.asarray(rotate_vector(q, inertia * omega)))
    L = np.array(L)

    mag = np.linalg.norm(L, axis=1)
    mag0 = mag[0]
    assert np.max(np.abs(mag - mag0) / mag0) < 1e-3, "|L| not conserved"

    # Direction wander (angle from initial L), in degrees.
    cos = (L @ L[0]) / (mag * mag0)
    wander_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    assert np.max(wander_deg) < 1.0, (
        f"world-frame L direction wandered {np.max(wander_deg):.2f} deg "
        f"(pre-fix ~120 deg; correct kinematics keep it fixed)"
    )


def test_quat_derivative_matches_dcm_rate_at_generic_attitude():
    """q_dot must satisfy Rdot = R skew(omega_body) at a generic attitude.

    R(q) is body->NED (v_ned = R v_body); for body-frame omega the DCM rate is
    Rdot = R skew(omega). Identity q hides the bug (both conventions agree); a
    tilted attitude with multi-axis omega exposes it.
    """
    q = euler_to_quat(jnp.array([0.5, 0.4, 0.3]))
    q = q / jnp.linalg.norm(q)
    omega = jnp.array([0.5, 2.0, 0.3])

    qdot = quat_derivative(q, omega)

    eps = 1e-6
    q_plus = q + eps * qdot
    q_plus = q_plus / jnp.linalg.norm(q_plus)
    R = np.asarray(quat_to_dcm(q))
    R_plus = np.asarray(quat_to_dcm(q_plus))
    Rdot_fd = (R_plus - R) / eps

    Rdot_true = R @ _skew(np.asarray(omega))  # body-frame omega

    err = np.max(np.abs(Rdot_fd - Rdot_true))
    assert err < 1e-4, (
        f"Rdot mismatch {err:.3e}: q_dot does not satisfy Rdot = R skew(omega) "
        f"(pre-fix q_dot matches the world-frame form skew(omega) R)"
    )
