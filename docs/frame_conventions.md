# Frame Conventions

Authoritative reference for the coordinate frames, attitude representation,
and sign conventions used throughout `fomodynamics`.

**Related Documentation:**
- [README.md — Core conventions](../README.md#core-conventions) — quick recap (frames, units, quaternions, sign gotchas)
- [control_guide.md](control_guide.md) — LQR controller tuning and integrator selection
- [simulator_models.md](simulator_models.md) — model state vectors

---

## 1. Overview

`fomodynamics` uses aerospace-standard coordinate frames:

- **World Frame:** NED (North-East-Down)
- **Body Frame:** FRD (Forward-Right-Down)
- **Quaternion Convention:** Scalar-first `[qw, qx, qy, qz]`
- **Units:** SI internally; angles in radians

If you are bridging to libraries with different conventions (z-up worlds,
FLU bodies, scalar-last quaternions), the transforms are non-trivial — they
involve rotation matrices that must be carefully derived to preserve
physical meaning, plus a kinematic-consistency layer for angular velocity
and torque. The full derivation lives in private notes alongside the
benchmark adapters that use it.

---

## 2. World Frame: NED (North-East-Down)

```
                North (N)
                   ^
                   |
                   |
    West  <--------+--------> East (E)
                   |
                   |
                   v
                 Down (D)

    Gravity direction: +D (positive Down)
```

| Axis | Direction | Notes |
|------|-----------|-------|
| X | North | Primary horizontal axis |
| Y | East | Secondary horizontal axis |
| Z | Down | Positive toward Earth center |

**Gravity:** g = +9.8 m/s² in the +D direction (down is positive).

**Altitude:** Increasing altitude means `pos_d` *decreases* (more negative). This is the most common sign-error source for newcomers from robotics or graphics backgrounds where +Z is up.

---

## 3. Body Frame: FRD (Forward-Right-Down)

```
        Body Frame (FRD)
        ================

              x (Forward)
              ^
              |
              |
         -----+-----> y (Right)
              |
              |
              v
              z (Down)
```

| Axis | Direction | Rotation Name |
|------|-----------|---------------|
| X | Forward (nose) | Roll axis |
| Y | Right (starboard) | Pitch axis |
| Z | Down (belly) | Yaw axis |

**Positive rotations** follow the right-hand rule about each axis:

- **+Roll (φ > 0):** Right wing down, left wing up
- **+Pitch (θ > 0):** Nose up, tail down
- **+Yaw (ψ > 0):** Nose right (clockwise from above)

---

## 4. Quaternion Convention (Scalar-First)

`fomodynamics` uses scalar-first quaternions: **q = [qw, qx, qy, qz]**

| Component | Interpretation |
|-----------|----------------|
| qw | Scalar (cosine of the half-angle) |
| qx | Vector x component |
| qy | Vector y component |
| qz | Vector z component |

**Identity quaternion:** `[1, 0, 0, 0]` — no rotation, body aligned with world.

**Quaternion to rotation matrix** (active body→world rotation):

```
       ⎡ 1-2(qy²+qz²)    2(qxqy-qzqw)    2(qxqz+qyqw) ⎤
R(q) = ⎢ 2(qxqy+qzqw)    1-2(qx²+qz²)    2(qyqz-qxqw) ⎥
       ⎣ 2(qxqz-qyqw)    2(qyqz+qxqw)    1-2(qx²+qy²) ⎦
```

Many other libraries (ROS, Eigen, most graphics / game engines) use
**scalar-last** ordering `[qx, qy, qz, qw]`. When bridging, reorder
explicitly:

```python
q_fmd = jnp.array([q_other[3], q_other[0], q_other[1], q_other[2]])
```

---

## 5. State Vector Reference (6-DOF)

The 6-DOF rigid body state vector has 13 components:

| Index | Name | Description | Frame |
|-------|------|-------------|-------|
| 0-2 | pos_n, pos_e, pos_d | Position | NED world |
| 3-5 | vel_u, vel_v, vel_w | Linear velocity | FRD body |
| 6-9 | qw, qx, qy, qz | Attitude quaternion | scalar-first |
| 10-12 | omega_p, omega_q, omega_r | Angular velocity | FRD body |

**Derived outputs:**

- Roll (φ), Pitch (θ), Yaw (ψ) — Euler angles extracted from the quaternion (ZYX order). Use the quaternion as the source of truth; compute Euler angles only for display, since they have a singularity at ±90° pitch.

---

## 6. Angular Velocity and Torque Sign Conventions

Angular velocity **ω = [p, q, r]** and torque **τ = [τ_roll, τ_pitch, τ_yaw]** are expressed in the body frame.

| Component | Axis | Positive Direction (Right-Hand Rule) |
|-----------|------|--------------------------------------|
| p (roll rate) | Body X (forward) | Right wing drops, left wing rises |
| q (pitch rate) | Body Y (right) | Nose rises, tail drops |
| r (yaw rate) | Body Z (down) | Nose moves right (CW from above) |

**Relationship:** A positive torque about an axis produces positive angular acceleration about that axis:

```
τ = I · α    (I is the inertia tensor, α is angular acceleration)
```

---

## 7. Quaternion comparison

When comparing orientations from two sources, prefer geodesic distance
over component-wise comparison or Euler angles:

```python
def quaternion_geodesic_distance(q1, q2):
    """Geodesic distance between quaternions (handles q/-q ambiguity)."""
    dot = jnp.abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    return 2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0))
```

This handles the `q ≡ -q` ambiguity automatically, returns an angle in
`[0, π]` radians, has no gimbal-lock issues, and is more numerically stable
than going through Euler angles.
