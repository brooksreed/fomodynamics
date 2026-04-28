# BLUR Frame Conventions

This document provides the authoritative reference for coordinate frame conventions used in BLUR and their relationship to external systems (SCG/PyBullet).

**Related Documentation:**
- [CLAUDE.md](../CLAUDE.md) - Core design principles including NED/FRD conventions
- [benchmark_validation.md](benchmark_validation.md) - Cross-library validation methodology
- [control_guide.md](control_guide.md) - LQR controller tuning and integrator selection
- [simulator_models.md](simulator_models.md) - Model documentation and state vectors

---

## 1. Overview

BLUR uses aerospace-standard coordinate frames:
- **World Frame:** NED (North-East-Down)
- **Body Frame:** FRD (Forward-Right-Down)
- **Quaternion Convention:** Scalar-first (qw, qx, qy, qz)

External benchmarks (SCG/PyBullet) use robotics-standard frames:
- **World Frame:** z-up (right-handed, gravity along -z). In our BLUR↔SCG benchmarks we align this as **NEU** (North-East-Up): x→North, y→East, z→Up.
- **Body Frame:** FLU (Forward-Left-Up)
- **Attitude:** Euler angles (roll, pitch, yaw)

The transforms between these conventions are not simple sign flips—they involve rotation matrices that must be carefully derived to preserve physical meaning.

---

## 2. BLUR Conventions (Authoritative Definitions)

### 2.1. World Frame: NED (North-East-Down)

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

**Altitude:** Increasing altitude means pos_d *decreases* (more negative).

### 2.2. Body Frame: FRD (Forward-Right-Down)

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

### 2.3. Quaternion Convention (Scalar-First)

BLUR uses scalar-first quaternions: **q = [qw, qx, qy, qz]**

| Component | Interpretation |
|-----------|----------------|
| qw | Scalar (cosine half-angle) |
| qx | Vector x component |
| qy | Vector y component |
| qz | Vector z component |

**Identity quaternion:** [1, 0, 0, 0] = no rotation (body aligned with world)

**Quaternion to rotation matrix:**
```
       ⎡ 1-2(qy²+qz²)    2(qxqy-qzqw)    2(qxqz+qyqw) ⎤
R(q) = ⎢ 2(qxqy+qzqw)    1-2(qx²+qz²)    2(qyqz-qxqw) ⎥
       ⎣ 2(qxqz-qyqw)    2(qyqz+qxqw)    1-2(qx²+qy²) ⎦
```

### 2.4. State Vector Reference

The BLUR 6-DOF rigid body state vector has 13 components:

| Index | Name | Description | Frame |
|-------|------|-------------|-------|
| 0-2 | pos_n, pos_e, pos_d | Position | NED world |
| 3-5 | vel_u, vel_v, vel_w | Linear velocity | FRD body |
| 6-9 | qw, qx, qy, qz | Attitude quaternion | - |
| 10-12 | omega_p, omega_q, omega_r | Angular velocity | FRD body |

**Derived outputs:**
- Roll (φ), Pitch (θ), Yaw (ψ) - Euler angles extracted from quaternion (ZYX order)

### 2.5. Angular Velocity and Torque Sign Conventions

Angular velocity **ω = [p, q, r]** and torque **τ = [τ_roll, τ_pitch, τ_yaw]** are expressed in the body frame.

| Component | Axis | Positive Direction (Right-Hand Rule) |
|-----------|------|--------------------------------------|
| p (roll rate) | Body X (forward) | Right wing drops, left wing rises |
| q (pitch rate) | Body Y (right) | Nose rises, tail drops |
| r (yaw rate) | Body Z (down) | Nose moves right (CW from above) |

**Relationship:** A positive torque about an axis produces positive angular acceleration about that axis:
```
τ = I · α  (where I is inertia tensor, α is angular acceleration)
```

---

## 3. External System Conventions

### 3.1. SCG/PyBullet (z-up world, FLU body)

**World Frame: z-up (inertial)**

```
                  z (Up)
                   ^
                   |
                   |
                   +--------> y (East, right when facing +x/North)
                  /
                 /
                v
               x (North / forward in benchmarks)

    Gravity direction: -z (negative z)
```

| Axis | Direction | Notes |
|------|-----------|-------|
| X | North (benchmark alignment) | In BLUR↔SCG benchmarks we align SCG x with BLUR North |
| Y | East (benchmark alignment) | In BLUR↔SCG benchmarks we align SCG y with BLUR East |
| Z | Up | Positive away from Earth |

**Body Frame: FLU (Forward-Left-Up)**

```
        Body Frame (FLU)
        ================

              x (Forward)
              ^
              |
              |
    y (Left) <-----+
                   |
                   |
                   v
              z (Up, out of page shown as down for visibility)

    Note: z actually points UP (out of the page/screen)
```

| Axis | Direction | Rotation Name |
|------|-----------|---------------|
| X | Forward (nose) | Roll axis |
| Y | Left (port) | Pitch axis |
| Z | Up (dorsal) | Yaw axis |

**Positive rotations** follow the right-hand rule about each axis:
- **+Roll (φ > 0):** Left wing up, right wing down
- **+Pitch (θ > 0):** Nose up (same as FRD)
- **+Yaw (ψ > 0):** Nose left (CCW from above)

**SCG State Vector (12 components, interleaved):**

| Index | Name | Description |
|-------|------|-------------|
| 0, 1 | x, x_dot | Position and velocity (x) |
| 2, 3 | y, y_dot | Position and velocity (y) |
| 4, 5 | z, z_dot | Position and velocity (z) |
| 6-8 | φ, θ, ψ | Euler angles (roll, pitch, yaw) |
| 9-11 | p, q, r | Angular velocity (body frame) |

---

## 4. Transform Derivations

This section derives the mathematical transforms between BLUR (NED/FRD) and SCG (z-up/FLU) conventions using rotation matrices.

### 4.1. World Frame Transform Matrix S

The transform from z-up world to NED world is a reflection in the z-axis:

```
Position mapping:
  x_ned = x_zup        (North = x)
  y_ned = y_zup        (East = y)
  z_ned = -z_zup       (Down = -Up)

Therefore:
       ⎡ 1   0   0 ⎤
  S =  ⎢ 0   1   0 ⎥  = diag(1, 1, -1)
       ⎣ 0   0  -1 ⎦

  v_ned = S · v_zup
  v_zup = S · v_ned    (S is its own inverse: S² = I)
```

### 4.2. Body Frame Transform Matrix T

The transform from FLU body to FRD body is a 180° rotation about the x-axis:

```
Axis mapping:
  x_frd =  x_flu       (Forward = Forward)
  y_frd = -y_flu       (Right = -Left)
  z_frd = -z_flu       (Down = -Up)

Therefore:
       ⎡ 1   0   0 ⎤
  T =  ⎢ 0  -1   0 ⎥  = diag(1, -1, -1)
       ⎣ 0   0  -1 ⎦

  v_frd = T · v_flu
  v_flu = T · v_frd    (T is its own inverse: T² = I)
```

**Visual:**
```
    FRD <--T = diag(1,-1,-1)--> FLU

    This is a 180° rotation about the x-axis (forward).
    T is its own inverse: T · T = I
```

### 4.3. Rotation Matrix Transform

A rotation matrix R transforms vectors from body frame to world frame:
```
v_world = R · v_body
```

To convert rotation matrices between conventions:
```
R_blur = S · R_scg · T

Where:
  S = diag(1, 1, -1)    (z-up world → NED world)
  T = diag(1, -1, -1)   (FLU body → FRD body)
```

**Derivation:**
```
Given: v_zup = R_scg · v_flu

We want: v_ned = R_blur · v_frd

Substituting:
  S · v_zup = R_blur · (T · v_flu)

Therefore:
  S · R_scg · v_flu = R_blur · T · v_flu

For all v_flu:
  R_blur = S · R_scg · T
```

### 4.4. Quaternion/Attitude Transform

**Euler Angles (ZYX convention):**
```
FRD (BLUR)         FLU (SCG)
-----------        ----------
φ_frd = -φ_flu     Roll: NEGATED (y,z axes opposite)
θ_frd = -θ_flu     Pitch: NEGATED (empirically validated in Phase 11.5 Step 5)
ψ_frd = +ψ_flu     Yaw: UNCHANGED (empirically validated in Phase 11.5 Step 5)
```

**Why roll is negated:** In FRD, +roll means right wing down. In FLU, +roll means left wing up. These are physically opposite, so the sign must change.

**Why pitch is negated:** In FRD, +pitch means nose up (rotation about +y/starboard). In FLU, +y points left, so +pitch (rotation about +y) means nose down. Same physical rotation, opposite sign convention. Validated empirically against SCG/PyBullet reference trajectories.

**Why yaw differs from the naive intuition:** A naive “pure vector” frame flip suggests yaw should negate, but benchmark validation against SCG/PyBullet reference data showed yaw is **unchanged** for the conventions used in our SCG↔BLUR comparison pipeline.

### 4.5. Angular Velocity Transform

Angular velocity **ω = [p, q, r]** is a body-frame vector, BUT its transform must be **kinematically consistent** with the Euler angle representation used in each frame.

#### 4.5.1 Pure Vector Transform (Geometric)

For the same physical angular velocity vector expressed in different body coordinate frames:

```
Pure vector transform (geometric):
  ω_frd = T · ω_flu
        = [ p_flu, -q_flu, -r_flu ]
```

#### 4.5.2 Kinematically Consistent Transform (Used in Practice)

When omega is coupled to Euler angle dynamics (as in quaternion integration from Euler rates), the transform must account for the Euler angle sign conventions:

```
Euler angle transform:
  φ_blur = -φ_scg    (roll negated)
  θ_blur = -θ_scg    (pitch negated)
  ψ_blur = +ψ_scg    (yaw unchanged)

Angular velocity transform:
  p_blur = -p_scg    (roll rate negated)
  q_blur = -q_scg    (pitch rate negated)
  r_blur = +r_scg    (yaw rate unchanged)
```

**Verification (Phase 11.5 Step 5):** The correct transforms were derived empirically by comparing BLUR simulation output against SCG/PyBullet reference data. The key test scenarios were:

1. **Pure rotation** (initial omega = [1.0, 0.5, 0.2] rad/s): Quaternion error reduced from 54.77° to 1.29°
2. **Step roll torque**: Quaternion error reduced from 174.56° to 1.00°, omega sign matches

The earlier theory (Phase 11.5 Step 1) predicted `[-p, -q, -r]` based on Euler angle derivatives, but empirical testing against reference data showed `[-p, -q, +r]` is correct. Similarly for Euler angles: the theory predicted `[-φ, θ, -ψ]` but empirical testing showed `[-φ, -θ, +ψ]`.

### 4.6. Torque Transform

Torque **τ = [τ_roll, τ_pitch, τ_yaw]** must be consistent with the angular velocity transform. Since torque produces angular acceleration (τ/I = α), and angular acceleration integrates to angular velocity, the torque transform follows the same pattern as omega:

```
Torque transform (consistent with omega):
  τ_roll_blur = -τ_roll_scg    (roll torque negated)
  τ_pitch_blur = -τ_pitch_scg  (pitch torque negated)
  τ_yaw_blur = +τ_yaw_scg      (yaw torque unchanged)
```

This ensures that when SCG applies positive roll torque and gets positive roll rate, BLUR's converted torque produces the equivalent dynamics that transform back correctly.

---

## 5. Quick Reference Tables

### 5.1. Position and Velocity

| Quantity | BLUR → SCG | SCG → BLUR |
|----------|------------|------------|
| x (forward) | x_scg = pos_n | pos_n = x_scg |
| y (lateral) | y_scg = pos_e | pos_e = y_scg |
| z (vertical) | z_scg = -pos_d | pos_d = -z_scg |
| vx (world) | vx_scg = v_ned[0] | See note¹ |
| vy (world) | vy_scg = v_ned[1] | See note¹ |
| vz (world) | vz_scg = -v_ned[2] | See note¹ |

¹ BLUR uses body-frame velocity; conversion requires rotation by attitude quaternion.

### 5.2. Attitude (Euler Angles)

| Quantity | BLUR → SCG | SCG → BLUR |
|----------|------------|------------|
| Roll (φ) | φ_scg = -φ_blur | φ_blur = -φ_scg |
| Pitch (θ) | θ_scg = -θ_blur | θ_blur = -θ_scg |
| Yaw (ψ) | ψ_scg = +ψ_blur | ψ_blur = +ψ_scg |

### 5.3. Angular Velocity

| Quantity | BLUR → SCG | SCG → BLUR |
|----------|------------|------------|
| Roll rate (p) | p_scg = -p_blur | p_blur = -p_scg |
| Pitch rate (q) | q_scg = -q_blur | q_blur = -q_scg |
| Yaw rate (r) | r_scg = +r_blur | r_blur = +r_scg |

**Note:** Roll and pitch negated; yaw unchanged. Verified empirically in Phase 11.5 Step 5.

### 5.4. Torque

| Quantity | BLUR → SCG | SCG → BLUR |
|----------|------------|------------|
| Roll torque (τ_roll) | τ_roll_scg = -τ_roll_blur | τ_roll_blur = -τ_roll_scg |
| Pitch torque (τ_pitch) | τ_pitch_scg = -τ_pitch_blur | τ_pitch_blur = -τ_pitch_scg |
| Yaw torque (τ_yaw) | τ_yaw_scg = +τ_yaw_blur | τ_yaw_blur = +τ_yaw_scg |

**Note:** Follows same pattern as angular velocity (torque produces angular acceleration).

### 5.5. Example: 15-Degree Roll

| System | Roll Angle | Angular Vel | Physical Meaning |
|--------|------------|-------------|------------------|
| BLUR (FRD) | φ = +15° | p = +5°/s | Right wing 15° down, rolling further right-wing-down |
| SCG (FLU) | φ = -15° | p = -5°/s | Same physical orientation; roll and pitch negated, yaw same |

Same physical state; note that **both** the Euler roll angle and omega are negated for kinematic consistency.

---

## Appendix A: Common Pitfalls

### A.1. Vector Transform vs Kinematic Consistency

A natural intuition is that angular velocity is a vector and should transform as:
```python
# Vector transform (T = diag(1,-1,-1)):
p_blur = p_scg      # p same
q_blur = -q_scg     # q negated
r_blur = -r_scg     # r negated
```

However, trajectory validation shows this produces **incorrect dynamics** (mirrored rotations). The empirically validated transform used in BLUR↔SCG adapters is:
```python
# Empirically validated (Phase 11.5 Step 5):
p_blur = -p_scg
q_blur = -q_scg
r_blur = +r_scg
```

**Key point:** The transform must be consistent globally across Euler, omega, torque, and quaternion comparisons used in the benchmark pipeline. In Phase 11.5 Step 5, the consistent set was found to be:
- Euler: `[-φ, -θ, +ψ]`
- Omega: `[-p, -q, +r]`
- Torque: `[-τ_roll, -τ_pitch, +τ_yaw]`

### A.2. Distinguishing Physical Vectors from Kinematic Quantities

While omega is geometrically a vector in 3D space, in the context of attitude representation:
- **Pure vector transform:** Would apply T @ omega = [p, -q, -r]
- **Benchmark consistency (used in BLUR↔SCG validation):** Requires `[-p, -q, +r]` for agreement with SCG/PyBullet reference data

If you're comparing angular velocity values between BLUR and SCG, remember that the same physical rotation rate will have **all** omega components negated between the two systems.

### A.3. Forgetting Velocity Frame Conversion

BLUR stores body-frame velocity; SCG stores world-frame velocity. Conversion requires:

```python
# BLUR body velocity → SCG world velocity
vel_world_ned = rotate_vector(quat, vel_body)
vx_scg = vel_world_ned[0]
vy_scg = vel_world_ned[1]
vz_scg = -vel_world_ned[2]  # NED → z-up

# SCG world velocity → BLUR body velocity
vel_world_ned = jnp.array([vx_scg, vy_scg, -vz_scg])
vel_body = rotate_vector_inverse(quat, vel_world_ned)
```

### A.4. Sign of Gravity

| System | Gravity Expression | Notes |
|--------|-------------------|-------|
| BLUR (NED) | g = +9.8 m/s² in +D | Down is positive |
| SCG (z-up) | g = -9.8 m/s² in z | Up is positive, gravity points down |

The magnitude is the same; only the sign convention differs.

---

## 7. Quaternion Format Conventions (Phase 11.5)

### 7.1. Quaternion Ordering

| System | Format | Order | Notes |
|--------|--------|-------|-------|
| BLUR | Scalar-FIRST | `[w, x, y, z]` | Standard aerospace convention |
| PyBullet/SCG | Scalar-LAST | `[x, y, z, w]` | Graphics/game engine convention |
| Eigen (C++) | Scalar-LAST | `[x, y, z, w]` | Internal storage order |
| ROS | Scalar-LAST | `[x, y, z, w]` | tf2_ros message format |

### 7.2. Quaternion Comparison Methodology

When comparing orientations between BLUR and PyBullet/SCG, use geodesic distance instead of comparing Euler angles:

```python
def quaternion_geodesic_distance(q1, q2):
    """Geodesic distance between quaternions (handles q/-q ambiguity)."""
    dot = jnp.abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    return 2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0))
```

**Key advantages:**
- Handles q/-q ambiguity automatically (same rotation, opposite quaternions)
- Returns angle in radians [0, π]
- No gimbal lock issues
- More numerically stable than Euler comparison

### 7.3. Quaternion Conversion Between Frames

The conversion between PyBullet and BLUR quaternions was derived empirically by comparing BLUR simulation output against SCG/PyBullet reference data (Phase 11.5 Step 5):

```python
def pybullet_quat_to_blur_quat(q_pyb):
    """Convert [x,y,z,w] PyBullet to [w,x,y,z] BLUR."""
    # Reorder and apply sign transforms [w, -x, -y, z]:
    return jnp.array([
        q_pyb[3],   # w unchanged
        -q_pyb[0],  # x negated
        -q_pyb[1],  # y negated
        q_pyb[2],   # z unchanged
    ])
```

**Note:** The earlier theoretical derivation predicted `[w, -x, y, -z]` based on roll/yaw Euler angle sign flips, but empirical testing with pure rotation scenarios showed `[w, -x, -y, z]` is correct. The difference comes from the complex relationship between quaternion components and Euler angles in the ZYX convention.

### 7.4. Why Euler Angles as Derived View

For benchmark comparison, Euler angles should be computed from BLUR quaternions as a **derived view**, not used as an intermediate in frame transforms:

1. **Gimbal lock:** Euler angles have singularities at ±90° pitch
2. **Numerical precision:** Quaternion-to-Euler-to-quaternion introduces error
3. **Ambiguity:** Multiple Euler representations for same rotation

The recommended comparison pipeline:
1. Get raw quaternion from PyBullet (saved in reference data)
2. Convert to BLUR frame using `pybullet_quat_to_blur_quat()`
3. Compare using `quaternion_geodesic_distance()`
4. Compute Euler angles from BLUR quaternion **only** for display

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-17 | Claude | Initial creation (Phase 11.5 Step 0) |
| 2026-01-17 | Claude | Added Section 7: Quaternion Format Conventions (Phase 11.5 Step 2) |
| 2026-01-17 | Claude | Corrected transforms based on empirical testing (Phase 11.5 Step 5): Euler `[-φ,-θ,+ψ]`, Omega `[-p,-q,+r]`, Torque `[-τ_roll,-τ_pitch,+τ_yaw]`, Quaternion `[w,-x,-y,z]` |
