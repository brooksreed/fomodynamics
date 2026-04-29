# Core Concepts

This section introduces the foundational conventions and mathematical representations used throughout fomodynamics. Understanding these concepts is essential before working with any of the simulator models.

**Estimated reading time: 1 hour**

---

## Related Documentation

- [Frame Conventions (authoritative reference)](../../frame_conventions.md) - Complete coordinate frame documentation with transform derivations
- [README — Core conventions](../../../../README.md#core-conventions) - Frames, units, quaternions, sign gotchas
- [Simulator Models](../../simulator_models.md) - Detailed model documentation and parameter classes

---

## Table of Contents

1. [Why These Conventions?](#1-why-these-conventions)
2. [Frame Conventions](#2-frame-conventions)
3. [State Vectors](#3-state-vectors)
4. [Quaternion Attitude Representation](#4-quaternion-attitude-representation)
5. [SI Units Internally](#5-si-units-internally)
6. [Circular-Aware Math](#6-circular-aware-math)
7. [Putting It Together](#7-putting-it-together)

---

## 1. Why These Conventions?

fomodynamics uses aerospace-standard conventions for several reasons:

1. **Consistency with literature:** Most flight dynamics textbooks and control papers use NED/FRD frames. This makes it easier to implement algorithms from published research.

2. **Physical intuition:** The conventions align with how pilots and sailors think: "nose up" is positive pitch, "right wing down" is positive roll, "turn right" is positive yaw.

3. **Numerical stability:** Quaternions avoid gimbal lock that plagues Euler angles at extreme pitch attitudes (common in acrobatic flight or hydrofoiling).

4. **Interoperability:** SI units internally with conversions for display means data flows cleanly between fomodynamics packages and external tools.

These choices may feel unfamiliar if you come from robotics (which often uses z-up frames) or game development (which often uses different quaternion conventions). The [Frame Conventions](../../frame_conventions.md) document provides complete transform derivations for cross-library validation.

---

## 2. Frame Conventions

fomodynamics uses two primary coordinate frames:

### 2.1. World Frame: NED (North-East-Down)

The **NED** frame is an inertial reference frame fixed to the Earth's surface:

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

**Key implications:**

- **Gravity** acts in the +D direction: `g = [0, 0, +9.81]` m/s^2
- **Altitude increase** means `pos_d` *decreases* (becomes more negative)
- A vehicle at sea level has `pos_d = 0`; a vehicle 10m above sea level has `pos_d = -10`

### 2.2. Body Frame: FRD (Forward-Right-Down)

The **FRD** frame is fixed to the vehicle and moves with it:

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

| Axis | Direction | Rotation Name | Angular Rate |
|------|-----------|---------------|--------------|
| X | Forward (nose) | Roll axis | p |
| Y | Right (starboard) | Pitch axis | q |
| Z | Down (belly) | Yaw axis | r |

**Positive rotations** follow the right-hand rule about each axis:

| Rotation | Positive Direction | Physical Meaning |
|----------|-------------------|------------------|
| Roll (+phi) | Right wing down | Banking into a right turn |
| Pitch (+theta) | Nose up | Climbing |
| Yaw (+psi) | Nose right | Turning right (clockwise from above) |

### 2.3. Why NED/FRD?

The NED/FRD convention is standard in:

- **Aerospace:** FAA, NASA, most flight dynamics textbooks
- **Marine:** Fossen's marine dynamics texts, most ship autopilot systems
- **Military:** MIL-HDBK standards, autopilot specifications

Alternative conventions you may encounter:

| System | World Frame | Body Frame | Example Users |
|--------|-------------|------------|---------------|
| fomodynamics | NED | FRD | Aerospace, marine |
| Robotics | z-up | FLU | ROS, Drake |
| OpenGL | y-up | varies | Game engines |

See the [Frame Conventions](../../frame_conventions.md) document for the authoritative reference on fomodynamics's frames.

---

## 3. State Vectors

fomodynamics's 6-DOF rigid body simulator uses a 13-element state vector:

### 3.1. State Vector Structure

| Index | Name | Description | Frame | Units |
|-------|------|-------------|-------|-------|
| 0 | `pos_n` | North position | NED world | m |
| 1 | `pos_e` | East position | NED world | m |
| 2 | `pos_d` | Down position | NED world | m |
| 3 | `vel_u` | Forward velocity | FRD body | m/s |
| 4 | `vel_v` | Right velocity | FRD body | m/s |
| 5 | `vel_w` | Down velocity | FRD body | m/s |
| 6 | `qw` | Quaternion scalar | - | - |
| 7 | `qx` | Quaternion x | - | - |
| 8 | `qy` | Quaternion y | - | - |
| 9 | `qz` | Quaternion z | - | - |
| 10 | `omega_p` | Roll rate | FRD body | rad/s |
| 11 | `omega_q` | Pitch rate | FRD body | rad/s |
| 12 | `omega_r` | Yaw rate | FRD body | rad/s |

### 3.2. Why Body-Frame Velocities?

fomodynamics stores linear velocity in the body frame (not world frame) because:

1. **Forces are computed in body frame:** Lift, drag, thrust act along body axes
2. **Simpler equations of motion:** No rotation matrix in the velocity dynamics
3. **Physical intuition:** Pilots and sailors think in terms of forward speed, sideslip, and climb rate

To get world-frame velocity, rotate body velocity by the attitude quaternion:

```python
from fmd.core import rotate_vector

# Body-frame velocity
vel_body = state[3:6]  # [vel_u, vel_v, vel_w]

# Attitude quaternion
quat = state[6:10]  # [qw, qx, qy, qz]

# World-frame velocity
vel_world = rotate_vector(quat, vel_body)  # [vel_n, vel_e, vel_d]
```

### 3.3. Derived Outputs

Euler angles are **derived** from the quaternion, not stored in the state:

```python
from fmd.core import quat_to_euler
import numpy as np

# Get Euler angles from state
quat = state[6:10]
euler = quat_to_euler(quat)  # Returns [roll, pitch, yaw] in radians

# Convert to degrees for display
roll_deg = np.degrees(euler[0])
pitch_deg = np.degrees(euler[1])
yaw_deg = np.degrees(euler[2])
```

Euler angles are computed using ZYX convention (yaw-pitch-roll sequence).

---

## 4. Quaternion Attitude Representation

### 4.1. Why Quaternions?

fomodynamics uses quaternions instead of Euler angles for attitude representation:

| Issue | Euler Angles | Quaternions |
|-------|--------------|-------------|
| Gimbal lock | Yes (at +/-90 deg pitch) | No |
| Interpolation | Non-linear, can twist | Linear (SLERP) |
| Composition | Three trig operations | One multiplication |
| Numerical drift | Accumulates | Easy to renormalize |
| Singularities | Multiple | None (q and -q are same) |

**Gimbal lock** is particularly problematic for:
- Acrobatic flight (loops, rolls)
- Hydrofoiling boats (can pitch steeply)
- Quadrotors recovering from flips

### 4.2. Scalar-First Convention

fomodynamics uses **scalar-first** quaternion ordering: `q = [qw, qx, qy, qz]`

| Component | Name | Interpretation |
|-----------|------|----------------|
| `qw` | Scalar | `cos(theta/2)` where theta is rotation angle |
| `qx` | Vector X | `sin(theta/2) * axis_x` |
| `qy` | Vector Y | `sin(theta/2) * axis_y` |
| `qz` | Vector Z | `sin(theta/2) * axis_z` |

The **identity quaternion** (no rotation, body aligned with world) is:

```python
q_identity = [1, 0, 0, 0]  # qw=1, qx=qy=qz=0
```

**Other conventions you may encounter:**

| Library | Order | Notes |
|---------|-------|-------|
| fomodynamics | [w, x, y, z] | Scalar-first (aerospace standard) |
| ROS | [x, y, z, w] | Scalar-last |
| Eigen (C++) | [x, y, z, w] | Scalar-last (internal storage) |
| Most game engines | [x, y, z, w] | Scalar-last |

### 4.3. Core Quaternion Operations

The `fmd.core` module provides quaternion utilities:

```python
from fmd.core import (
    quat_multiply,      # Hamilton product: q1 * q2
    quat_conjugate,     # Inverse for unit quaternions
    quat_normalize,     # Ensure unit length
    quat_derivative,    # Compute dq/dt from angular velocity
    rotate_vector,      # Rotate vector from body to world frame
    rotate_vector_inverse,  # Rotate vector from world to body frame
    identity_quat,      # Return [1, 0, 0, 0]
    quaternion_distance,    # Geodesic distance between orientations
)
```

**Example: Rotate a body-frame vector to world frame**

```python
import numpy as np
from fmd.core import rotate_vector, euler_to_quat

# Vehicle pitched up 30 degrees
pitch_rad = np.radians(30)
quat = euler_to_quat([0, pitch_rad, 0])  # [roll, pitch, yaw]

# Unit vector pointing forward in body frame
forward_body = np.array([1, 0, 0])

# Where does "forward" point in world frame?
forward_world = rotate_vector(quat, forward_body)
print(f"Forward in world: N={forward_world[0]:.3f}, E={forward_world[1]:.3f}, D={forward_world[2]:.3f}")
# Output: Forward in world: N=0.866, E=0.000, D=-0.500
# (pointing north and up, as expected for nose-up pitch)
```

### 4.4. Quaternion Dynamics

The quaternion derivative is computed from angular velocity:

```
dq/dt = 0.5 * omega_quat * q
```

where `omega_quat = [0, p, q, r]` is the angular velocity as a pure quaternion.

```python
from fmd.core import quat_derivative
import numpy as np

# Current attitude (identity)
quat = np.array([1.0, 0.0, 0.0, 0.0])

# Angular velocity: rolling at 1 rad/s
omega = np.array([1.0, 0.0, 0.0])  # [p, q, r]

# Quaternion derivative
dq_dt = quat_derivative(quat, omega)
print(f"dq/dt = {dq_dt}")
# Output: dq/dt = [0.  0.5 0.  0. ]
```

### 4.5. Quaternion to Rotation Matrix

The rotation matrix R transforms vectors from body frame to world frame:

```
v_world = R @ v_body
```

```python
from fmd.core import quat_to_dcm
import numpy as np

# 45-degree yaw (heading northeast)
quat = np.array([0.924, 0, 0, 0.383])  # cos(22.5 deg), 0, 0, sin(22.5 deg)

# Direction cosine matrix
R = quat_to_dcm(quat)

# Forward in body frame -> world frame
forward_world = R @ np.array([1, 0, 0])
print(f"Heading: N={forward_world[0]:.3f}, E={forward_world[1]:.3f}")
# Output: Heading: N=0.707, E=0.707 (northeast)
```

---

## 5. SI Units Internally

### 5.1. The SI Principle

**All data is stored in SI units internally. Conversion happens only for display.**

| Quantity | Internal Unit | Display Unit |
|----------|---------------|--------------|
| Position | meters (m) | meters |
| Velocity | meters/second (m/s) | knots (for marine) |
| Angle | radians (rad) | degrees |
| Angular velocity | radians/second (rad/s) | degrees/second |
| Mass | kilograms (kg) | kilograms |
| Force | newtons (N) | newtons |
| Torque | newton-meters (Nm) | newton-meters |

### 5.2. Why SI Internally?

1. **Physics equations work directly:** F=ma uses kg, m/s^2, N
2. **No conversion errors:** Calculations stay in one unit system
3. **Trigonometry works:** `sin()`, `cos()` expect radians
4. **Interoperability:** Most scientific libraries (NumPy, JAX, SciPy) assume SI

### 5.3. Unit Conversion Utilities

The `fmd.core.units` module provides conversion functions:

```python
from fmd.core import convert_to_si, convert_from_si

# Speed: knots to m/s (internal storage)
speed_knots = 20.0
speed_ms = convert_to_si(speed_knots, "speed")  # 10.29 m/s

# Angle: degrees to radians (internal storage)
heading_deg = 45.0
heading_rad = convert_to_si(heading_deg, "heading")  # 0.785 rad

# For display: convert back
speed_display = convert_from_si(speed_ms, "speed")  # 20.0 knots
heading_display = convert_from_si(heading_rad, "heading")  # 45.0 degrees
```

### 5.4. Common Conversions Reference

| Quantity | To SI | From SI |
|----------|-------|---------|
| Knots to m/s | x * 0.51444 | x / 0.51444 |
| Degrees to radians | x * pi/180 | x * 180/pi |
| RPM to rad/s | x * 2*pi/60 | x * 60/(2*pi) |

**Best practice:** Keep all calculations in SI. Only convert when:
- Reading user input (e.g., "set heading to 45 degrees")
- Displaying results (e.g., "speed: 20 knots")
- Interfacing with external systems that require specific units

---

## 6. Circular-Aware Math

### 6.1. The Problem with Angles

Angles wrap around. Consider computing the difference between two headings:

```python
# Naive subtraction
heading1 = 359  # degrees
heading2 = 1    # degrees
diff = heading1 - heading2  # = 358 degrees (WRONG!)
# Should be -2 degrees (359 is 2 degrees counter-clockwise from 1)
```

This matters for:
- Heading error in autopilots
- Angular velocity estimation from position data
- Filtering and interpolation of angle time series

### 6.2. Circular Subtraction

The `circular_subtract` function computes the shortest angular difference:

```python
from fmd.core import circular_subtract
import numpy as np

# Convert to radians (SI units internally)
heading1 = np.radians(359)
heading2 = np.radians(1)

# Circular subtraction: 359 deg - 1 deg
diff = circular_subtract(heading1, heading2)
print(f"Difference: {np.degrees(diff):.1f} degrees")
# Output: Difference: -2.0 degrees

# The other direction: 1 deg - 359 deg
diff2 = circular_subtract(heading2, heading1)
print(f"Difference: {np.degrees(diff2):.1f} degrees")
# Output: Difference: 2.0 degrees
```

### 6.3. Wrapping Angles

The `wrap_angle` function constrains angles to a specified range:

```python
from fmd.core import wrap_angle
import numpy as np

# Default range: [-pi, pi)
angle = np.radians(270)  # 270 degrees
wrapped = wrap_angle(angle)
print(f"270 deg wrapped: {np.degrees(wrapped):.1f} degrees")
# Output: 270 deg wrapped: -90.0 degrees

# Multiple angles at once
angles = np.radians([0, 90, 180, 270, 360, 450])
wrapped = wrap_angle(angles)
print(f"Wrapped: {np.degrees(wrapped)}")
# Output: Wrapped: [  0.  90. -180. -90.  0.  90.]
```

### 6.4. Circular Mean

The `circular_mean` function computes the mean direction of angles:

```python
from fmd.core import circular_mean
import numpy as np

# Mean of 350 deg and 10 deg
angles = np.radians([350, 10])
mean = circular_mean(angles)
print(f"Mean of 350 and 10: {np.degrees(mean):.1f} degrees")
# Output: Mean of 350 and 10: 0.0 degrees (correct!)

# Naive arithmetic mean would give: (350 + 10) / 2 = 180 degrees (WRONG!)
```

### 6.5. Available Circular Operations

```python
from fmd.core import (
    wrap_angle,              # Constrain angle to range
    unwrap_angle,            # Remove discontinuities from angle sequence
    circular_subtract,       # Compute shortest angular difference
    circular_mean,           # Compute mean direction
    angle_difference_to_vector,  # Convert heading difference to lateral/longitudinal
)
```

### 6.6. When to Use Circular Math

| Operation | Use Circular Math? | Function |
|-----------|-------------------|----------|
| Heading error | Yes | `circular_subtract` |
| Average heading | Yes | `circular_mean` |
| Filtering angles | Unwrap first | `unwrap_angle` then filter |
| Quaternion interpolation | Use SLERP | (quaternions don't wrap) |
| Roll/pitch near limits | Usually not | Typically stay in [-pi/2, pi/2] |

---

## 7. Putting It Together

Here is a complete example that demonstrates all core concepts:

```python
"""Core concepts demonstration: Creating and interpreting a 6-DOF state."""

import numpy as np
from fmd.core import (
    euler_to_quat,
    quat_to_euler,
    rotate_vector,
    circular_subtract,
    convert_to_si,
    convert_from_si,
)

# ============================================================
# 1. Define initial state in human-readable units
# ============================================================

# Position: 100m north, 50m east, 10m altitude (remember: altitude = -pos_d)
pos_n = 100.0  # meters
pos_e = 50.0   # meters
pos_d = -10.0  # meters (negative = above sea level)

# Velocity: 15 knots forward, no sideslip, no climb
speed_knots = 15.0
vel_u = convert_to_si(speed_knots, "speed")  # Convert to m/s
vel_v = 0.0  # No sideslip (m/s)
vel_w = 0.0  # No vertical speed (m/s)

# Attitude: 5 deg roll, 10 deg pitch up, heading 045 (northeast)
roll_deg = 5.0
pitch_deg = 10.0
yaw_deg = 45.0

# Convert to radians and then to quaternion
euler_rad = np.array([
    np.radians(roll_deg),
    np.radians(pitch_deg),
    np.radians(yaw_deg),
])
quat = euler_to_quat(euler_rad)

# Angular velocity: gentle right turn at 3 deg/s
omega_p = 0.0  # No roll rate
omega_q = 0.0  # No pitch rate
omega_r = np.radians(3.0)  # 3 deg/s yaw rate

# ============================================================
# 2. Assemble the 13-element state vector
# ============================================================

state = np.array([
    pos_n, pos_e, pos_d,           # Position (NED)
    vel_u, vel_v, vel_w,           # Velocity (body frame)
    quat[0], quat[1], quat[2], quat[3],  # Quaternion (scalar-first)
    omega_p, omega_q, omega_r,     # Angular velocity (body frame)
])

print("State vector (13 elements):")
print(state)

# ============================================================
# 3. Extract and interpret state components
# ============================================================

# Position
position_ned = state[0:3]
altitude = -position_ned[2]  # Convert pos_d to altitude
print(f"\nPosition: N={position_ned[0]:.1f}m, E={position_ned[1]:.1f}m, Alt={altitude:.1f}m")

# Velocity in body frame
velocity_body = state[3:6]
speed_ms = np.linalg.norm(velocity_body)
speed_display = convert_from_si(speed_ms, "speed")
print(f"Speed: {speed_display:.1f} knots ({speed_ms:.2f} m/s)")

# Attitude: quaternion to Euler
attitude_quat = state[6:10]
euler_out = quat_to_euler(attitude_quat)
print(f"Attitude: Roll={np.degrees(euler_out[0]):.1f} deg, "
      f"Pitch={np.degrees(euler_out[1]):.1f} deg, "
      f"Yaw={np.degrees(euler_out[2]):.1f} deg")

# Angular velocity
omega = state[10:13]
print(f"Turn rate: {np.degrees(omega[2]):.1f} deg/s")

# ============================================================
# 4. Compute derived quantities
# ============================================================

# World-frame velocity
vel_world = rotate_vector(attitude_quat, velocity_body)
print(f"\nWorld velocity: N={vel_world[0]:.2f} m/s, E={vel_world[1]:.2f} m/s, D={vel_world[2]:.2f} m/s")

# Heading error to a target
target_heading = np.radians(90)  # Due east
current_heading = euler_out[2]   # Current yaw
heading_error = circular_subtract(target_heading, current_heading)
print(f"Heading error to due east: {np.degrees(heading_error):.1f} deg")
```

**Expected output:**

```
State vector (13 elements):
[  100.            50.           -10.             7.71666667     0.
     0.             0.99144486     0.01918082     0.09406228     0.08418598
     0.             0.             0.05235988]

Position: N=100.0m, E=50.0m, Alt=10.0m
Speed: 15.0 knots (7.72 m/s)
Attitude: Roll=5.0 deg, Pitch=10.0 deg, Yaw=45.0 deg
Turn rate: 3.0 deg/s

World velocity: N=5.27 m/s, E=5.27 m/s, D=-1.34 m/s
Heading error to due east: 45.0 deg
```

---

## Summary

| Concept | fomodynamics Convention | Key Point |
|---------|----------------|-----------|
| World frame | NED (North-East-Down) | +D is down, altitude = -pos_d |
| Body frame | FRD (Forward-Right-Down) | Forces computed here |
| Quaternion | Scalar-first [qw, qx, qy, qz] | Identity = [1, 0, 0, 0] |
| State vector | 13 elements | [pos, vel, quat, omega] |
| Units | SI internally | radians, m/s, kg |
| Angle math | Circular-aware | Use `circular_subtract` for differences |

**Next steps:**

- [03 - Simple Pendulum](03_simple_pendulum.md): Apply these concepts to your first simulation
- [06 - Planar Quadrotor](06_planar_quadrotor.md): See a 2D flight dynamics example
- [Frame Conventions](../../frame_conventions.md): Deep dive into coordinate transforms
