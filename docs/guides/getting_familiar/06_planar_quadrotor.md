# Planar Quadrotor (2D Flight Dynamics)

**Estimated reading time: 1 hour**

This section introduces the planar quadrotor model, a 2D simplification of quadrotor flight dynamics. Operating in the x-z plane with pitch rotation, the planar quadrotor captures the essential physics of flight control while remaining tractable for analysis and visualization.

---

## Related Documentation

- [Frame Conventions](../../frame_conventions.md) - Coordinate frame definitions
- [Simulator Models](../../simulator_models.md) - Complete model documentation
- [Core Concepts](02_core_concepts.md) - State vectors and SI units
- [Control Guide](../../control_guide.md) - LQR design and timestep selection

---

## Overview

The planar quadrotor is a 6-state, 2-control system that models a quadrotor constrained to the x-z (vertical) plane. This model is useful for:

- Understanding fundamental flight dynamics before tackling 3D
- Control algorithm development and testing
- Visualization and debugging
- Validation against benchmark environments (Safe Control Gym, PyBullet)

**Why study 2D before 3D?** The planar quadrotor eliminates roll and yaw dynamics, quaternion attitude representation, and 3D Coriolis terms. This allows you to focus on the essential physics: thrust produces acceleration, differential thrust produces rotation, and gravity pulls down.

---

## State and Control Vectors

### State Vector (6 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | Horizontal position | m |
| 1 | z | Vertical position (positive = up) | m |
| 2 | theta | Pitch angle (positive = nose up) | rad |
| 3 | x_dot | Horizontal velocity | m/s |
| 4 | z_dot | Vertical velocity | m/s |
| 5 | theta_dot | Pitch rate | rad/s |

**Coordinate convention:** The planar quadrotor uses a right-hand coordinate system where:
- **x** is horizontal (positive = right/forward)
- **z** is vertical (positive = up)
- **theta** is pitch angle (positive = nose up, counterclockwise rotation when viewed from the right)

This differs from the NED convention used by the 3D quadrotor. In NED, down is positive. The 2D model uses the more intuitive "up is positive" convention.

### Control Vector (2 elements)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | T1 | Right rotor thrust | N |
| 1 | T2 | Left rotor thrust | N |

The two rotors are positioned symmetrically at distance `arm_length` from the center of mass.

---

## Physics Model

### Equations of Motion

The planar quadrotor dynamics are governed by three coupled equations:

**Translational dynamics (Newton's second law):**

$$\ddot{x} = -\frac{T}{m} \sin(\theta)$$

$$\ddot{z} = \frac{T}{m} \cos(\theta) - g$$

**Rotational dynamics (Euler's equation):**

$$\ddot{\theta} = \frac{M}{I}$$

where:
- $T = T_1 + T_2$ is the total thrust
- $M = (T_1 - T_2) \cdot L$ is the pitch moment
- $L$ is the arm length (rotor to center of mass)
- $I$ is the pitch moment of inertia
- $m$ is the vehicle mass
- $g$ is gravitational acceleration

### Physical Interpretation

**Thrust direction:** Both rotors produce thrust along the body's negative z-axis (upward in the body frame). When the vehicle is level, this thrust points straight up. When tilted, the thrust has horizontal and vertical components.

**Pitch control:** Differential thrust creates a moment about the center of mass:
- More thrust on T1 (right rotor) produces positive moment (nose up)
- More thrust on T2 (left rotor) produces negative moment (nose down)

**Coupling:** Translation and rotation are coupled through the pitch angle. To move horizontally, you must first tilt, which redirects thrust. This coupling is fundamental to quadrotor control.

---

## Creating a Planar Quadrotor

### Using Parameter Presets

BLUR provides several preset configurations:

```python
from fmd.simulator import PlanarQuadrotor
from fmd.simulator.params import (
    PLANAR_QUAD_TEST_DEFAULT,  # 1kg, easy calculations
    PLANAR_QUAD_CRAZYFLIE,     # Crazyflie nano quadcopter
    PLANAR_QUAD_HEAVY,         # 2kg photography drone
    PLANAR_QUAD_SCG,           # Safe Control Gym compatible
)

# Create quadrotor with test parameters
quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

print(f"Mass: {quad.mass} kg")
print(f"Arm length: {quad.arm_length} m")
print(f"Pitch inertia: {quad.inertia_pitch} kg*m^2")
print(f"Gravity: {quad.g} m/s^2")
```

**Preset Properties:**

| Preset | Mass (kg) | Arm (m) | Inertia (kg*m^2) | Use Case |
|--------|-----------|---------|------------------|----------|
| `PLANAR_QUAD_TEST_DEFAULT` | 1.0 | 0.25 | 0.01 | Testing, easy mental math |
| `PLANAR_QUAD_CRAZYFLIE` | 0.030 | 0.0397 | 1.4e-5 | Crazyflie validation |
| `PLANAR_QUAD_HEAVY` | 2.0 | 0.30 | 0.04 | Stress testing |
| `PLANAR_QUAD_SCG` | 0.027 | 0.0397 | 1.4e-5 | Safe Control Gym parity |

### Custom Parameters

```python
from fmd.simulator.params import PlanarQuadrotorParams

params = PlanarQuadrotorParams(
    mass=0.5,           # kg
    arm_length=0.15,    # m
    inertia_pitch=0.005 # kg*m^2
)
quad = PlanarQuadrotor(params)
```

### JAX-Traceable Construction

For gradient-based optimization through parameters:

```python
# Use from_values() to avoid attrs validation (allows JAX tracing)
quad = PlanarQuadrotor.from_values(
    mass=1.0,
    arm_length=0.25,
    inertia_pitch=0.01,
)
```

---

## Hover Equilibrium

### The Hover Condition

At hover, the quadrotor maintains a fixed position with zero velocity and zero pitch. This requires:

1. **Zero net vertical force:** Total thrust equals weight
2. **Zero net moment:** Equal thrust from both rotors
3. **Level attitude:** Pitch angle is zero

**Equilibrium control:**

$$T_1 = T_2 = \frac{mg}{2}$$

**Verification:**

```python
from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl
from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
import jax.numpy as jnp

quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

# Verify hover thrust
T_hover = quad.hover_thrust_per_rotor()
print(f"Hover thrust per rotor: {T_hover:.4f} N")
print(f"Total hover thrust: {quad.hover_thrust_total():.4f} N")
print(f"Vehicle weight: {quad.mass * quad.g:.4f} N")

# Verify equilibrium (state derivative is zero)
state = quad.default_state()  # All zeros
control = quad.hover_control()  # [T_hover, T_hover]
deriv = quad.forward_dynamics(state, control)

print(f"\nState derivative at hover: {deriv}")
# Should be all zeros (within numerical precision)
```

### Equilibrium Stability

Hover is an **unstable equilibrium**. Any small perturbation (e.g., a gust of wind tilting the vehicle) will cause it to drift and eventually crash without active control. This is why quadrotors require continuous feedback control to maintain hover.

**Why unstable?** When tilted, the thrust has a horizontal component but the control is unchanged. The horizontal force accelerates the vehicle sideways, and without correction, the situation worsens.

---

## Running Simulations

### Basic Simulation

```python
from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl
from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT
import jax.numpy as jnp

quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)

# Start at hover
initial_state = quad.default_state()
control = ConstantControl(quad.hover_control())

# Simulate for 5 seconds at 1000 Hz
result = simulate(
    quad,
    initial_state,
    dt=0.001,
    duration=5.0,
    control=control
)

print(f"Simulation: {len(result.times)} timesteps")
print(f"Final state: {result.states[-1]}")
```

### Creating Custom States and Controls

```python
# Create specific initial conditions
state = quad.create_state(
    x=0.0,          # Start at origin
    z=2.0,          # 2 meters altitude
    theta=0.1,      # Slight pitch (nose up)
    x_dot=0.0,
    z_dot=0.0,
    theta_dot=0.0
)

# Create specific control
control = quad.create_control(
    T1_val=5.0,  # Right rotor thrust
    T2_val=5.0   # Left rotor thrust
)
```

### Freefall Simulation

With zero thrust, the quadrotor undergoes freefall:

```python
from fmd.simulator import ZeroControl

# Start at 10m altitude
initial = quad.create_state(z=10.0)

# No thrust
result = simulate(
    quad,
    initial,
    dt=0.001,
    duration=1.0,
    control=ZeroControl()
)

# Verify freefall physics: z = z0 - 0.5*g*t^2
t = result.times[-1]
expected_z = 10.0 - 0.5 * quad.g * t**2
actual_z = float(result.states[-1, 1])  # z is index 1

print(f"Expected z: {expected_z:.4f} m")
print(f"Actual z: {actual_z:.4f} m")
print(f"Error: {abs(actual_z - expected_z) * 1000:.4f} mm")
```

---

## Energy Analysis

### Energy Computation

The planar quadrotor model includes energy calculation methods:

```python
# Total mechanical energy
state = quad.create_state(z=5.0, x_dot=2.0, z_dot=1.0, theta_dot=0.5)
total_energy = quad.energy(state)
kinetic_energy = quad.kinetic_energy(state)
potential_energy = quad.potential_energy(state)

print(f"Kinetic energy: {kinetic_energy:.4f} J")
print(f"Potential energy: {potential_energy:.4f} J")
print(f"Total energy: {total_energy:.4f} J")
```

**Energy components:**

$$KE = \frac{1}{2}m(\dot{x}^2 + \dot{z}^2) + \frac{1}{2}I\dot{\theta}^2$$

$$PE = mgz$$

$$E = KE + PE$$

### Energy Conservation in Freefall

Without thrust, mechanical energy is conserved:

```python
import jax

# Start with potential energy
initial = quad.create_state(z=10.0)
result = simulate(quad, initial, dt=0.0001, duration=1.0)

# Compute energy at each timestep
energies = jax.vmap(quad.energy)(result.states)
initial_energy = float(energies[0])

# Check conservation
max_deviation = float(jnp.max(jnp.abs(energies - initial_energy)))
print(f"Initial energy: {initial_energy:.4f} J")
print(f"Max energy deviation: {max_deviation:.6f} J")
print(f"Relative error: {max_deviation / initial_energy:.2e}")
```

### Power Balance

Power is the rate of energy change. The thrust power equals the dot product of thrust force and velocity:

```python
state = quad.create_state(z_dot=2.0)  # Moving upward
control = quad.hover_control()

# Power delivered by thrust
P_thrust = quad.power_thrust(state, control)

# Power removed by gravity
P_gravity = quad.power_gravity(state)

print(f"Thrust power: {P_thrust:.4f} W")
print(f"Gravity power: {P_gravity:.4f} W")
print(f"Net power: {P_thrust + P_gravity:.4f} W")
```

**At hover equilibrium with vertical motion:**
- Moving up: Thrust does positive work, gravity does negative work
- Moving down: Thrust does negative work, gravity does positive work
- At hover with zero velocity: No mechanical power transfer

---

## Control Scenarios

### Differential Thrust (Rotation)

To rotate without changing vertical motion:

```python
T_hover = quad.hover_thrust_per_rotor()

# Equal total thrust (hover), but differential
# T1 > T2 produces positive moment (nose up)
delta_T = 0.5  # N
control = jnp.array([T_hover + delta_T, T_hover - delta_T])

initial = quad.default_state()
result = simulate(
    quad,
    initial,
    dt=0.001,
    duration=0.5,
    control=ConstantControl(control)
)

# Check rotation
print(f"Final pitch: {jnp.degrees(result.states[-1, 2]):.2f} degrees")
print(f"Final pitch rate: {jnp.degrees(result.states[-1, 5]):.2f} deg/s")
```

**Expected behavior:** The quadrotor rotates at constant angular acceleration:

$$\ddot{\theta} = \frac{(T_1 - T_2) \cdot L}{I} = \frac{\Delta T \cdot L}{I}$$

### Tilted Hover

When tilted but applying hover thrust:

```python
theta_initial = 0.2  # radians (~11.5 degrees)
initial = quad.create_state(theta=theta_initial)

result = simulate(
    quad,
    initial,
    dt=0.001,
    duration=1.0,
    control=ConstantControl(quad.hover_control())
)

print(f"Final x: {float(result.states[-1, 0]):.4f} m")
print(f"Final z: {float(result.states[-1, 1]):.4f} m")
```

**What happens?**
1. Thrust has horizontal component: $F_x = -T \sin(\theta)$
2. Thrust has reduced vertical component: $F_z = T \cos(\theta) < mg$
3. Vehicle accelerates horizontally (negative x direction)
4. Vehicle descends because $T \cos(\theta) < mg$

---

## Utility Methods

### Flight Path Analysis

```python
# State with horizontal and vertical velocity
state = quad.create_state(x_dot=5.0, z_dot=3.0, theta=0.2)

# Speed magnitude
speed = quad.speed(state)
print(f"Speed: {speed:.4f} m/s")

# Flight path angle (angle of velocity vector)
gamma = quad.flight_path_angle(state)
print(f"Flight path angle: {jnp.degrees(gamma):.2f} degrees")

# Angle of attack (pitch minus flight path angle)
alpha = quad.angle_of_attack(state)
print(f"Angle of attack: {jnp.degrees(alpha):.2f} degrees")
```

---

---

## Key Takeaways

1. **6 states, 2 controls:** Position (x, z), pitch (theta), and their derivatives; two rotor thrusts

2. **Hover equilibrium:** T1 = T2 = mg/2 is unstable; requires active control

3. **Coupled dynamics:** Translation requires rotation; you must tilt to move horizontally

4. **Energy analysis:** Kinetic + potential energy; conserved in freefall

5. **Power balance:** Thrust power = F dot v; gravity power = -mg*z_dot

6. **Validation:** Matches Safe Control Gym and PyBullet within established tolerances

---

## Companion Notebook

For hands-on exploration, see the companion notebook `notebooks/getting_familiar/gf_06_planar_quadrotor.ipynb` which covers:
- Hover stabilization and equilibrium verification
- Thrust response and differential thrust
- Energy and power analysis plots
- Freefall validation against analytical solutions

---

## What's Next

- Validation Methodology (see `docs/public/dev/testing.md`) - Tolerance hierarchy and cross-library validation
- [Control Guide](../../control_guide.md) - LQR design, eigenvalue analysis, and timestep selection
