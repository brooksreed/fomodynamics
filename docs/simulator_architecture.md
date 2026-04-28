# Simulator Architecture

The `fmd.simulator` package provides 6-DOF (six degrees of freedom) dynamic system simulation for marine vehicles, with a focus on foiling sailboats.

**Related documentation:**
- [Simulator Models](simulator_models.md) - Reference for all public physics models (Boat2D, Cartpole, PlanarQuadrotor, SimplePendulum, Box1D, Moth3D)
- [JAX Simulator Guide](jax_simulator_guide.md) - JIT compilation, autodiff, control interface

## Features

- **JAX-Based**: JIT compilation for fast simulations, automatic differentiation for gradients
- **Pure Physics Engine**: No plotting or visualization - outputs raw simulation data
- **6-DOF Rigid Body Dynamics**: Full position, velocity, orientation (quaternion), and angular velocity
- **Force Accumulator Pattern**: Modular component-based architecture for forces and moments
- **RK4 Integration**: Classic 4th-order Runge-Kutta with `jax.lax.scan`
- **NED Frame**: Standard North-East-Down aerospace coordinate system
- **Analysis Integration**: Output format compatible with `fmd.analysis` package

## JAX and JIT Compilation

The simulator uses JAX for high-performance computing:

- **JIT Compilation**: First run compiles to optimized code, subsequent runs are fast
- **Autodiff**: Compute gradients through simulations for sensitivity analysis and optimization
- **Vectorization**: Use `jax.vmap` to run many simulations in parallel

For detailed JAX usage, see the [JAX Simulator Guide](jax_simulator_guide.md).

### Quick JAX Example

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M
import jax
import jax.numpy as jnp

# Gradient of final angle w.r.t. initial angle
def loss(initial_angle):
    pendulum = SimplePendulum(PENDULUM_1M)
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0] ** 2

sensitivity = jax.grad(loss)(0.3)
```

## Architecture

### State Vector (13 elements)

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0-2 | `pos_n, pos_e, pos_d` | Position in NED frame | m |
| 3-5 | `vel_u, vel_v, vel_w` | Velocity in body frame | m/s |
| 6-9 | `qw, qx, qy, qz` | Quaternion (scalar-first) | - |
| 10-12 | `omega_p, omega_q, omega_r` | Angular velocity in body frame | rad/s |

### Dynamic Systems

The simulator provides several pre-built dynamic systems:

- **`RigidBody6DOF`**: Full 6-DOF rigid body with quaternion attitude
- **`Boat2D`**: Planar rigid body with Coriolis coupling (6 states)
- **`PlanarQuadrotor`**: 2D quadrotor with pitch dynamics (6 states)
- **`Moth3D`**: Hydrofoiling Moth 3DOF longitudinal model (5 states)
- **`Cartpole`**: Classic 4-state inverted-pendulum benchmark
- **`SimplePendulum`**: 2-state pendulum for testing/validation

### Force Components

Forces and moments are computed by modular `ForceElement` components:

- **`Gravity`**: Constant gravitational force in NED frame
- **`LinearDrag`**: Velocity-proportional drag in body frame
- Custom components: Inherit from `JaxForceElement` base class

### Equations of Motion

```
ṗ = R(q) · v              Position derivative
v̇ = F/m - ω × v           Velocity derivative (Coriolis term)
q̇ = 0.5 · Ω ⊗ q           Quaternion derivative
ω̇ = I⁻¹(M - ω × Iω)       Angular velocity derivative (Euler's equation)
```

Where:
- `p` = position in NED frame
- `v` = velocity in body frame
- `q` = quaternion attitude (scalar-first)
- `ω` = angular velocity in body frame
- `R(q)` = rotation matrix from body to NED
- `F` = total force in body frame
- `M` = total moment in body frame
- `I` = inertia tensor (diagonal)

## Quick Start

```python
from fmd.simulator import RigidBody6DOF, simulate, create_state, Gravity
import jax.numpy as jnp

# Create a 1kg rigid body with gravity
mass = 1.0
body = RigidBody6DOF(
    mass=mass,
    inertia=jnp.array([1.0, 1.0, 1.0]),  # Diagonal inertia tensor
    components=[Gravity(mass=mass)],
)

# Set initial state: 100m above ground, at rest
initial_state = create_state(
    position=jnp.array([0, 0, -100]),  # NED: negative D = above ground
    velocity=jnp.array([0, 0, 0]),
)

# Run simulation
result = simulate(body, initial_state, dt=0.01, duration=4.5)

# Access results
print(f"Final position: {result.states[-1, 0:3]}")
print(f"Final velocity: {result.states[-1, 3:6]}")
```

## Examples

See the `examples/` directory for runnable examples:

- **`drop_test.py`**: Free-fall under gravity, validates against analytical solution
- **`pendulum.py`**: Simple pendulum demonstrating energy conservation
- **`spinning_disk.py`**: 3D gyroscopic motion (frisbee throw)

Run examples:
```bash
uv run python examples/drop_test.py
uv run python examples/pendulum.py
uv run python examples/spinning_disk.py
```

## Boat2D Model

A 6-state planar rigid body with physically realistic Coriolis coupling terms.

### State Vector (6 elements)

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0 | `x` | North position | m |
| 1 | `y` | East position | m |
| 2 | `psi` | Heading angle | rad |
| 3 | `u` | Surge velocity (forward) | m/s |
| 4 | `v` | Sway velocity (starboard) | m/s |
| 5 | `r` | Yaw rate | rad/s |

### Control Vector

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0 | `thrust` | Forward thrust | N |
| 1 | `yaw_moment` | Yaw torque | N·m |

### Dynamics

The Boat2D model uses coupled planar rigid body dynamics with Coriolis terms:

**Kinematics:**
```
ẋ = u·cos(ψ) - v·sin(ψ)
ẏ = u·sin(ψ) + v·cos(ψ)
ψ̇ = r
```

**Dynamics (with Coriolis coupling):**
```
u̇ = T/m - (Dᵤ/m)·u + r·v
v̇ = -(Dᵥ/m)·v - r·u
ṙ = τ/Iᵤᵤ - (Dᵣ/Iᵤᵤ)·r
```

The `+rv` and `-ru` terms are Coriolis coupling - they create sway velocity during turns, which is physically correct for a planar rigid body.

### Analytical Methods

```python
from fmd.simulator import Boat2D
from fmd.simulator.params import BOAT2D_TEST_DEFAULT

boat = Boat2D(BOAT2D_TEST_DEFAULT)

# Steady-state surge: u_ss = thrust / drag_surge
boat.steady_state_surge(thrust=50.0)  # Returns 5.0 m/s

# Surge time constant: tau = mass / drag_surge
boat.surge_time_constant()  # Returns 10.0 s

# Yaw time constant: tau = izz / drag_yaw
boat.yaw_time_constant()  # Returns 10.0 s
```

### Outputs

The model computes derived outputs via `get_outputs()`:
- `speed_over_ground`: Magnitude of NED velocity
- `course_over_ground`: Direction of NED velocity

## PlanarQuadrotor Model

A 6-state 2D quadrotor restricted to the x-z plane with pitch rotation. See [Simulator Models](simulator_models.md#3-planarquadrotor---2d-quadrotor) for the full state vector, control vector, and equations of motion.

### Usage

```python
import jax.numpy as jnp
from fmd.simulator import PlanarQuadrotor, simulate, ConstantControl
from fmd.simulator.params import PLANAR_QUAD_TEST_DEFAULT

quad = PlanarQuadrotor(PLANAR_QUAD_TEST_DEFAULT)
control = ConstantControl(quad.hover_control())
result = simulate(quad, quad.default_state(), dt=0.001, duration=5.0, control=control)
```

### Convenience Methods

```python
quad.hover_thrust_total()       # Returns m*g (sum of both rotors)
quad.hover_thrust_per_rotor()   # Returns m*g/2
quad.hover_control()            # Returns [m*g/2, m*g/2]
```

## Output Format

CSV columns (all SI units):
```
time,pos_n,pos_e,pos_d,vel_u,vel_v,vel_w,qw,qx,qy,qz,omega_p,omega_q,omega_r,roll,pitch,yaw
```

The `roll`, `pitch`, `yaw` columns are derived from the quaternion for convenience.

## Verification Tests

The test suite includes physics verification:

- **Drop Test**: Free-fall under gravity matches analytical solution
- **Spin Test**: Demonstrates intermediate axis instability (Dzhanibekov effect)
- **Energy Conservation**: Rotational kinetic energy conserved for torque-free motion
- **Quaternion Normalization**: Quaternion remains unit length during simulation

## Adding New Components

Create custom force/moment components by inheriting from `JaxForceElement`:

```python
import equinox as eqx
import jax.numpy as jnp
from fmd.simulator.components import JaxForceElement

class MyDragForce(JaxForceElement):
    """Custom quadratic drag force."""

    drag_coeff: float  # Will be traced by JAX, can differentiate

    def compute(self, t, state, control):
        """Compute force and moment in body frame.

        Args:
            t: Time (scalar)
            state: State vector (13 elements for RigidBody6DOF)
            control: Control vector

        Returns:
            Tuple of (force, moment), each shape (3,)
        """
        # Extract body velocity
        vel = state[3:6]
        speed = jnp.linalg.norm(vel)

        # Quadratic drag: F = -c * |v| * v
        force = -self.drag_coeff * speed * vel
        moment = jnp.zeros(3)

        return force, moment
```

**Important**: All code in `compute()` must be JIT-compatible (no Python if/else on array values).
See the [JAX Simulator Guide](jax_simulator_guide.md) for details.

## Coordinate Frames

### NED (North-East-Down)

The simulator uses the NED coordinate frame, standard in aerospace:

- **N (North)**: +X points north
- **E (East)**: +Y points east
- **D (Down)**: +Z points toward Earth center

Important: **Altitude increases as `pos_d` decreases** (more negative = higher up).

### Body Frame

The body frame is fixed to the vehicle:

- **X (forward)**: Points toward bow/nose
- **Y (starboard)**: Points to right side
- **Z (down)**: Points toward keel/belly

### Quaternion Convention

Scalar-first quaternion convention: `[qw, qx, qy, qz]`

The quaternion represents rotation from body frame to NED frame.
