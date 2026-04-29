# Extending BLUR

**Estimated reading time: 1 hour**

This guide covers how to extend BLUR with new force components, dynamic models, and parameter classes. BLUR uses a modular architecture where force elements are composable, models inherit from base classes, and parameters are immutable with validation.

---

## Related Documentation

- [README](../../../../README.md) - Quick start, conventions, install
- [docs/simulator_models.md](../../simulator_models.md) - Complete model reference
- [dev/adding_components.md](../../dev/adding_components.md) - Adding new components, models, schemas
- [dev/testing.md](../../dev/testing.md) - Testing and validation patterns

---

## Extension Architecture

BLUR provides three main extension points:

| Extension Point | Base Class | Purpose |
|-----------------|------------|---------|
| Force Elements | `JaxForceElement` | Add new forces/moments (thrust, drag, lift) |
| Dynamic Systems | `JaxDynamicSystem` | Add new vehicle/system types |
| Parameters | `attrs` frozen classes | Configure models with validated, immutable data |

All extensions must be JAX-compatible: pure functions, no Python control flow in hot paths, and compatible with JIT compilation and automatic differentiation.

---

## Adding a Force Element

Force elements are the building blocks of 6-DOF rigid body models. Each element computes forces and moments in the body frame.

### The `JaxForceElement` Base Class

```python
# From fmd/simulator/components/base.py

class JaxForceElement(eqx.Module):
    """Abstract base class for JAX force/moment generating components.

    Each JaxForceElement computes forces and moments acting on a rigid body.
    Forces and moments are returned in the body frame.

    Conventions:
        - Coordinate frame: NED (North-East-Down)
        - Forces/moments in body frame
        - State indices: POS 0-2, VEL 3-5, QUAT 6-9, OMEGA 10-12
        - Signature: (t, state, control) for diffrax compatibility
    """

    @abstractmethod
    def compute(
        self,
        t: float,
        state: Array,
        control: Array,
    ) -> Tuple[Array, Array]:
        """Compute force and moment vectors in body frame.

        Returns:
            Tuple of (force, moment) vectors in body frame coordinates.
            Force: [Fx, Fy, Fz] in Newtons, shape (3,)
            Moment: [Mx, My, Mz] in Newton-meters, shape (3,)
        """
        pass
```

### Example: Gravity Force

A simple example from the codebase:

```python
from fmd.simulator.components.base import JaxForceElement
from fmd.simulator.quaternion import rotate_vector_inverse
from fmd.simulator.params.base import STANDARD_GRAVITY
import jax.numpy as jnp

class JaxGravity(JaxForceElement):
    """Gravitational force component.

    In NED frame, gravity acts in the +Z direction (down).
    The force is transformed to body frame using the body's orientation.

    Attributes:
        mass: Mass of the body (kg)
        g: Gravitational acceleration (m/s^2)
    """

    mass: float
    g: float = STANDARD_GRAVITY

    def compute(self, t, state, control):
        # Extract quaternion from state (indices 6:10)
        quat = state[6:10]

        # Gravity vector in NED frame: [0, 0, mg] (down is positive Z)
        gravity_ned = jnp.array([0.0, 0.0, self.mass * self.g])

        # Transform to body frame
        force_body = rotate_vector_inverse(quat, gravity_ned)

        # Gravity produces no moment about CoM
        moment_body = jnp.zeros(3)

        return force_body, moment_body
```

### Example: Velocity-Squared Drag

A more complex force that depends on state:

```python
class JaxQuadrotorDrag(JaxForceElement):
    """Aerodynamic drag opposing motion in body frame.

    F_drag = -c * ||v|| * v

    Attributes:
        drag_coeff: Drag coefficient (kg/m)
    """

    drag_coeff: float

    def __init__(self, drag_coeff: float):
        self.drag_coeff = drag_coeff

    def compute(self, t, state, control):
        vel_body = state[3:6]

        # Safe norm computation (avoids division by zero)
        speed_sq = jnp.sum(vel_body**2)
        speed = jnp.sqrt(speed_sq + 1e-12)

        # Drag opposes velocity, scales with speed squared
        drag_force = -self.drag_coeff * speed * vel_body

        # No moment from simple drag model
        return drag_force, jnp.zeros(3)
```

### Key Patterns for Force Elements

1. **All forces/moments in body frame** - The rigid body equations expect body-frame forces
2. **Use state indices consistently** - Position 0-2, velocity 3-5, quaternion 6-9, omega 10-12
3. **Avoid Python control flow** - Use `jnp.where()` instead of `if/else`
4. **Handle edge cases numerically** - Add small epsilons to avoid division by zero
5. **Return exactly two 3-vectors** - `(force, moment)`, each shape `(3,)`

---

## Adding a New Model

New models can either extend `JaxDynamicSystem` directly (for simple systems) or extend `RigidBody6DOFJax` (for 3D vehicles with force accumulation).

### The `JaxDynamicSystem` Base Class

```python
# From fmd/simulator/base.py

class JaxDynamicSystem(eqx.Module):
    """Abstract base class for JAX-compatible dynamical systems.

    The system is defined by: dx/dt = f(t, x, u)

    Conventions:
        - Coordinate frame: NED (North-East-Down)
        - Quaternion: Hamilton convention, scalar-first [qw, qx, qy, qz]
        - Units: SI (m, m/s, rad, rad/s)
        - Signature: (state, control, t=0.0) with time optional
    """

    state_names: Tuple[str, ...] = eqx.field(static=True)
    control_names: Tuple[str, ...] = eqx.field(static=True, default=())

    @abstractmethod
    def forward_dynamics(self, state, control, t=0.0) -> Array:
        """Compute state derivative: dx/dt = f(x, u, t)."""
        pass

    @property
    def num_states(self) -> int:
        return len(self.state_names)

    @property
    def num_controls(self) -> int:
        return len(self.control_names)

    def default_state(self) -> Array:
        """Default initial state (zeros). Override for non-zero defaults."""
        return jnp.zeros(self.num_states)

    def default_control(self) -> Array:
        """Default control input (zeros). Override for non-zero defaults."""
        return jnp.zeros(self.num_controls)

    def post_step(self, state: Array) -> Array:
        """Post-process state after integration (e.g., quaternion normalization)."""
        return state
```

### Simple Model Example: Pendulum

The pendulum shows a minimal model implementation:

```python
from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import SimplePendulumParams
import jax.numpy as jnp
import equinox as eqx

class SimplePendulumJax(JaxDynamicSystem):
    """Simple pendulum (point mass on massless rod).

    State vector (2 elements):
        [0] theta     - Angle from vertical (rad)
        [1] theta_dot - Angular velocity (rad/s)

    Equation of motion:
        theta_ddot = -(g/L) * sin(theta)
    """

    length: float
    g: float = 9.80665

    # Static metadata
    state_names: Tuple[str, ...] = eqx.field(
        static=True, default=("theta", "theta_dot")
    )
    control_names: Tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(self, params: SimplePendulumParams):
        """Initialize from validated parameters."""
        self.length = params.length
        self.g = params.g

    def forward_dynamics(self, state, control, t=0.0):
        theta = state[0]
        theta_dot = state[1]
        theta_ddot = -(self.g / self.length) * jnp.sin(theta)
        return jnp.array([theta_dot, theta_ddot])

    def energy(self, state) -> float:
        """Total mechanical energy (for validation)."""
        theta, theta_dot = state[0], state[1]
        T = 0.5 * self.length**2 * theta_dot**2  # Kinetic
        V = self.g * self.length * (1 - jnp.cos(theta))  # Potential
        return T + V

    def period_small_angle(self) -> float:
        """Theoretical period for small oscillations."""
        return 2 * jnp.pi * jnp.sqrt(self.length / self.g)
```

### Composable 6-DOF Model: Quadrotor

For 3D vehicles, extend `RigidBody6DOFJax` and compose force elements:

```python
from fmd.simulator.rigid_body import RigidBody6DOFJax
from fmd.simulator.components import JaxGravity, JaxQuadrotorThrust, JaxQuadrotorDrag
from fmd.simulator.params import QuadrotorParams

class QuadrotorJax(RigidBody6DOFJax):
    """Quadrotor UAV dynamics.

    State vector (13 elements, inherited from RigidBody6DOFJax):
        [0:3]   pos_n, pos_e, pos_d - NED position (m)
        [3:6]   vel_u, vel_v, vel_w - Body velocity (m/s)
        [6:10]  qw, qx, qy, qz - Attitude quaternion
        [10:13] omega_p, omega_q, omega_r - Body angular velocity (rad/s)

    Control vector (4 elements):
        [0] thrust   - Total thrust magnitude (N)
        [1] tau_roll - Roll moment (N*m)
        [2] tau_pitch - Pitch moment (N*m)
        [3] tau_yaw  - Yaw moment (N*m)
    """

    g: float
    drag_coeff: float

    control_names: Tuple[str, ...] = eqx.field(
        static=True, default=("thrust", "tau_roll", "tau_pitch", "tau_yaw")
    )

    def __init__(self, params: QuadrotorParams):
        self.g = params.g
        self.drag_coeff = params.drag_coeff

        # Build component list - the key pattern!
        components = [
            JaxGravity(mass=params.mass, g=params.g),
            JaxQuadrotorThrust(),
        ]

        if params.drag_coeff > 0:
            components.append(JaxQuadrotorDrag(params.drag_coeff))

        # Initialize base class with mass, inertia, and components
        super().__init__(
            mass=params.mass,
            inertia=jnp.asarray(params.inertia_matrix),
            components=components,
        )

    def hover_thrust(self) -> float:
        """Thrust required for level hover: T = m*g."""
        return self.mass * self.g

    def hover_control(self) -> Array:
        """Control vector for hover at level attitude."""
        return jnp.array([self.hover_thrust(), 0.0, 0.0, 0.0])

    def default_control(self) -> Array:
        """Default control is hover."""
        return self.hover_control()
```

### Key Patterns for Models

1. **Inherit from the right base class**
   - `JaxDynamicSystem` for simple/custom state vectors
   - `RigidBody6DOFJax` for 3D vehicles with standard 13-state layout

2. **Use Equinox module fields**
   - Mutable fields: `mass: float` (default)
   - Static fields: `state_names: Tuple[str, ...] = eqx.field(static=True)`

3. **Compose force elements for 6-DOF models**
   - Pass `components` to the base class constructor
   - Forces are accumulated automatically in `forward_dynamics`

4. **Provide validation methods**
   - `energy()` for conservation checks
   - `period_small_angle()` or similar analytical solutions

5. **Support JAX-traceable construction**
   - Provide `from_values()` classmethod for gradient computation
   - Bypasses attrs validation which is not JAX-traceable

---

## Parameter Classes

BLUR uses `attrs` frozen classes for model parameters, providing immutability and validation at construction time.

### Base Validators

BLUR provides standard validators in `fmd.simulator.params.base`:

```python
# Scalar validators
from fmd.simulator.params.base import (
    is_finite,      # Not NaN or Inf
    positive,       # > 0
    non_negative,   # >= 0
)

# Array validators
from fmd.simulator.params.base import (
    is_finite_array,   # All elements finite
    is_3vector,        # Shape is (3,)
    positive_array,    # All elements > 0
    is_valid_inertia,  # Valid inertia tensor (diagonal or 3x3)
    to_float_array,    # Converter to numpy array
)

# Constants
from fmd.simulator.params.base import (
    STANDARD_GRAVITY,     # 9.80665 m/s^2
    AIR_DENSITY_SL,       # 1.225 kg/m^3
    WATER_DENSITY_FRESH,  # 1000.0 kg/m^3
)
```

### Simple Parameter Class: Pendulum

```python
import attrs
from fmd.simulator.params.base import STANDARD_GRAVITY, is_finite, positive

@attrs.define(frozen=True, slots=True)
class SimplePendulumParams:
    """Immutable parameters for SimplePendulum dynamics.

    Attributes:
        length: Pendulum length (m). Must be positive.
        g: Gravitational acceleration (m/s^2). Must be positive.
    """

    length: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "m", "description": "Pendulum length"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravitational acceleration"},
    )

    @property
    def period_small_angle(self) -> float:
        """Theoretical period: T = 2*pi*sqrt(L/g)."""
        return 2 * np.pi * np.sqrt(self.length / self.g)

    def with_length(self, length: float) -> "SimplePendulumParams":
        """Return new params with updated length."""
        return attrs.evolve(self, length=length)
```

### Complex Parameter Class: Quadrotor

```python
import attrs
import numpy as np
from numpy.typing import NDArray

@attrs.define(frozen=True, slots=True, eq=False)
class QuadrotorParams:
    """Immutable parameters for Quadrotor 6-DOF dynamics.

    Inertia accepts either [Ixx, Iyy, Izz] diagonal or full 3x3 tensor.

    Attributes:
        mass: Vehicle mass (kg). Must be positive.
        inertia: Moments of inertia (kg*m^2).
        drag_coeff: Aerodynamic drag coefficient (kg/m).
        g: Gravitational acceleration (m/s^2).
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Vehicle mass"},
    )
    inertia: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_valid_inertia],
        metadata={"unit": "kg*m^2", "description": "Inertia tensor"},
    )
    drag_coeff: float = attrs.field(
        default=0.0,
        validator=[is_finite, non_negative],
        metadata={"unit": "kg/m", "description": "Drag coefficient"},
    )
    g: float = attrs.field(
        default=STANDARD_GRAVITY,
        validator=[is_finite, positive],
        metadata={"unit": "m/s^2", "description": "Gravity"},
    )

    @property
    def hover_thrust(self) -> float:
        """Thrust for level hover: T = m*g."""
        return self.mass * self.g

    @property
    def inertia_matrix(self) -> NDArray:
        """Full 3x3 inertia matrix."""
        if self.inertia.shape == (3,):
            return np.diag(self.inertia)
        return self.inertia

    # Custom __eq__ and __hash__ for numpy array handling
    def __eq__(self, other):
        if not isinstance(other, QuadrotorParams):
            return NotImplemented
        return (
            self.mass == other.mass
            and np.array_equal(self.inertia, other.inertia)
            and self.drag_coeff == other.drag_coeff
            and self.g == other.g
        )

    def __hash__(self):
        return hash((self.mass, self.inertia.tobytes(), self.drag_coeff, self.g))
```

### Key Patterns for Parameters

1. **Always use `frozen=True`** - Immutability prevents accidental modification

2. **Chain validators** - `validator=[is_finite, positive]` runs in order

3. **Use converters for arrays** - `converter=to_float_array` ensures consistent dtype

4. **Provide `with_*` methods** - Use `attrs.evolve()` for modified copies

5. **Custom `__eq__`/`__hash__` for arrays** - The default attrs equality does not work with numpy arrays

6. **Add computed properties** - Derived values like `hover_thrust`, `inertia_matrix`

7. **Include metadata** - Units and descriptions for documentation

---

## Testing New Models

Testing physics models requires both unit tests and physics validation tests.

### Physics Validation Patterns

#### 1. Equilibrium Tests

Verify that the system stays at rest when it should:

```python
def test_pendulum_equilibrium():
    """Pendulum at rest stays at rest."""
    pendulum = SimplePendulum(PENDULUM_1M)

    # At rest, hanging straight down
    state = jnp.array([0.0, 0.0])
    deriv = pendulum.forward_dynamics(state, jnp.array([]))

    np.testing.assert_allclose(deriv, jnp.zeros(2), atol=1e-15)
```

#### 2. Energy Conservation

For conservative systems, energy should be preserved:

```python
def test_energy_conservation():
    """Energy is conserved during simulation."""
    pendulum = SimplePendulum(PENDULUM_1M)
    initial = jnp.array([0.5, 0.0])  # Released from rest

    result = simulate(pendulum, initial, dt=0.001, duration=5.0)

    # Compute energy at each timestep
    energies = jax.vmap(pendulum.energy)(result.states)
    initial_energy = energies[0]

    # Energy should be constant within tolerance
    np.testing.assert_allclose(
        energies, initial_energy * jnp.ones_like(energies), rtol=1e-4
    )
```

#### 3. Known Analytical Solutions

Compare against analytical formulas:

```python
def test_small_angle_period():
    """Period matches analytical for small angles."""
    pendulum = SimplePendulum(PENDULUM_1M)
    theta0 = 0.05  # Small angle

    result = simulate(pendulum, jnp.array([theta0, 0.0]), dt=0.001, duration=10.0)

    # Find zero crossings to measure period
    theta = np.array(result.states[:, 0])
    times = np.array(result.times)

    crossings = []
    for i in range(1, len(theta)):
        if theta[i-1] > 0 and theta[i] <= 0:
            t = times[i-1] + (times[i] - times[i-1]) * (
                theta[i-1] / (theta[i-1] - theta[i])
            )
            crossings.append(t)

    measured_period = crossings[1] - crossings[0]
    theoretical_period = float(pendulum.period_small_angle())

    np.testing.assert_allclose(measured_period, theoretical_period, rtol=1e-3)
```

#### 4. Angular Momentum Conservation

For torque-free rotation:

```python
def test_angular_momentum_conservation():
    """Angular momentum magnitude conserved for torque-free rotation."""
    quad = Quadrotor.from_values(
        mass=1.0,
        inertia=jnp.array([0.01, 0.01, 0.01]),  # Symmetric
        g=0.0,  # Zero gravity
    )

    initial = create_state(
        angular_velocity=jnp.array([0.5, 0.3, 0.7])
    )
    control = ConstantControl(jnp.zeros(4))

    result = simulate(quad, initial, dt=0.001, duration=2.0, control=control)

    # Compute angular momentum magnitude at each step
    L_magnitudes = []
    for state in result.states:
        omega = state[10:13]
        L = 0.01 * omega  # I * omega for symmetric inertia
        L_magnitudes.append(np.linalg.norm(L))

    initial_L = L_magnitudes[0]
    max_drift = np.max(np.abs(np.array(L_magnitudes) - initial_L) / initial_L)

    assert max_drift < 0.001  # 0.1% tolerance
```

### JIT Compilation Tests

Ensure models work with JAX JIT:

```python
def test_jit_compatible():
    """Model can be JIT compiled."""
    pendulum = SimplePendulum(PENDULUM_1M)

    @jax.jit
    def run_sim(p, s0):
        return simulate(p, s0, dt=0.01, duration=1.0)

    result = run_sim(pendulum, jnp.array([0.3, 0.0]))
    assert jnp.all(jnp.isfinite(result.states))
```

### Gradient Tests

Verify autodiff works through the model:

```python
def test_gradient_through_simulation():
    """Can compute gradient w.r.t. initial state."""
    pendulum = SimplePendulum(PENDULUM_1M)

    def loss(theta0):
        initial = jnp.array([theta0, 0.0])
        result = simulate(pendulum, initial, dt=0.01, duration=1.0)
        return result.states[-1, 0] ** 2

    grad = jax.grad(loss)(0.3)
    assert jnp.isfinite(grad)
```

---

## Practical Walkthrough: Adding a Thruster

Let us walk through adding a simple thruster force element that could be used for a marine vehicle.

### Step 1: Define the Force Element

```python
# fmd/simulator/components/thruster.py

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Tuple

from fmd.simulator.components.base import JaxForceElement


class JaxThruster(JaxForceElement):
    """Single thruster force element.

    Applies thrust force at a specified position in body frame.
    Thrust magnitude comes from control input.

    The thruster direction is fixed along body -X (forward thrust).
    Off-axis mounting creates both force and moment.

    Attributes:
        position: Thruster position in body frame [x, y, z] (m)
        direction: Thrust direction in body frame (unit vector)
        max_thrust: Maximum thrust force (N)
        control_index: Index in control vector for this thruster

    Control:
        control[control_index]: Thrust command (0 to 1, normalized)
    """

    position: Array     # Mounting position [x, y, z]
    direction: Array    # Thrust direction (unit vector)
    max_thrust: float   # Maximum thrust (N)
    control_index: int  # Index in control vector

    def __init__(
        self,
        position: Array,
        direction: Array = None,
        max_thrust: float = 100.0,
        control_index: int = 0,
    ):
        """Initialize thruster.

        Args:
            position: Mounting position in body frame [x, y, z] (m)
            direction: Thrust direction (default: forward, [-1, 0, 0])
            max_thrust: Maximum thrust force (N)
            control_index: Index in control vector for this thruster
        """
        self.position = jnp.asarray(position)
        self.direction = (
            jnp.asarray(direction) if direction is not None
            else jnp.array([-1.0, 0.0, 0.0])  # Forward thrust
        )
        # Normalize direction
        self.direction = self.direction / jnp.linalg.norm(self.direction)
        self.max_thrust = max_thrust
        self.control_index = control_index

    def compute(
        self,
        t: float,
        state: Array,
        control: Array,
    ) -> Tuple[Array, Array]:
        """Compute thruster force and moment.

        Args:
            t: Current time (unused)
            state: State vector (unused for simple thruster)
            control: Control vector with thrust command at control_index

        Returns:
            Tuple of (force, moment) in body frame
        """
        # Get thrust command from control vector (clamp to [0, 1])
        thrust_cmd = jnp.clip(control[self.control_index], 0.0, 1.0)

        # Compute thrust force
        thrust_magnitude = thrust_cmd * self.max_thrust
        force = thrust_magnitude * self.direction

        # Moment from off-axis thrust: M = r x F
        moment = jnp.cross(self.position, force)

        return force, moment
```

### Step 2: Create Parameter Class

```python
# fmd/simulator/params/thruster.py

import attrs
import numpy as np
from numpy.typing import NDArray

from fmd.simulator.params.base import (
    is_finite,
    is_finite_array,
    is_3vector,
    positive,
    to_float_array,
)


@attrs.define(frozen=True, slots=True)
class ThrusterParams:
    """Parameters for a single thruster.

    Attributes:
        position: Mounting position in body frame [x, y, z] (m)
        direction: Thrust direction in body frame (normalized automatically)
        max_thrust: Maximum thrust force (N)
    """

    position: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={"unit": "m", "description": "Mounting position"},
    )
    direction: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_3vector],
        metadata={"unit": "", "description": "Thrust direction (normalized)"},
    )
    max_thrust: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "N", "description": "Maximum thrust"},
    )

    @direction.validator
    def _normalize_direction(self, attribute, value):
        """Validate that direction has non-zero magnitude."""
        if np.linalg.norm(value) < 1e-10:
            raise ValueError("direction must have non-zero magnitude")

    @property
    def direction_normalized(self) -> NDArray:
        """Normalized thrust direction."""
        return self.direction / np.linalg.norm(self.direction)
```

### Step 3: Write Tests

```python
# tests/simulator/components/test_thruster.py

import pytest
import numpy as np
import jax.numpy as jnp

from fmd.simulator.components.thruster import JaxThruster
from fmd.simulator.params.thruster import ThrusterParams


class TestJaxThruster:
    """Tests for JaxThruster force element."""

    def test_zero_thrust(self):
        """Zero control produces zero force/moment."""
        thruster = JaxThruster(
            position=jnp.array([0.0, 0.0, 0.0]),
            max_thrust=100.0,
        )
        state = jnp.zeros(13)
        control = jnp.array([0.0])

        force, moment = thruster.compute(0.0, state, control)

        np.testing.assert_allclose(force, jnp.zeros(3), atol=1e-15)
        np.testing.assert_allclose(moment, jnp.zeros(3), atol=1e-15)

    def test_full_thrust_centerline(self):
        """Full thrust on centerline produces force, no moment."""
        thruster = JaxThruster(
            position=jnp.array([0.0, 0.0, 0.0]),  # At CoM
            max_thrust=100.0,
        )
        state = jnp.zeros(13)
        control = jnp.array([1.0])  # Full thrust

        force, moment = thruster.compute(0.0, state, control)

        # Force along -X (forward)
        np.testing.assert_allclose(force, jnp.array([-100.0, 0.0, 0.0]), rtol=1e-10)
        # No moment at CoM
        np.testing.assert_allclose(moment, jnp.zeros(3), atol=1e-15)

    def test_off_axis_moment(self):
        """Off-axis thruster produces moment."""
        # Thruster on starboard side, 1m from centerline
        thruster = JaxThruster(
            position=jnp.array([0.0, 1.0, 0.0]),  # Y = +1m (starboard)
            direction=jnp.array([-1.0, 0.0, 0.0]),  # Forward thrust
            max_thrust=100.0,
        )
        state = jnp.zeros(13)
        control = jnp.array([1.0])

        force, moment = thruster.compute(0.0, state, control)

        # Force still forward
        np.testing.assert_allclose(force, jnp.array([-100.0, 0.0, 0.0]), rtol=1e-10)

        # Moment: r x F = [0, 1, 0] x [-100, 0, 0] = [0, 0, 100]
        # Positive yaw moment (turns right, as expected from starboard thruster)
        np.testing.assert_allclose(moment, jnp.array([0.0, 0.0, 100.0]), rtol=1e-10)

    def test_clamps_thrust_command(self):
        """Thrust command is clamped to [0, 1]."""
        thruster = JaxThruster(
            position=jnp.array([0.0, 0.0, 0.0]),
            max_thrust=100.0,
        )
        state = jnp.zeros(13)

        # Over-saturated command
        force, _ = thruster.compute(0.0, state, jnp.array([2.0]))
        assert jnp.linalg.norm(force) == pytest.approx(100.0)

        # Negative command
        force, _ = thruster.compute(0.0, state, jnp.array([-1.0]))
        np.testing.assert_allclose(force, jnp.zeros(3), atol=1e-15)


class TestThrusterParams:
    """Tests for ThrusterParams validation."""

    def test_valid_params(self):
        """Can create valid params."""
        params = ThrusterParams(
            position=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            max_thrust=100.0,
        )
        assert params.max_thrust == 100.0

    def test_rejects_zero_direction(self):
        """Rejects zero direction vector."""
        with pytest.raises(ValueError, match="non-zero magnitude"):
            ThrusterParams(
                position=[0.0, 0.0, 0.0],
                direction=[0.0, 0.0, 0.0],
                max_thrust=100.0,
            )

    def test_rejects_negative_thrust(self):
        """Rejects negative max thrust."""
        with pytest.raises(ValueError, match="positive"):
            ThrusterParams(
                position=[0.0, 0.0, 0.0],
                direction=[1.0, 0.0, 0.0],
                max_thrust=-10.0,
            )
```

### Step 4: Export in `__init__.py`

```python
# In fmd/simulator/components/__init__.py
from fmd.simulator.components.thruster import JaxThruster

# In fmd/simulator/params/__init__.py
from fmd.simulator.params.thruster import ThrusterParams
```

---

## Checklist for New Extensions

When adding new components or models, verify:

- [ ] **Base class inheritance** - Correct base class for the extension type
- [ ] **JAX compatibility** - No Python if/else in compute paths; use `jnp.where()`
- [ ] **Body frame convention** - Forces and moments in body frame for 6-DOF
- [ ] **State indices** - Consistent with 13-state layout (pos, vel, quat, omega)
- [ ] **Parameter validation** - Use attrs frozen classes with validators
- [ ] **Unit tests** - Cover zero/equilibrium, edge cases, typical operation
- [ ] **Physics validation** - Energy conservation, analytical solutions, equilibrium
- [ ] **JIT test** - Verify JIT compilation works
- [ ] **Gradient test** - Verify autodiff works through the component
- [ ] **Exports** - Added to appropriate `__init__.py` files
- [ ] **Documentation** - Docstrings with attributes, conventions, examples

---

## Summary

BLUR's extension architecture centers on three patterns:

1. **Force Elements** (`JaxForceElement`) - Modular force/moment computations that compose into 6-DOF models

2. **Dynamic Systems** (`JaxDynamicSystem`, `RigidBody6DOFJax`) - State evolution with configurable force elements

3. **Parameters** (`attrs` frozen classes) - Immutable, validated configuration

All extensions must maintain JAX compatibility for JIT compilation and automatic differentiation. Test new components with equilibrium checks, energy conservation, and known analytical solutions to build confidence in the physics.

---

## Next Steps

- Review the existing models in `fmd/simulator/` for patterns and conventions
- Explore [docs/simulator_models.md](../../simulator_models.md) for complete model documentation
- See `dev/testing.md` - Testing and validation patterns for detailed testing philosophy
