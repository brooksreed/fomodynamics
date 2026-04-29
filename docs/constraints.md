# Constraint System

The `fmd.simulator.constraints` module provides a constraint abstraction system for optimization and control. Constraints represent inequalities `g(t, x, u) <= 0` used in:

- **Optimization** (iLQR, gradient descent) via relaxation functions
- **MPC** via constraint sets with symbolic forms
- **Simulation** enforcement via clip/project methods

## Design Philosophy

### Constraints vs Forces

**Constraints are NOT forces.** This is a fundamental design decision:

- **Constraints**: Inequalities for optimization/control (`g(t,x,u) <= 0`)
- **Forces**: Continuous dynamics elements (`JaxForceElement.compute() -> force, moment`)

For contact physics, penalty forces, and collision dynamics, use `JaxForceElement` components in `fmd.simulator.components`. The constraint system is for optimization boundaries, not physics simulation.

### Sign Convention

All constraints follow the standard optimization convention:
- `value() <= 0`: Constraint is **satisfied**
- `value() > 0`: Constraint is **violated**

Violation magnitude is `max(value, 0)` (i.e., `relu(value)`).

### Capabilities vs Categories

- **Capabilities** (typed enum): What enforcement methods a constraint supports (used for dispatch)
- **Categories** (semantic enum): How to organize/group constraints (documentation only)

## Quick Start

```python
from fmd.simulator.constraints import (
    BoxConstraint, ScalarBound, ConstraintSet,
    quadratic_penalty, smooth_relu_penalty
)
import jax.numpy as jnp

# Create bounds on control inputs
throttle = BoxConstraint("throttle", index=0, lower=0.0, upper=1.0, on_state=False)
steering = BoxConstraint("steering", index=1, lower=-0.5, upper=0.5, on_state=False)

# Combine into a set
constraints = ConstraintSet([throttle, steering])

# Check feasibility
state = jnp.zeros(6)
control = jnp.array([0.5, 0.3])
is_ok = constraints.is_feasible(t=0.0, x=state, u=control)

# Get maximum violation
violation = constraints.max_violation(t=0.0, x=state, u=control)

# For optimization: convert to cost function
cost_fn = quadratic_penalty(throttle, weight=100.0)
cost = cost_fn(0.0, state, control)
```

## Available Constraints

### Bound Constraints (`bounds.py`)

#### BoxConstraint
Two-sided bound on a single state or control element.

```python
# State position bound: -10 <= pos_x <= 10
pos_bound = BoxConstraint("pos_x", index=0, lower=-10.0, upper=10.0)

# Control throttle bound: 0 <= throttle <= 1
throttle = BoxConstraint("throttle", index=0, lower=0.0, upper=1.0, on_state=False)
```

Returns 2-element vector `[lower - val, val - upper]`.

#### ScalarBound
One-sided bound on a single element.

```python
# Maximum velocity: v <= 10
max_vel = ScalarBound("max_vel", index=3, bound=10.0, is_upper=True)

# Minimum throttle: throttle >= 0
min_throttle = ScalarBound("min_throttle", index=0, bound=0.0,
                           is_upper=False, on_state=False)
```

Returns scalar: `val - bound` (upper) or `bound - val` (lower).

### Physical Constraints (`physical.py`)

#### KeepOutZone
Spherical exclusion zone for obstacle avoidance.

```python
# Stay outside 5m sphere at (10, 20, -5) NED
obstacle = KeepOutZone(
    name="obstacle_1",
    center=jnp.array([10.0, 20.0, -5.0]),
    radius=5.0
)
```

Returns scalar: `radius - distance` (negative when outside).

#### HalfSpaceConstraint
Planar constraint (e.g., ground plane, altitude ceiling).

```python
# Ground plane in NED: pos_d <= 0 (stay above ground)
ground = HalfSpaceConstraint(
    name="ground",
    normal=jnp.array([0.0, 0.0, 1.0]),  # Points down
    offset=0.0
)

# Altitude ceiling: pos_d >= -100 (stay below 100m)
ceiling = HalfSpaceConstraint(
    name="ceiling",
    normal=jnp.array([0.0, 0.0, -1.0]),  # Points up
    offset=100.0
)
```

## Relaxation Functions

Convert constraints to differentiable cost functions for optimization.

### quadratic_penalty
```python
cost_fn = quadratic_penalty(constraint, weight=100.0)
# Cost = weight * sum(relu(g)^2)
```
- Zero when feasible
- C1 continuous (gradient = 0 at boundary)
- Quadratic growth with violation

### log_barrier
```python
cost_fn = log_barrier(constraint, scale=1.0)
# Cost = -scale * sum(log(-g))
```
- Only valid in strictly feasible region
- Returns `inf` if violated
- For interior point methods

### smooth_relu_penalty
```python
cost_fn = smooth_relu_penalty(constraint, weight=100.0, softness=0.1)
# Cost = weight * softness * sum(softplus(g / softness))
```
- Differentiable everywhere (C-infinity)
- Small non-zero cost even when feasible
- Good for gradient-based optimization

### exact_penalty
```python
cost_fn = exact_penalty(constraint, weight=100.0)
# Cost = weight * sum(relu(g))
```
- L1 penalty (linear growth)
- Non-smooth at boundary
- Exact for large enough weight

### augmented_lagrangian
```python
cost_fn, update_fn = augmented_lagrangian(constraint, weight=10.0)
# For ALM optimization
```
- Returns cost function and multiplier update
- For augmented Lagrangian methods

## ConstraintSet

Collection of constraints with utilities.

```python
constraints = ConstraintSet([c1, c2, c3])

# Filter by capability
clippable = constraints.by_capability(Capability.HARD_CLIP)

# Filter by category
state_bounds = constraints.by_category(ConstraintCategory.STATE_BOUND)

# Evaluate all
values = constraints.all_values(t, x, u)  # dict[str, Array]

# Aggregate metrics
max_viol = constraints.max_violation(t, x, u)
is_feas = constraints.is_feasible(t, x, u, tol=1e-6)
```

## Coordinate Conventions

fomodynamics uses **NED (North-East-Down)** coordinates:
- `pos_n, pos_e, pos_d` at state indices 0, 1, 2 (for RigidBody6DOF)
- **+D is DOWN**: altitude increase means `pos_d` decreases
- Water surface at `pos_d = 0` means "above water" is `pos_d <= 0`

When creating physical constraints, be mindful of these conventions:
- Ground plane normal points DOWN (+D direction)
- Ceiling constraint normal points UP (-D direction)

## Creating Custom Constraints

Extend `AbstractConstraint` for custom constraints:

```python
from fmd.simulator.constraints import AbstractConstraint, Capability, ConstraintCategory
import equinox as eqx
import jax.numpy as jnp

class MyConstraint(AbstractConstraint):
    """Custom constraint: x[0]^2 + x[1]^2 <= radius^2."""

    radius: float

    def __init__(self, name: str, radius: float):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", ConstraintCategory.SAFETY)
        object.__setattr__(self, "capabilities", frozenset({Capability.HAS_SYMBOLIC_FORM}))
        object.__setattr__(self, "radius", radius)

    def value(self, t: float, x, u):
        # Returns: x[0]^2 + x[1]^2 - radius^2
        # <= 0 when inside circle
        return x[0]**2 + x[1]**2 - self.radius**2
```

### Implementing Clip Support

To support `Capability.HARD_CLIP`, implement the `clip` method:

```python
def clip(self, t: float, x, u):
    # Project to feasible set
    norm = jnp.sqrt(x[0]**2 + x[1]**2)
    scale = jnp.minimum(1.0, self.radius / jnp.maximum(norm, 1e-10))
    x = x.at[0].set(x[0] * scale)
    x = x.at[1].set(x[1] * scale)
    return x, u
```

## Testing Constraints

Use `ConstraintTestHelper` for comprehensive testing:

```python
from fmd.simulator.constraints import ConstraintTestHelper

def test_my_constraint():
    c = MyConstraint("test", radius=1.0)

    # JIT compatibility
    ConstraintTestHelper.check_jit_compatible(c)

    # vmap compatibility (for MPC batching)
    ConstraintTestHelper.check_vmap_compatible(c)

    # Differentiability
    ConstraintTestHelper.check_differentiable(c, t=0.0, x=x, u=u)

    # Sign convention
    ConstraintTestHelper.check_sign_convention(
        c,
        feasible_points=[(0.0, x_ok, u)],
        infeasible_points=[(0.0, x_bad, u)],
    )
```

## Future Integration Roadmap

### Phase 2: Simulation Integration
- `rk4_step_with_constraints()` for constraint enforcement during simulation
- Clip/project options in `simulate()`

### Phase 3: Optimization Integration
- `constraint_cost()` builder for trajectory optimization
- Integration with the public `MultipleShootingOCP` infrastructure

### Phase 4: CasADi Export
- `symbolic_value()` methods on constraints
- `export_to_opti()` for MPC problems
- `compare_jax_casadi()` test utility completion

### Phase 5: Rate Limits
- Augmented state for `u_prev` tracking
- `ControlRateLimit` constraint
