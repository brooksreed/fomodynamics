# Cartpole: The Classic Control Benchmark

**Estimated reading time: 1 hour**

This section introduces the Cartpole (inverted pendulum) model, a 4-state coupled dynamical system that serves as the classic benchmark in control theory and reinforcement learning. Unlike the simple pendulum, the cartpole couples two mechanical subsystems, cart and pole, creating richer dynamics where control inputs affect both subsystems simultaneously.

---

## Related Documentation

- [Simulator Models](../../simulator_models.md) - Complete Cartpole reference
- [Simple Pendulum](03_simple_pendulum.md) - 2-state pendulum for comparison
- [Control Guide](../../control_guide.md) - LQR design and timestep selection for cartpole balancing

---

## Overview

The cartpole consists of a cart on a frictionless track with a pole attached at the pivot. A horizontal force applied to the cart is the only control input. The challenge is that pushing the cart causes the pole to tip, and conversely, the pole swinging affects the cart's motion.

```
         |     <- pole (mass m_p, length 2l)
         |
         |
        / \
       /   \
   ==========   <- cart (mass m_c)
      [F ->]    <- control force
   ==================== track
```

Key features:
- **4 states**: Cart position and velocity, pole angle and angular velocity
- **1 control**: Horizontal force on cart
- **Coupled dynamics**: Cart motion affects pole, pole affects cart
- **Two equilibria**: Upright (unstable) and hanging (stable)

---

## State and Control Vectors

### State Vector (4 elements)

| Index | Symbol | Name | Description | Units |
|-------|--------|------|-------------|-------|
| 0 | x | Position | Cart position along track | m |
| 1 | x_dot | Velocity | Cart velocity | m/s |
| 2 | theta | Angle | Pole angle from vertical | rad |
| 3 | theta_dot | Angular velocity | Pole angular velocity | rad/s |

**Sign conventions:**
- `theta = 0`: Pole is upright (unstable equilibrium)
- `theta = pi`: Pole is hanging down (stable equilibrium)
- Positive `theta`: Pole tilted clockwise (to the right)

### Control Vector (1 element)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | F | Horizontal force on cart | N |

**Sign convention:** Positive force pushes the cart to the right.

---

## Physics

### The Coupling Problem

What makes the cartpole interesting is the coupling between cart and pole motion. Consider what happens when you push the cart to the right (positive F):

1. **Direct effect on cart**: The cart accelerates to the right
2. **Reaction on pole**: As the cart accelerates, the pole tips backward (negative theta)
3. **Pole affects cart**: The tipping pole exerts a horizontal force on the cart

This bidirectional coupling is captured in the Barto-Sutton-Anderson equations of motion.

### Equations of Motion

The dynamics follow the classic formulation from Barto, Sutton, and Anderson (1983):

**Angular acceleration:**

$$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \cdot \text{temp}}{l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}$$

where the coupling term is:

$$\text{temp} = \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}$$

**Cart acceleration:**

$$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{m_c + m_p}$$

Notice how the cart acceleration depends on the angular acceleration - this is the essence of the coupling.

### Understanding the Coupling Terms

Breaking down the equations reveals the physical meaning:

1. **Centrifugal term** (`m_p * l * theta_dot^2 * sin(theta)`): The rotating pole exerts a centrifugal force on the cart

2. **Reaction term** (`-m_p * l * theta_ddot * cos(theta)`): Angular acceleration of the pole creates a horizontal reaction force on the cart

3. **Effective mass** (`4/3 - m_p * cos^2(theta) / (m_c + m_p)`): The effective rotational inertia changes with pole angle

---

## Two Equilibrium Points

The cartpole has two distinct equilibria where all derivatives are zero:

### Upright Equilibrium (Unstable)

State: `[x, x_dot, theta, theta_dot] = [0, 0, 0, 0]`

At this point:
- Pole is vertical, pointing up
- No motion
- Any perturbation causes divergence

This is an **unstable equilibrium** - the pole will fall if disturbed. Balancing here requires active control.

### Hanging Equilibrium (Stable)

State: `[x, x_dot, theta, theta_dot] = [0, 0, pi, 0]`

At this point:
- Pole is vertical, pointing down
- No motion
- Perturbations cause oscillation around this point

This is a **stable equilibrium** - the pole naturally returns here. It behaves like a simple pendulum hanging from a moving cart.

---

## Using the Cartpole Model

### Creating a Cartpole

```python
from fmd.simulator import Cartpole, simulate, ConstantControl
from fmd.simulator.params import CARTPOLE_CLASSIC, CartpoleParams
import jax.numpy as jnp

# Use the classic preset (matches OpenAI Gym)
cartpole = Cartpole(CARTPOLE_CLASSIC)

# Or create custom parameters
params = CartpoleParams(
    mass_cart=2.0,    # Cart mass (kg)
    mass_pole=0.2,    # Pole mass (kg)
    pole_length=0.75, # Half-length to pole COM (m)
    g=9.80665,        # Gravity (m/s^2)
)
custom_cartpole = Cartpole(params)
```

### Available Parameter Presets

| Preset | m_c (kg) | m_p (kg) | l (m) | Notes |
|--------|----------|----------|-------|-------|
| `CARTPOLE_CLASSIC` | 1.0 | 0.1 | 0.5 | OpenAI Gym standard |
| `CARTPOLE_HEAVY_POLE` | 1.0 | 0.5 | 0.5 | Harder to balance |
| `CARTPOLE_LONG_POLE` | 1.0 | 0.1 | 1.0 | Easier to balance |

**Note on pole length:** The `pole_length` parameter is the **half-length** to the center of mass. The full pole length is `2 * pole_length`.

### Basic Simulation

```python
# Start with a small tilt from upright
initial_state = jnp.array([0.0, 0.0, 0.1, 0.0])

# Simulate with no control (unforced dynamics)
result = simulate(cartpole, initial_state, dt=0.01, duration=5.0)

print(f"Initial angle: {initial_state[2]:.3f} rad")
print(f"Final angle: {result.states[-1, 2]:.3f} rad")
# The pole will have fallen significantly
```

### Simulation with Control

```python
# Apply a constant force to the right
control = ConstantControl(jnp.array([5.0]))  # 5 N

result = simulate(
    cartpole,
    initial_state,
    dt=0.01,
    duration=5.0,
    control=control
)

# Cart will move right, pole dynamics will be affected
print(f"Final cart position: {result.states[-1, 0]:.2f} m")
print(f"Final pole angle: {result.states[-1, 2]:.3f} rad")
```

---

## Model Utilities

### Equilibrium States

```python
# Get the two equilibrium states
upright = cartpole.upright_state()   # [0, 0, 0, 0]
hanging = cartpole.hanging_state()   # [0, 0, pi, 0]

print(f"Upright: {upright}")
print(f"Hanging: {hanging}")
```

### Energy Computation

The total mechanical energy is the sum of cart kinetic energy, pole kinetic energy, and pole potential energy:

```python
import jax

# Compute energy at each timestep
energies = jax.vmap(cartpole.energy)(result.states)

# For conservative system (no damping), energy should be constant
print(f"Initial energy: {energies[0]:.4f} J")
print(f"Final energy: {energies[-1]:.4f} J")
print(f"Energy drift: {abs(energies[-1] - energies[0]):.6f} J")
```

Energy reference:
- At upright equilibrium (theta=0): `PE = m_p * g * l` (maximum PE)
- At hanging equilibrium (theta=pi): `PE = -m_p * g * l` (minimum PE)

### Pole Position Utilities

```python
# Get pole tip position in world frame
state = jnp.array([1.0, 0.0, 0.3, 0.0])  # Cart at x=1, pole tilted
x_tip, y_tip = cartpole.pole_tip_position(state)
print(f"Pole tip: ({x_tip:.3f}, {y_tip:.3f}) m")

# Get pole center of mass position
x_com, y_com = cartpole.pole_com_position(state)
print(f"Pole COM: ({x_com:.3f}, {y_com:.3f}) m")
```

### Linearized Properties

For small angles, the system can be linearized. The linearized natural frequency is:

```python
omega = cartpole.linearized_frequency()  # rad/s
period = cartpole.linearized_period()    # seconds

print(f"Linearized frequency: {omega:.3f} rad/s")
print(f"Linearized period: {period:.3f} s")
```

This is analogous to the simple pendulum, but here it describes the unstable oscillation mode at the upright equilibrium.

---

## Demonstrating the Physics

### 1. Upright Equilibrium is Unstable

```python
# Start perfectly upright
perfect_upright = jnp.array([0.0, 0.0, 0.0, 0.0])

# Compute derivative - should be exactly zero
deriv = cartpole.forward_dynamics(perfect_upright, jnp.zeros(1))
print(f"Derivative at equilibrium: {deriv}")
# All zeros - this is an equilibrium point

# Now add a tiny perturbation
tiny_tilt = jnp.array([0.0, 0.0, 0.001, 0.0])  # 0.001 rad ~ 0.06 degrees

result = simulate(cartpole, tiny_tilt, dt=0.001, duration=2.0)
final_angle = result.states[-1, 2]

print(f"Initial tilt: {tiny_tilt[2]:.4f} rad")
print(f"Final angle after 2s: {final_angle:.3f} rad")
# The pole has fallen significantly - exponential divergence
```

### 2. Hanging Equilibrium is Stable

```python
# Start slightly displaced from hanging
near_hanging = jnp.array([0.0, 0.0, jnp.pi + 0.1, 0.0])

result = simulate(cartpole, near_hanging, dt=0.001, duration=5.0)

# Plot or examine: the angle oscillates around pi
angles = result.states[:, 2]
print(f"Min angle: {min(angles):.3f} rad")
print(f"Max angle: {max(angles):.3f} rad")
# Bounded oscillation around pi - stable
```

### 3. Cart-Pole Coupling

```python
# Demonstrate how force affects both cart and pole
upright = jnp.zeros(4)
force = jnp.array([10.0])  # Strong push to the right

deriv = cartpole.forward_dynamics(upright, force)
print(f"Cart acceleration: {deriv[1]:.3f} m/s^2")   # Positive (cart goes right)
print(f"Pole acceleration: {deriv[3]:.3f} rad/s^2") # Negative (pole tips back)

# The coupling: cart accelerates right, pole tips backward (opposite direction)
```

### 4. Energy Conservation

```python
# Verify energy conservation without control
initial = jnp.array([0.0, 0.0, 0.5, 0.0])  # Moderate initial tilt

# Use small timestep for accuracy
result = simulate(cartpole, initial, dt=0.0001, duration=1.0)
energies = jax.vmap(cartpole.energy)(result.states)

initial_energy = energies[0]
max_drift = jnp.max(jnp.abs(energies - initial_energy))
relative_drift = max_drift / initial_energy

print(f"Initial energy: {initial_energy:.6f} J")
print(f"Max energy drift: {max_drift:.8f} J")
print(f"Relative drift: {relative_drift:.2e}")
# Should be very small (< 1e-3) with small timestep
```

---

## OpenAI Gym Equivalence

The fomodynamics Cartpole implementation matches the OpenAI Gym equations exactly. This enables direct validation and comparison with RL environments.

### Equation Equivalence

The Gym implementation uses the same Barto-Sutton-Anderson formulation:

```python
# OpenAI Gym (pseudocode):
# temp = (force + polemass_length * theta_dot^2 * sin(theta)) / total_mass
# theta_acc = (gravity * sin(theta) - cos(theta) * temp) /
#             (length * (4/3 - pole_mass * cos^2(theta) / total_mass))
# x_acc = temp - polemass_length * theta_acc * cos(theta) / total_mass
```

fomodynamics uses algebraic rearrangement for numerical stability but produces identical results.

### Golden Value Verification

The implementation has been verified against manually computed values:

| State | Control | x_ddot | theta_ddot |
|-------|---------|--------|------------|
| theta=0.1, at rest | F=0 | -0.0712266147 | 1.5748530266 |
| theta=0, at rest | F=10 | 9.7560975610 | -14.6341463415 |

```python
# Verify golden values
state1 = jnp.array([0.0, 0.0, 0.1, 0.0])
deriv1 = cartpole.forward_dynamics(state1, jnp.zeros(1))
print(f"theta_ddot at theta=0.1, F=0: {deriv1[3]:.10f}")
# Expected: 1.5748530266

state2 = jnp.zeros(4)
deriv2 = cartpole.forward_dynamics(state2, jnp.array([10.0]))
print(f"x_ddot at theta=0, F=10: {deriv2[1]:.10f}")
# Expected: 9.7560975610
```

---

## JIT Compilation and Autodiff

### JIT-Compiled Simulation

```python
import jax

@jax.jit
def run_simulation(initial_state):
    return simulate(cartpole, initial_state, dt=0.01, duration=2.0)

# First call compiles
result1 = run_simulation(jnp.array([0.0, 0.0, 0.1, 0.0]))

# Subsequent calls use cached compilation
result2 = run_simulation(jnp.array([0.0, 0.0, 0.2, 0.0]))
```

### Gradient Through Dynamics

Autodiff enables gradient-based control optimization:

```python
def loss(initial_theta):
    """Loss: final angle squared."""
    initial = jnp.array([0.0, 0.0, initial_theta, 0.0])
    result = simulate(cartpole, initial, dt=0.01, duration=0.5)
    return result.states[-1, 2] ** 2

# Gradient of final angle w.r.t. initial angle
grad_loss = jax.grad(loss)
theta0 = 0.1
gradient = grad_loss(theta0)
print(f"d(final_theta^2)/d(initial_theta) at theta0=0.1: {gradient:.4f}")
```

### Gradient Through Parameters

For differentiable model design, use `from_values`:

```python
def loss_wrt_length(pole_length):
    """Loss as function of pole length."""
    cp = Cartpole.from_values(
        mass_cart=1.0,
        mass_pole=0.1,
        pole_length=pole_length,
    )
    initial = jnp.array([0.0, 0.0, 0.1, 0.0])
    result = simulate(cp, initial, dt=0.01, duration=1.0)
    return result.states[-1, 2] ** 2

# How does final angle depend on pole length?
grad_wrt_length = jax.grad(loss_wrt_length)
sensitivity = grad_wrt_length(0.5)
print(f"d(loss)/d(pole_length): {sensitivity:.4f}")
# Longer poles fall slower (easier to balance)
```

---

## Vectorized Simulation

Use `jax.vmap` to simulate multiple initial conditions in parallel:

```python
def simulate_to_final(initial):
    result = simulate(cartpole, initial, dt=0.01, duration=2.0)
    return result.states[-1]

# Batch of initial tilts
initial_tilts = jnp.array([0.05, 0.10, 0.15, 0.20])
initials = jnp.zeros((4, 4)).at[:, 2].set(initial_tilts)

# Vectorized simulation
final_states = jax.vmap(simulate_to_final)(initials)

print("Initial tilts:", initial_tilts)
print("Final angles:", final_states[:, 2])
# Larger initial tilts lead to larger final angles
```

---

## Control Challenges

The cartpole presents several canonical control problems:

### Balance Problem

**Goal:** Keep the pole upright (near theta=0) starting from small deviations.

This is a local stabilization problem suitable for LQR:
- Linearize around the upright equilibrium
- Design feedback gain K
- Apply u = -K @ (state - equilibrium)

### Swing-Up Problem

**Goal:** Swing the pole from hanging (theta=pi) to upright (theta=0).

This is a global control problem requiring nonlinear methods:
- Energy-based swing-up to inject energy
- Switch to LQR near upright for final stabilization
- Suitable for iLQR trajectory optimization

### Position Control

**Goal:** Move the cart to a target position while keeping the pole balanced.

This combines trajectory tracking with balance:
- Reference trajectory for cart position
- Pole must stay near vertical throughout

These problems are explored in the companion notebook and the Control Overview section.

---

## Summary

The cartpole introduces several important concepts:

1. **Coupled dynamics**: Two subsystems (cart, pole) that affect each other
2. **Underactuation**: Only 1 control input for 4 states
3. **Multiple equilibria**: Stable and unstable fixed points
4. **The balance problem**: Quintessential control challenge

Key equations:
- Angular acceleration: $\ddot{\theta} = \frac{g \sin\theta + \cos\theta \cdot \text{temp}}{l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}$
- Coupling term: $\text{temp} = \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}$

Key API:
- `Cartpole(params)` - Create from parameters
- `simulate(cartpole, initial, dt, duration)` - Run simulation
- `cartpole.energy(state)` - Total mechanical energy
- `cartpole.upright_state()` / `cartpole.hanging_state()` - Equilibrium states

---

## Next Steps

- **[06 - Planar Quadrotor](06_planar_quadrotor.md)**: 2D flight dynamics with thrust and torque control
- **[Control Guide](../../control_guide.md)**: LQR design, eigenvalue analysis, and timestep selection
- **Companion notebook**: `notebooks/public/getting_familiar/gf_04_cartpole.ipynb` provides interactive visualization of the dynamics, OpenAI Gym comparison, and LQR balance demonstration

---

## References

1. Barto, A. G., Sutton, R. S., and Anderson, C. W. (1983). "Neuronlike adaptive elements that can solve difficult learning control problems." *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(5), 834-846.

2. OpenAI Gym CartPole-v1: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
