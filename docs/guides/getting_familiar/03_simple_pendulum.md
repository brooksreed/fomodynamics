# Simple Pendulum

**Estimated reading time: 30 minutes**

The simple pendulum is the most basic dynamic system in fomodynamics. With only 2 states, it provides an ideal introduction to fomodynamics's simulation workflow while still demonstrating important concepts like energy conservation and period validation.

---

## Related Documentation

- [README](../../../../README.md) - Quick start, installation, core conventions
- [docs/simulator_models.md](../../simulator_models.md) - Full model reference
- [Companion notebook](../../../../notebooks/public/getting_familiar/gf_03_simple_pendulum.ipynb) - Interactive examples

---

## Physics Overview

The simple pendulum consists of a point mass on a massless rod swinging under gravity.

### State Vector

The pendulum has 2 states:

| Index | Name | Symbol | Units | Description |
|-------|------|--------|-------|-------------|
| 0 | `theta` | $\theta$ | rad | Angle from vertical (positive = clockwise) |
| 1 | `theta_dot` | $\dot{\theta}$ | rad/s | Angular velocity |

### Equation of Motion

The pendulum follows the nonlinear differential equation:

$$\ddot{\theta} = -\frac{g}{L} \sin(\theta)$$

where:
- $g$ = gravitational acceleration (9.80665 m/s^2 by default)
- $L$ = pendulum length (meters)

### Small-Angle Approximation

For small angles ($\theta \ll 1$ rad), $\sin(\theta) \approx \theta$, giving simple harmonic motion:

$$\ddot{\theta} \approx -\frac{g}{L} \theta$$

This yields the analytical period:

$$T = 2\pi\sqrt{\frac{L}{g}}$$

For a 1-meter pendulum: $T \approx 2.006$ seconds.

---

## Creating a Pendulum

### Using Parameter Presets

fomodynamics provides ready-to-use pendulum configurations:

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M, PENDULUM_2M
import jax.numpy as jnp

# Create a 1-meter pendulum
pendulum = SimplePendulum(PENDULUM_1M)

print(f"Length: {pendulum.length} m")
print(f"Gravity: {pendulum.g} m/s^2")
print(f"Theoretical period: {pendulum.period_small_angle():.4f} s")
```

Available presets:
- `PENDULUM_1M` - 1 meter length
- `PENDULUM_2M` - 2 meter length
- `SECONDS_PENDULUM` - Length tuned for 2-second period

### Custom Parameters

Create a pendulum with specific parameters:

```python
from fmd.simulator.params import SimplePendulumParams

params = SimplePendulumParams(length=1.5, g=9.81)
pendulum = SimplePendulum(params)
```

---

## Running a Simulation

### Basic Simulation

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M
import jax.numpy as jnp

# Create pendulum
pendulum = SimplePendulum(PENDULUM_1M)

# Initial state: displaced 0.1 rad (~6 degrees), at rest
initial_state = jnp.array([0.1, 0.0])

# Simulate for 5 seconds with 1ms timestep
result = simulate(pendulum, initial_state, dt=0.001, duration=5.0)

print(f"Simulation steps: {len(result.times)}")
print(f"Final angle: {result.states[-1, 0]:.4f} rad")
```

### Accessing Results

The simulation returns a `JaxSimulationResult` with:

```python
# Time array
times = result.times  # Shape: (N,)

# State trajectory
theta = result.states[:, 0]      # Angle over time
theta_dot = result.states[:, 1]  # Angular velocity over time

# Get Cartesian position of the bob
x, y = pendulum.cartesian_position(result.states[-1])
```

---

## Validation

### Period Validation

For small angles, the measured period should match the theoretical value:

```python
import numpy as np

# Small initial angle for linear regime
initial_state = jnp.array([0.05, 0.0])  # ~3 degrees
result = simulate(pendulum, initial_state, dt=0.001, duration=10.0)

# Find zero crossings (positive to negative)
theta = np.array(result.states[:, 0])
times = np.array(result.times)

crossings = []
for i in range(1, len(theta)):
    if theta[i-1] > 0 and theta[i] <= 0:
        # Linear interpolation for precise crossing time
        t = times[i-1] + (times[i] - times[i-1]) * (
            theta[i-1] / (theta[i-1] - theta[i])
        )
        crossings.append(t)

# Period is time between consecutive crossings
measured_period = crossings[1] - crossings[0]
theoretical_period = pendulum.period_small_angle()

print(f"Theoretical period: {theoretical_period:.4f} s")
print(f"Measured period: {measured_period:.4f} s")
print(f"Error: {abs(measured_period - theoretical_period) * 1000:.2f} ms")
```

### Energy Conservation

The total mechanical energy should remain constant:

```python
import jax

# Compute energy at each timestep
energies = jax.vmap(pendulum.energy)(result.states)

initial_energy = energies[0]
max_drift = jnp.max(jnp.abs(energies - initial_energy))

print(f"Initial energy: {initial_energy:.6f} J")
print(f"Max energy drift: {max_drift:.2e} J")
print(f"Relative drift: {max_drift / initial_energy:.2e}")
```

The energy formula for a unit mass pendulum is:

$$E = \frac{1}{2}L^2\dot{\theta}^2 + gL(1 - \cos\theta)$$

The first term is kinetic energy; the second is potential energy (zero at the bottom).

---

## Key Concepts

### Why Start Here?

The simple pendulum demonstrates fomodynamics fundamentals without added complexity:

1. **Minimal state**: Only 2 states to track
2. **No control inputs**: Pure physics, no control design needed
3. **Analytical validation**: Exact formulas for period and energy
4. **Nonlinear dynamics**: The $\sin(\theta)$ term shows nonlinear behavior at large angles

### What to Notice

- The period increases for larger initial angles (nonlinear effect)
- Energy is conserved to high precision (RK4 integrator)
- Smaller timesteps give more accurate period measurements

---

## Next Steps

The companion notebook `gf_03_simple_pendulum.ipynb` provides interactive examples:
- Period validation against theory
- Energy conservation plots
- Large-angle behavior exploration

Continue to [04 - Cartpole](04_cartpole.md) to see a 4-state coupled system with control inputs.
