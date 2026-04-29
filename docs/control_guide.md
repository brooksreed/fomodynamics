# Control Guide

This document provides comprehensive guidance for LQR controller design, integrator selection, and tuning strategies for the `fmd` simulation environment.

**Related Documentation:**
- [frame_conventions.md](frame_conventions.md) - Coordinate frame conventions
- [simulator_models.md](simulator_models.md) - Model documentation and state vectors
- [timestep_guide.md](timestep_guide.md) - Timestep selection and stability constraints

---

## 1. LQR Controller Design

### 1.1 Basic Usage

fomodynamics provides LQR (Linear Quadratic Regulator) controllers for stabilization and trajectory tracking. The control law is:

```
u = u_ref - K @ (x - x_ref)
```

where K is the feedback gain matrix, x_ref is the reference state, and u_ref is the feedforward control.

**Example: Cartpole Stabilization**

```python
from fmd.simulator import Cartpole, simulate
from fmd.simulator.lqr import LQRController
from fmd.simulator.params import CARTPOLE_CLASSIC
import jax.numpy as jnp

# Create system
cartpole = Cartpole(CARTPOLE_CLASSIC)
x_eq = cartpole.upright_state()
u_eq = jnp.array([0.0])

# Design LQR controller with Q/R weights
Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))  # State cost
R = jnp.array([[0.1]])                            # Control cost

controller = LQRController.from_linearization(
    cartpole, x_eq, u_eq, Q, R
)

# Simulate from perturbed initial condition
x0 = jnp.array([0.0, 0.0, 0.1, 0.0])  # Small angle perturbation
result = simulate(cartpole, x0, dt=0.01, duration=5.0, control=controller)
```

### 1.2 Q and R Weight Selection

The Q/R weight ratio determines closed-loop response speed. Larger Q or smaller R creates faster response but requires smaller timesteps.

**Starting Point Guidelines:**

| Use Case | Q diagonal | R diagonal | Notes |
|----------|-----------|-----------|-------|
| Cartpole stabilization | `[1, 1, 10, 10]` | `0.1` | Higher weight on angle states |
| Pendulum swing-up | `[10, 1]` | `0.01` | Position more important |
| PlanarQuadrotor hover | `1.0` all states | `1e2` | Smooth control, depends on inertia |
| **Moth 3DOF (default)** | `[100, 100, 10, 10]` | `[50, 500]` | Smooth control, low noise amplification |

**Intuition for Q/R Selection:**

- **Q diagonal elements**: Penalize deviation from reference. Higher values = faster correction.
  - Position states: Typically 1.0-10.0
  - Velocity states: Often same as position, or lower for smoother motion
  - Angle states: Often 10x position for inverted systems

- **R diagonal elements**: Penalize control effort. Higher values = more conservative.
  - Start high (1e4-1e6) for stability
  - Reduce gradually to improve tracking
  - Never reduce below what your timestep can handle

### 1.3 Continuous vs Discrete LQR

fomodynamics supports both continuous-time and discrete-time LQR design:

```python
# Continuous-time LQR (default)
controller = LQRController.from_linearization(
    system, x_eq, u_eq, Q, R
)

# Discrete-time LQR
controller = LQRController.from_linearization(
    system, x_eq, u_eq, Q, R,
    discrete=True,
    dt=0.01  # Required for discretization
)
```

**When to use each:**

| Type | Use When |
|------|----------|
| Continuous | General purpose, RK4 integration, production |
| Discrete | Matching specific hardware update rates, Euler integration |

---

## 2. Timestep Selection and Stability

### 2.1 The Fundamental Constraint

Explicit integrators (RK4, Euler) have stability limits based on the closed-loop system eigenvalues:

```
RK4:   dt < 2.785 / |lambda_max|
Euler: dt < 2.0 / |lambda_max|
```

where lambda_max is the fastest (most negative real part) closed-loop eigenvalue.

**Key insight**: Aggressive LQR tuning (high Q/R ratio) creates fast eigenvalues that may require microsecond-level timesteps!

### 2.2 Eigenvalue-Based Timestep Selection

The closed-loop eigenvalues are computed from:

```python
from scipy import linalg
A_cl = A - B @ K  # Closed-loop A matrix
eigenvalues = linalg.eigvals(A_cl)
max_eigenvalue_mag = max(abs(eigenvalues.real))
```

**Warning**: For systems with very small inertia (gram-scale rotational dynamics), aggressive R values can produce eigenvalues that exceed 10,000 rad/s, requiring sub-millisecond timesteps. Always check the eigenvalue/dt limit for your specific platform before running long simulations.

### 2.3 RK4 Stability Region

The RK4 stability region in the complex plane shows where `|lambda * dt|` must lie for stable integration:

```
  Im(lambda*dt)
        |
   2.0  +       .......
        |     ..       ..
   1.5  +    .           .
        |   .     RK4     .
   1.0  +  .   Stability   .
        |  .    Region     .
   0.5  + .                 .
        | .                 .
   0.0  +--.------+------.-+---> Re(lambda*dt)
        | .    -2.785    . |
  -0.5  + .                 .
        |  .               .
  -1.0  +  .               .
        |   .             .
  -1.5  +    .           .
        |     ..       ..
  -2.0  +       .......
```

The stability boundary intersects the negative real axis at approximately -2.785. For pure real eigenvalues (typical in LQR):

```
dt_max = 2.785 / |lambda_max|
```

### 2.4 Checking Stability Before Simulation

Always verify your timestep before running a simulation:

```python
import jax.numpy as jnp
from fmd.simulator import Cartpole, simulate
from fmd.simulator.lqr import LQRController
from fmd.simulator.linearize import linearize
from fmd.simulator.params import CARTPOLE_CLASSIC
from scipy import linalg

cartpole = Cartpole(CARTPOLE_CLASSIC)
x_eq = cartpole.upright_state()
u_eq = jnp.array([0.0])

Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0]))
R = jnp.array([[0.1]])

controller = LQRController.from_linearization(cartpole, x_eq, u_eq, Q, R)

# Check closed-loop stability
A, B = linalg.eigvals, None  # placeholder for clarity below
A_mat, B_mat = linearize(cartpole, x_eq, u_eq)
A_cl = A_mat - B_mat @ controller.K
eigenvalues = linalg.eigvals(A_cl)
max_eig = max(abs(eigenvalues.real))
print(f"Max eigenvalue magnitude: {max_eig:.1f} rad/s")
print(f"Recommended dt < {2.785 / max_eig * 1000:.2f} ms (RK4)")

# Use with appropriate timestep
x0 = jnp.array([0.0, 0.0, 0.1, 0.0])
result = simulate(cartpole, x0, dt=0.01, duration=5.0, control=controller)
```

---

## 3. Integrator Selection

fomodynamics provides multiple integrators for different use cases:

| Integrator | Function | Order | Accuracy | Use When |
|------------|----------|-------|----------|----------|
| RK4 | `simulate()` | 4th | O(dt^4) | Default, production simulations |
| Forward Euler | `simulate_euler()` | 1st | O(dt) | Simple debugging, hardware-realistic timing |
| Semi-implicit Euler | `simulate_symplectic()` | 1st | O(dt) | Energy-preserving, conservative systems |
| Euler substepped | `simulate_euler_substepped()` | 1st | O(dt) | Hardware-realistic control timing |
| Symplectic substepped | `simulate_symplectic_substepped()` | 1st | O(dt) | Hardware-realistic + energy preserving |

### 3.1 RK4 (Default)

```python
from fmd.simulator import simulate

result = simulate(
    system, x0,
    dt=0.01,      # 10ms timestep
    duration=5.0,
    control=controller
)
```

**Characteristics:**
- 4th-order accurate: error ~ O(dt^4)
- Good for smooth dynamics
- Stability limit: dt < 2.785/|lambda_max|
- Recommended for production simulations

### 3.2 Forward Euler

```python
from fmd.simulator import simulate_euler

result = simulate_euler(
    system, x0,
    dt=0.001,     # 1ms timestep (needs smaller dt)
    duration=5.0,
    control=controller
)
```

**Characteristics:**
- 1st-order accurate: error ~ O(dt)
- Stability limit: dt < 2.0/|lambda_max| (stricter than RK4)
- Useful for hardware-realistic timing analysis at small dt
- Simple to understand and debug

### 3.3 Semi-implicit (Symplectic) Euler

```python
from fmd.simulator import simulate_symplectic

result = simulate_symplectic(
    system, x0,
    dt=0.01,
    duration=10.0,
    control=controller
)
```

**Algorithm:**
```
v_new = v_old + dt * a(x_old, v_old)   # Update velocity first
x_new = x_old + dt * v_new             # Use NEW velocity for position
```

**Characteristics:**
- Preserves symplectic structure (Hamiltonian systems)
- Energy oscillates around true value rather than drifting
- Good for long simulations of conservative systems (pendulums, orbital mechanics)

**Note:** System must define `position_indices` and `velocity_indices` properties.

### 3.4 Substepped Integration

Real embedded systems run control at different rates than physics:

```python
from fmd.simulator import simulate_euler_substepped

result = simulate_euler_substepped(
    system, x0,
    dt_sim=0.001,      # 1ms physics timestep (1000 Hz)
    dt_control=0.02,   # 20ms control update (50 Hz)
    duration=5.0,
    control=controller
)
```

**Timing Diagram:**

```
Control rate (50 Hz)    |--------20ms--------|--------20ms--------|
                              ctrl_0               ctrl_1
                              |                    |
Physics rate (1000 Hz)  |1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|...
                         ^                   ^
                         dt_sim=1ms          dt_sim=1ms

Within each 20ms control period:
  - Control held constant (zero-order hold)
  - Physics steps 20 times at 1ms each
  - Output sampled at control rate
```

**Use cases:**
- Hardware-in-the-loop simulation
- Matching embedded controller timing
- Testing controller robustness to delayed updates

### 3.5 Integrator Comparison Summary

```
Accuracy vs Computation Cost:

High    |  RK4 (4th order)
        |    * Best for smooth dynamics
        |    * 4 derivative evaluations per step
        |
Accuracy|
        |  Symplectic Euler (1st order)
        |    * Energy preserving
        |    * 2 derivative evaluations
        |
        |  Forward Euler (1st order)
Low     |    * Simplest, least accurate
        |    * 1 derivative evaluation
        +---------------------------------->
          Low                          High
                  Computation Cost
```

---

## 4. Troubleshooting

### 4.1 Unstable Simulation

**Symptoms:** State values explode, oscillate wildly, or show high-frequency noise

**Diagnosis checklist:**

1. **Check eigenvalues:**
   ```python
   from scipy import linalg
   A_cl = A - B @ K
   eigenvalues = linalg.eigvals(A_cl)
   print(f"Eigenvalues: {eigenvalues}")
   print(f"Max real: {max(eigenvalues.real):.1f}")  # Should be negative!
   print(f"Max magnitude: {max(abs(eigenvalues)):.1f}")
   ```

2. **Check timestep:**
   ```python
   max_eig = max(abs(eigenvalues.real))
   dt_limit_rk4 = 2.785 / max_eig
   dt_limit_euler = 2.0 / max_eig
   print(f"Your dt: {dt}")
   print(f"RK4 limit: {dt_limit_rk4}")
   print(f"Euler limit: {dt_limit_euler}")
   ```

3. **Check Q/R conditioning:**
   ```python
   # Q and R should be positive definite
   print(f"Q eigenvalues: {linalg.eigvals(Q)}")  # All positive
   print(f"R eigenvalues: {linalg.eigvals(R)}")  # All positive
   ```

**Solutions:**
- Reduce timestep (most common fix)
- Increase R (makes controller less aggressive)
- Decrease Q (same effect)
- Use symplectic integrator for conservative systems

### 4.2 NaN Values

**Symptoms:** States become NaN during simulation

**Common causes:**

1. **Division by zero in dynamics:**
   - Check for zero mass/inertia
   - Check for zero denominators in state derivative

2. **Quaternion denormalization:**
   - fomodynamics auto-normalizes, but extreme dt can break this
   - Reduce timestep significantly

3. **Extreme control values:**
   ```python
   # Check control bounds
   print(f"Max control: {result.controls.max()}")
   print(f"Min control: {result.controls.min()}")
   ```

4. **Invalid initial condition:**
   - Quaternion must be unit: |q| = 1
   - Check for NaN in initial state

**Debug approach:**
```python
# Run with very small dt to find when NaN occurs
for i in range(len(result.times)):
    if jnp.any(jnp.isnan(result.states[i])):
        print(f"NaN at t={result.times[i]:.4f}")
        print(f"Previous state: {result.states[i-1]}")
        break
```

### 4.3 Slow Convergence

**Symptoms:** System stabilizes but takes too long to reach setpoint

**Diagnosis:**

1. **Check eigenvalue magnitudes:**
   ```python
   # Real parts should be reasonably negative
   real_parts = eigenvalues.real
   time_constants = -1.0 / real_parts  # Approximate settling time
   print(f"Slowest time constant: {max(time_constants):.2f} s")
   ```

2. **Check Q/R ratio:**
   - Low Q relative to R = slow response
   - Increase Q or decrease R (carefully!)

**Solutions:**
- Increase Q weights (especially for slow states)
- Decrease R (but check timestep requirement first!)
- Use TVLQRController for trajectory tracking

### 4.4 Steady-State Error

**Symptoms:** System stabilizes but not at the reference point

**Common causes:**

1. **Wrong equilibrium point:**
   - Verify x_ref is actually an equilibrium
   - Check u_ref provides correct feedforward

2. **Model mismatch:**
   - Parameters (mass, inertia) don't match actual system
   - Missing disturbances (wind, friction)

3. **Integrator drift:**
   - LQR is proportional-only (no integral action)
   - Consider LQI (LQR with integrator augmentation)

**Verification:**
```python
# Check that x_ref is an equilibrium
deriv = system.forward_dynamics(x_ref, u_ref)
print(f"State derivative at ref: {deriv}")  # Should be ~0
```

---

## 5. Moth-Specific Timestep Guidance

The Moth 3DOF model has speed-dependent eigenvalues that constrain the RK4 timestep. See [timestep_guide.md](timestep_guide.md) for the full timestep architecture.

**Key points:**

- Default Moth dt is 5ms (`MOTH_DEFAULT_DT` from `fmd.simulator`)
- At 10 m/s, the fast pitch eigenvalue is ~280 rad/s (max stable dt = 9.6ms)
- At 20 m/s, max stable dt drops to 4.7ms -- use dt=0.002 or smaller
- LQR gains are designed via discrete-time DARE with ZOH discretization at 5ms
- Use `validate_simulation_dt()` to check for timestep mismatches before simulation

**Eigenvalue table:**

| Speed (m/s) | Fast eigenvalue (rad/s) | Max stable RK4 dt (ms) |
|-------------|------------------------|-------------------------|
| 6 | ~100 | 27.9 |
| 10 | ~280 | 9.6 |
| 14 | ~415 | 6.7 |
| 20 | ~587 | 4.7 |

---

## 6. Advanced Topics

### 6.1 Time-Varying LQR (TVLQR)

For aggressive maneuvers where dynamics vary along the trajectory:

```python
from fmd.simulator.lqr import TVLQRController

# Create TVLQR from reference trajectory
controller = TVLQRController.from_trajectory(
    system,
    times=t_ref,      # (T,) time points
    x_refs=x_ref,     # (T, n) reference states
    u_refs=u_ref,     # (T, m) reference controls
    Q=Q,
    R=R,
)

# Gains are recomputed at each trajectory point
result = simulate(system, x0, dt=0.01, duration=T, control=controller)
```

### 6.2 Trajectory LQR (Fixed Gain)

For mild maneuvers, a single gain suffices:

```python
from fmd.simulator.lqr import TrajectoryLQRController

controller = TrajectoryLQRController.from_trajectory(
    system,
    times=t_ref,
    x_refs=x_ref,
    u_refs=u_ref,
    Q=Q,
    R=R,
    linearize_at=0,  # Use linearization from first point
)
```

### 6.3 Constraint-Aware Simulation

The `fmd` simulator supports constraint enforcement during simulation:

```python
from fmd.simulator.constraints import ConstraintSet, BoxConstraint
from fmd.simulator.constraints.base import Capability

# Define constraints (example: cartpole force limit)
constraints = ConstraintSet([
    BoxConstraint(
        name="force_limits",
        u_min=jnp.array([-15.0]),
        u_max=jnp.array([15.0]),
    )
])

# Simulate with constraint enforcement
result = simulate(
    system, x0, dt=0.01, duration=5.0,
    control=controller,
    constraints=constraints,
    enforcement=Capability.HARD_CLIP,
)
```

---

## References

**Internal documentation:**
- [frame_conventions.md](frame_conventions.md) - Coordinate frame transforms
- [simulator_models.md](simulator_models.md) - Model documentation and state vectors
- [timestep_guide.md](timestep_guide.md) - Timestep selection details

**Source code:**
- `src/fmd/simulator/lqr.py` - LQR controller implementation
- `src/fmd/simulator/integrator.py` - Integrator implementations

**External references:**
- Anderson & Moore, "Optimal Control" - LQR theory
- Tedrake, "Underactuated Robotics" - Applied control for robotics
