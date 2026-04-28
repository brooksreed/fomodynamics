# JAX Simulator User Guide

This guide explains the JAX-based simulator in BLUR, including how to use it effectively and understand its JAX-specific features.

**Related documentation:**
- [Simulator Architecture](simulator_architecture.md) - Core architecture, state vectors, equations of motion
- [Simulator Models](simulator_models.md) - Reference for all physics models (Boat2D, Quadrotor, etc.)

## Quick Start

```python
from fmd.simulator import SimplePendulum, simulate, ConstantControl
from fmd.simulator.params import PENDULUM_1M
import jax.numpy as jnp

# Create a model
pendulum = SimplePendulum(PENDULUM_1M)

# Run a simulation
initial_state = jnp.array([0.3, 0.0])  # 0.3 rad, zero velocity
result = simulate(pendulum, initial_state, dt=0.01, duration=5.0)

# Access results
print(f"Final angle: {result.states[-1, 0]:.4f} rad")
```

## Why JAX?

The BLUR simulator uses JAX for three key capabilities:

1. **JIT Compilation**: Simulations are compiled to optimized machine code, typically 10-100x faster than pure Python
2. **Automatic Differentiation**: Compute gradients through simulations for sensitivity analysis, optimization, and control
3. **Vectorization**: Run many simulations in parallel with `jax.vmap`

## JIT Compilation

### What Is JIT?

JIT (Just-In-Time) compilation transforms Python code into optimized machine code the first time it runs. This means:

- First call includes compilation time (may take 1-10 seconds)
- Subsequent calls with same input shapes are fast
- Functions must be "JIT-compatible" (see limitations below)

### How to Use JIT

The `simulate()` function is already JIT-compatible:

```python
# This compiles on first call, then runs fast
result = simulate(model, initial_state, dt=0.01, duration=5.0)
```

For explicit JIT (e.g., parameter sweeps):

```python
import jax

@jax.jit
def run_simulation(initial_angle):
    initial = jnp.array([initial_angle, 0.0])
    return simulate(pendulum, initial, dt=0.01, duration=5.0)

# First call compiles
result1 = run_simulation(0.3)  # Slow (compiles)
result2 = run_simulation(0.5)  # Fast (uses cached compilation)
```

### JIT Limitations

JIT requires "traceable" code - no Python control flow that depends on array values:

```python
# BAD - Python if/else on array value
def bad_derivative(t, state, control):
    if state[0] > 0:  # ERROR: can't trace this
        return jnp.array([1.0, 0.0])
    else:
        return jnp.array([0.0, 1.0])

# GOOD - JAX conditional
def good_derivative(t, state, control):
    return jnp.where(state[0] > 0,
                     jnp.array([1.0, 0.0]),
                     jnp.array([0.0, 1.0]))
```

Other limitations:
- No Python `for` loops over arrays (use `jax.lax.fori_loop` or vectorization)
- No Python `dict` or `list` access with array indices
- All operations must be JAX-compatible (`jnp` instead of `np`)

### Debugging JIT Issues

Temporarily disable JIT to debug:

```python
import jax

# Disable JIT globally
jax.config.update("jax_disable_jit", True)

# Now Python if/else works (but slowly)
result = simulate(model, initial, dt=0.01, duration=5.0)

# Re-enable for production
jax.config.update("jax_disable_jit", False)
```

Use `jax.debug.print` for JIT-safe debugging:

```python
import jax

def my_derivative(self, t, state, control):
    jax.debug.print("t={t}, state={s}", t=t, s=state)
    return ...
```

## Automatic Differentiation

### Computing Gradients

Use `jax.grad` to differentiate through simulations:

```python
import jax
from fmd.simulator.params import PENDULUM_1M

def loss(initial_angle):
    """Loss function: final kinetic energy."""
    pendulum = SimplePendulum(PENDULUM_1M)
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    final_velocity = result.states[-1, 1]
    return 0.5 * final_velocity ** 2

# Gradient of loss w.r.t. initial angle
grad_fn = jax.grad(loss)
sensitivity = grad_fn(0.3)
```

### Parameter Gradients

To differentiate w.r.t. model parameters, use `from_values()`:

```python
def loss_wrt_length(length):
    """Loss as function of pendulum length."""
    # from_values() avoids attrs validation during JIT tracing
    pendulum = SimplePendulum.from_values(length=length, g=9.80665)
    initial = jnp.array([0.5, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0] ** 2  # Squared final angle

# Gradient w.r.t. pendulum length
grad_length = jax.grad(loss_wrt_length)(1.0)
```

### Higher-Order Derivatives

```python
# Hessian (second derivative)
hessian_fn = jax.hessian(loss)
H = hessian_fn(0.3)

# Jacobian of vector-valued function
def trajectory_loss(initial):
    result = simulate(pendulum, initial, dt=0.01, duration=1.0)
    return result.states[-1]  # Final state (vector)

jacobian = jax.jacobian(trajectory_loss)(jnp.array([0.3, 0.0]))
```

## Equinox Modules

BLUR uses [Equinox](https://docs.kidger.site/equinox/) for JAX-compatible classes.

### What Are Equinox Modules?

Equinox modules are Python classes that work with JAX's transformations (JIT, grad, vmap). They're "PyTrees" - JAX's way of representing nested data structures.

### Creating Custom Force Components

```python
import equinox as eqx
import jax.numpy as jnp
from fmd.simulator.components import JaxForceElement

class MyCustomForce(JaxForceElement):
    """Custom force component."""

    # Numeric parameters (will be traced by JAX)
    coefficient: float

    # Non-numeric configuration (static, not traced)
    name: str = eqx.field(static=True, default="my_force")

    def compute(self, t, state, control):
        """Compute force and moment in body frame.

        Args:
            t: Time (scalar)
            state: 13-element state [pos(3), vel(3), quat(4), omega(3)]
            control: Control vector

        Returns:
            Tuple of (force, moment), each shape (3,)
        """
        # Extract body velocity
        vel_u, vel_v, vel_w = state[3], state[4], state[5]

        # Compute drag force (example)
        force = -self.coefficient * jnp.array([vel_u, vel_v, vel_w])
        moment = jnp.zeros(3)

        return force, moment
```

### Using Custom Components

```python
from fmd.simulator import RigidBody6DOF, Gravity

my_force = MyCustomForce(coefficient=0.5)

body = RigidBody6DOF(
    mass=1.0,
    inertia=jnp.array([0.1, 0.1, 0.1]),
    components=[Gravity(mass=1.0), my_force],
)
```

### The `@eqx.field(static=True)` Pattern

Use `static=True` for non-numeric configuration that shouldn't be traced:

```python
class MyModel(eqx.Module):
    mass: float              # Will be traced, can differentiate
    name: str = eqx.field(static=True)  # Not traced, for configuration
```

## Control Interface

### Why Functions Don't Work

Python functions can't be JIT-compiled because their behavior can't be traced:

```python
# BAD - Python function as control
def my_control(t, state):
    return jnp.array([10.0, 0.0])

result = simulate(boat, initial, control=my_control)  # Won't work with JIT
```

### Using Control Classes

BLUR provides Equinox-based control classes:

```python
from fmd.simulator import ConstantControl, PiecewiseConstantControl

# Constant control
control = ConstantControl(jnp.array([50.0, 0.0]))  # [thrust, yaw_moment]

# Time-varying control (piecewise constant)
times = jnp.array([0.0, 1.0, 2.0, 3.0])
values = jnp.array([
    [10.0, 0.0],   # t < 1.0
    [50.0, 5.0],   # 1.0 <= t < 2.0
    [50.0, -5.0],  # 2.0 <= t < 3.0
    [10.0, 0.0],   # t >= 3.0
])
control = PiecewiseConstantControl(times, values)

result = simulate(boat, initial, dt=0.01, duration=5.0, control=control)
```

### Creating Custom Control Schedules

```python
import equinox as eqx
from fmd.simulator.control import ControlSchedule

class SinusoidalControl(ControlSchedule):
    """Sinusoidal thrust control."""

    amplitude: float
    frequency: float
    offset: float

    @property
    def size(self) -> int:
        return 2  # [thrust, yaw_moment]

    def __call__(self, t, state):
        thrust = self.offset + self.amplitude * jnp.sin(2 * jnp.pi * self.frequency * t)
        return jnp.array([thrust, 0.0])
```

## Float64 Mode

### Why Float64?

BLUR enforces 64-bit floating point for numerical accuracy in physics simulations. This is critical for:

- Long simulations where errors accumulate
- Gradient computation through many timesteps
- Comparing to analytical solutions

### Import Order

BLUR automatically enables float64 mode when you import `fmd.simulator`. However, if you import JAX first, you may get float32:

```python
# GOOD - import fmd.simulator first
from fmd.simulator import SimplePendulum
import jax.numpy as jnp

# BAD - may get float32
import jax.numpy as jnp
from fmd.simulator import SimplePendulum
```

### Checking Float Mode

```python
import jax.numpy as jnp

arr = jnp.array(1.0)
print(f"dtype: {arr.dtype}")  # Should be float64
```

### Troubleshooting Float32 Issues

If you see `float32` or numerical precision issues:

```python
# Manually enable float64 (before any JAX imports)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
# Now jnp.array(1.0).dtype == float64
```

## FP32 / Mixed-Precision Policy

BLUR defaults to float64 for maximum numerical accuracy. However, float32 can provide
significant speedups -- especially on GPU, where consumer hardware (e.g., RTX 4070 SUPER)
has ~32x more FP32 than FP64 compute throughput.

The following policy is based on benchmarks run on an RTX 4070 SUPER with Moth3D
(see `benchmarks/gpu_fp32/report/benchmark_report.md` for full data).

### When FP32 is safe

Use `FMD_USE_FLOAT32=1` for workflows where **relative behavior** matters more than
absolute precision:

- **Parameter sweeps and Monte Carlo** -- batch >= 10, where you compare across runs
- **RL training episodes** -- reward signals are relative
- **Exploratory visualization** -- seeing trends, not exact values
- **Sensitivity analysis** -- relative rankings are preserved

FP32 accuracy on Moth3D at 10s: max state deviation < 3e-05 (calm), < 2e-04 (waves).
At 60s: max deviation < 2e-04 (calm), < 8e-03 (waves). No NaN or divergence observed.

### When to keep FP64

- **Gradient-based optimization** -- FP32 gradients can be noisy for small perturbations
- **Long-horizon calibration** -- drift accumulates over minutes of sim time
- **Validation against analytical solutions** -- need full precision for comparison
- **Control design (LQR/MPC)** -- Riccati solvers are sensitive to conditioning
- **Any workflow comparing to physical measurements**

### Mixed-Precision Strategies

Two alternatives provide accuracy/speed tradeoffs (available in `benchmarks/gpu_fp32/integrator_mixed.py`):

- **Strategy A (FP32 forces + FP64 state)**: Forces computed in FP32, state accumulation
  in FP64. Near-FP64 accuracy (max deviation < 5e-07 at 10s) with ~1.1x speedup. Best for
  workflows needing accuracy with partial FP32 benefit.

- **Kahan compensated summation**: All FP32, but uses Kahan summation to reduce accumulation
  error. 10-30x better accuracy than plain FP32 (max deviation < 1e-05 at 10s) with
  FP32-like speed.

### GPU Batch-Size Crossover

GPU overhead makes it slower than CPU for small batches. The crossover where GPU wins:

| Scenario | Crossover (all precisions) |
|----------|---------------------------|
| Open-loop (calm) | batch >= 500 |
| Open-loop (waves) | batch >= 100 |
| Closed-loop (LQG) | batch >= 500-1000 |

At large batches, GPU speedup is substantial:

| Configuration | B=5000 throughput | vs CPU FP64 |
|---------------|-------------------|-------------|
| GPU + FP32 | ~7900 sims/s | 10.3x |
| GPU + FP64 | ~4500 sims/s | 5.9x |
| CPU + FP32 | ~1060 sims/s | 1.4x |
| CPU + FP64 | ~770 sims/s | baseline |

### Enabling FP32

```bash
# Environment variable (must be set before JAX import)
FMD_USE_FLOAT32=1 python my_script.py
```

```python
# Or in code (before importing fmd.simulator)
import os
os.environ["FMD_USE_FLOAT32"] = "1"
from fmd.simulator import Moth3D, simulate
```

## GPU/TPU Configuration

### Checking Available Devices

```python
import jax

print(jax.devices())  # List of available devices
# e.g., [CpuDevice(id=0)] or [CudaDevice(id=0)]
```

### Performance Considerations

- **CPU**: Works everywhere, good for development
- **GPU**: Much faster for large simulations and batched runs
- **Compilation overhead**: First run includes JIT compilation time

### Troubleshooting GPU

If GPU isn't detected:

1. Check CUDA installation: `nvidia-smi`
2. Check JAX GPU package: `pip list | grep jax`
3. Install JAX with CUDA support:
   ```bash
   pip install "jax[cuda12]"
   ```

Force CPU-only:

```python
import jax
jax.config.update("jax_platform_name", "cpu")
```

## Simulation Results

### SimulationResult vs RichSimulationResult

- `SimulationResult`: JIT-safe, contains arrays only
- `RichSimulationResult`: Has metadata (state names, system info)

```python
from fmd.simulator import simulate, result_with_meta

# Basic simulation (JIT-safe)
result = simulate(model, initial, dt=0.01, duration=5.0)
# result.times, result.states, result.controls

# Add metadata for analysis
rich_result = result_with_meta(model, result)
# rich_result.state_names, rich_result.control_names, etc.
```

### Converting to NumPy/Analysis

```python
import numpy as np
from fmd.simulator.output import result_to_datastream

# Convert JAX arrays to numpy
times_np = np.asarray(result.times)
states_np = np.asarray(result.states)

# Convert to DataStream for analysis
rich_result = result_with_meta(model, result)
stream = result_to_datastream(rich_result)
```

## Common Patterns

### Parameter Sweep

```python
import jax
import jax.numpy as jnp

def simulate_with_length(length):
    pendulum = SimplePendulum.from_values(length=length, g=9.80665)
    initial = jnp.array([0.5, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0]  # Final angle

# Vectorize over lengths
lengths = jnp.array([0.5, 1.0, 1.5, 2.0])
final_angles = jax.vmap(simulate_with_length)(lengths)
```

### Optimization

```python
import jax
import jax.numpy as jnp
from jax import grad

def loss(params):
    """Objective: minimize final angle deviation."""
    length, initial_angle = params
    pendulum = SimplePendulum.from_values(length=length, g=9.80665)
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0] ** 2

# Gradient descent
params = jnp.array([1.0, 0.5])
lr = 0.01
for _ in range(100):
    g = grad(loss)(params)
    params = params - lr * g
```

## Long-Lived Process Workflows

### The Cost Model

Every JAX simulation pays overhead beyond the actual compute. Understanding where time goes is essential for choosing the right workflow.

| Category | Typical cost | Cached? | Notes |
|----------|-------------|---------|-------|
| Process init (XLA backend, device discovery, float64 enable) | ~300-500ms | No — paid every Python process | Dominates one-shot scripts |
| Cold compile (tracing + XLA optimization + codegen) | ~200ms (Moth3D) | Yes — `FMD_JAX_CACHE_DIR` cuts to ~100ms | Scales with model complexity, NOT sim duration |
| Persistent cache hit (tracing + disk cache lookup) | ~50-100ms | — | Still traces Python; skips XLA optimization |
| JIT'd execution | ~1-5ms | — | Same compiled kernel runs for any duration |

A one-shot `python my_sim.py` pays the full ~600-800ms before doing 1-5ms of work. A long-lived process pays it once, then every subsequent call is ~1-5ms.

**Key insight:** Compile cost depends on model complexity and `(dt, duration)` signature, not on sim duration. A 1s and 10s Moth3D sim have identical compile times because `lax.scan` traces its body once and runs it N times.

### Why Long-Lived Processes Matter

After the first simulation in a process, every subsequent call with the same `(dt, duration)` hits the warm JIT cache and costs ~1-5ms. The crossover is immediate: **as soon as you'd run the script twice, use a long-lived process.**

This is the single most impactful workflow change for interactive iteration. Parameter tuning, trim exploration, and control design all involve running many simulations with small changes — exactly the pattern where JIT cache reuse pays off.

### Jupyter/IPython Pattern

The recommended workflow for exploratory simulation work:

```python
# Cell 1: Setup (runs once, slow — ~600ms)
from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3
import jax.numpy as jnp

moth = Moth3D(MOTH_BIEKER_V3)
state0 = moth.default_state()

# Warm up JIT cache
_ = simulate(moth, state0, dt=0.005, duration=2.0)

# Cell 2+: Iterate (runs many times, fast — ~1-5ms each)
result = simulate(moth, state0, dt=0.005, duration=2.0)
# plot, analyze, tweak params, re-run...
```

Cell 1 pays the full startup + compile cost. Every subsequent cell execution is just the JIT'd kernel.

### Structuring Exploratory Work

Rules for fast iteration:

1. **Build the model once.** Don't reconstruct `Moth3D(...)` in each cell — the constructor itself is cheap but it can trigger retrace if JIT sees a "new" model instance.
2. **Use `from_values()` for parameter changes inside JIT.** `Moth3D(params)` runs attrs validation, which breaks inside traced code. `Moth3D.from_values(...)` skips validation and is JIT-safe.
3. **Keep the same `(dt, duration)` when possible.** Different values trigger separate compiles. The canonical combos (see `docs/dev/testing.md`) are pre-warmed in the test suite.
4. **Don't restart the kernel unless you need to.** Every kernel restart pays the full ~500ms init cost again.

### Understanding Cost Categories

When something feels slow, diagnose which cost you're paying:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| First call in process is slow (~600ms) | Process init + cold compile | Expected; use long-lived process |
| First call is slow, subsequent calls fast | Cold compile, then cache hit | Expected; normal JIT behavior |
| Every call is slow (~200ms each) | Cache miss — retracing | Check for lambda closures, new model instances, or changing shapes |
| Every call is slow (~400ms each) | Process restart each time | Use Jupyter/IPython instead of running scripts |

### Diagnostic: `JAX_LOG_COMPILES=1`

The single most useful debugging tool for JIT performance:

```bash
JAX_LOG_COMPILES=1 python my_script.py
```

This prints a line every time JAX actually compiles. If you see:
- **One compile, then silence** — working correctly, subsequent calls use cache
- **Multiple compiles** — cache miss problem (closures, different shapes, new model instances)
- **No compiles but still slow** — it's process startup or Python overhead, not JIT

Use this liberally when something feels slower than expected.

### When to Use One-Shot Scripts

Long-lived processes aren't always the right choice:

- **CI / automated testing** — process isolation prevents JAX state leakage between tests
- **Batch overnight sweeps** — clean state per run, the ~600ms startup is negligible against multi-minute sweeps
- **Reproducible reports** — a standalone script is a clean unit of work for archiving
- **Any run where you need a clean JAX state** — tests use process isolation via `scripts/run_tests.py` for exactly this reason

**Rule of thumb:** if you're iterating, use a notebook. If you're producing an artifact, use a script.

### Performance Reference

Measured on CPU (WSL, Ryzen 5900X), float64:

| Scenario | Eager mode | JIT (cold) | JIT (warm) |
|----------|-----------|------------|------------|
| Moth3D 1s sim (200 steps) | ~193ms | ~200ms compile + 0.3ms run | ~1-2ms |
| Moth3D 10s sim (2000 steps) | ~198ms | ~200ms compile + 1.4ms run | ~1-5ms |
| SimplePendulum 1s sim | ~30ms | ~50ms compile + 0.1ms run | ~0.1ms |
| Cartpole 1s sim | ~35ms | ~60ms compile + 0.1ms run | ~0.1ms |

vmap throughput (CPU, batch=100):
- Cartpole: **4.6x** faster than Python loop
- Moth3D (batch=50): **1.9x** faster than Python loop

GPU throughput (RTX 4070 SUPER, Moth3D open-loop calm, dt=0.005, 10s sim):
- GPU FP32, batch=5000: **7893 sims/s** (10.3x vs CPU FP64)
- GPU FP64, batch=5000: **4483 sims/s** (5.9x vs CPU FP64)
- CPU FP32, batch=5000: **1065 sims/s** (1.4x vs CPU FP64)
- CPU FP64, batch=5000: **766 sims/s** (baseline)

See `benchmarks/gpu_fp32/report/benchmark_report.md` for full crossover and accuracy data.

## Regenerating Golden Values

If you modify the simulator and need to update regression tests:

```bash
uv run python tests/simulator/jax/generate_golden_values.py > tests/simulator/jax/golden_values.py
```

This captures numerical outputs from the current implementation for regression testing.
