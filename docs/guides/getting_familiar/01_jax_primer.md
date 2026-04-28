# JAX Primer for BLUR Users

This section introduces the essential JAX concepts you need to understand BLUR's simulator. You don't need to be a JAX expert to use BLUR, but understanding these fundamentals will help you write efficient code and debug issues.

**Estimated reading time: 30 minutes**

---

## Related Documentation

For a deeper dive into JAX usage in BLUR's simulator, see:
- [JAX Simulator Guide](../../jax_simulator_guide.md) - Complete reference for simulation, control schedules, and Equinox modules

---

## Why JAX?

BLUR uses JAX instead of plain NumPy for three key capabilities:

1. **JIT Compilation**: Functions are compiled to optimized machine code, typically 10-100x faster than interpreted Python
2. **Automatic Differentiation**: Compute exact gradients through simulations for sensitivity analysis, optimization, and control design
3. **Vectorization**: Run many simulations in parallel with a single function call

These capabilities make JAX ideal for robotics applications where you need to:
- Run thousands of Monte Carlo simulations
- Optimize trajectories using gradient-based methods
- Compute sensitivity of outcomes to initial conditions or parameters

---

## Key Concept: `jax.numpy`

JAX provides `jax.numpy` (typically imported as `jnp`), a drop-in replacement for NumPy that supports JAX transformations:

```python
import jax.numpy as jnp
import numpy as np

# NumPy array
np_arr = np.array([1.0, 2.0, 3.0])

# JAX array - same syntax
jax_arr = jnp.array([1.0, 2.0, 3.0])

# Most operations work identically
np_result = np.sin(np_arr) + np.cos(np_arr)
jax_result = jnp.sin(jax_arr) + jnp.cos(jax_arr)
```

**When to use which:**
- Use `jax.numpy` inside functions that will be JIT-compiled or differentiated
- Use regular `numpy` for data loading, plotting, and I/O operations

---

## Key Concept: `jax.jit`

JIT (Just-In-Time) compilation transforms Python functions into optimized machine code the first time they run:

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    """Uncompiled Python - runs slowly."""
    for _ in range(100):
        x = jnp.sin(x) + jnp.cos(x)
    return x

@jax.jit
def fast_function(x):
    """JIT-compiled - runs fast after first call."""
    for _ in range(100):
        x = jnp.sin(x) + jnp.cos(x)
    return x

x = jnp.ones(1000)

# First call includes compilation time (may take 1-2 seconds)
result1 = fast_function(x)

# Subsequent calls are fast (milliseconds)
result2 = fast_function(x)
```

**Key points:**
- First call compiles the function (slow)
- Subsequent calls with the same input shapes reuse the compiled version (fast)
- Compilation is cached based on input shapes and dtypes

### When BLUR uses JIT

BLUR's `simulate()` function is JIT-compatible. You can wrap simulation calls in `@jax.jit` for parameter sweeps:

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M
import jax
import jax.numpy as jnp

pendulum = SimplePendulum(PENDULUM_1M)

@jax.jit
def run_simulation(initial_angle):
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0]  # Return final angle

# First call compiles
final1 = run_simulation(0.3)

# Subsequent calls are fast
final2 = run_simulation(0.5)
final3 = run_simulation(0.7)
```

---

## Key Concept: `jax.vmap`

`vmap` (vectorized map) automatically transforms a function to operate over batches:

```python
import jax
import jax.numpy as jnp

def single_computation(x):
    """Process a single input."""
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

# Manual loop - slow
inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
results_loop = jnp.array([single_computation(x) for x in inputs])

# vmap - fast, parallel
batched_computation = jax.vmap(single_computation)
results_vmap = batched_computation(inputs)
```

**Why vmap matters for BLUR:**

You can run multiple simulations in parallel:

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M
import jax
import jax.numpy as jnp

pendulum = SimplePendulum(PENDULUM_1M)

def simulate_with_angle(initial_angle):
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0]  # Final angle

# Vectorize over initial angles
batch_simulate = jax.vmap(simulate_with_angle)

# Run 100 simulations in parallel
initial_angles = jnp.linspace(0.1, 1.5, 100)
final_angles = batch_simulate(initial_angles)
```

This is much faster than a Python for-loop, especially on GPU.

---

## Key Concept: `jax.grad`

`grad` computes gradients automatically through any differentiable JAX function:

```python
import jax
import jax.numpy as jnp

def loss(x):
    """A simple loss function: x^2."""
    return x ** 2

# Compute gradient: d/dx(x^2) = 2x
grad_loss = jax.grad(loss)

x = 3.0
print(f"loss(3) = {loss(x)}")           # 9.0
print(f"grad(3) = {grad_loss(x)}")       # 6.0
```

**Why gradients matter for BLUR:**

Gradients enable optimization and sensitivity analysis through simulations:

```python
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import PENDULUM_1M
import jax
import jax.numpy as jnp

pendulum = SimplePendulum(PENDULUM_1M)

def final_angle_loss(initial_angle):
    """Loss: squared final angle deviation from zero."""
    initial = jnp.array([initial_angle, 0.0])
    result = simulate(pendulum, initial, dt=0.01, duration=5.0)
    return result.states[-1, 0] ** 2

# Gradient of loss w.r.t. initial angle
grad_fn = jax.grad(final_angle_loss)

# How sensitive is the final angle to initial conditions?
sensitivity = grad_fn(0.3)
print(f"d(loss)/d(initial_angle) at 0.3 rad = {sensitivity:.4f}")
```

This is how optimal control algorithms like iLQR compute control corrections.

---

## BLUR-Specific Pattern: `jax.lax.scan`

BLUR's simulator uses `jax.lax.scan` for efficient integration loops. This is JAX's way of writing compiled for-loops.

### The Problem with Python Loops

Python for-loops don't compile well:

```python
# Conceptual Python version (not how BLUR works)
def simulate_python_loop(initial_state, times, dt):
    states = [initial_state]
    state = initial_state
    for t in times[1:]:
        state = rk4_step(state, t, dt)  # One integration step
        states.append(state)
    return jnp.stack(states)
```

This creates many small operations that can't be optimized together.

### The `jax.lax.scan` Solution

`scan` is a functional loop primitive that compiles to efficient code:

```python
import jax
import jax.numpy as jnp

def scan_example():
    """Compute cumulative sum using scan."""

    def step_fn(carry, x):
        """One step of the loop.

        Args:
            carry: Accumulated value passed between iterations
            x: Current input from the sequence

        Returns:
            (new_carry, output_to_collect)
        """
        new_carry = carry + x
        return new_carry, new_carry  # Update carry, collect this value

    inputs = jnp.array([1, 2, 3, 4, 5])
    initial_carry = 0

    final_carry, outputs = jax.lax.scan(step_fn, initial_carry, inputs)

    return outputs  # [1, 3, 6, 10, 15]
```

### How BLUR Uses `scan`

Here's a simplified version of BLUR's integration loop:

```python
import jax
import jax.numpy as jnp

def simulate_with_scan(system, initial_state, times):
    """Simplified version of BLUR's simulate() function."""

    def step_fn(carry, idx):
        """Single integration step."""
        state, t_prev = carry
        t_curr = times[idx]
        dt = t_curr - t_prev

        # RK4 integration step
        new_state = rk4_step(system, state, t_prev, dt)

        return (new_state, t_curr), new_state

    # Run the loop
    _, states = jax.lax.scan(
        step_fn,
        (initial_state, times[0]),  # Initial carry
        jnp.arange(1, len(times)),   # Loop indices
    )

    # Prepend initial state
    return jnp.vstack([initial_state[None, :], states])
```

**Why `scan` matters:**
- The entire loop compiles to a single optimized operation
- Enables gradient computation through the full trajectory
- Works efficiently with `vmap` for batch simulations

You don't need to write `scan` loops yourself to use BLUR, but understanding this pattern helps when reading BLUR's source code or writing custom integrators.

---

## Common Gotchas

### Gotcha 1: Traced Values in Control Flow

JIT compilation "traces" your function to understand its structure. Array values become abstract placeholders during tracing, so you can't use them in Python control flow:

```python
import jax
import jax.numpy as jnp

# BAD - Python if/else on array value
@jax.jit
def bad_function(x):
    if x > 0:  # ERROR: x is a traced value!
        return x + 1
    else:
        return x - 1

# GOOD - Use jnp.where for conditional logic
@jax.jit
def good_function(x):
    return jnp.where(x > 0, x + 1, x - 1)
```

**The error message looks like:**
```
ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected
```

**Solution:** Use `jnp.where()` or `jax.lax.cond()` for conditionals inside JIT-compiled functions.

### Gotcha 2: Pure Functions Required

JAX functions must be "pure" - they can't have side effects like modifying global variables or printing:

```python
import jax
import jax.numpy as jnp

# BAD - side effects (print, global modification)
results = []

@jax.jit
def bad_function(x):
    print(f"Processing {x}")  # Won't work as expected
    results.append(x)          # Won't work
    return x + 1

# GOOD - return all outputs, no side effects
@jax.jit
def good_function(x):
    return x + 1

# For debugging, use jax.debug.print
@jax.jit
def debuggable_function(x):
    jax.debug.print("Processing x={x}", x=x)
    return x + 1
```

### Gotcha 3: In-Place Updates Don't Work

JAX arrays are immutable. You can't modify them in place:

```python
import jax.numpy as jnp

x = jnp.array([1, 2, 3])

# BAD - NumPy-style in-place update
# x[0] = 10  # This raises an error!

# GOOD - Use .at[].set() to create a new array
x_new = x.at[0].set(10)  # Returns [10, 2, 3]
```

### Debugging JIT Issues

If you're getting tracing errors, temporarily disable JIT:

```python
import jax

# Disable JIT globally for debugging
jax.config.update("jax_disable_jit", True)

# Your code runs as regular Python now
result = my_jitted_function(x)

# Re-enable for production
jax.config.update("jax_disable_jit", False)
```

---

## Float64 Precision in BLUR

BLUR uses 64-bit floating point by default for numerical accuracy in physics simulations. This is configured automatically when you import from `fmd.simulator`:

```python
# This import enables float64 mode
from fmd.simulator import SimplePendulum

import jax.numpy as jnp
print(jnp.array(1.0).dtype)  # float64
```

**Important:** If you import JAX before BLUR, you may get float32 precision:

```python
# BAD order - may get float32
import jax.numpy as jnp
from fmd.simulator import SimplePendulum

# GOOD order - guaranteed float64
from fmd.simulator import SimplePendulum
import jax.numpy as jnp
```

For more details on precision and GPU configuration, see [CLAUDE.md](../../../../CLAUDE.md).

---

## Summary

| Concept | Purpose | BLUR Usage |
|---------|---------|------------|
| `jax.numpy` | NumPy-like arrays for JAX | All array operations in simulator |
| `jax.jit` | Compile functions for speed | Wrapping simulation calls |
| `jax.vmap` | Batch operations automatically | Running parallel simulations |
| `jax.grad` | Automatic differentiation | Optimal control, sensitivity analysis |
| `jax.lax.scan` | Efficient compiled loops | Integration loop in `simulate()` |

**Rules of thumb:**
- Functions must be pure (no side effects)
- Use `jnp.where()` instead of Python `if/else` on array values
- Arrays are immutable; use `.at[].set()` for updates
- First JIT call is slow (compilation); subsequent calls are fast

---

## Next Steps

Now that you understand JAX fundamentals, continue to:
- [02 - Core Concepts](02_core_concepts.md) - Learn BLUR's coordinate frames and state conventions

---

## Companion Notebook

For hands-on practice with JAX concepts, work through the companion notebook:

**`notebooks/getting_familiar/gf_01_jax_basics.ipynb`**

The notebook covers:
- JIT timing comparison (seeing the speedup yourself)
- vmap batching example (running parallel simulations)
- Computing gradients through dynamics
