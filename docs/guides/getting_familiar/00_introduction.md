# Introduction to fomodynamics

**Estimated reading time: 30 minutes**

**Related Documentation:**
- [README](../../../../README.md) - Quick start and installation
- [README — Core conventions](../../../../README.md#core-conventions) - Frames, units, quaternions
- [Moth Simulation Vision](../../moth_simulation_vision.md) - Long-term project vision

---

## What is fomodynamics?

**fomodynamics** (import name `fmd`) is a Python library for control and estimation of 3D vehicles, with a focus on hydrofoiling sail and power boats. It combines physics simulation with telemetry analysis, providing a unified environment for developing and validating control algorithms.

`fomodynamics` is built on [JAX](https://jax.readthedocs.io/), giving it automatic differentiation, just-in-time compilation, and GPU acceleration out of the box. This makes it well-suited for both classical optimization approaches (like Model Predictive Control) and learning-based methods (like Reinforcement Learning).

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **6-DOF Rigid Body Simulation** | Quaternion-based attitude, RK4 integration, modular force components |
| **Vehicle Models** | Pendulum, cartpole, boat (2D), planar quadrotor, hydrofoiling Moth (3DOF) |
| **Automatic Differentiation** | Gradients through simulations for optimization and learning |
| **Telemetry Analysis** | Schema-based data loading, circular-aware math, multi-rate alignment |
| **Control Algorithms** | Discrete LQR, EKF estimation, direct multiple-shooting OCP (IPOPT) |

### Design Principles

`fomodynamics` follows several core design principles that inform its architecture:

1. **SI Units Internally** - All data is stored in SI units (meters, radians, m/s). Conversions to display units (knots, degrees) happen only at presentation time.

2. **NED Coordinate Frame** - Position uses North-East-Down (NED), with body-frame velocities in Forward-Right-Down (FRD). This is the standard aerospace convention.

3. **Quaternion Attitude** - Orientation is represented with scalar-first quaternions (qw, qx, qy, qz), avoiding gimbal lock issues inherent in Euler angles.

4. **Circular-Aware Math** - Angular quantities wrap correctly: `359 deg - 1 deg = -2 deg`, not `358 deg`.

5. **JAX Compatibility** - Core math avoids Python control flow (if/else) in performance-critical paths, enabling JIT compilation and automatic differentiation.

---

## Target Use Cases

`fomodynamics` is designed for several interconnected use cases:

### Control Prototyping

Design and test control algorithms in simulation before deploying to hardware. `fomodynamics` supports:
- Discrete LQR for linearized stabilization
- Direct multiple-shooting OCP (via CasADi + IPOPT) for trajectory optimization
- EKF / Kalman estimation for state estimation

### Hydrofoiling Boats

The primary application driving `fomodynamics`'s development is simulating hydrofoiling Moths (high-performance sailing dinghies). This involves:
- Modeling foil hydrodynamics with lift/drag curves
- Ride height control through flap actuation
- Balancing sail forces against foil forces

See the [Moth Simulation Vision](../../moth_simulation_vision.md) for the full roadmap.

### Learning Algorithms

`fomodynamics`'s JAX foundation makes it suitable for reinforcement learning research:
- Gradients through dynamics enable policy gradient methods
- Fast JIT-compiled rollouts for sample efficiency
- Vectorized simulation with `jax.vmap` for parallel environments

### Data Analysis

Beyond simulation, `fomodynamics` includes tools for analyzing real telemetry:
- Schema-based data loading for multiple formats
- Circular-aware filtering and statistics
- Comparison between simulated and recorded data

---

## Prerequisites

To get the most out of this guide, you should be comfortable with:

### Required

- **Python** - Familiarity with modern Python (3.10+), including type hints and dataclasses
- **NumPy** - Array operations and broadcasting
- **Basic Dynamics** - Newton's laws, rigid body kinematics (position, velocity, acceleration)
- **Linear Algebra** - Matrices, vectors, matrix multiplication

### Helpful (but not required)

- **Control Theory** - State-space representation, feedback control
- **JAX** - JIT compilation, automatic differentiation (covered briefly in [01_jax_primer.md](01_jax_primer.md))
- **Quaternions** - Unit quaternions for rotation representation (covered in [02_core_concepts.md](02_core_concepts.md))

If you are unfamiliar with JAX, start with the [JAX Primer](01_jax_primer.md) section before diving into the model documentation.

---

## Installation

`fomodynamics` uses [uv](https://docs.astral.sh/uv/) for package management. The recommended setup is WSL2 Ubuntu 24.04 for Windows users.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/brooksreed/fomodynamics.git
cd fomodynamics

# Install all dependencies
uv sync

# Verify installation
uv run python -c "import fmd; print('fomodynamics installed successfully')"
```

### Excluding Dependencies

```bash
# Skip specific groups if needed
uv sync --no-group viz3d         # skip rerun-sdk
uv sync --no-group optimization  # skip CasADi
```

### GPU Support (Optional)

`fomodynamics` automatically detects and uses GPU when available. GPU support requires separate installation:

```bash
# Install `fomodynamics` first
uv sync

# Then enable GPU support (requires NVIDIA GPU with drivers)
bash cuda-setup.sh

# Verify GPU is detected
uv run python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

See [jax_simulator_guide.md](../../jax_simulator_guide.md) for detailed GPU configuration and memory management.

### Verify Your Installation

Run a quick simulation to confirm everything works:

```python
import jax.numpy as jnp
from fmd.simulator import SimplePendulum, simulate
from fmd.simulator.params import SimplePendulumParams

# Create a simple pendulum (1 meter length, standard gravity)
params = SimplePendulumParams(length=1.0)
pendulum = SimplePendulum(params)

# Initial state: displaced 30 degrees from vertical
x0 = jnp.array([jnp.radians(30.0), 0.0])  # [theta, theta_dot]

# Simulate for 5 seconds
result = simulate(pendulum, x0, dt=0.01, duration=5.0)

# Check the result
import numpy as np
print(f"Simulated {len(result.times)} timesteps")
print(f"Final angle: {np.degrees(float(result.states[-1, 0])):.1f} degrees")
```

---

## Package Structure

`fomodynamics` is organized into a few main subpackages, all under the `fmd` import name:

```
fmd/
├── core/           # Shared math, units, and abstractions
├── simulator/      # JAX-based 6-DOF dynamics simulation
└── analysis/       # Telemetry loading, filtering, and plotting
```

### fmd.core

The foundation layer providing shared utilities:

- **Quaternion math** - Operations like `quat_multiply`, `quat_rotate`, conversion to/from Euler angles
- **Circular operations** - `circular_subtract`, `circular_mean`, `wrap_angle` for handling angles
- **Units** - SI unit definitions and display conversions
- **Abstractions** - `DynamicSystem` abstract base class that all models implement

### fmd.simulator

The physics simulation layer:

- **Models** - Implementations of various vehicles (pendulum, cartpole, boats, quadrotors)
- **Integrators** - RK4 (default), Euler, and symplectic integrators
- **Components** - Modular force/moment elements (Gravity, Thrust, Drag)
- **Control** - Discrete LQR and EKF estimation
- **Parameters** - Immutable `attrs` classes for model configuration

### fmd.analysis

The data analysis layer:

- **Loaders** - Schema-based loading of CSV telemetry files
- **DataStream** - Circular-aware operations on time-series data
- **Plotting** - Time-series visualization with domain-specific presets

---

## Guide Roadmap

This guide is organized to build understanding progressively, from fundamentals to advanced topics.

### Foundation (Start Here)

| Section | Time | What You Will Learn |
|---------|------|---------------------|
| [00_introduction.md](00_introduction.md) | 30 min | You are here |
| [01_jax_primer.md](01_jax_primer.md) | 30 min | JAX basics: JIT, vmap, grad |
| [02_core_concepts.md](02_core_concepts.md) | 1 hr | Frames, quaternions, units, state vectors |

### Models (Choose Based on Interest)

| Section | Time | System | Key Concepts |
|---------|------|--------|--------------|
| [03_simple_pendulum.md](03_simple_pendulum.md) | 30 min | 2-state | Period validation, energy conservation |
| [04_cartpole.md](04_cartpole.md) | 1 hr | 4-state | Coupled dynamics, equilibrium analysis |
| [06_planar_quadrotor.md](06_planar_quadrotor.md) | 1 hr | 6-state | 2D flight, thrust modeling |

### Control and Validation

| Section | Time | What You Will Learn |
|---------|------|---------------------|
| [10_extending_blur.md](10_extending_blur.md) | 1 hr | Adding new models and components |
| [../../control_guide.md](../../control_guide.md) | 1 hr | LQR design, eigenvalue analysis, timestep selection |

### Suggested Reading Paths

**"I want to run simulations quickly"**
> 00 (intro) -> 02 (core concepts) -> pick a model (03-07) -> 09 (validation)

**"I want to understand the control algorithms"**
> 00 (intro) -> 02 (core concepts) -> 04 (cartpole) -> 06 (planar quadrotor) -> ../../control_guide.md

**"I want to add new physics models"**
> 00 (intro) -> 02 (core concepts) -> 03 (pendulum) -> 10 (extending)

**"I want to trust the simulation results"**
> 02 (core concepts) -> 09 (validation) -> then model sections of interest

---

## Your First Simulation

Let us run a slightly more interesting simulation to get a feel for `fomodynamics`. This example simulates a 2D boat accelerating from rest.

```python
import jax.numpy as jnp
from fmd.simulator import Boat2D, simulate, ConstantControl
from fmd.simulator.params import Boat2DParams

# Create a boat with specific parameters
params = Boat2DParams(
    mass=100.0,       # kg
    izz=50.0,         # yaw moment of inertia (kg*m^2)
    drag_surge=10.0,  # surge damping (kg/s)
    drag_sway=20.0,   # sway damping (kg/s)
    drag_yaw=5.0,     # yaw damping (kg*m^2/s)
)
boat = Boat2D(params)

# Apply constant thrust and a small yaw moment
# Control: [surge_thrust (N), yaw_moment (N*m)]
control = ConstantControl(jnp.array([50.0, 2.0]))

# Simulate from rest for 30 seconds
x0 = boat.default_state()  # [x, y, psi, u, v, r] = all zeros
result = simulate(boat, x0, dt=0.01, duration=30.0, control=control)

# Analyze the results
import numpy as np
final_state = np.asarray(result.states[-1])

print(f"Final position: N={final_state[0]:.1f}m, E={final_state[1]:.1f}m")
print(f"Final heading: {np.degrees(final_state[2]):.1f} degrees")
print(f"Final surge velocity: {final_state[3]:.2f} m/s")

# Theoretical steady-state surge velocity
u_ss = 50.0 / 10.0  # thrust / drag
print(f"Expected steady-state: {u_ss:.2f} m/s")
```

This example demonstrates several `fomodynamics` patterns:
- Creating a model with parameters via `attrs` classes
- Using `ConstantControl` for open-loop simulation
- Accessing results through the `SimulationResult` object
- Validating against analytical solutions

---

## Running Tests

`fomodynamics` has an extensive test suite. Running tests is a good way to verify your installation and explore the codebase:

```bash
# Run all tests (note: some tests are slow)
uv run pytest tests/ -v

# Run fast tests only (skip @pytest.mark.slow tests)
uv run pytest tests/ -v -m "not slow"

# Run tests for a specific package
uv run pytest tests/core/ -v
uv run pytest tests/simulator/ -v
uv run pytest tests/analysis/ -v

# Run integration tests
uv run pytest tests/test_integration.py -v
```

See [overall_testing_approach.md](../../overall_testing_approach.md) and [dev/testing.md](../../dev/testing.md) for more details on test organization and JAX memory management.

---

## Getting Help

If you encounter issues:

1. **Check the documentation** - [README.md](../../../../README.md) and this guide
2. **Run the tests** - `uv run pytest tests/ -v -m "not slow"` to verify your installation
3. **Check existing docs** - See the [docs/](../../) folder for specialized topics

### Key Reference Documents

| Document | Purpose |
|----------|---------|
| [README.md](../../../../README.md) | Quick start, installation, basic examples, core conventions |
| [frame_conventions.md](../../frame_conventions.md) | Authoritative frame reference |
| [control_guide.md](../../control_guide.md) | LQR tuning, stability analysis |
| [simulator_models.md](../../simulator_models.md) | Complete model documentation |

---

## Next Steps

You are now ready to dive into fomodynamics. We recommend starting with:

1. **[01_jax_primer.md](01_jax_primer.md)** - If you are new to JAX
2. **[02_core_concepts.md](02_core_concepts.md)** - Essential for understanding all `fomodynamics` models
3. **A model of your choice** - Start with [03_simple_pendulum.md](03_simple_pendulum.md) for the simplest example, or jump to [06_planar_quadrotor.md](06_planar_quadrotor.md) for 2D flight dynamics

Welcome to fomodynamics!
