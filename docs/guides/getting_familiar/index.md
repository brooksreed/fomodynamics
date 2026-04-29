# Getting Familiar with fomodynamics

Welcome to the Getting Familiar guide for **fomodynamics** (Foiling Moth Dynamics). This index helps you navigate the guide sections based on your goals and interests.

---

## Guide Sections

### Foundations

| Section | Title | Time | Description |
|---------|-------|------|-------------|
| [00](00_introduction.md) | Introduction | 30 min | fomodynamics overview, goals, prerequisites, installation, and guide roadmap |
| [01](01_jax_primer.md) | JAX Primer | 30 min | Essential JAX concepts for fomodynamics users: JIT compilation, pytrees, and functional patterns |
| [02](02_core_concepts.md) | Core Concepts | 1 hr | NED frame conventions, quaternion attitude representation, SI units, and state vector layout |

### Model Tutorials

| Section | Title | Time | Description |
|---------|-------|------|-------------|
| [03](03_simple_pendulum.md) | Simple Pendulum | 30 min | Your first fomodynamics simulation with the simplest model |
| [04](04_cartpole.md) | Cartpole | 1 hr | A 4-state coupled system introducing control challenges |
| [06](06_planar_quadrotor.md) | Planar Quadrotor | 1 hr | 2D flight dynamics with thrust and torque control |

### Advanced Topics

| Section | Title | Time | Description |
|---------|-------|------|-------------|
| [10](10_extending_fomodynamics.md) | Extending fomodynamics | 1 hr | Adding new dynamic models and force elements |

---

## Reading Paths

Choose a path based on what you want to accomplish:

### "I want to run simulations"

Start here if you want to get simulations running quickly.

1. [00 - Introduction](00_introduction.md) - Setup and orientation
2. [02 - Core Concepts](02_core_concepts.md) - Understand the coordinate frames and state vectors
3. Pick a model tutorial:
   - [03 - Simple Pendulum](03_simple_pendulum.md) for the gentlest start
   - [04 - Cartpole](04_cartpole.md) for a coupled control system
   - [06 - Planar Quadrotor](06_planar_quadrotor.md) for 2D flight dynamics
4. [dev/testing.md](../../../dev/testing.md) - Verification and testing patterns

**Estimated time: 3-4 hours**

### "I want to understand controls"

Start here if you want to design controllers for fomodynamics systems.

1. [00 - Introduction](00_introduction.md) - Setup and orientation
2. [02 - Core Concepts](02_core_concepts.md) - State vectors and dynamics foundations
3. [04 - Cartpole](04_cartpole.md) - Work through a coupled control example
4. [06 - Planar Quadrotor](06_planar_quadrotor.md) - 2D flight dynamics
5. [../../control_guide.md](../../control_guide.md) - LQR design, eigenvalue analysis, and timestep selection

**Estimated time: 5 hours**

### "I want to add new physics"

Start here if you want to extend fomodynamics with custom models or force elements.

1. [00 - Introduction](00_introduction.md) - Setup and orientation
2. [02 - Core Concepts](02_core_concepts.md) - Conventions your model must follow
3. [03 - Simple Pendulum](03_simple_pendulum.md) - See how a minimal model is structured
4. [10 - Extending fomodynamics](10_extending_fomodynamics.md) - Create your own models and components

**Estimated time: 3 hours**

### "I want to trust the results"

Start here if you need to validate fomodynamics against other tools or verify correctness.

1. [02 - Core Concepts](02_core_concepts.md) - Understand what fomodynamics is computing
2. [dev/testing.md](../../../dev/testing.md) - Testing philosophy, tolerance tiers, and cross-validation
3. Model sections for physics derivations:
   - [03 - Simple Pendulum](03_simple_pendulum.md) - Analytical solutions
   - [04 - Cartpole](04_cartpole.md) - Golden values and analytical equilibria
   - [06 - Planar Quadrotor](06_planar_quadrotor.md) - 2D thrust and moment dynamics

**Estimated time: 3 hours**

---

## Prerequisites

Before starting this guide, you should have:

- Python 3.10+ installed
- Basic familiarity with NumPy array operations
- Understanding of differential equations (conceptual level)
- fomodynamics installed per the [README](../../../../README.md)

No prior JAX experience is required; section 01 covers what you need.

---

## Total Reading Time

The complete guide takes approximately **12 hours** to work through. However, most users will follow a specific reading path taking **3-5 hours**.

---

## Getting Help

- Check the [README](../../../../README.md) for quick start, install, and core conventions
- Review `docs/frame_conventions.md` for detailed coordinate frame documentation
- See `docs/simulator_models.md` for model-specific parameter references
