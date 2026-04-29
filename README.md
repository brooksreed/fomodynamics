# fomodynamics

**Fo**iling **Mo**th **Dynamics** — a Python library for simulation, control,
and analysis of foiling Moth sailboats and related 3D vehicles.

> **Status: alpha.** This package is pre-release and the API may change
> without notice. The repository this README ships from is the staging area
> for an open-source split; expect rough edges and missing documentation
> until the first tagged release.

## What it is

`fomodynamics` (import name `fmd`) bundles:

- **A 3-DOF Moth foiling dynamics model** with main-foil + rudder lift/drag,
  hull buoyancy, mechanical wand–flap linkage, and a CasADi-based trim solver.
- **Generic 6-DOF rigid-body simulation infrastructure** built on JAX with
  RK4 / Euler / symplectic integrators, force-component composition, wave
  and environment models, constraints, and a vmap-based parameter sweep
  framework.
- **Simple / teaching models**: pendulum, cartpole, planar quadrotor,
  Boat2D, box-1D — each with matching CasADi mirrors for parity tests.
- **Classical control**: discrete LQR, EKF / Kalman estimation, generic
  multiple-shooting OCP infrastructure suitable for cartpole-style MPC.
- **Telemetry analysis**: schema-aware CSV loading, circular-aware
  filtering, time-series plotting, and a Rerun-based 3D viewer.

## Install

The package isn't on PyPI yet. From this monorepo:

```bash
uv sync                  # installs all dev + analysis + viz3d extras
```

Once published, the install will be:

```bash
pip install fomodynamics
```

## Quick start

### Simulation — a falling 6-DOF body

```python
import jax.numpy as jnp
from fmd.simulator import RigidBody6DOF, simulate, create_state, Gravity

body = RigidBody6DOF(
    mass=10.0,
    inertia=jnp.array([1.0, 2.0, 3.0]),
    components=[Gravity(mass=10.0)],
)
result = simulate(body, create_state(), dt=0.01, duration=10.0)
print(result.states[-1, 0:3])  # NED position [N, E, D]
```

### Moth — foiling sailboat

```python
import jax.numpy as jnp
from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3

moth = Moth3D(MOTH_BIEKER_V3)
result = simulate(moth, moth.default_state(), dt=0.01, duration=10.0)
```

State vector (3-DOF Moth): `[pos_d, theta, w, q]` — vertical position
(NED, +D = down), pitch, body-frame heave rate, pitch rate.

### Cartpole + LQR

```python
import jax.numpy as jnp
from fmd.simulator import Cartpole, simulate
from fmd.simulator.lqr import LQRController
from fmd.simulator.params import CARTPOLE_CLASSIC

system = Cartpole(CARTPOLE_CLASSIC)
controller = LQRController.from_linearization(
    system,
    x_eq=jnp.array([0.0, jnp.pi, 0.0, 0.0]),  # upright
    u_eq=jnp.array([0.0]),
    Q=jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1])),
    R=jnp.diag(jnp.array([0.01])),
)
result = simulate(
    system,
    jnp.array([0.0, jnp.pi - 0.2, 0.0, 0.0]),
    dt=0.01,
    duration=5.0,
    control=controller,
)
```

### Trim solver (CasADi)

```python
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.params import MOTH_BIEKER_V3

result = find_moth_trim(MOTH_BIEKER_V3, u_forward=8.0)
print(result.state, result.control)
```

## Core conventions

- **Frames**: NED world (`x=N, y=E, z=D`), FRD body (`x=Forward, y=Right,
  z=Down`). `+pos_d` means deeper / lower; `-pos_d` means rising.
- **Quaternions**: scalar-first `[qw, qx, qy, qz]`.
- **Units**: SI internally (m, m/s, radians, kg, N). Display conversions
  (knots, degrees) only at the I/O boundary.
- **Circular math**: helpers in `fmd.core.operations` for angle subtraction,
  mean, wrapping.
- **Numerical precision**: float64 by default. Override with
  `FMD_USE_FLOAT32=1`.

### Sign gotchas

The conventions above are aerospace-standard but trip up newcomers from
robotics or graphics backgrounds. Common bug sources:

- **+D is *down* in NED.** Altitude *increase* means `pos_d` *decreases*
  (more negative). A boat rising out of the water has a *more negative*
  `pos_d`. This is the most frequent sign-error source.
- **Angles are in radians internally.** Convert to degrees only for
  display / human-readable output.
- **Quaternion is scalar-first** `[qw, qx, qy, qz]` — *not* the
  `[qx, qy, qz, qw]` order used by ROS, Eigen, or most game engines.
  When bridging, reorder explicitly: `q_fmd = [q_other[3], q_other[0],
  q_other[1], q_other[2]]`.
- **Use circular-aware ops for angles**: `fmd.core.operations.circular_subtract`,
  `circular_mean`, `wrap_angle` — naïve subtraction breaks at the
  ±π wrap point.

## Documentation

Detailed reference docs live under `docs/public/`:

- `simulator_architecture.md` — overall design, components, integrators.
- `simulator_models.md` — every model's state/control vectors and presets.
- `frame_conventions.md` — NED/FRD vs other libraries.
- `moth_modeling.md`, `moth_3dof_equations.md`, `trim_solver.md`,
  `wand_linkage_kinematics.md` — moth-specific physics.
- `physical_intuition_guide.md` — sanity-check checklists for dynamics
  work.
- `control_guide.md` — LQR design, tuning, integrator selection.
- `jax_simulator_guide.md`, `timestep_guide.md` — JAX patterns and
  timestep selection.
- `dev/{adding_components,moth_reference,testing,visualization}.md` —
  developer-facing references.
- `guides/getting_familiar/` — tutorial series paired with the public
  notebooks under `notebooks/public/getting_familiar/`.

## Examples and notebooks

- `examples/` — runnable scripts (`pendulum.py`, `drop_test.py`,
  `spinning_disk.py`, `moth_sim_smoke_test.py`, `moth_open_loop.py`,
  `moth_lqg_calm_water.py`, `moth_geometry.py`, `plot_moth_smoke_test.py`).
- `notebooks/public/` — Jupyter notebooks
  (`moth_observability_analysis.ipynb`, `getting_familiar/*`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: open an issue,
keep changes small and focused, run `uv run pytest -m "not slow"` before
sending a PR.

## License

MIT. See [LICENSE](LICENSE).
