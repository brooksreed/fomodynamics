# Adding Components Reference

Templates and conventions for adding new force components, models, schemas, and scripts.

**Related docs:**
- [../simulator_models.md](../simulator_models.md) — state vectors, model details, presets

---

## New Force/Moment Component

```python
import equinox as eqx
import jax.numpy as jnp
from fmd.simulator.components import JaxForceElement

class MyForce(JaxForceElement):
    my_param: float  # Equinox module fields

    def compute(self, t, state, control):
        force = jnp.array([0.0, 0.0, 0.0])   # Body frame
        moment = jnp.array([0.0, 0.0, 0.0])  # Body frame
        return force, moment
```

## New Data Schema

Create in `fmd/analysis/loaders/schemas/`:

```python
from fmd.analysis.loaders.base import Schema

class MySchema(Schema):
    @property
    def name(self) -> str:
        return "my_schema"

    @property
    def required_columns(self) -> list[str]:
        return ["time", "col1", "col2"]

    def normalize(self, df):
        # Convert to SI units, add 'time' column
        ...
```

## JAX↔CasADi Model Equivalence

When creating a new dynamics model that has both JAX and CasADi implementations, you MUST add equivalence tests to `tests/simulator/casadi/test_equivalence.py`. These tests verify that both implementations produce identical results.

**Three-Level Testing:**
- **Level 0 (Derivatives):** 100+ random states produce identical `forward_dynamics` outputs
- **Level 1 (Jacobians):** A, B matrices match at 50+ random points
- **Level 2 (Trajectories):** 100+ RK4 steps produce identical trajectories

**Tolerances (from `tests/simulator/casadi/conftest.py`):**
- Derivatives: rtol=1e-12, atol=1e-14
- Jacobians: rtol=1e-10, atol=1e-12
- Trajectories: rtol=1e-10, atol=1e-12

See `TestCartpoleEquivalence` or `TestBox1DEquivalence` in `tests/fmd/simulator/casadi/test_equivalence.py` for reference implementations.

## Parameter System

Model parameters are managed through immutable `attrs` classes in `fmd.simulator.params`:

- All params classes are `frozen=True` (immutable)
- Values validated at construction (positive mass, valid inertia, no NaN/Inf)
- Use `attrs.evolve()` to create modified copies

See [docs/simulator_models.md](../simulator_models.md) for available parameter classes and presets.

## Script Output Convention

All scripts save output to `results/<script-name>/<timestamp>/`:
- Default: `results/moth-smoke-test/2026-02-20_163000/`
- Override: `--output-dir <path>`
- Timestamp format: `%Y-%m-%d_%H%M%S`
- `results/` is gitignored — don't commit generated output

Viz scripts (.rrd) save directly to `results/` without timestamps (overwritten per session).

`tests/validation/report.py` is an exception — its output goes to `docs/plans/.../validation/`
because those are committed documentation artifacts.

When adding new scripts that produce file output:
- Add `--output-dir` flag defaulting to `results/<script-name>/<timestamp>/`
- Use `os.makedirs(output_dir, exist_ok=True)`
- Print the output path so the user can find it

## Package Boundaries — Full Table

See also the import direction rules in [CLAUDE.md](../../CLAUDE.md).

| If you're adding... | Put it in... | Future package |
|---------------------|-------------|----------------|
| Generic sim infrastructure (base classes, integrator, control) | `fmd.simulator` | blur-core |
| Generic analysis (DataStream, processing, plots) | `fmd.analysis` | blur-core |
| Core math (quaternion, circular ops, units) | `fmd.core` | blur-core |
| Tow boogie forces, params, model changes | Existing tow boogie files | blur-tow-boogie |
| Moth forces, params, model changes | Existing moth files | blur-moth |
| Generic OCP infrastructure (MultipleShootingOCP), estimation | `fmd.ocp`, `fmd.estimation` | fomodynamics |
| New toy models (generic, educational) | `fmd.simulator` | blur-core |
| New marine vehicle model | New model files (follow tow boogie pattern) | Its own package |

If you're unsure where something belongs, ask. When in doubt, keep it
out of blur-core (it's easier to promote code to core later than to
remove it).
