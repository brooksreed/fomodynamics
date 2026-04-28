# Testing Approach

This document describes the overall testing philosophy and structure for the BLUR codebase.

## Philosophy

| Concern | Mechanism | Purpose |
|---------|-----------|---------|
| **Correctness** | pytest | Physics validation, edge cases, error paths |
| **Usability** | Notebooks | API works as documented, examples execute |
| **Sanity** | Manual | Physical behavior looks reasonable (largely manual) |

## Test Organization

### Directory Structure

```
tests/
├── core/              # Quaternion, circular ops, units
├── simulator/
│   ├── jax/          # JAX implementations + bridge tests
│   ├── moth/         # Moth model tests (params, forces, damping)
│   ├── casadi/       # CasADi equivalence tests
│   └── _legacy/      # NumPy reference (archived)
├── analysis/          # Data loading, filtering, operations
│   └── moth/         # Moth-specific analysis tests
├── control/           # LQR controller tests
├── estimation/        # EKF, LQG estimation tests
│   └── moth/         # Moth estimation tests
├── ocp/               # Optimal control problem tests
│   └── moth/         # Moth OCP tests
├── test_integration.py    # Cross-package pipeline tests
└── test_notebooks.py      # Notebook execution + validation
```

### Test Categories

| Category | Purpose | Example Files |
|----------|---------|---------------|
| Unit | Single function/class correctness | `test_quaternion.py`, `test_params.py` |
| Integration | Cross-package pipelines | `test_integration.py` |
| Bridge | JAX vs NumPy equivalence | [`test_bridge.py`](https://github.com/brooksreed/blur/blob/main/tests/simulator/jax/test_bridge.py) |
| Notebook | API usability, examples work | `test_notebooks.py` |

## Running Tests

### Common Commands

```bash
# All tests
uv run pytest tests/ -v

# Fast (skip notebooks)
uv run pytest -m "not notebooks"

# Only notebook smoke tests (fast)
uv run pytest -m notebooks_smoke -v

# Full notebook validation (slow)
uv run pytest -m notebooks_validation -v

# Specific package
uv run pytest tests/core/ -v
uv run pytest tests/simulator/ -v
uv run pytest tests/analysis/ -v
```

### Markers

| Marker | Description |
|--------|-------------|
| `slow` | Tests that take >30s each (moth LQR, estimation). Skip with `-m "not slow"` |
| `notebooks` | All notebook tests |
| `notebooks_smoke` | Fast notebooks only (01-02) |
| `notebooks_validation` | Full execution with output validation |

### Marker Usage

```python
@pytest.mark.notebooks
@pytest.mark.notebooks_smoke
def test_notebook_smoke(notebook_name):
    """Fast smoke test for notebooks."""
    ...

@pytest.mark.notebooks
@pytest.mark.notebooks_validation
def test_notebook_produces_output(notebook_name):
    """Validation test requiring full execution."""
    ...
```

## Notebook Testing

See [`notebooks/TESTING.md`](../notebooks/TESTING.md) for notebook-specific documentation.

**Key points**:
- Notebooks 05 (JIT performance) and 06 (autodiff sensitivity) are **excluded from smoke tests** due to expensive timing/optimization loops
- Session-scoped caching: notebooks execute once per session, results reused
- Per-notebook timeouts configurable via `BLUR_NOTEBOOK_TIMEOUT` env var

**Configuration** (from [`test_notebooks.py`](https://github.com/brooksreed/blur/blob/main/tests/test_notebooks.py)):

```python
# Fast notebooks for smoke tests
# Note: Exploration notebooks (03_boat_2d, 04_quadrotor, etc.) were consolidated
# into the getting_familiar/ series and are not tested via this mechanism.
NOTEBOOKS_SMOKE = ["01_explore_atlas_data", "02_filter_comparison"]

# Slow notebooks excluded from smoke tests
NOTEBOOKS_SLOW = ["05_jax_jit_performance", "06_autodiff_sensitivity"]
```

## Physical Validation

### Automated Checks

Tests verify known physical properties:

| Test | Verification |
|------|--------------|
| Pendulum period | Small-angle period matches $2\pi\sqrt{L/g}$ |
| Energy conservation | Total energy stays constant for conservative systems |
| Drop test | Free-fall matches analytical $\frac{1}{2}gt^2$ |
| Hover equilibrium | Quadrotor with hover thrust stays at origin |

### Manual Validation

Physical behavior sanity checks are **largely manual**:

- Visual inspection of simulation trajectories
- Notebook plots for qualitative behavior
- Parameter sensitivity exploration

This is an area for potential improvement - more automated physics validation tests could be added.

## Tolerance Levels

Different tolerances for different test types:

| Tolerance | Value | Use Case |
|-----------|-------|----------|
| `DERIV_RTOL` | 1e-12 | Pure math (derivative computation) |
| `TRAJ_RTOL` | 1e-10 | Trajectory (RK4 accumulates error) |
| `ANALYTICAL_RTOL` | 1e-6 | Analytical property verification |

See [`tests/simulator/jax/conftest.py`](https://github.com/brooksreed/blur/blob/main/tests/simulator/jax/conftest.py) for tolerance constants.

## Adding New Tests

### New Model

1. Add unit tests in `tests/simulator/jax/test_<model>_jax.py`
2. Add bridge tests in `test_bridge.py` (if legacy implementation exists)
3. Add integration test in `test_integration.py`
4. Optionally add exploration notebook

### New Analysis Feature

1. Add unit tests in `tests/analysis/test_<feature>.py`
2. Ensure circular ops use `test_operations.py`

### New Notebook

1. Add to `NOTEBOOKS_SMOKE` or `NOTEBOOKS_SLOW` in `test_notebooks.py`
2. Add validation tests if needed

## Related Documentation

- [`docs/plans_archive/legacy_to_jax_validation.md`](plans_archive/legacy_to_jax_validation.md) - Details on JAX migration validation
- [`notebooks/TESTING.md`](../notebooks/TESTING.md) - Notebook-specific testing guide
