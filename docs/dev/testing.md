# Testing Reference

Practical commands, tiers, and JAX memory management for running BLUR tests.

**Related docs:**
- [../overall_testing_approach.md](../overall_testing_approach.md) — testing philosophy and structure
- [../timestep_guide.md](../timestep_guide.md) — timestep selection and RK4 stability

---

## Test Organization

- Unit tests for each package in `tests/{package}/`
- Moth tests in model-specific subdirs: `tests/{simulator,estimation,ocp,analysis}/moth/`
- Integration tests in `tests/test_integration.py`
- Test both success and error paths
- Physics verification tests (pendulum period, drop test, energy conservation)
- Xfail markers use category tags in the reason string: `[trim-dep]`, `[casadi-pending]`, `[threshold-stale]`, `[viz-dep]`, `[pre-existing]`. Always include a category tag when adding xfails.

## Test Tiers

| Tier | Command | Time | Tests | Use Case |
|------|---------|------|-------|----------|
| Smoke | `uv run pytest tests/core/ tests/analysis/ --ignore=tests/analysis/moth` | <30s | ~400 | Quick sanity check |
| Moth fast | `uv run pytest tests/simulator/moth tests/analysis/moth -m "not slow"` | ~60s | ~300 | Moth params, model, forces |
| Fast (parallel) | `uv run pytest -m "not slow" -n 4 --dist=loadfile` | ~2-5 min | ~2075 | Development iteration |
| Fast (seq) | `uv run pytest -m "not slow"` | ~5-10 min | ~2075 | Development iteration (no xdist) |
| Full | `uv run pytest` | varies | ~2200 | Pre-merge validation (run in batches if memory-constrained) |

### Timing reality check

The timing estimates above are approximate and can vary significantly. Key facts:

- **JAX JIT memory accumulates** across a long single-process run. If the full suite swap-thrashes on your machine, split it into smaller invocations by directory (e.g. `uv run pytest tests/simulator`, then `uv run pytest tests/ocp`, etc.).
- **The slowest test areas** are moth estimation and moth LQR tests. These dominate the "slow" marker.
- **Avoid running multiple pytest processes concurrently** — they compete for memory and push each other into swap.

If you notice timing that doesn't match these estimates, file an issue.

## Canonical (dt, duration) Combos

Tests should use one of these 10 canonical `(dt, duration)` pairs unless a specific value is required for physics reasons. Using canonical combos lets the session-scoped pre-warm fixtures pay the JIT compile cost once at session start, so individual tests hit the warm cache.

| # | dt | duration | Rationale |
|---|------|----------|-----------|
| 1 | 0.01 | 1.0 | Primary standard |
| 2 | 0.01 | 0.5 | Short-sim standard |
| 3 | 0.001 | 1.0 | High-accuracy standard |
| 4 | 0.01 | 2.0 | Medium-duration |
| 5 | 0.01 | 5.0 | Long-sim standard |
| 6 | 0.001 | 2.0 | High-accuracy medium |
| 7 | 0.001 | 5.0 | High-accuracy long |
| 8 | 0.005 | 2.0 | Moth standard |
| 9 | 0.005 | 1.0 | Moth short-sim |
| 10 | 0.005 | 5.0 | Moth long-sim |

The canonical combos are defined in `tests/conftest.py` as `CANONICAL_SIM_COMBOS`.

### Guidance for new tests

- **Default to a canonical combo.** Pick the one closest to your needs.
- **If your test needs a specific dt for physics** (e.g., energy conservation at dt=0.0001, rate-limit arithmetic at dt=0.02), use the specific value and document why.
- **Duration doesn't usually matter** for non-physics tests. If you just need "some simulation output," use (0.01, 0.5) or (0.01, 1.0).

### Documented exceptions

These tests intentionally use non-canonical combos:

| Combo | Files | Reason |
|-------|-------|--------|
| dt=0.0001 | cartpole, quadrotor | Energy conservation — needs very small dt |
| dt=0.0005 | bicycle | High-speed (30 m/s) stability — needs fine timestep |
| dt=0.02 | integrator_constraints | dt value is part of rate-limit arithmetic assertions |
| dt=0.1 | grad | Deliberately testing coarse timestep behavior |
| dt=0.3 | integrator_time_grid | Non-divisible dt edge case test |
| (0.001, 0.01/0.02) | moth_model, moth_open_loop, linearize | Linearization/equilibrium over very short windows |
| (0.005, 0.02) | moth_forces, moth_open_loop | Equilibrium test over very short unstable window |
| (0.01, 30.0/40.0) | validation tests | Long convergence — needs extended duration |
| Computed durations | pendulum, boat2d, trajectory_tracking | Physics-driven (3.5x period, 10x tau, etc.) |

### Pre-warm fixtures

Two session-scoped autouse fixtures pre-warm the JIT cache at session start:

- **`_prewarm_jit_cache`** (root conftest): Runs `SimplePendulum` for all 7 non-moth combos (dt=0.01, dt=0.001). Warms the JIT cache for the `simulate()` + RK4 integrator code paths. Note: different model pytree structures (e.g., Quadrotor vs SimplePendulum) may still trigger separate compiles.
- **`_prewarm_moth_jit_cache`** (moth conftest): Runs `Moth3D(MOTH_BIEKER_V3)` for the 3 moth combos (dt=0.005). Moth3D has a different pytree structure and needs its own warm-up.

Both fixtures catch `ImportError` gracefully, so tests still work if JAX is not installed.

## Batched Execution (OOM-safe)

If a long single-process run swap-thrashes, split by directory to keep JAX JIT memory bounded:

```bash
uv run pytest tests/core tests/analysis --ignore=tests/analysis/moth
uv run pytest tests/simulator -m "not slow"
uv run pytest tests/ocp -m "not slow"
uv run pytest tests/control -m "not slow"
uv run pytest tests/estimation -m "not slow"
uv run pytest -m slow                          # only the slow markers, in isolation
```

For per-test process isolation (CPU only — incompatible with GPU mode), add `--forked`:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/control -v --forked
```

## Recommended Development Workflow

```bash
# Quick check after small changes (<30s)
uv run pytest tests/core tests/analysis --ignore=tests/analysis/moth -v

# Moth-specific iteration (~60s)
uv run pytest tests/simulator/moth tests/analysis/moth -m "not slow"

# Standard development iteration (~2-5 min parallel)
uv run pytest -m "not slow" -n 4 --dist=loadfile -v

# Before creating PR — full suite (split into batches if memory-constrained)
uv run pytest
```

## JAX Memory Management

JAX JIT compilation can consume significant memory, especially with slow-marked optimization tests. For CPU runs, add `--forked` to isolate each test in a subprocess (`--forked` is incompatible with GPU mode due to `os.fork()` + CUDA conflicts).

### GPU Support

BLUR auto-detects GPU when CUDA-enabled JAX is installed. Install separately via `bash cuda-setup.sh` or `uv pip install "jax[cuda12]"`. GPU memory is auto-limited to 65% (important on WSL where GPU also drives display).

### Configuration

Key environment variables:
- `JAX_PLATFORMS=cpu` — force CPU (recommended default for tests)
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` — adjust GPU memory fraction
- `FMD_USE_FLOAT32=1` — use float32 instead of float64 (~2x faster on GPU, less numerical precision)

Use `FMD_DTYPE` / `FMD_NP_DTYPE` from `fmd.core` for consistent dtype across the codebase. Default is float64, required for numerical stability in long simulations and RK4 integration.

### Running Tests

**Default to CPU for tests** — CPU is typically faster than GPU for BLUR's problem sizes:
```bash
env JAX_PLATFORMS=cpu uv run pytest tests/control/ -v --forked
```

**Test timing reference:**
- `tests/simulator/moth/`: ~16 min for LQR tests, ~31 min for estimation tests

**If you encounter OOM errors:** run tests in smaller batches, reduce GPU memory fraction (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`), force CPU (`JAX_PLATFORMS=cpu`), or clear JAX cache (`rm -rf ~/.cache/jax /tmp/jax*`).
