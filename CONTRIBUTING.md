# Contributing to fomodynamics

Thanks for your interest in `fomodynamics`. This is an alpha-stage project
and the API is not yet stable, so contributions that surface breakage or
sharpen the public surface are particularly welcome.

## Filing issues

Open issues at [https://github.com/brooksreed/blur/issues](https://github.com/brooksreed/blur/issues)
(URL will move once the public repo is extracted). When reporting a bug,
please include:

- A minimal reproducer (10–20 lines of Python is ideal).
- The output of `python -c "import jax; print(jax.devices(), jax.__version__)"`.
- The Python version and OS (`python --version`, `uname -a`).
- Whether you used `uv sync` or `pip install`.

For feature requests, describe the use-case first and the API shape
second — concrete scenarios are easier to triage than abstract API
proposals.

## Dev environment setup

The repository ships with a `uv`-based workflow. From a fresh clone:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo>
cd blur                          # repo dir; package is `fomodynamics`
uv sync                          # installs all dev + analysis + viz3d extras
```

The full developer setup (WSL2, CUDA, full vehicle list, internal scripts)
lives in [`README_PRIVATE.md`](README_PRIVATE.md).

## Running tests

```bash
# Fast suite (~5 min) — run this before sending a PR
uv run pytest -m "not slow" -n 4 --dist=loadfile

# Public-only public-package tests
uv run pytest tests/fmd -m "not slow"

# A single test file
uv run pytest tests/fmd/simulator/moth/test_model.py -v
```

The full test suite is process-isolated and takes ~3.5 hours; you don't
need to run it for most PRs. See `docs/public/dev/testing.md` for the
breakdown.

## Code style

Repo-wide rules are documented in [`CLAUDE.md`](CLAUDE.md). Highlights:

- **Simple over clever.** Prefer readable code; comments only where the
  intent isn't obvious from the code.
- **SI units internally.** Convert to display units (degrees, knots) only
  at the I/O boundary. See `docs/public/frame_conventions.md`.
- **Quaternions are scalar-first.** `[qw, qx, qy, qz]` everywhere.
- **Circular math.** Use `fmd.core.operations` helpers for angle
  subtraction / mean / wrap. Never subtract angles with `-`.
- **No mutable defaults.** Especially never `default=np.array(...)` on
  attrs fields — use `factory=lambda: np.array(...)`.

## Pull requests

- Branch from `main`. Keep PRs focused — one logical change per PR.
- Add or update tests for any behavior change. Tests live under `tests/`.
- Update relevant docs under `docs/` when adding new public surface.

## Where to put new code

Quick guide:

| What you're adding | Where it goes |
|---|---|
| Generic simulator infra (integrators, frames, control schedules) | `src/fmd/simulator/` |
| Moth physics (forces, params, scenarios, metrics) | `src/fmd/simulator/` (moth_*) |
| Simple / teaching model (pendulum-flavored) | `src/fmd/simulator/` |
| LQR / linearization | `src/fmd/simulator/` |
| Generic OCP / multiple-shooting | `src/fmd/ocp/` |
| Estimation (KF / EKF / measurement model) | `src/fmd/estimation/` |
| Analysis / loaders / plots | `src/fmd/analysis/` |

If unsure, open a discussion before doing the work — it's far cheaper to
adjust scope early than to retract something from a public release.

## License

By contributing, you agree that your contributions will be licensed under
the project's MIT license.
