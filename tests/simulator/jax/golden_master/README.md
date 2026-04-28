# Golden master NPZ files (public)

This directory holds the public golden-master reference data exercised by
tests under `tests/fmd/simulator/jax/`. The files here are byte-for-byte
copies of (a subset of) the private golden masters in
`tests/private/simulator/jax/golden_master/`.

## Files

- `scg_cartpole_true.npz` — referenced by
  `tests/fmd/simulator/jax/test_euler_exact_match.py`
  (`TestCartpoleEulerExactMatch`) and by
  `tests/private/simulator/jax/test_quadrotor3d_scg_library_exact_match.py`.
- `scg_quadrotor3d_true.npz` — referenced by both as well.

## Why duplicated?

The duplication is intentional. The public/private test trees should not
cross-reference each other at runtime — that would break the Phase 7
isolated staging directory, where only `tests/fmd/` ships and
`tests/private/` is absent. Keeping a copy of each NPZ in the public tree
lets `tests/fmd/` collect and run with `tests/private/` removed.

When regenerating either NPZ (e.g. via
`tests/private/simulator/jax/generate_golden_values.py`), copy the new
bytes into both locations so they remain in sync.

## Background on the broader benchmark fixtures

The richer benchmark fixtures (`scg_cartpole_reference.npz`,
`scg_quadrotor_reference.npz`, `f1tenth_bicycle_reference.npz`,
`blur_lqr_gains.npz`, the `scg_planar_quadrotor_*` and
`scg_quadrotor3d_*` PyBullet/symbolic variants) live exclusively under
`tests/private/simulator/jax/golden_master/`. They are generated from
private benchmark adapters (safe-control-gym, f1tenth_gym, pybullet) and
are not part of the public test surface. See the README in the private
directory for the regeneration commands and on-disk format.
