# Moth Trim Solver

Technical reference for the Moth 3DOF trim solver (`fmd.simulator.trim`).

## Overview

The trim solver finds steady-state equilibrium points where all state derivatives are approximately zero. The current, canonical implementation (`find_moth_trim`, `calibrate_moth_thrust`, `calibrate_moth_thrust_table`, all in `fmd.simulator.trim_casadi`/`trim_calibration`) uses CasADi/IPOPT — see the [CasADi/IPOPT Trim Solver](#casadiipopt-trim-solver) section below for the current two-phase algorithm, weights, and API.

> **Historical note:** the sections immediately below (through "Bounds and Variable Scaling") describe the **retired SciPy solver** (L-BFGS-B primary + SLSQP polish, `fmd.simulator.trim.find_trim`) that `find_moth_trim` used to wrap. `find_moth_trim` is now a drop-in CasADi replacement of the same name — SciPy-specific kwargs are accepted and silently ignored. These sections are kept for the objective-function design rationale (term roles, weight-sensitivity analysis), which broadly motivated the CasADi Phase 1 penalty design, but the concrete weight *values* and the "Optimizer Details" (SciPy L-BFGS-B/SLSQP) no longer reflect the current implementation. Two exceptions inside this block are current, not historical: "Typical Results" (the calibrated thrust curve, CasADi-sourced and dated 2026-07-16) and "Analytical xdot Correction" / "Analytical xdot Correction (Detailed)" (rewritten to describe the current exact CasADi formulation).

Two modes:
- **Default trim** (`find_moth_trim`): Find equilibrium at a given thrust. Used for simulation setup and control design.
- **Thrust calibration** (`calibrate_moth_thrust`): Find the minimum-drag equilibrium by treating thrust as a free optimization variable.

## State and Constraints

Moth 3DOF state: `[pos_d, theta, w, q, u]` (ride height, pitch, heave vel, pitch rate, surge vel).

### Kinematic constraints (always applied)

```
q = 0           (zero pitch rate at trim)
w = u * tan(θ)  (kinematic consistency: heave = surge * sin(pitch))
```

These are hard constraints applied before every objective evaluation. The `w` value at trim is **not a residual** — it's a constrained kinematic variable. At θ = 0.34°, w = 10.0 * tan(0.34°) = 0.059 m/s. This is physically correct.

### What the solver actually minimizes

The xdot (state derivative) residuals are the true measure of trim quality:

| Component | Meaning | Typical value at trim | Scale |
|-----------|---------|----------------------|-------|
| pos_d_dot | Vertical velocity | ~0 (exact via w constraint) | 0.05 m/s |
| theta_dot | Pitch rate | 0 (exact, equals q which is constrained) | 0.035 rad/s |
| w_dot | Heave acceleration | ~0.001 m/s² | 0.5 m/s² |
| q_dot | Pitch acceleration | ~0.00003 rad/s² | 0.35 rad/s² |
| u_dot | Surge acceleration | ~0.002 m/s² | 0.2 m/s² |

## Objective Function

The objective is a weighted sum of normalized squared terms:

```
J = Σ(xdot_i / XDOT_SCALE_i)²           # residual: xdot ≈ 0
  + w_u * ((u - u_target) / U_SCALE)²    # speed pin
  + w_theta * (θ / THETA_SCALE)²         # pitch regularization
  + w_flap * (flap / FLAP_SCALE)²        # flap regularization
  + w_elev * (elev / ELEV_SCALE)²        # elevator regularization
  + w_pos_d * ((pos_d - ref) / POS_D_SCALE)²  # ride height
  + w_drag * (hydro_drag / DRAG_SCALE)²  # drag minimization (default mode)
  + w_thrust * (thrust / THRUST_SCALE)²  # thrust minimization (calibration mode)
```

### Term roles

**Residual terms** (unweighted, always active): The xdot/scale terms are the primary objective. At a good trim point, these contribute < 0.001 to J.

**Speed pin** (`w_u=50.0`): Strongly pins surge velocity to the target speed. Prevents the optimizer from "cheating" by changing speed to reduce forces.

**State/control regularization** (`w_theta`, `w_flap`, `w_elev`, `w_pos_d`): Break the null space — when multiple equilibria exist at a given speed, prefer the one with small angles and near-reference ride height. These are **not** constraints, they're soft preferences.

**Drag/thrust minimization** (`w_drag` or `w_thrust`): In the null space of the residual, prefer the minimum-drag operating point. In default mode, this is done via `w_drag` (hydrodynamic drag extraction). In calibration mode, thrust minimization subsumes drag minimization (at equilibrium, thrust = drag).

### Weight sets

| Weight | Default | Calibration | Role |
|--------|---------|-------------|------|
| w_u | 50.0 | 50.0 | Speed pin (strong, both modes) |
| w_theta | 5.0 | 0.5 | Pitch reg (relaxed during calibration to let physics drive θ) |
| w_flap | 1.0 | 0.1 | Flap reg (relaxed during calibration) |
| w_elev | 0.5 | 0.1 | Elevator reg (relaxed during calibration) |
| w_pos_d | 0.5 | 0.1 | Ride height reg (relaxed during calibration) |
| w_drag | 0.01 | 0.0 | Drag min (off during calibration, subsumed by thrust) |
| w_thrust | — | 0.01 | Thrust min (null-space breaker) |

### Why w_thrust = 0.01 works (and lower values work too)

The theta regularization (w_theta) indirectly minimizes thrust: pushing θ toward 0° reduces foil AOA, which reduces drag, which reduces the thrust needed for equilibrium. This means even w_thrust=0.0 converges to the same solution (tested: 0.0, 0.001, 0.01 all give thrust ≈ 124.7N at 10 m/s).

The w_thrust term is insurance against edge cases where theta regularization alone doesn't fully break the null space. At 0.01, the thrust term contributes ~0.02 to the objective at typical thrust values — genuinely a null-space breaker, not a competing objective.

**Sensitivity (at 10 m/s):**

| w_thrust | Thrust | Fx residual | Notes |
|----------|--------|-------------|-------|
| 0.0 | 124.7 N | -0.014 N | Best force balance |
| 0.001 | 124.7 N | -0.025 N | Identical solution |
| 0.01 | 124.3 N | -0.195 N | Current default — good |
| 0.1 | 120.8 N | -1.9 N | Starting to bias solution |
| 1.0 | 100.8 N | -15.4 N | Broken — force imbalance |

## Thrust Calibration

### Architecture

Single-shot optimization with thrust as a free variable (6 DOFs: pos_d, theta, flap, elevator, u, thrust). Replaced a previous two-level Newton architecture where an inner solver found equilibrium at fixed thrust and an outer Newton loop iterated on thrust.

### Analytical xdot correction

Thrust enters the EOM through the sail force's NED-horizontal-to-body-frame rotation (`F_bx = thrust*cos(theta)`, `F_bz = thrust*sin(theta)`), so it is not simply additive in body-x. Rather than modifying the force model, the CasADi solver (`trim_casadi._build_xdot_expr`) builds the base `xdot` from a **zero-thrust** model and adds the full thrust contribution symbolically:

```python
f_bx = thrust * cos(theta)
f_bz = thrust * sin(theta)
xdot[W] += f_bz / m_eff_heave                            # heave acceleration
xdot[Q] += (ce_z * f_bx - ce_x * f_bz) / i_eff            # pitch acceleration (full moment arm)
xdot[U] += f_bx / m_eff_surge                             # surge acceleration
```

This is exact at all pitch angles (no small-angle approximation, no delta-from-baked-thrust) since it is built from a zero-thrust baseline rather than perturbing an existing nonzero-thrust solution. Note it touches **three** states — `w_dot`, `q_dot`, and `u_dot` — not just surge and pitch: thrust has a body-z component whenever theta != 0, so it also contributes to heave.

### Model parameters

| Parameter | Expression | Source |
|-----------|-----------|--------|
| m_eff_surge | total_mass + added_mass_surge | `Moth3DCasadiExact` |
| m_eff_heave | total_mass + added_mass_heave | `Moth3DCasadiExact` |
| i_eff | iyy + added_inertia_pitch | `Moth3DCasadiExact` |
| ce_x, ce_z | sail_ce_position - cg_offset | `Moth3DCasadiExact` |

The base `xdot` is built from a **zero-thrust** model (`sail_thrust_coeff=1e-10`, no lookup table — see `_zero_thrust_model`), not a "baked" nonzero thrust, so there is no delta/perturbation term to track — the full thrust force/moment is added directly (see `_build_xdot_expr` above).

### Post-solve validation

Because `_build_xdot_expr` is exact (not a linearized correction), post-solve force balance is just `max_xdot_residual` on `CalibrationTrimResult` — the max absolute value of the full 5-state `xdot` (including the thrust contribution) at the solution. No separate "corrected" residual bookkeeping is needed.

## Optimizer Details

### Two-phase approach

1. **L-BFGS-B** (primary): Quasi-Newton with box constraints. Handles the smooth, well-conditioned landscape well. SciPy finite-difference gradients (faster than JAX eager-mode for ~6 variables).
2. **SLSQP** (polish): Short run from L-BFGS-B solution. Can improve slightly due to different step-size strategy.

### Why SciPy FD > JAX gradients for this problem

At ~4-6 decision variables, JAX eager-mode gradient overhead dominates. SciPy finite-difference is ~2.5x faster. JAX gradients are available via `use_jax_grad=True` but off by default.

### Adaptive regularization

When state targets are pinned (`target_theta` and/or `target_pos_d`), the problem has fewer free variables. Control weights are reduced to avoid fighting the residual minimization.

## Typical Results

### Calibrated thrust curve (MOTH_BIEKER_V3 preset, physics-correctness batch)

> **Snapshot**: 2026-07-16, branch `physics-correctness` (post
> WAVE-AOA/ETA-DEPTH/QUAT/TRIM-NULL/FSL fixes; free-surface lift on).
> Calibrated with the CasADi/IPOPT two-phase solver, **pinned at
> pos_d = DEFAULT_POS_D_REF (-1.40 m)**, 30 deg heel, cold start (no seeds).
> The pinned solve stays on the primary trim branch by construction — the
> free (regularized) solve is branch-ambiguous above ~18 m/s, where cold
> starts can land on a nose-down secondary branch with 17-29% lower thrust.

| Speed (m/s) | Thrust (N) |
|-------------|------------|
| 6.0 | 47.6 |
| 7.0 | 50.3 |
| 8.0 | 56.3 |
| 9.0 | 64.9 |
| 10.0 | 75.5 |
| 11.0 | 87.9 |
| 12.0 | 102.0 |
| 13.0 | 117.7 |
| 14.0 | 134.8 |
| 15.0 | 153.4 |
| 16.0 | 173.4 |
| 17.0 | 194.9 |
| 18.0 | 217.6 |
| 19.0 | 241.8 |
| 20.0 | 267.3 |

**Observations:**

- Thrust is monotonically increasing across the full 6-20 m/s range, approximately following a u^2 drag law.
- Pitch decreases smoothly from +4.0 deg (6 m/s, nose-up for lift at low dynamic pressure) through zero near 14 m/s to -0.49 deg at 20 m/s; flap and elevator stay within +/-0.5 deg across the band.
- Leeward tip depth at trim grows from +2.2 cm (6 m/s, the binding ventilation-margin case) to +7.4 cm (20 m/s); depth_factor 0.998 everywhere.
- All 15 speeds converge with residuals < 1e-8 (hard-constraint phase).
- Historical note: the pre-AoA-fix model was non-monotonic at low speed (235, 155, 124, 111, 108 N at 6-10 m/s) from lift-into-surge leakage via a conflated AoA; that was fixed in 2026-03.

**Open-loop stability:** The system has a positive real eigenvalue at all speeds (~+0.33 to +0.58 rad/s), with instability increasing with speed. Time constants of 1.7-3.0 s are physically reasonable for an open-loop unstable foiling boat. See `tests/simulator/moth/test_damping.py` for current reference eigenvalues.

### Convergence properties

- Seed invariant: initial guesses of 30, 70, 150N converge within 10N at 10 m/s
- Monotonic: thrust increases monotonically with speed from 10+ m/s (validated 10-20 m/s)
- Force balance: |total_fx| < 0.3N at 8+ m/s

## Bounds and Variable Scaling

### Decision variable bounds

| Variable | Lower | Upper | Units | Notes |
|----------|-------|-------|-------|-------|
| pos_d | -0.6 | 0.5 | m | Ride height (NED, negative = above waterline) |
| theta | -0.3 | 0.3 | rad | Pitch angle (~±17°) |
| u | u_target - 0.5 | u_target + 0.5 | m/s | Surge velocity (tight band) |
| main_flap | MAIN_FLAP_MIN | MAIN_FLAP_MAX | rad | From moth_3d module constants |
| rudder_elevator | RUDDER_ELEV_MIN | RUDDER_ELEV_MAX | rad | From model control_bounds or defaults |
| thrust_total | 10.0 | 500.0 | N | Calibration mode only |

### Objective normalization scales

Each term is divided by its scale before squaring, so the optimizer sees O(1) gradients:

| Scale | Value | Units | Rationale |
|-------|-------|-------|-----------|
| XDOT_SCALE[pos_d_dot] | 0.05 | m/s | Vertical velocity |
| XDOT_SCALE[theta_dot] | 0.035 | rad/s | ~2 deg/s pitch rate |
| XDOT_SCALE[w_dot] | 0.5 | m/s² | Heave acceleration |
| XDOT_SCALE[q_dot] | 0.35 | rad/s² | ~20 deg/s² pitch acceleration |
| XDOT_SCALE[u_dot] | 0.2 | m/s² | Surge acceleration |
| U_SCALE | 0.25 | m/s | Speed deviation |
| THETA_SCALE | 0.035 | rad | ~2° pitch |
| FLAP_SCALE | 0.052 | rad | ~3° flap |
| ELEV_SCALE | 0.035 | rad | ~2° elevator |
| POS_D_SCALE | 0.05 | m | Ride height |
| DRAG_SCALE | 20.0 | N | Hydrodynamic drag |
| THRUST_SCALE | 50.0 | N | ~half typical thrust |

## Analytical xdot Correction (Detailed)

### Why correct analytically rather than modifying the model?

Thrust enters the equations of motion through `MothSailForce.compute_moth()` (in `moth_forces.py`). The sail force is applied in the NED horizontal plane and rotated to body frame by pitch angle: `F_bx = F_sail * cos(theta)`, `F_bz = F_sail * sin(theta)`. At small trim theta (~0-4 deg), `cos(theta) ≈ 1` so the thrust contribution is approximately linear in the body-x direction. The delta between any two thrust values produces a near-exact xdot correction at typical trim angles.

### Derivation

Let `xdot_base = forward_dynamics(state, control)` with the baked-in thrust `T_base`. We want xdot at a different thrust `T_total`:

```
delta_T = T_total - T_base
```

Thrust is body-x only (forward in FRD frame). In the 3DOF longitudinal model:

1. **Surge acceleration** (`u_dot`): thrust adds `delta_T / m_eff_surge` where `m_eff_surge = total_mass + added_mass_surge` (the effective mass including added mass in surge, from `forward_dynamics` line ~530).

2. **Pitch acceleration** (`q_dot`): thrust acts at CE height, producing a moment `delta_T * ce_z` where `ce_z = sail_ce_position_z - cg_offset_z` (the vertical offset from CG to sail centre of effort). The pitch acceleration contribution is `ce_z * delta_T / i_eff` where `i_eff = composite_iyy + added_inertia_pitch`.

3. **Other states**: `pos_d_dot`, `theta_dot` (= q), and `w_dot` are unaffected because thrust has no direct vertical force component (body-x only) and `theta_dot = q = 0` at trim.

### Exactness conditions

The correction is exact when `q = 0` (no Coriolis cross-coupling between surge thrust and pitch/heave). At trim, q is constrained to zero, so the correction is exact at the solution. During optimization iterations where q may be slightly nonzero, there is a small approximation error, but this vanishes at convergence.

## Calibrating the Thrust Table

`calibrate_moth_thrust_table` (in `fmd.simulator.trim_calibration`) sweeps a list of target speeds and runs `calibrate_moth_thrust` at each, optionally warm-starting from a JSON seed cache.

```python
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_calibration import calibrate_moth_thrust_table

results = calibrate_moth_thrust_table(
    MOTH_BIEKER_V3,
    speeds=range(6, 21),
    seed_path="trim_seeds.json",  # optional warm-start cache
)
```

### Seed cache

`seed_path` is a JSON file mapping `"<speed>" -> [pos_d, theta, w, q, u, flap, elev, thrust]` (an 8-vector). When the file exists, matching entries are used as IPOPT initial guesses; converged solutions are written back at the end of the sweep (preserving entries for speeds that were not run). Pass `save_seeds=False` to disable the write-back, or `seed_path=None` to disable the seed mechanism entirely.

The seed cache is for warm-start *speedup* — given different starts, IPOPT can converge to different points within the residual null-space, so seed-loaded results may differ by a few percent from cold-start results. Both are valid trims (residual ≪ 1e-6); the regularization weights determine which one is preferred.

### Script usage

```bash
# Default sweep: 6..20 m/s in 1 m/s steps, ./trim_seeds.json as cache
uv run python scripts/calibrate_thrust_table.py

# Custom speed list, custom output dir
uv run python scripts/calibrate_thrust_table.py --speeds 8 10 12 --output-dir ./calib_today

# Disable seed loading/saving
uv run python scripts/calibrate_thrust_table.py --no-seeds

# Regenerate report.md and plot from a saved results.json (no re-solve)
uv run python scripts/calibrate_thrust_table.py --from-dir ./calib_today
```

The script writes `thrust_table.csv`, `results.json`, `report.md`, and `plots/thrust_table_comparison.png`, and prints a paste-ready snippet for `presets.py` plus an old-vs-new comparison table.

### Continuation behaviour

Each speed's solve is independent — there is no speed-to-speed propagation within a single sweep. The seed cache provides per-speed warm-starts across runs, which is usually what you want: a once-good seed at a difficult speed (e.g., 8 m/s, 20 m/s) keeps converging well on subsequent re-runs. If you want true speed-to-speed continuation in a single run, call `find_casadi_trim_sweep` directly.

## No-Surge vs Surge vs Calibration Modes

The trim solver operates in three modes with different free variable sets:

### No-surge mode (default)

```
Free:  pos_d, theta, main_flap, rudder_elevator  (4 variables)
Fixed: u = u_forward (exact), q = 0, w = u*tan(theta)
```

Used for: simulation initialization, control design. Thrust is read from the preset's lookup table at `u_forward`. The solver finds the equilibrium state and control at fixed speed and thrust.

### Surge mode

```
Free:  pos_d, theta, u, main_flap, rudder_elevator  (5 variables)
Fixed: q = 0, w = u*tan(theta)
```

Enabled by `surge_enabled=True` on the Moth3D model. Speed `u` becomes a free variable with tight bounds (`u_forward ± 0.5 m/s`). The solver finds the equilibrium including the natural speed the boat settles at for the given thrust.

### Calibration mode

```
Free:  pos_d, theta, u, main_flap, rudder_elevator, thrust_total  (6 variables)
Fixed: q = 0, w = u*tan(theta)
```

Enabled by `calibrate_thrust=True`. Requires a model constructed with `surge_enabled=True`. Thrust becomes a free variable (bounds [10, 500] N). Regularization weights are relaxed (theta, flap, elevator, pos_d all at 10-20% of default) to let the physics drive the solution rather than regularization preferences. The thrust minimization term (`w_thrust=0.01`) breaks the null space, finding the minimum-drag equilibrium.

### How the modes interact

Calibration mode produces thrust values that populate the lookup table used by no-surge and surge modes. The workflow is:

1. Run calibration sweep → thrust table (speeds, values)
2. Update preset with new table
3. No-surge/surge modes read thrust from the updated table

The calibration zeros hull buoyancy (`hull_buoyancy_coeff=0.0`) because buoyancy is a placeholder that shouldn't bias the thrust estimate. No-surge/surge modes use the preset's actual buoyancy setting.

## CasADi/IPOPT Trim Solver

An independent trim solver using CasADi/IPOPT (`fmd.simulator.trim_casadi`). Finds the same equilibria as the SciPy solver but uses a different optimization approach.

### Two-Phase Strategy

The solver uses a two-phase approach for robust convergence with exact feasibility:

**Phase 1 (penalty)**: Relaxed tolerance, penalty formulation for robust convergence.
```
J = sum((xdot_i / xdot_scale_i)²)                            # residual
  + W_THRUST * (thrust / thrust_scale)²                       # quadratic thrust reg
  + W_CTRL * ((flap / flap_scale)² + (elev / elev_scale)²)   # control reg
```
- No equality constraints — bounds only
- Geometry-derived initial guess (pos_d from foil tip-at-surface calculation)
- IPOPT: `tol=1e-6`, `max_iter=300`

**Phase 2 (hard constraint)**: Warm-started from Phase 1, enforces exact `xdot=0`.
```
min  W_CTRL * ((flap / flap_scale)² + (elev / elev_scale)²)
s.t. xdot[0,2,3,4] = 0   (4 equality constraints)
     lb <= z <= ub
```
- Drops `theta_dot` constraint (index 1, equals q which is pinned to 0 by bounds) to avoid rank-deficient Jacobian
- Unscaled constraints (IPOPT handles scaling internally)
- IPOPT: `tol=1e-8`, `max_iter=500`, warm start enabled
- Thrust is NOT in the Phase 2 objective — minimizing `thrust²` creates a zero-thrust attractor that overwhelms constraint enforcement

### Characteristic Scales

The `CharacteristicScales` frozen dataclass defines physical scales for each variable and residual component. These are in human-friendly units (degrees, meters, Newtons) with SI conversion properties:

| Scale | Default | Units | Role |
|-------|---------|-------|------|
| `theta_deg` | 1.0 | ° | Pitch angle |
| `flap_deg` | 3.0 | ° | Main flap deflection |
| `elev_deg` | 2.0 | ° | Rudder elevator deflection |
| `pos_d_m` | 0.05 | m | Ride height |
| `thrust_N` | 100.0 | N | Thrust |
| `w_ms` | 0.1 | m/s | Heave velocity |
| `pos_d_dot_ms` | 0.05 | m/s | Vertical velocity rate |
| `theta_dot_rads` | 0.035 | rad/s | Pitch rate |
| `w_dot_ms2` | 0.5 | m/s² | Heave acceleration |
| `q_dot_rads2` | 0.35 | rad/s² | Pitch acceleration |
| `u_dot_ms2` | 0.2 | m/s² | Surge acceleration |

Custom scales can be passed via the `scales` parameter to `find_casadi_trim`.

### Objective Weights

**Phase 1** (penalty — weights must be small so residual dominates):

| Weight | Value | Role |
|--------|-------|------|
| `W_THRUST` | `1e-8` | Quadratic thrust regularization |
| `W_CTRL` | `1e-8` | Control uniqueness |

**Phase 2** (hard constraint — only control regularization):

| Weight | Value | Role |
|--------|-------|------|
| `W_HARD_CTRL` | `1e-3` | Control effort minimization in null space |

No thrust weight in Phase 2. Including `min thrust²` causes IPOPT to drive thrust toward zero, entering infeasible feasibility restoration.

### Why Quadratic Thrust Penalty (Not Linear)

A linear `min(thrust)` objective has constant gradient pointing toward `thrust=0`, creating a zero-thrust attractor where IPOPT converges to an infeasible boundary. The **quadratic** penalty `(thrust/scale)²` has zero gradient at `thrust=0`, avoiding this attractor.

### Post-Solve Sanity Checks

The solver runs sanity checks that produce warnings (not errors):
- Thrust outside [10, 400] N
- |theta| > 10 deg
- pos_d outside [-2.0, -0.1] m
- Flap or elevator at bound
- Kinematic inconsistency: |w - u*tan(theta)| > 1e-4

### Success Criterion

Success requires both: Phase 2 IPOPT reports `Solve_Succeeded` or `Solved_To_Acceptable_Level`, AND `max(|xdot|) < 1e-6`. This replaces the old single-phase approach that relied solely on physical residual.

### Geometry-Derived Initial Guess

The initial guess for `pos_d` is computed from foil geometry rather than tuning parameters:
```
tip_rise = (main_foil_span / 2) * sin(heel_angle)
pos_d_guess = tip_rise - main_foil_body_z * cos(heel_angle) + 0.05m
```

This places the main foil's leeward tip at the water surface, then lowers the boat 5cm for initial submergence. Other initial values: `theta=0, w=0, q=0, flap=0.05rad, elev=0.02rad, thrust=100N`.

### Seed System (`z0`)

`find_casadi_trim()` accepts an optional `z0: np.ndarray` (8-vector) as initial guess. When provided, it replaces the geometry-derived initial guess and is clipped to variable bounds. This enables:

- **Calibration bootstrapping**: `scripts/calibrate_thrust_table.py` loads/saves seeds from `src/fmd/simulator/params/trim_seeds.json`. After calibration, converged z-vectors are written back as seeds for future runs.
- **Warm-starting constrained trim**: `run_configuration_comparison()` seeds constrained-control trim from the baseline trim solution, dramatically improving convergence when controls are pinned.

The seed file is keyed by speed string (e.g., `"10.0"`) mapping to 8-element arrays `[pos_d, theta, w, q, u, flap, elev, thrust]`. Use `--no-seeds` to disable.

### Fixed Controls (`fixed_controls`)

`find_casadi_trim()` accepts `fixed_controls: dict[str, float]` to pin control surfaces at specific values. Supported keys: `"main_flap"`, `"rudder_elevator"`. Implementation: sets `lbz[idx] = ubz[idx] = value`, which IPOPT handles natively as a fixed variable.

This replaces the old SciPy `find_trim` approach for constrained-control trim, achieving residuals < 1e-10 vs ~0.25 with SciPy.

### Reporting

The `trim_report.py` module generates unified JSON + Markdown reports for both CasADi and SciPy sweep results. Reports include per-speed results tables, per-phase solver statistics, and warnings.

### History

The solver evolved through three architectures:
1. **Two-phase feasibility/thrust** (commit `4c12b61`): Phase 1 feasibility + Phase 2 thrust optimization with proximity regularization.
2. **Single-phase penalty** (commit `d444095`): Simplified to single-phase with characteristic-value scaling. Worked but hit `max_iter=2000` at 5/8 speeds due to penalty/KKT mismatch.
3. **Two-phase penalty/hard-constraint** (current): Phase 1 penalty for robust warm start, Phase 2 hard constraints for exact feasibility. All speeds converge with `Solve_Succeeded` and residuals < 1e-9.

See `docs/plans/casadi_trim_infeasibility_fix_20260313/` for the full analysis.

## Known Limitations

1. **Hull buoyancy zeroed during calibration**: The calibration mode sets `hull_buoyancy_coeff=0.0` to avoid biasing thrust estimates with the simplified buoyancy model. This means calibrated thrust values reflect a foiling-only force balance.

2. **Fixed heel angle**: All trim modes use a fixed heel angle (default 30°). The lateral force balance is not solved — heel is assumed constant.

3. **Analytical correction exact everywhere**: The CasADi thrust xdot correction (`_build_xdot_expr`) is built from a zero-thrust baseline plus the full thrust force/moment, so it is exact at all pitch angles and all optimization iterates, not just at the q=0 trim point. (This superseded an earlier SciPy-era delta/small-angle correction that was only exact at q=0; historical text describing that limitation has been corrected — see "Analytical xdot Correction" above.)

4. **Ventilation not modeled**: Foil ventilation (air entrainment at low immersion) is not modeled. At very low ride heights or high speeds where immersion is minimal, the simulation may be optimistic about foil performance.

5. **Added mass at t=0**: Added mass terms use their full steady-state values from t=0. In reality, added mass develops over time as the flow field establishes.

6. **(Historical, SciPy solver only)** SLSQP ill-conditioning: SLSQP stalls at ~5 iterations when objective gradients are O(1e5). This applied to the retired SciPy solver's two-phase approach (L-BFGS-B primary, SLSQP polish) and does not apply to the current CasADi/IPOPT solver.

7. **Multiple equilibria / local minima**: The trim problem can have multiple solutions at a given speed. The regularization terms (theta, flap, elevator penalties) select among them by preferring small deflections, but this preference can lead the solver to a physically suboptimal equilibrium — particularly at low speeds where the physics demands larger deflections than the regularization rewards.

8. **(Historical)** Continuation sweep can follow suboptimal branches: an earlier `calibrate_moth_thrust_table()` used each speed's solution as the initial guess for the next (in-loop continuation), which could bias the solver toward a branch that becomes suboptimal at other speeds (e.g. a smooth negative-elevator branch dominating over a lower-residual positive-elevator solution). **This no longer applies**: the current implementation has no speed-to-speed propagation within a sweep (see "Continuation behaviour" above) — each speed solves independently, optionally warm-started only from the on-disk seed cache. The multi-solution risk from item 7 still exists per-speed; the diagnostic remains useful: if a trim result shows negative rudder flow AoA (i.e., `2θ + elevator < 0`), the rudder is producing downward lift and nose-up moment — working against both weight support and pitch balance, a strong signal of a local minimum rather than a physics-driven equilibrium. See `physical_intuition_guide.md` § "Control input → physical effect" for the sign chain. The C1.F/C1.G work found a related but distinct branch-ambiguity (a nose-down secondary trim branch at u>=18 m/s in the *free* pos_d solve, not an elevator-sign artifact); pinning pos_d at `DEFAULT_POS_D_REF` (see "Calibrated thrust curve" above) sidesteps it by construction.

## API Reference

All of the below live in `fmd.simulator.trim_casadi` (`calibrate_moth_thrust_table` in `fmd.simulator.trim_calibration`). `find_moth_trim` is a drop-in replacement for the old SciPy API of the same name — it accepts and silently ignores SciPy-specific kwargs (`pos_d_guess`, `prev_trim`, etc.) so old call sites keep working.

### find_moth_trim

```python
find_moth_trim(
    params: MothParams,
    u_forward: float = 10.0,
    target_theta: float | None = None,
    target_pos_d: float | None = None,
    heel_angle: float | None = None,  # default 30 deg
    z0: np.ndarray | None = None,
    fixed_controls: dict[str, float] | None = None,
    **kwargs,
) -> CasadiTrimResult
```

Primary trim-finding API — CasADi two-phase solver. Returns `CasadiTrimResult` with optimized state, control, thrust, residual, and warnings.

### calibrate_moth_thrust

```python
calibrate_moth_thrust(
    params: MothParams,
    target_u: float,
    heel_angle: float = np.deg2rad(30.0),
    z0: np.ndarray | None = None,
    target_pos_d: float | None = DEFAULT_POS_D_REF,
    **kwargs,
) -> CalibrationTrimResult
```

Single-speed calibration. Since CasADi always solves for thrust as a free variable, this wraps `find_casadi_trim` and validates the result. Pinned at `target_pos_d` (default `DEFAULT_POS_D_REF`) by default — see "Calibrated thrust curve" above for why; pass `target_pos_d=None` for the legacy free-ride-height calibration.

### calibrate_moth_thrust_table

```python
calibrate_moth_thrust_table(
    params: MothParams,
    speeds: Iterable[float],
    *,
    seed_path: str | os.PathLike | None = None,
    save_seeds: bool = True,
    heel_angle: float = np.deg2rad(30.0),
    verbose: bool = True,
) -> list[CalibrationTrimResult]
```

Sweeps `speeds`, calling `calibrate_moth_thrust` at each. With `seed_path`, loads matching entries from a JSON cache as IPOPT initial guesses and (unless `save_seeds=False`) writes converged solutions back at the end of the sweep, preserving entries for speeds that were not run. See "Calibrating the Thrust Table" above for the seed-cache format and the bundled script.

### validate_trim_result

```python
validate_trim_result(
    result: CasadiTrimResult,
    u_target: float,
) -> list[str]
```

Post-solve plausibility checks: pitch angle vs speed-dependent limit, pos_d range, thrust near bounds, and controls at bounds. Starts from `result.warnings` (solver-level warnings) and appends physical-plausibility warnings. Returns list of warning strings (empty if all pass).

### validate_thrust_sweep

```python
validate_thrust_sweep(
    speeds: np.ndarray | tuple | list,
    thrusts: np.ndarray | tuple | list,
    monotonic_tol: float = 2.0,
    jump_fraction: float = 0.5,
) -> list[str]
```

Checks a thrust table for monotonicity (all adjacent pairs) and sharp jumps (>50% between adjacent speeds). Returns list of warning strings.

### CasadiTrimResult

```python
@dataclass
class CasadiTrimResult:
    state: np.ndarray           # Trim state vector [pos_d, theta, w, q, u]
    control: np.ndarray         # Trim control [main_flap, rudder_elevator]
    thrust: float
    residual: float              # max(|xdot|) at solution
    solve_time: float            # wall-clock seconds for entire solve
    success: bool
    iter_count: int = -1         # IPOPT iteration count
    ipopt_stats: dict = ...
    diagnostics: dict = ...      # includes leeward_tip_depth, depth_factor
    warnings: list[str] = ...
    phases: list[PhaseInfo] = ...  # per-phase (penalty/hard_constraint) stats
```

### CalibrationTrimResult

```python
@dataclass
class CalibrationTrimResult:
    speed: float                 # Forward speed (m/s)
    thrust: float                # Calibrated thrust (N)
    trim: CasadiTrimResult       # Full trim result
    max_xdot_residual: float     # max(|xdot|) at the solution (N/rad/etc, per-state)
    warnings: list[str]          # Combined warnings
```
