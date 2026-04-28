# Moth Trim Solver

Technical reference for the Moth 3DOF trim solver (`fmd.simulator.trim`).

## Overview

The trim solver finds steady-state equilibrium points where all state derivatives are approximately zero. It uses SciPy optimization (L-BFGS-B primary + SLSQP polish) to minimize a scale-aware multi-term objective.

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

Thrust enters the EOM linearly. Rather than modifying the force model, the solver applies an analytical correction to the xdot output:

```python
thrust_delta = thrust_total - baked_thrust
xdot[U] += thrust_delta / m_eff_surge      # surge acceleration
xdot[Q] += ce_z * thrust_delta / i_eff     # pitch acceleration (moment arm)
```

This is exact at the trim point where q = 0 (no Coriolis interaction). The correction only affects u_dot and q_dot because thrust is body-x only, with a pitch moment from the CE height offset.

### Model parameters

| Parameter | Expression | Source |
|-----------|-----------|--------|
| m_eff_surge | total_mass + added_mass_surge | forward_dynamics line 530 |
| i_eff | composite_iyy + added_inertia_pitch | forward_dynamics line 531 |
| ce_z | sail.ce_position_z - cg_offset[2] | moth_forces line 519 |
| baked_thrust | sail.thrust_coeff | moth_forces line 527 (constant path) |

### Post-solve validation

After calibration, force balance is checked via `compute_aux`:
```
total_fx_corrected = total_fx_from_model + (calibrated_thrust - baked_thrust)
```

The `total_fx_residual` on `CalibrationTrimResult` reflects this corrected balance.

## Optimizer Details

### Two-phase approach

1. **L-BFGS-B** (primary): Quasi-Newton with box constraints. Handles the smooth, well-conditioned landscape well. SciPy finite-difference gradients (faster than JAX eager-mode for ~6 variables).
2. **SLSQP** (polish): Short run from L-BFGS-B solution. Can improve slightly due to different step-size strategy.

### Why SciPy FD > JAX gradients for this problem

At ~4-6 decision variables, JAX eager-mode gradient overhead dominates. SciPy finite-difference is ~2.5x faster. JAX gradients are available via `use_jax_grad=True` but off by default.

### Adaptive regularization

When state targets are pinned (`target_theta` and/or `target_pos_d`), the problem has fewer free variables. Control weights are reduced to avoid fighting the residual minimization.

## Typical Results

### Calibrated thrust curve (MOTH_BIEKER_V3 preset, post AoA fix + measured geometry)

> **Snapshot**: 2026-03-13, branch `feature/foil-force-decomposition-fix`.
> Geometry: hull_cg_from_bow=1.99, main_foil_from_bow=1.57, strut_depth=1.03, main_foil_cl0=0.15. AoA decomposition fix applied (alpha_geo/alpha_eff). Calibrated at 30 deg heel with `surge_enabled=True`.

| Speed (m/s) | Thrust (N) |
|-------------|------------|
| 6.0 | 52.3 |
| 7.0 | 54.8 |
| 8.0 | 61.3 |
| 9.0 | 70.6 |
| 10.0 | 82.4 |
| 11.0 | 96.2 |
| 12.0 | 111.8 |
| 13.0 | 129.2 |
| 14.0 | 148.3 |
| 15.0 | 168.9 |
| 16.0 | 191.1 |
| 17.0 | 214.7 |
| 18.0 | 239.8 |
| 19.0 | 266.3 |
| 20.0 | 294.2 |

**Observations:**

- Thrust is monotonically increasing across the full 6-20 m/s range, approximately following a u^2 drag law.
- The AoA decomposition fix eliminated the non-monotonic low-speed behavior (old table had 235, 155, 124, 111, 108 N at 6-10 m/s). The old model had artificial drag from lift-into-surge leakage via the conflated AoA, which inflated low-speed thrust requirements.
- Thrust minimum at 6 m/s (52.3 N). Low-speed values are dramatically lower than the pre-fix model (~52 vs ~235 N at 6 m/s).
- Pitch angle decreases monotonically with speed (alpha proportional to 1/u^2), as expected.
- Residuals < 0.02 at 8+ m/s; 6-7 m/s are at the foiling envelope edge with higher residuals.

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

## Sweep Modes

Three sweep modes for calibrating the thrust lookup table, each with different speed-to-speed strategies:

### Continuation (`calibrate_moth_thrust_table`)

Fastest (~2 min). Each speed seeds from the previous speed's solution.

- **Best for**: Routine use at 8+ m/s, incremental recalibration.
- **Risk**: Error propagation at low speeds. The low-to-high sweep starts at 6 m/s with a cold start. If 6 m/s converges to a local minimum (e.g., negative elevator), that bad solution seeds 7 m/s, etc.
- **Details**: `pos_d_guess`, `theta_guess`, `main_flap_guess`, `rudder_elevator_guess`, and `thrust_guess` are all carried forward. Default sweep: 6-20 m/s in 1 m/s steps.

### Robust (`calibrate_moth_thrust_table_robust`)

Gold standard (~9 min with 3 seeds). Independent multistart at each speed -- no continuation.

- **Best for**: Establishing new reference tables, validating after parameter changes, or when continuation results look suspicious (e.g., negative rudder AoA at low speeds).
- **How it works**: At each speed, runs `calibrate_moth_thrust_multistart()` which tries the default guess plus 3 physics-informed seeds (varying flap and elevator). Keeps the best result.
- **Physics-informed seeds**: Explore the positive-elevator branch that the default guess misses at low speeds. Seeds use progressively higher flap (0.05, 0.10, 0.15 rad) and positive elevator (0.02, 0.04, 0.06 rad).
- **Performance**: (1 default + 3 seeds) x 15 speeds x ~10s = ~10 min.
- **Reference table output**: Pass `return_reference_table=True` to get a 3-tuple `(speeds, thrusts, reference_table)`. The reference table can be fed directly to `calibrate_moth_thrust_table_seeded()` for fast subsequent runs.

### Seeded (`calibrate_moth_thrust_table_seeded`)

Fast + independent (~2.5 min). Each speed seeded from a reference table.

- **Best for**: Routine recalibration after small parameter changes, when a known-good reference table exists.
- **How it works**: Each speed gets its initial guess from the reference table (or nearest-neighbor fallback from closest entry). Single solve per speed, no continuation.
- **Reference table**: Dict mapping speed -> {pos_d_guess, theta_guess, main_flap_guess, rudder_elevator_guess, thrust_guess}. If not provided, falls back to the preset's existing thrust lookup values.
- **Performance**: 1 seed x 15 speeds x ~10s = ~2.5 min.

### When to use each mode

| Situation | Recommended Mode |
|-----------|-----------------|
| First calibration of new geometry | Robust |
| Checking if low-speed trim looks wrong | Robust |
| Small parameter tweak (CG, foil area) | Seeded (with robust reference) |
| Routine recalibration, 8+ m/s only | Continuation |
| Generating reference for seeded mode | Robust |

### Script usage

```bash
# Continuation (default)
python scripts/calibrate_thrust_table.py

# Robust multistart
python scripts/calibrate_thrust_table.py --mode robust

# Seeded from preset
python scripts/calibrate_thrust_table.py --mode seeded

# Custom speeds
python scripts/calibrate_thrust_table.py --mode robust --speeds 6 7 8 9 10
```

## Continuation Strategy (Detail)

### calibrate_moth_thrust_table()

The table calibration function sweeps across speeds using continuation: each speed uses the previous speed's solution as its initial guess. This is critical because:

1. **Cold-start convergence**: At higher speeds (>14 m/s), the optimizer with default initial guesses may find suboptimal local minima. The continuation from a nearby speed provides a much better starting point.

2. **Sweep direction**: Default sweep is low-to-high (6→20 m/s). Low speeds converge reliably from cold-start. Each subsequent speed only changes by 1 m/s, keeping the initial guess close to the solution.

3. **What gets continued**: `pos_d_guess`, `theta_guess`, `main_flap_guess`, `rudder_elevator_guess`, and `thrust_guess` are all seeded from the previous result. This captures the smooth variation of all trim variables with speed.

4. **Failure handling**: By default, failures are surfaced to the caller (warning fields on results and script-level handling). The convenience script does not silently inject default trim values.

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

3. **Analytical correction exact only at q=0**: The thrust xdot correction is exact at the trim point (q=0 constraint) but approximate during intermediate optimization steps where q may be slightly nonzero.

4. **Ventilation not modeled**: Foil ventilation (air entrainment at low immersion) is not modeled. At very low ride heights or high speeds where immersion is minimal, the simulation may be optimistic about foil performance.

5. **Added mass at t=0**: Added mass terms use their full steady-state values from t=0. In reality, added mass develops over time as the flow field establishes.

6. **SLSQP ill-conditioning**: SLSQP stalls at ~5 iterations when objective gradients are O(1e5). The two-phase approach (L-BFGS-B primary, SLSQP polish) mitigates this, but SLSQP may not improve on L-BFGS-B for all speed points.

7. **Multiple equilibria / local minima**: The trim problem can have multiple solutions at a given speed. The regularization terms (theta, flap, elevator penalties) select among them by preferring small deflections, but this preference can lead the solver to a physically suboptimal equilibrium — particularly at low speeds where the physics demands larger deflections than the regularization rewards.

8. **Continuation sweep can follow suboptimal branches**: The `calibrate_moth_thrust_table()` sweep uses each speed's solution as the initial guess for the next. This continuation is critical for convergence at high speeds but can bias the solver toward a branch that becomes suboptimal at other speeds. Specifically, the low-to-high sweep starts at 6 m/s with a cold start, and the solution found there (with extreme negative elevator) seeds 7 m/s, etc. The negative-elevator branch is smooth and the solver follows it, but a positive-elevator solution with lower residual may exist. **Diagnostic**: if a trim result shows negative rudder flow AoA (i.e., `2θ + elevator < 0`), the rudder is producing downward lift and nose-up moment — working against both weight support and pitch balance. This is a strong signal of a local minimum, not a physics-driven equilibrium. See `physical_intuition_guide.md` § "Control input → physical effect" for the sign chain.

## API Reference

### find_moth_trim

```python
find_moth_trim(
    moth: JaxDynamicSystem,
    u_forward: float = 10.0,
    pos_d_guess: float = -0.25,
    theta_guess: float = 0.02,
    main_flap_guess: float = 0.03,
    rudder_elevator_guess: float = 0.01,
    tol: float = 1e-10,
    target_theta: Optional[float] = None,
    target_pos_d: Optional[float] = None,
    regularization_weights: Optional[dict] = None,
    u_bounds_margin: float = 0.5,
    use_jax_grad: bool = False,
    calibrate_thrust: bool = False,
    thrust_guess: float = 70.0,
) -> TrimResult
```

Primary trim-finding API. Returns `TrimResult` with optimized state, control, residual, and warnings. When `calibrate_thrust=True`, also populates `calibrated_thrust`.

### calibrate_moth_thrust

```python
calibrate_moth_thrust(
    params: MothParams,
    target_u: float,
    thrust_guess: float = 70.0,
    heel_angle: float | None = None,  # default 30°
    fx_tol: float = 1.0,
    pos_d_guess: float = -0.25,
    theta_guess: float = 0.02,
    main_flap_guess: float = 0.03,
    rudder_elevator_guess: float = 0.01,
    tol: float = 1e-10,
) -> CalibrationTrimResult
```

Single-speed calibration. Builds a Moth3D with `surge_enabled=True`, runs `find_moth_trim` with `calibrate_thrust=True`, and validates the force balance. Returns `CalibrationTrimResult`.

### calibrate_moth_thrust_table

```python
calibrate_moth_thrust_table(
    params: MothParams,
    speeds: Sequence[float] | None = None,  # default 6..20
    **kwargs,  # passed to calibrate_moth_thrust
) -> tuple[tuple[float, ...], tuple[float, ...]]
```

Multi-speed calibration with continuation. Returns `(speeds_tuple, thrusts_tuple)` suitable for `MothParams.sail_thrust_speeds` and `sail_thrust_values`.

### calibrate_moth_thrust_multistart

```python
calibrate_moth_thrust_multistart(
    params: MothParams,
    target_u: float,
    n_seeds: int = 3,
    **kwargs,  # passed to calibrate_moth_thrust
) -> CalibrationTrimResult
```

Single-speed calibration with multistart. Tries the default guess plus physics-informed seeds (varying flap and elevator). Returns the best result across all seeds.

### calibrate_moth_thrust_table_robust

```python
calibrate_moth_thrust_table_robust(
    params: MothParams,
    speeds: Sequence[float] | None = None,  # default 6..20
    n_seeds: int = 3,
    return_reference_table: bool = False,
    **kwargs,  # passed to calibrate_moth_thrust_multistart
) -> tuple[tuple[float, ...], tuple[float, ...]]  # or 3-tuple if return_reference_table=True
```

Multi-speed calibration with independent multistart at each speed. No continuation between speeds. Gold standard for establishing reference tables.

### calibrate_moth_thrust_table_seeded

```python
calibrate_moth_thrust_table_seeded(
    params: MothParams,
    speeds: Sequence[float] | None = None,  # default 6..20
    reference_table: dict[float, dict] | None = None,
    **kwargs,  # passed to calibrate_moth_thrust
) -> tuple[tuple[float, ...], tuple[float, ...]]
```

Multi-speed calibration with per-speed seeding from a reference table. Each speed gets its own independent seed. If `reference_table` is None, builds a minimal one from the preset's thrust lookup values.

### validate_trim_result

```python
validate_trim_result(
    state: np.ndarray,
    control: np.ndarray,
    moth: JaxDynamicSystem,
    u_forward: float,
    total_fx_threshold: float = 1.0,
    calibrated_thrust: Optional[float] = None,
) -> list[str]
```

Post-solve plausibility checks: force balance (total_fx, total_my), state ranges (theta, pos_d), control saturation, and thrust-at-bounds. Returns list of warning strings (empty if all pass).

### validate_thrust_sweep

```python
validate_thrust_sweep(
    speeds: np.ndarray,
    thrusts: np.ndarray,
    monotonic_tol: float = 2.0,
    jump_fraction: float = 0.5,
) -> list[str]
```

Checks a thrust table for monotonicity (all adjacent pairs) and sharp jumps (>50% between adjacent speeds). Low-speed non-monotonicity (e.g., 6-7 m/s dip) is physical and expected. Returns list of warning strings.

### TrimResult

```python
@dataclass
class TrimResult:
    state: np.ndarray           # Trim state vector [pos_d, theta, w, q, u]
    control: np.ndarray         # Trim control [main_flap, rudder_elevator]
    residual: float             # L2 norm of xdot at trim
    success: bool               # Optimizer convergence flag
    optimize_result: OptimizeResult  # Full scipy result
    warnings: list[str]         # Plausibility warnings
    calibrated_thrust: Optional[float]  # Only when calibrate_thrust=True
```

### CalibrationTrimResult

```python
@dataclass
class CalibrationTrimResult:
    speed: float                # Forward speed (m/s)
    thrust: float               # Calibrated thrust (N)
    trim: TrimResult            # Full trim result
    total_fx_residual: float    # Corrected force balance (N)
    warnings: list[str]         # Combined warnings
```
