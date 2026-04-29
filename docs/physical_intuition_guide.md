# Physical Intuition Guide for Dynamics Work

Use this guide when working on dynamic models, force components, trim behavior, validation, and interpretation of simulation/tests.

This is a skeptical reference, not a source of truth. Treat every claim as provisional until cross-checked.

## One-Page Field Checklist

Use this when time is tight. If any answer is "no" or "unsure," do not trust the conclusion yet.

- **Frames clear?** You can state frame, axis direction, and sign for each logged variable.
- **Units coherent?** Every equation term is dimensionally consistent in SI.
- **Instantaneous physics right?** Small perturbations produce expected derivative signs.
- **Numerics behaving?** No obvious integration artifacts (step-size dependence, NaN masking).
- **Independent cross-check done?** At least one second path agrees (derivation, canonical helper, or parity test).
- **Test can fail?** New/updated test is non-vacuous and fails against the old bug.
- **Explanation falsified?** You attempted at least one plausible alternative hypothesis.
- **Scope bounded?** You explicitly state where the claim is valid (regime/speed/trim/mode).

## Purpose

- Prevent high-confidence wrong conclusions in physics/modeling work.
- Provide fast sanity checks before and after code changes.
- Standardize how we debug and validate dynamic behavior in fomodynamics.

## Non-Negotiable Mindset

- Verify before explaining: do not invent a story from one plot.
- Prefer first principles over pattern matching from past docs.
- Assume old notes can be stale or wrong, even if detailed.
- Seek disconfirming evidence: actively try to falsify your current hypothesis.
- Separate "looks plausible" from "is verified."

## Source Hierarchy (Most to Least Trustworthy)

1. Canonical implementation used in production paths.
2. Independent derivation or known analytical identity.
3. Focused tests that would fail if the hypothesis is wrong.
4. Cross-implementation parity checks (for fomodynamics: JAX vs CasADi).
5. Diagnostic scripts and plots.
6. Reports/scratchpads/handoffs/retros.

When levels disagree, trust higher levels first and investigate why.

## fomodynamics Physical Invariants and Core Conventions

- Units are SI internally (`m`, `s`, `rad`, `N`).
- Coordinate frame is NED (`+D` means down).
- Quaternion convention is scalar-first (`qw, qx, qy, qz`).
- Angles are circular quantities; use circular-aware ops where needed.
- Moth depth/height intuition:
  - More negative `pos_d` means the boat is higher (less down).
  - "Upward force" often appears as negative body-z force (`Fz < 0` in FRD).

### Moth 3DOF depth geometry

The main foil is 1.82m below the **boat** CG in body frame (body z = +1.82). The 75 kg sailor sitting 0.2m above boat CG shifts the **system** CG upward by 0.12m. The effective main foil distance from system CG is ~1.94m (body z), which at 30° heel projects to ~1.68m in NED depth.

See `docs/moth_vertical_geometry.md` for the full vertical layout and hull-datum coordinate system.

**Foil depth approximation** (small theta, 30° heel, BIEKER_V3 preset):

```
main_foil_depth  ≈ pos_d + 1.68
rudder_depth     ≈ pos_d + 1.64
```

Full formula: `pos_d + eff_z * cos(heel) * cos(theta) - eff_x * sin(theta)`

| pos_d | Main foil depth | Rudder depth | What's happening |
|-------|----------------|--------------|------------------|
| 0.0 | 1.68m | 1.64m | Hull submerged, foils deep |
| -0.94 | 0.74m | 0.70m | Hull just clear of water |
| -1.30 | 0.38m | 0.34m | Normal foiling trim |
| -1.44 | 0.24m | 0.20m | Main tip surfacing at 30° heel |
| -1.68 | 0.00m | — | Main foil center at surface |

**Ventilation geometry** (at 30° heel): leeward tip rises by `(span/2) * sin(heel) = 0.475 * sin(30°) = 0.2375m` above foil center. Ventilation onset when foil_depth < 0.2375m → pos_d < ~-1.44 (main foil). Rudder (span 0.68m) has max_submergence = 0.17m, so it ventilates at pos_d < ~-1.47.

**Hull contact**: `hull_contact_depth = 0.94m` (computed from `hull_cg_above_bottom - combined_cg_offset[2]`). Hull enters water when `pos_d > -0.94`.

**Key regime difference**: The foiling pos_d regime is ~[-1.68, -0.94], centered around ~-1.3. This is a significant shift from the old geometry where foiling was at pos_d ~ -0.26.

If behavior looks weird, check conventions before touching parameters.

## Reference Frames and Sign Conventions (Expanded)

When debugging, write these explicitly in notes/tests before interpreting numbers.

### fomodynamics frame quick map

- **World frame (NED)**: `x=N`, `y=E`, `z=D`.
- **Body frame (FRD)**: `x=Forward`, `y=Right/Starboard`, `z=Down`.
- **Rerun display frame**: transformed from NED/FRD (`x=E`, `y=N`, `z=Up`).

### Sign interpretation rules that prevent common mistakes

- In NED, larger `pos_d` means lower in the water; smaller/more-negative `pos_d` means higher.
- In FRD, negative `Fz` is upward lift; positive `Fz` is downward force.
- Pitch/heel terms can look sign-inverted after frame transforms; confirm with a single-point rotation check.
- Any "height above water" metric is a derived quantity; always trace back to its `pos_d`/depth formula.

### Frame audit mini-procedure

1. For each variable in a claim, record: frame, positive direction, units.
2. Pick one synthetic state and compute expected sign by hand.
3. Verify implementation matches hand-computed sign in that state.
4. Verify plotting/viz layer uses the same convention (or explicit transform).
5. Add/keep one regression test that would fail on axis swap or sign flip.

### High-value frame sanity tests

- **Zero-angle identity**: at zero pitch/heel/yaw, transformed quantities reduce to unrotated forms.
- **Odd/even checks**: terms with `sin` should flip with angle sign; pure `cos` factors should not.
- **Round-trip check**: transform to visualization frame and back (or equivalent invariant) should preserve geometry.
- **Limit check**: small-angle approximation and full trig formula should agree near zero.

## 90-Second Sanity Checklist

Run this before deep debugging:

1. **Sign check**: if state/force increases, does direction physically match NED/FRD assumptions?
2. **Magnitude check**: are values in realistic order-of-magnitude (not 10x-1000x off)?
3. **Units check**: any deg/rad or N/kN mismatch?
4. **Limit check**: does behavior approach expected extremes (zero speed, high speed, shallow/deep submergence)?
5. **Monotonic-local check**: if one input is nudged slightly, does the immediate derivative move in expected direction?
6. **Energy/momentum/consistency check** (where applicable): does a conserved or bounded quantity drift unexpectedly?

If 2+ checks fail, stop and re-derive before coding further.

## Skeptical Debug Procedure (Use in Order)

1. **State hypothesis clearly**
   - Example: "Missing `cos(heel)` term in foil depth causes 2D-vs-3D mismatch."
2. **Create a minimal reproduction**
   - Smallest state/control/time horizon that still shows the issue.
3. **Do paper math**
   - Derive the expected sign/scaling from geometry and force balance.
4. **Instrument intermediate terms**
   - Log exactly the terms needed to test the hypothesis (not everything).
5. **Cross-check through independent path**
   - Compare with canonical helper function or alternate implementation.
6. **Falsification test**
   - Add a test that would fail if your hypothesis is wrong.
7. **Only then patch**
   - Keep fix minimal and local; avoid parameter retuning until physics is fixed.
8. **Re-validate at multiple levels**
   - Unit tests, equivalence tests, scenario scripts, and a targeted plot/report.

## Cross-Validation Ladder for Model Changes

- **L0: Algebra/geometry**
  - Independent derivation from frame transforms and definitions.
- **L1: Instantaneous dynamics**
  - `forward_dynamics` directionality and sign tests.
- **L2: Differential structure**
  - Jacobian sanity where relevant (finite and expected local trends).
- **L3: Trajectory behavior**
  - Short rollout for immediate response + longer rollout for emergent coupling.
- **L4: Cross-implementation parity**
  - JAX/CasADi equivalence for derivatives/Jacobians/trajectories.
- **L5: Operational scenario**
  - Script-level behavior and interpretation report.

Do not skip to L5 and declare success.

## Interpreting Plots and Tests Without Fooling Yourself

- A clean plot is not proof; ask what would produce the same shape for the wrong reason.
- Distinguish:
  - instantaneous effect at fixed state,
  - re-trimmed equilibrium effect,
  - closed-loop controlled effect.
- A passing test can be vacuous; check that it can fail for the bug it claims to detect.
- If a result depends on optimizer convergence, report residuals and possible local-minimum sensitivity.
- If a metric drifts slowly, test whether drift comes from residual trim imbalance vs true instability.

## fomodynamics-Specific Pitfalls Seen Repeatedly

- NED sign confusion for `pos_d` and height-above-water interpretations.
- Mixing raw geometry with CG-adjusted geometry in diagnostics/tests.
- Assuming script scratchpad conclusions are final (some are stale).
- Treating optimizer success flags as ground truth without residual checks.
- Interpreting fixed-speed (`surge_disabled`) behavior as if surge coupling were active.
- Declaring monotonic relationships globally when only locally verified.

## Practical fomodynamics Debug/Validation Workflow

Use this sequence for Moth/dynamics work:

1. Run targeted simulator tests around modified physics.
2. Run non-slow suite for regression.
3. Run relevant CasADi equivalence tests for changed dynamics.
4. Reproduce behavior with script(s):
   - `examples/moth_open_loop.py`
   - `scripts/moth_configuration_sweep.py --plot` (private)
   - `scripts/moth_dynamics_sweep.py --report --plots` (private)
   - `scripts/moth_lqg_viz.py --scenario <name>` (private)
5. Compare against known baseline report/plot, but treat baseline as a hypothesis.
6. Record what is verified vs inferred.

## Evidence Template for Any Physics Claim

When writing a plan/scratchpad/report, include:

- **Claim**: one sentence.
- **Why expected (first principles)**: 2-5 lines.
- **Direct evidence**: test names + key values.
- **Independent cross-check**: second path/result.
- **Could still be wrong if**: explicit caveat list.
- **Next falsification step**: one concrete check to break the claim.

## Handling Contradictory Documentation

If two internal docs disagree:

1. Mark both as untrusted.
2. Re-run the smallest reproducible check from code/tests.
3. Update the stale doc with explicit "superseded by" note.
4. Preserve contradiction context in scratchpad to prevent repeated confusion.

## External V&V Principles Worth Importing

- Verification is "solving equations right"; validation is "solving the right equations."
- Verification should use analytic/benchmark solutions where possible, not only code-to-code agreement.
- Separate bug finding (error evaluation) from numerical error estimation (convergence/error bands).
- Credibility depends on intended use; incremental predictions are often easier to trust than absolute ones.
- Conserved quantities are useful alarms, but not sufficient proof of correctness.

## References (Starter Set)

- NASA CFD V&V overview and verification assessment:
  - https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html
  - https://www.grc.nasa.gov/WWW/wind/valid/tutorial/verassess.html
- MFiX V&V manual introduction (AIAA-style V&V framing):
  - https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/introduction.html
- Conserved quantities as simulation checks:
  - http://spiff.rit.edu/classes/phys559/lectures/conserved/conserved.html
- ASME V&V 20 family (VVUQ standards context):
  - https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer

## Appendix: fomodynamics Case Patterns to Reuse

These are patterns to emulate, not absolute truths.

### Pattern A: Geometry mismatch between physics and visualization

- Symptom: 2D diagnostic and 3D viz disagree on depth/height.
- Fast checks:
  - compare both paths against a canonical geometry helper;
  - verify rotation order and frame mapping on one test point;
  - confirm CG offset is applied consistently in both code paths.
- Required closure: one regression test reproducing the original mismatch and passing after fix.

### Pattern B: Instantaneous derivative vs re-trimmed equilibrium confusion

- Symptom: "control input causes opposite behavior" depending on analysis.
- Fast checks:
  - separate fixed-state derivative test from re-trim solver test;
  - report both moment direction and resulting trimmed state shift;
  - ensure docs/tests label which interpretation is being used.
- Required closure: paired tests, one instantaneous and one re-trimmed.

### Pattern C: Hidden coupling revealed by enabling a state

- Symptom: behavior looks stable in constrained mode but diverges in fully dynamic mode.
- Fast checks:
  - run constrained vs unconstrained with identical initial conditions;
  - compare force balance terms that were previously masked;
  - test short-horizon first to isolate directionality before long-horizon divergence.
- Required closure: document which mode each claim applies to.

## Appendix: Moth 3DOF Force and Drag Model Reference

> **Snapshot**: 2026-03-13, branch `feature/foil-force-decomposition-fix`, post-commit `1ee36cc`.
> **Source**: `src/fmd/simulator/components/moth_forces.py`.
> **Preset values updated** for measured geometry (hull_cg_from_bow=1.99, main_foil_from_bow=1.57, strut_depth=1.03) and AoA decomposition fix (alpha_geo/alpha_eff separation).
> **If the code has changed since this snapshot**, re-audit the formulas below against the source before relying on them.

This section documents every surge-direction (u) force in the Moth 3DOF model. Use it when interpreting open-loop surge results, debugging deceleration, or assessing model fidelity.

### Force assembly (`Moth3D._compute_step_terms`)

```
gravity_fx = -total_mass * g * sin(theta)
gravity_fz =  total_mass * g * cos(theta)

total_fx = f_foil[0] + f_rudder[0] + f_sail[0] + f_hull[0] + gravity_fx
total_fz = f_foil[2] + f_rudder[2] + f_sail[2] + f_hull[2] + gravity_fz
```

When `surge_enabled=True` (**default**): `u_dot = total_fx / m_eff_surge - q*w`.
When `surge_enabled=False`: `u_dot = 0`, `u` follows the scheduled speed profile.

Note: `f_sail[0]` now includes the NED→body rotation: `f_sail[0] = F_sail * cos(theta)`,
and `f_sail[2] = F_sail * sin(theta)`.

### 1. Main foil (`MothMainFoil`, line 162)

**AoA decomposition**:
```
w_local = w - q * eff_pos_x + w_orbital
alpha_geo = arctan2(w_local, u_safe)        # geometric flow angle (for force rotation)
alpha_eff = flap_effectiveness * main_flap + w_local / u_safe  # effective AoA (for polar)
```

**Lift/drag coefficients** (no stall — CL is unbounded linear):
```
cl = (cl0 + cl_alpha * alpha_eff) * depth_factor
cd = cd0 + cl² / (π * ar * oswald)
```

**Body-frame forces** (full rotation by alpha_geo):
```
fx = -drag * cos(alpha_geo) + lift * sin(alpha_geo)
fz = -drag * sin(alpha_geo) - lift * cos(alpha_geo)
```

Surge force contributions:
- **Profile drag**: `cd0 * q_dyn * area * cos(alpha_geo)` — speed-dependent, AoA-independent, **unaffected by ventilation**.
- **Induced drag**: `CL²/(π·AR·e) * q_dyn * area * cos(alpha_geo)` — quadratic in CL, scales as `depth_factor²` under ventilation.
- **Lift-forward-tilt**: `+lift * sin(alpha_geo)` — forward component of lift at positive alpha_geo (thrust-producing during foiling).

**Preset** (MOTH_BIEKER_V3): `cd0=0.006`, `cl_alpha=5.7`, `cl0=0.15`, `AR=10.7`, `e=0.85`, `area=0.08455 m²`.

### 2. Rudder elevator (`MothRudderElevator`, line 307)

Same alpha_geo/alpha_eff decomposition as main foil:
```
w_local = w - q * eff_pos_x + w_orbital
alpha_geo = arctan2(w_local, u_safe)
alpha_eff = rudder_elevator_angle + w_local / u_safe
rudder_cl = cl_alpha * alpha_eff * depth_factor
rudder_cd = cd0 + rudder_cl² / (π * ar * oswald)
rudder_fx = -rudder_drag * cos(alpha_geo) + rudder_lift * sin(alpha_geo)
rudder_fz = -rudder_drag * sin(alpha_geo) - rudder_lift * cos(alpha_geo)
```

**Preset**: `cd0=0.008`, `cl_alpha=5.0`, `AR=9.1`, `e=0.85`, `area=0.051 m²`.

### 3. Sail thrust (`MothSailForce`, line 444)

Sail thrust is applied in the **NED horizontal plane**, then rotated to body frame using pitch angle theta. This models the physical reality that sail force direction is set by the wind, not the hull pitch.

```
f_sail = interp(thrust_table, u_forward)  # if table present
f_sail = thrust_coeff + thrust_slope * u_forward  # fallback when table empty

# NED→body rotation
force_bx = f_sail * cos(theta)
force_bz = f_sail * sin(theta)
moment_y = ce_z * f_sail * cos(theta)
```

**Preset** (`MOTH_BIEKER_V3`): uses a 15-point lookup table
(`sail_thrust_speeds=6..20`, `sail_thrust_values=46.9..239.2 N`) with
fallback `thrust_coeff=74.8 N` (10 m/s reference). Monotonically increasing
across the full 6-20 m/s range.

### 4. Hull contact drag + buoyancy (`MothHullDrag`, line 519)

**Hull drag** (active only when hull touches water):
```
immersion = max(0, pos_d + contact_depth)
hull_drag = drag_coeff * immersion
```

**Buoyancy** (two-point, decomposed to body frame):
```
f_buoy = (buoyancy_coeff/2) * immersion_at_point   (upward in world)
buoy_fx = f_buoy * sin(theta)                       (body x component)
```

**Preset**: `drag_coeff=500 N/m`, `contact_depth=0.94 m` (dynamic, from `hull_cg_above_bottom - cg_offset[2]`), `buoyancy_coeff=5000 N/m`.

When foiling (`pos_d < -0.94 m`), hull drag and buoyancy are both zero.

### 5. Gravity projection

```
gravity_fx = -total_mass * g * sin(theta)
```

Nose-down (θ < 0): small forward assist. Nose-up (θ > 0): opposes surge. This is the pitch-to-speed coupling that creates phugoid-like energy exchange.

### Key insight: ventilation L/D collapse

Ventilation reduces both CL and profile drag (`cd0`) by `depth_factor`. Induced drag scales as `depth_factor²`. At partial ventilation, lift drops much faster than total drag → severe L/D degradation. This, combined with induced drag blowup at high AoA, produces the "stall-like" behavior seen in surge simulations. **There is no actual stall model** — CL is unbounded linear.

### What's NOT modeled

| Missing physics | Effect | Notes |
|----------------|--------|-------|
| Aerodynamic stall | CL continues growing past real-world stall angle | Invalid above ~10-12° AoA |
| Parasitic aerodynamic drag | No hull windage or rig drag (`ρ_air·Cd·A·u²`) | No quadratic speed governor above water |
| Sail aerodynamics | Sail force is prescribed by calibrated thrust table (not apparent-wind aero model) | No explicit sail CL/CD or trim-angle dependence |
| Ventilation hysteresis | Ventilation onset/recovery is symmetric (smooth tanh) | Real ventilation is harder to recover from |
| Wave-induced drag | No added resistance from waves in surge direction | Only wave orbital velocity affects AoA |

### Moth 3DOF Pitch Trim Intuition

> **Snapshot**: 2026-03-13, branch `feature/foil-force-decomposition-fix`, commit `1ee36cc`.
> Illustrative numbers from the AoA decomposition fix + measured geometry. Exact values will change with future geometry or trim solver updates.

#### Foil AoA formulas and the w/u coupling

Both foils compute two separate angles of attack (at trim, with q ≈ 0 and no waves):

```
alpha_geo  = arctan(w / u)                   # geometric flow angle (for force rotation)
main_alpha_eff   = flap_eff * flap + w/u     # effective AoA (for polar, includes control)
rudder_alpha_eff = elevator       + w/u      # effective AoA (for polar, includes control)
```

At steady trim the boat flies at constant altitude, so the velocity vector is horizontal. In body frame this means `w = u·tan(θ) ≈ u·θ` for small angles. Therefore **`w/u ≈ θ`**:

```
alpha_geo        ≈ θ                         (geometric flow angle)
main_alpha_eff   ≈ θ + flap_eff * flap       (effective, determines lift/drag magnitude)
rudder_alpha_eff ≈ θ + elevator              (effective, determines lift/drag magnitude)
```

**Key change from pre-fix model**: The old model used a single `aoa = theta + control + w/u ≈ 2θ + control` for both polar lookup and force rotation. The fix separates these: `alpha_eff` (with control surface, for polar) and `alpha_geo` (without control surface, for rotation). The `2θ` double-counting no longer occurs because the geometric flow angle is computed via `arctan(w/u) ≈ θ`, not `θ + w/u ≈ 2θ`. Control surface deflections correctly affect only the lift/drag magnitude, not the force rotation direction.

#### Control input → physical effect (sign chain reference)

Trace each control input through the force model to its final effect on lift and pitching moment. Use this to sanity-check trim results. Sign conventions: fz < 0 = upward lift (FRD), My > 0 = nose-up (FRD).

| Input | alpha_eff effect | Lift direction (fz) | Pitch moment (My) | Physical meaning |
|-------|-----------|--------------------|--------------------|-----------------|
| +flap | +main alpha_eff (x0.5) | Up (fz < 0) | Nose-UP (+My) | More forward lift, raises bow |
| +elevator | +rudder alpha_eff (x1.0) | Up (fz < 0) | Nose-DOWN (-My) | More aft lift, pushes tail up |
| +theta | +alpha_geo and +alpha_eff (via w/u) | Up (both foils) | Mixed | Raises both foil AoAs; net depends on balance |

**Key implication**: Positive elevator simultaneously (a) adds upward lift (helping support weight) and (b) creates nose-down moment (balancing the main foil's nose-up tendency). Negative elevator does the opposite — downward rudder lift and nose-up moment.

**Note on alpha_geo vs alpha_eff**: Control surface deflections only affect alpha_eff (and thus lift/drag magnitude). They do not change alpha_geo (the force rotation angle). This means flap/elevator adjust how much force the foil produces, but the direction of that force depends only on the geometric flow angle.

**Warning — sign reversal at extreme negative elevator**: If the elevator goes far enough negative to flip the rudder's alpha_eff below zero (i.e., `theta + elevator < 0`), the rudder produces *downward* lift and a *nose-up* moment. This reverses the rudder's normal role in the moment balance — it opposes both weight support and pitch trim. If a trim result shows negative rudder alpha_eff, the solver has likely found a suboptimal local minimum. See `docs/trim_solver.md` § Known Limitations for how the continuation sweep can produce this at low speeds.

#### Pitch moment balance and rudder AoA vs speed

The pitch moment about the CG sums to zero at trim. The dominant terms (sign convention: positive M_y = nose up in FRD):

| Component | Arm | Moment sign | Trend with speed |
|-----------|-----|-------------|------------------|
| Main foil lift | r_x > 0 (fwd of CG) | Nose UP | ~Constant (supports weight) |
| Sail thrust | r_z < 0 (above CG) | Nose DOWN | Increases (more thrust) |
| Rudder lift | r_x < 0 (aft of CG) | Nose DOWN (when AoA > 0) | Adjusts for balance |

**Note**: The rudder moment sign assumes positive rudder AoA (upward lift aft of CG → nose-down). At extreme negative elevator where rudder AoA flips negative, the rudder moment reverses to nose-UP. See the sign chain table above.

Since main foil lift is roughly constant (weight support) and sail nose-down moment grows with speed, the **net moment that the rudder must provide decreases with speed** — less nose-down correction needed at high speed. But rudder force scales with V², so the **required rudder alpha_eff decreases even faster** with speed:

```
required_rudder_alpha_eff ∝ required_moment / V²
```

This produces a monotonically decreasing rudder alpha_eff with speed. After the AoA decomposition fix, `alpha_eff = elevator + w/u ≈ elevator + θ` (not `2θ + elevator` as in the old model). The illustrative values below reflect the post-fix model with measured geometry.

#### Why elevator can be negative at low speed

At low foiling speeds (e.g. 8 m/s), theta is relatively large because the main foil needs more AoA to generate weight-supporting lift at lower dynamic pressure. The `w/u ≈ θ` coupling contributes to the rudder's alpha_eff. If the total `θ + w/u ≈ 2θ` (in alpha_eff, which includes the w/u term) exceeds what the moment balance requires, the elevator must go negative to compensate.

This is physically correct: the boat's pitch attitude already contributes enough to the rudder's effective AoA, and the elevator trims it back down. Whether this matches real-world observation depends on whether the actual theta at 8 m/s is as high as the model predicts — a rider using more flap or having different CG placement could achieve the same lift with less theta, which would make elevator positive.

#### Flap vs theta tradeoff

The trim solver chooses between theta and flap to achieve the required main foil alpha_eff. For the main foil:

```
main_alpha_eff ≈ θ + 0.5 * flap     (flap_effectiveness = 0.5, w/u ≈ θ term included)
```

Each degree of theta contributes to both alpha_geo (force direction) and alpha_eff (force magnitude via w/u). Each degree of flap provides only 0.5° of alpha_eff but does not affect alpha_geo or the rudder.

The trim solver's regularization weights (theta penalty is ~10x flap per radian) partially counteract this, but theta still dominates at low speed because it simultaneously satisfies both the force and moment balance.

**Key takeaway**: if the model produces undesirable low-speed elevator behavior, the most direct lever is reducing theta — either by increasing `cl0` (camber), increasing `flap_effectiveness`, or adjusting the trim solver to favor flap over theta at low speeds.

## Report Interpretation Focus

The model is most accurate in the **clean foiling regime** — foils submerged, hull above water, no ventilation or hull contact. When interpreting impulse responses or open-loop simulations:

- **Focus on the first 1-2 seconds** after a perturbation. The pitch-heave coupling, non-minimum-phase heave responses, and initial force balance are physically realistic and the most informative.
- **Ventilation and hull contact models are coarse approximations.** They provide qualitatively correct boundary behavior (lift loss when foils emerge, drag increase when hull enters water) but the detailed dynamics in those regimes should not be over-analyzed.
- **Surge speed coupling** is dominated by gravity body-frame projection: `gravity_fx = -mg·sin(θ)`. Nose-up pitch retards forward motion; nose-down pitch accelerates it. Strut immersion drag is a secondary effect.
