# Interpretation Skill — wand_vs_pid_waves

This is a **harness-agnostic** instruction file. Any AI agent that can
read files and write markdown can follow it — no Claude-Code-specific
tooling, no MCP servers, no special slash commands required.

## Purpose

Produce a clear, technically accurate narrative report (`report.md`)
in this folder from the artifacts that `run.py` generated:

- `recipe.md` — context, prerequisites, expected report structure
- `metrics.json` — exact numerical values (single-seed + 50-seed MC)
- `report_guidelines.txt` — physics guidance for this comparison
- `plots/*.png` — single-run dashboards, overlays, and MC distributions
- `run.py` — the regenerator script (read for sim configuration details)

## Inputs

```
docs/reports/wand_vs_pid_waves/
├── recipe.md
├── metrics.json
├── report_guidelines.txt
├── run.py
└── plots/
    ├── dashboard_mechanical.png
    ├── dashboard_pid_natural.png
    ├── dashboard_pid_deeper.png
    ├── compare_ride_height.png
    ├── compare_flap_command.png
    ├── compare_wand_angle.png
    ├── mc_ride_height_rms.png
    ├── mc_ride_height_rms_around_target.png
    ├── mc_breach_distribution.png
    ├── mc_flap_activity.png
    └── mc_pitch_speed.png
```

## Workflow

Follow these steps **in order**. Do not skip steps. Do not estimate
numbers from plots — always pull exact values from `metrics.json`.

### Step 1: Read `recipe.md`

Understand the setup, controllers being compared, and the expected
report structure. Note the prerequisites — your report should
restate them in the Setup section.

### Step 2: Read `metrics.json`

This is the source of truth for **every number** in the report.
Key fields:

- `setup` — trim state, wave preset, control bounds, sim parameters.
- `single_seed.metrics.<controller>` — extended metrics for seed=0
  (ride height stats, pitch stats, speed stats, foil tip depth,
  control effort, control rate, settling time, has_nan).
- `monte_carlo.<controller>` — per-seed arrays plus aggregate
  mean/std/median/min/max for: `ride_height_rms`,
  `ride_height_rms_around_target` (RMS vs own setpoint — use this
  for cross-setpoint comparison; `ride_height_rms` measures vs
  natural-trim pos_d and penalises any controller that targets
  a different depth), `ride_height_std`, `ride_height_mean`,
  `target_pos_d_m`, `depth_factor_mean`, `breach_count`,
  `breach_fraction`, `flap_rms`, `flap_saturation_fraction`,
  `pitch_rms_error`, `speed_loss_mean`.

Controllers in `metrics.json`:
- `mechanical` — passive linkage
- `pid_natural` — PID at natural trim pos_d
- `pid_deeper` — PID with `target_pos_d = foil_tip_at_surface + 0.30 m`
  (30 cm safety margin; NED-positive-down so + makes pos_d less
  negative = boat rides lower = foil more submerged)

Sanity-check the trim values against `setup.trim_state`. A plausible
foiling-moth trim has `pos_d` negative with magnitude of order 1 m.
Cross-check `setup.trim_state.theta_deg` (small positive pitch ≈ 1°
is normal). Each controller's `target_pos_d_m` is stored in
`monte_carlo.<controller>`.

**Important cross-setpoint caveat**: `pid_deeper.rms_vs_target` (0.189m
in the 50-seed run) is higher than `pid_natural.rms_vs_target` (0.091m).
This is a documented known limitation, NOT a bug. The deeper setpoint
shifts the pitch equilibrium from ~0.8° to ~3°, and the inversion was
calibrated at the natural trim theta. The theta-induced residual is
~8 cm in calm water, ~14 cm under waves. The safety-relevant metric is
`breach_count`, where pid_deeper dominates (1.7 vs 25 vs 86).

### Step 3: Read `report_guidelines.txt`

This file is the physics-checklist tailor-made for this report. It
documents:

- The NED convention (pos_d positive DOWN; "more negative pos_d"
  means BOAT RISING — never write "boat descending" when pos_d is
  becoming more negative).
- What ordering the plan **expected** between PID and mechanical
  wand on RMS, breaches, flap activity, etc.
- What to flag if the expected ordering is violated.

### Step 4: For each plot, write a short section

For every PNG in `plots/`, write a section in `report.md` describing:

1. **What it shows** (panels, axes, controllers compared).
2. **Key numbers** from `metrics.json` — quote them, do not eyeball.
3. **Physical interpretation** — what does the difference (or
   similarity) between controllers mean?

Group plots logically:

- **Single seed (seed=0)**: three dashboards (`dashboard_mechanical.png`,
  `dashboard_pid_natural.png`, `dashboard_pid_deeper.png`) and three
  comparison overlays. Discuss the time-series behaviour, phase
  relationship with wave forcing, controller saturation events.
- **Monte Carlo (50 seeds)**: `mc_*.png`. Discuss the distribution
  spread, paired comparison, outliers. Use `mc_ride_height_rms_around_target.png`
  (not `mc_ride_height_rms.png`) when comparing cross-setpoint tracking quality.

### Step 5: Cross-check NED signs

Read your draft and apply these checks:

1. Wherever you wrote "boat descending" or "boat sinking", confirm
   pos_d is **increasing** (becoming **less negative**). If pos_d
   is becoming more negative, the boat is **rising**.
2. Wherever you wrote "ride height of X.X m", clarify whether X.X
   is `pos_d` (negative for foiling) or "height above water"
   (positive). The report should be unambiguous about which.
3. A breach (`tip_depth < 0`) means the **leeward foil tip is
   above the water surface**. The boat is NOT necessarily above
   the surface — only the tip is.
4. "Mean depth factor 0.8" means **80% of the foil span is
   submerged on average**, not that the foil is 80% deep.

### Step 6: Write `report.md`

Save in this folder (`docs/reports/wand_vs_pid_waves/report.md`).
Suggested structure (mirrors `recipe.md` § "Report structure"):

```markdown
# wand_vs_pid_waves — report (regenerated <date>)

## Summary
- (3-5 bullets, lead with headline RMS / breach numbers)

## Setup
- Moth preset, trim state, wave preset, gains, sim parameters.
- Pull verbatim from metrics.json["setup"].

## Single-seed time series (seed=0)
- Walk through dashboard_mechanical.png + dashboard_pid.png.
- Walk through compare_ride_height.png, compare_flap_command.png,
  compare_wand_angle.png.

## Monte Carlo across 50 seeds
- mc_ride_height_rms.png — RMS distribution.
- mc_breach_distribution.png — breach count distribution.
- mc_flap_activity.png — flap RMS distribution.
- mc_pitch_speed.png — pitch RMS and speed-loss distributions.

## Mechanism
- Why does PID outperform on tracking? (closed-form inversion +
  integrator removes steady bias)
- Why is it different on breaches/saturation? Inspect the
  ride_height_mean values to compare steady-state setpoints —
  a controller that tracks trim exactly may breach more if the
  trim point is itself close to the foil-tip surface intersection.

## Tuning suggestions
- Pull from recipe.md § "What to tune" and adapt to the observed
  data.

## Caveats
- Single trim point (10 m/s, 30° heel, head seas only).
- Single wave preset (SF Bay moderate). Other conditions may flip
  the ordering.
- The PID inversion assumes trim attitude — under large heel
  excursions it will be biased. The mechanical wand has no
  attitude assumption.
```

## Physics primer (short)

- **NED frame**: x = north, y = east, z = down. pos_d > 0 means below
  the still-water reference. A foiling-moth trim has `pos_d` negative
  with magnitude of order 1 m — the exact value depends on the trim
  speed and preset and is recorded in `metrics.json` `setup.trim_state`.
  (Boat ~1 m above the still-water surface, hovering on the main foil.)
- **Wand kinematics**: the wand pivots at the bowsprit and trails to
  the water surface. Boat HIGH (pos_d more negative) → wand
  VERTICAL (small angle ~0) → flap UP (less lift). Boat LOW (pos_d
  less negative, foil tip closer to surface) → wand HORIZONTAL
  (larger angle) → flap DOWN (more lift).
- **Mechanical wand**: passive, fast (no sensor lag), fixed gain.
  Tends to leave a steady-state offset proportional to the bias
  between linkage geometry and the trim flap angle. Saturation comes
  from the linkage's geometry (arccos / arcsin limits).
- **PID**: tunable, has an integrator (zero steady-state error under
  bias), needs explicit `dt` for I/D terms. Differential term on a
  raw wand signal amplifies wave-orbital noise and can destabilise
  the boat — keep Kd=0 unless you also low-pass the wand.
- **Closed-form inversion (PID)**: under trim-attitude assumption
  (theta=trim_theta, constant heel), the wand-angle-to-pos_d mapping
  is a single trig function:
  `pos_d_est = -wand_pivot_z_body * cos(heel) - wand_length * cos(wand_angle) + offset`,
  where `offset` is a per-construction calibration that absorbs the
  trim-theta residual and makes the inversion pos_d-agnostic for any
  pos_d reached under theta=trim_theta. Residual bias arises when the
  operating theta differs from trim_theta (e.g., the deeper setpoint
  shifts pitch equilibrium by ~2°, producing ~8 cm calm-water bias).

## What to flag (do **not** silently accept)

If any of the following is observed, write a flag at the top of the
report instead of (or in addition to) the headline summary:

- **pid_natural has higher rms_vs_target than mechanical.** pid_natural
  should win on tracking (lowest rms_vs_target) — if not, there's a
  tuning issue or sign error in the inversion.
- **pid_deeper breach_count ≥ mechanical breach_count.** The whole
  point of the deeper setpoint is to suppress breaches. If this
  doesn't hold, the target is not deep enough or the theta-shift
  bias is overwhelming the safety margin.
- **Mean foil depth factor < 0.3 for any controller.** Foil is
  ventilating on average — wave amplitude is outside the envelope.
- **rms_vs_target ordering for pid_deeper > pid_natural.** This IS
  the expected outcome (theta-shift limitation) but still deserves
  a clear flag and explanation so readers don't interpret it as a
  bug. Do NOT silently accept — document the mechanism.
- **Asymmetric saturation** (flap pinned to only one limit > 50%
  of the time). Indicates a steady bias the integrator cannot
  recover from.
- **Physically suspicious numbers**: NaNs in any state, non-monotonic
  trim solve (extreme alpha angles), impossible speed loss
  (e.g. `speed_loss > trim_u`).
- **Controller divergence** (state magnitude growing without bound).

## Do NOT

- Re-run the simulation. Use the artifacts as-is. If they look wrong,
  flag it; do not regenerate without user approval.
- Fabricate numbers. If metrics.json is missing a field, say so in
  the report rather than guess.
- Paraphrase the recipe. The recipe is the source of truth on
  configuration and on the design intent; the report's job is to
  describe **what happened**.
- Write physics narratives without the NED sign cross-check in
  Step 5.
