# Recipe — wand_vs_pid_waves

## Purpose

Compare a passive mechanical wand-to-flap linkage against two wand-only
PID configurations (natural trim and deeper trim) under SF Bay moderate
waves. The report is the canonical fmd example of how to organise a
regeneratable analysis: a recipe + script + agent-readable interpretation
skill that any user can clone, re-run, and tweak as a starting point for
their own tuning.

## Prerequisites

- Moth model: `MOTH_BIEKER_V3`
- Trim solve: `design_moth_lqr(u_forward=10.0)` (heel=30 deg)
- Wave preset: `WAVE_SF_BAY_MODERATE` (Hs=0.5m, Tp=3.0s, JONSWAP
  gamma=4.0, Stokes 2nd order), head seas (`mean_direction=pi`)
- Controllers (all in `fmd.simulator.moth_scenarios`):
  - `create_mechanical_wand_config()` (default linkage,
    `pullrod_offset=0.005`)
  - `create_pid_wand_config()` — natural trim, default gains
    `Kp=0.6, Ki=0.1, Kd=0.0`, `target_pos_d = trim pos_d ≈ -1.40 m`
  - `create_pid_wand_config(target_pos_d=compute_tip_at_surface_pos_d() + 0.30)`
    — deeper trim; commands the boat 30 cm below the foil-tip
    ventilation threshold (NED-positive-down: `+ 0.30` makes pos_d
    *less* negative = boat rides lower = foil tip more submerged).
- Closed-form wand-to-height inversion (PID): under trim attitude
  (theta=trim_theta) and constant heel, the mapping is:
  `pos_d_est = -z_pivot * cos(heel) - L_wand * cos(wand_angle) + offset`
  where `offset` is a per-construction calibration that absorbs the
  trim-theta residual. The inversion is pos_d-agnostic for fixed theta.
  A known limitation: a different setpoint shifts the pitch equilibrium,
  introducing a theta-induced steady-state offset of order ~8 cm
  (documented in scratchpad, user-approved to accept and document).
- Simulation: 60s duration, dt=0.005s, paired wave seeds (all
  controllers see the same wave field per seed).
- Monte Carlo: 50 wave seeds (5 in `--quick`); all controllers run
  through the same `fmd.simulator.sweep.sweep_closed_loop` vmap.

## How to regenerate

```bash
# From the fmd repo root (or anywhere — script resolves paths)
JAX_PLATFORMS=cpu uv run --no-sync python docs/reports/wand_vs_pid_waves/run.py
# Smoke run (5 seeds, ~30s):
JAX_PLATFORMS=cpu uv run --no-sync python docs/reports/wand_vs_pid_waves/run.py --quick
```

To regenerate the narrative `report.md`, point an agent at
`interpretation_skill.md` in this folder. The skill reads
`metrics.json`, `report_guidelines.txt`, `recipe.md`, and the PNGs
in `plots/`.

## Output structure

```
docs/reports/wand_vs_pid_waves/
├── README.md                  one-page entry point
├── recipe.md                  this file
├── interpretation_skill.md    agent-harness-agnostic "how to write report.md"
├── run.py                     single regenerator script
├── metrics.json               aggregated numbers (seed-level + per-controller)
├── report_guidelines.txt      physics checklist for the interpretation skill
├── report.md                  generated narrative (committed)
└── plots/                     PNG artifacts
    ├── dashboard_mechanical.png       single-run 6-panel dashboard (seed 0)
    ├── dashboard_pid_natural.png      single-run 6-panel dashboard (seed 0)
    ├── dashboard_pid_deeper.png       single-run 6-panel dashboard (seed 0)
    ├── compare_ride_height.png        seed-0 three-way overlay
    ├── compare_flap_command.png       seed-0 three-way overlay
    ├── compare_wand_angle.png         seed-0 three-way overlay
    ├── mc_ride_height_rms.png         50-seed box+strip
    ├── mc_ride_height_rms_around_target.png  50-seed RMS vs own setpoint
    ├── mc_breach_distribution.png     50-seed box+strip
    ├── mc_flap_activity.png           50-seed box+strip
    └── mc_pitch_speed.png             50-seed box+strip (paired)
```

## Report structure (followed by `interpretation_skill.md`)

1. **Flagged caveats** — any unexpected results or orderings that differ
   from the plan's expectations. Write this section first; it prevents
   the reader from being misled by the headline numbers.
2. **Summary** — 3-5 bullets, lead with `breach_count` (safety metric)
   and `ride_height_rms_around_target` (intra-controller tracking).
3. **Setup** — trim state, wave preset, gains, sim parameters
   (pull verbatim from `metrics.json["setup"]`).
4. **Single-seed time series** (seed=0) — walk through
   `dashboard_mechanical.png`, `dashboard_pid_natural.png`,
   `dashboard_pid_deeper.png`, and the three comparison overlays.
5. **Monte Carlo (50 seeds)** — distribution box/strip plots for ride
   height RMS, breach count, flap activity, pitch RMS, speed loss.
   Reference the exact numbers from `metrics.json["monte_carlo"]`.
   Use `ride_height_rms_around_target` (not `ride_height_rms`) for
   cross-setpoint comparison — `ride_height_rms` penalises any
   controller whose setpoint differs from the natural trim.
6. **Mechanism** — why the deeper-trim PID dominates on breach count;
   why the PID at natural trim tracks perfectly but breaches more;
   why rms_vs_target for pid_deeper is higher than pid_natural (theta
   equilibrium shift at the deeper setpoint — see scratchpad).
7. **Tuning suggestions** — lead with deeper-trim as the recommendation.

## When to re-run

- Wand kinematics change (`WandLinkage` defaults, `wand_pivot_position`,
  wand length).
- Wave model changes (`WAVE_SF_BAY_MODERATE` preset, JONSWAP
  spectrum implementation, Stokes order).
- Moth trim solver change (different equilibrium point).
- `PIDController` algorithm change (e.g., anti-windup, derivative
  filter, wave-aware inversion).
- After tuning the default PID gains (`_DEFAULT_PID_KP/KI/KD` in
  `src/fmd/simulator/moth_scenarios.py`).

## What to tune

This report is a **starting point**, not a final answer. The canonical
recommendation is **deeper-trim PID**:

```python
from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.moth_scenarios import create_pid_wand_config

# NED-positive-down: + 0.30 makes pos_d less negative = boat rides lower
# = foil tip 30 cm below the ventilation threshold (not -0.30!)
target = compute_tip_at_surface_pos_d() + 0.30
sensor, estimator, controller = create_pid_wand_config(
    lqr, heel_angle=heel, target_pos_d=float(target)
)
```

The inversion bakes in `cos(heel)` on the pivot z-component, making it
pos_d-agnostic under theta=trim_theta. Known limitation: a different
setpoint shifts the pitch equilibrium, leaving an ~8 cm residual bias
(see scratchpad). The breach metric (1.7 vs 25 vs 86 per 60 s window)
tells the safety story better than rms_vs_target.

Other things to tune:

- **PID gains**: pass `Kp=, Ki=, Kd=` to `create_pid_wand_config`.
  Try Kd > 0 only if you first low-pass the wand angle (raw wand
  signal carries wave-orbital motion → destabilisation through D).
- **Wave preset**: swap `WAVE_SF_BAY_MODERATE` for
  `WAVE_SF_BAY_LIGHT` (lighter chop, shorter wavelengths), or
  build your own `WaveParams`.
- **Simulation duration / seeds**: 60s × 50 seeds is enough to see
  the qualitative ordering; for tighter confidence intervals raise
  `n_seeds` or `duration`.
- **Wand-only controllers + EKF**: replace `PassthroughEstimator`
  with an EKF to recover vertical velocity, then use it in a
  state-feedback law (effectively LQG). This is what
  `create_wand_only_config()` does.

## Physics checklist (cross-reference)

- NED signs: `pos_d` more negative = boat rising. Always.
- A breach is `tip_depth < 0`, where `tip_depth` is the leeward
  foil-tip NED depth (positive = submerged).
- "Ride height RMS" is std of pos_d about its target, not about
  its mean. The two differ when there is a steady-state bias.
- Mean depth factor near 1.0 means foil is mostly submerged.
- Flap saturation > ~10% indicates the controller is hitting its
  envelope; either de-rate the gains or expand the actuator
  bounds (model: see `MAIN_FLAP_MIN/MAX` in `src/fmd/simulator/moth_3d.py`).
