# Recipe — wand_vs_pid_waves

## Purpose

Compare a passive mechanical wand-to-flap linkage against a wand-only
PID controller under SF Bay moderate waves. The report is the canonical
fmd example of how to organise a regeneratable analysis: a recipe +
script + agent-readable interpretation skill that any user can clone,
re-run, and tweak as a starting point for their own tuning.

## Prerequisites

- Moth model: `MOTH_BIEKER_V3`
- Trim solve: `find_moth_trim(u_forward=10 m/s, heel_angle=30 deg)`
- Wave preset: `WAVE_SF_BAY_MODERATE` (Hs=0.5m, Tp=3.0s, JONSWAP
  gamma=4.0, Stokes 2nd order), head seas (`mean_direction=pi`)
- Controllers (both in `fmd.simulator.moth_scenarios`):
  - `create_mechanical_wand_config()` (default linkage,
    `pullrod_offset=0.005`)
  - `create_pid_wand_config()` with default gains
    `Kp=0.6, Ki=0.1, Kd=0.0` (rad-flap per m-height-error,
    per m·s, per m/s)
- Closed-form wand-to-height inversion (PID): assumes trim attitude
  (theta=0, heel=0), with a per-construction `wand_angle_offset`
  calibrated so the inversion reproduces `pos_d_target` at the trim
  wand angle (round-trip identity).
- Simulation: 60s duration, dt=0.005s, paired wave seeds (mechanical
  and PID see the same wave field per seed).
- Monte Carlo: 50 wave seeds (5 in `--quick`); both controllers run
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
    ├── dashboard_mechanical.png       single-run 6-panel dashboard
    ├── dashboard_pid.png              single-run 6-panel dashboard
    ├── compare_ride_height.png        seed-0 overlay
    ├── compare_flap_command.png       seed-0 overlay
    ├── compare_wand_angle.png         seed-0 overlay
    ├── mc_ride_height_rms.png         50-seed box+strip
    ├── mc_breach_distribution.png     50-seed box+strip
    ├── mc_flap_activity.png           50-seed box+strip
    └── mc_pitch_speed.png             50-seed box+strip (paired)
```

## Report structure (followed by `interpretation_skill.md`)

1. **Summary** — 3-5 bullets, lead with headline ride-height RMS and
   breach count.
2. **Setup** — trim state, wave preset, gains, sim parameters
   (pull verbatim from `metrics.json["setup"]`).
3. **Single-seed time series** (seed=0) — pos_d, theta, u, flap, depth
   factor for both controllers, with phase relationship to wave forcing.
4. **Monte Carlo (50 seeds)** — distribution box/violin plots for ride
   height RMS, breach count, flap activity, pitch RMS, speed loss.
   Reference the exact numbers from `metrics.json["monte_carlo"]`.
5. **Mechanism** — why the PID outperforms (or underperforms) on each
   metric; where it saturates; what the mechanical wand's offset means
   for safety margin.
6. **Tuning suggestions** — which gain to raise/lower, which wave
   preset is harder, what would benefit from a wave-aware sensor model.

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

This report is a **starting point**, not a final answer. Things you
might tune:

- **PID gains**: pass `Kp=, Ki=, Kd=` to `create_pid_wand_config`.
  Try Kd > 0 only if you first low-pass the wand angle (the raw
  signal carries wave-orbital motion, which destabilises the boat
  through D).
- **Wave preset**: swap `WAVE_SF_BAY_MODERATE` for
  `WAVE_SF_BAY_LIGHT` (lighter chop, shorter wavelengths), or
  build your own `WaveParams`.
- **Simulation duration / seeds**: 60s × 50 seeds is enough to see
  the qualitative ordering; for tighter confidence intervals raise
  `n_seeds` or `duration`.
- **Trim depth (safety margin)**: pass `target_pos_d` to
  `design_moth_lqr` to put the trim deeper relative to the foil
  ventilation point. By default trim sits ~7cm above the foil-tip
  surface intersection — under waves with peak excursion ~0.25m
  this is too shallow for either controller to stay submerged.
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
