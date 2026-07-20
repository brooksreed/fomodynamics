# Recipe — wand_vs_pid_waves

## Purpose

Compare a passive mechanical wand-to-flap linkage against three wand-only
feedback configurations (natural-trim PID, deeper-trim PID, and a tuned
proportional-only controller) under SF Bay moderate waves. The report is
the canonical fmd example of how to organise a regeneratable analysis: a
recipe + script + agent-readable interpretation skill that any user can
clone, re-run, and tweak as a starting point for their own tuning.

## Prerequisites

- Moth model: `MOTH_BIEKER_V3`
- Trim solve: `design_moth_lqr(u_forward=10.0)` (heel=30 deg)
- Wave preset: `WAVE_SF_BAY_MODERATE` (Hs=0.5m, Tp=3.0s, JONSWAP
  gamma=4.0, Stokes 2nd order), head seas (`mean_direction=pi`)
- Controllers (all in `fmd.simulator.moth_scenarios`), each calibrated
  and initialized at its **own** pinned trim:
  - `create_mechanical_wand_config()` — passive linkage; the
    `pullrod_offset` is **auto-tuned closed-form at the trim**
    (`WandLinkage.required_pullrod_offset`; 0.005508 m for
    BIEKER_V3 / 30° / u=10), so the natural trim is the linkage's
    exact calm equilibrium. Explicit overrides still win.
  - `create_pid_wand_config()` — natural trim, default gains
    `Kp=0.6, Ki=0.1, Kd=0.0`, `target_pos_d = trim pos_d ≈ -1.40 m`
  - `create_pid_wand_config(target_pos_d=compute_tip_at_surface_pos_d() + 0.30)`
    — deeper trim; commands the boat 30 cm below the foil-tip
    ventilation threshold (NED-positive-down: `+ 0.30` makes pos_d
    *less* negative = boat rides lower = foil tip more submerged).
    The factory calibrates theta_ref / flap / elevator at the pinned
    trim of the target depth, so the calm bias is mm-level at any depth
    (at equal speed the trim pitch is nearly depth-invariant).
  - `create_pid_wand_config(Kp=0.4, Ki=0.0, Kd=0.0)` — tuned
    proportional-only controller at the natural trim. Same wand sensor,
    inversion, setpoint, and own-trim calibration as the natural-trim
    PID; the only change is the gains. Kp comes from a paired-seed gain
    sweep at Ki = 0 on this exact setup: tracking is nearly flat in
    stiffness (Kp 0.2–1.5 spans ~8 % in RMS) with a shallow optimum at
    Kp ≈ 0.3–0.4, and stiffening 0.6 → 0.844 *worsens* tracking by
    2.1 mm — at Hs = 0.5 m the residual is disturbance-dominated, so
    Kp = 0.4 sits on the soft side of that plateau. With Ki = 0 it does
    not chase the wand inversion's wave-rectified height bias, so it
    tracks well at low flap effort.
- Thrust: **P speed governor** (`apply_speed_governor`) —
  `F = max(T0 + Kp*(u_target - u), 0)`, Kp = 40 N/(m/s), u_target = 10;
  T0 = pinned-trim thrust at each controller's own setpoint (75.5 N
  natural, 91.9 N deeper). This gives every configuration a surge
  equilibrium; the calibrated thrust table alone is a required-thrust
  curve with zero surge stiffness and must not be the dynamic law.
  `--captive` runs `surge_enabled=False` as a towing-tank diagnostic.
- Closed-form wand-to-height inversion (PID): under trim attitude
  (theta=theta_ref) and constant heel, the mapping is:
  `pos_d_est = -z_pivot * cos(heel) - L_wand * cos(wand_angle) + offset`
  where `offset` is a per-construction calibration that absorbs the
  trim-theta residual. theta_ref is the pinned-trim pitch at the
  controller's own setpoint, making the calm bias mm-level everywhere.
- Simulation: 60s duration, dt=0.005s, paired wave seeds (all
  controllers see the same wave field per seed); plant built with
  `enable_encounter_distance=True` (integrated encounter position
  feeds the wave-aware breach metric).
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
    ├── mc_pitch_speed.png             50-seed box+strip (paired)
    └── surge_psd.png                  seed-0 surge spectra (governor/wave bands)
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
6. **Mechanism** — why the deeper-trim PID wins on breach count (tip
   margin vs crest amplitude — a setpoint effect, not a control-law
   effect); what each controller pays in added resistance under the
   governor; why the mechanical linkage and the PIDs differ on
   wave-band tracking and flap activity.
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
- Governor changes (`apply_speed_governor`, Kp/u_target defaults) or
  per-setpoint trim-calibration changes (`setpoint_trim`,
  `WandLinkage.required_pullrod_offset`).

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

The factory solves the pinned trim at `target_pos_d` and calibrates the
inversion's theta_ref (plus flap/elevator references) there, so the
controller reaches its setpoint with mm-level calm bias at any depth in
the wand's range. Pair a non-natural setpoint with the speed governor
so it also has its thrust equilibrium. The deeper setpoint's payoff is a
~2x reduction in foil-tip breaches over a 50-s window, bought with extra
strut immersion (higher calm thrust); it also has the lowest ride-height
RMS about its own setpoint. See `report.md` for the exact numbers.

Other things to tune:

- **PID gains**: pass `Kp=, Ki=, Kd=` to `create_pid_wand_config`. On a
  wand-derived (relative) height sensor, a soft proportional-only law
  (Kp=0.4, Ki=0) tracks better than the default `Kp=0.6, Ki=0.1` and at
  lower flap effort — the integrator chases the inversion's
  wave-rectified height bias. This is the `p_tuned` config; the report
  quantifies the gap. Try Kd > 0 only if you first low-pass the wand
  angle (raw wand signal carries wave-orbital motion → destabilisation
  through D).
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
