# wand_vs_pid_waves — report (regenerated 2026-05-13)

## Flagged findings (read first)

**Breach ordering is non-obvious**: the mechanical wand has *fewer*
breaches (25.2/60s) than the natural-trim PID (85.9/60s), which is
reversed from the naive expectation that the PID should win everywhere.
The mechanism is geometric: the mechanical wand drifts 21 cm *lower*
than its commanded trim (pos_d −1.19 m vs target −1.40 m), giving
it 21 cm more safety margin against the foil-tip ventilation threshold
(−1.47 m). The PID's integrator correctly eliminates that bias — but at
the default trim point, "correct" means parking only 7 cm from
ventilation. Explained in § Mechanism.

**pid_deeper rms_vs_target > pid_natural rms_vs_target**: the deeper-trim
PID shows 0.189 m rms_vs_target (50-seed mean) versus pid_natural's
0.091 m. This is a known limitation: the deeper setpoint shifts the
pitch equilibrium from ~0.8° to ~3°; the inversion was calibrated at
the natural trim theta. The theta-induced residual is ~8 cm calm-water,
~14 cm under waves. The safety-relevant metric is breach_count (1.7 vs
25.2 vs 85.9), where pid_deeper dominates 15× over the mechanical wand
and 51× over pid_natural. User-approved to accept and document (see
scratchpad § Phase E.3-F.3).

## Summary

- **pid_deeper dominates on safety**: 1.7 breaches / 60 s (50-seed
  mean) vs 25.2 for mechanical and 85.9 for pid_natural. The 30 cm
  commanded safety margin is largely maintained despite the theta-shift
  bias (depth_factor_mean = 0.997 — foil nearly always submerged).
- **pid_natural dominates on ride-height tracking**: rms_vs_target
  0.091 m (50-seed mean), 2.8× better than mechanical (0.252 m) and
  2.1× better than pid_deeper (0.189 m). Integrator eliminates steady
  bias; mechanical wand cannot.
- **Both PIDs have lower flap activity than mechanical**: 0.099 and
  0.141 rad RMS vs 0.145 rad, with 1.2% / 15.3% saturation vs 17.0%.
  Mechanical wand passes wave-orbital wand motion through the linkage
  near-linearly; PID's proportional term softens the response.
- **pid_deeper recommended** for SF Bay moderate waves: achieves the
  safety margin the natural-trim PID lacks, at the cost of slightly
  higher rms_vs_target and a 30 cm deeper trim (more drag).

## Setup

Source: `metrics.json["setup"]`.

| Parameter | Value |
|---|---|
| Moth preset | `MOTH_BIEKER_V3` |
| Forward speed | 10.0 m/s |
| Heel angle | 30.0 deg |
| Timestep | 0.005 s |
| Duration | 60.0 s |
| Steady-state window | 10.0 s onwards |
| Monte Carlo seeds | 50 |
| Wave preset | `WAVE_SF_BAY_MODERATE` (Hs=0.5 m, Tp=3.0 s, JONSWAP γ=4.0, Stokes 2nd order) |
| Wave direction | π rad (head seas) |

Trim state (NED: pos_d negative = boat above water surface):

| State | Value |
|---|---|
| pos_d | −1.400 m (CG 1.40 m above still water; foil-tip ventilation at −1.47 m, 7 cm margin) |
| theta | +0.82 deg (nose-up) |
| w | +0.143 m/s |
| q | 0.0 deg/s |
| u | 10.0 m/s |

Trim control: flap = −0.14 deg, elevator = +0.14 deg.
Control bounds: flap ∈ [−10, +15] deg.

Controllers:

- **mechanical** (`create_mechanical_wand_config`): passive WandLinkage,
  `pullrod_offset = 0.005 m`. Target = natural trim pos_d = −1.40 m.
- **pid_natural** (`create_pid_wand_config`): Kp=0.6, Ki=0.1, Kd=0.0.
  Target = natural trim pos_d = −1.40 m.
- **pid_deeper** (`create_pid_wand_config`, `target_pos_d = −1.169 m`):
  same gains. Target = `compute_tip_at_surface_pos_d() + 0.30 m`
  = −1.469 + 0.30 = −1.169 m. In NED: +0.30 makes pos_d less
  negative → boat rides 0.30 m lower → foil tip 30 cm below
  the ventilation threshold instead of 7 cm.

## Single-seed time series (seed = 0)

### Per-controller dashboards

Each dashboard shows six panels: wave elevation (main foil + rudder),
pos_d, pitch, forward speed, flap command, and leeward-tip depth.

| Metric (seed 0, ss-window) | mechanical | pid_natural | pid_deeper |
|---|---|---|---|
| pos_d mean (m) | −1.111 | −1.402 | −0.993 |
| pos_d target (m) | −1.400 | −1.400 | −1.169 |
| Breach count | 5 | 30 | **0** |
| Foil-tip depth mean (m, NED) | 0.320 | 0.050 | 0.444 |
| Speed mean (m/s) | 5.35 | 8.74 | 5.18 |
| Flap mean (deg) | 6.10 | 2.55 | 5.63 |
| Flap std (deg) | 7.61 | 5.06 | 7.16 |
| Pitch mean (deg) | 3.14 | 1.42 | 2.69 |
| Pitch rms error (rad) | 0.057 | 0.038 | 0.046 |

NED-sign cross-check on the mechanical wand:
- pos_d mean = −1.111 m. Trim pos_d = −1.400 m. −1.111 is *less* negative
  → the boat is flying *lower* (CG 1.111 m above water vs 1.400 m at trim).
  "Mechanical wand parks the boat lower than trim" ✓ — foil is more
  submerged, hence 5 breaches vs 30 for pid_natural.
- Speed loss = 10 − 5.35 = 4.65 m/s. More drag from lower trim position. ✓

NED-sign cross-check on pid_natural:
- pos_d mean = −1.402 m ≈ target −1.400 m. Integrator removed steady
  bias as expected. ✓
- Foil-tip depth mean = 0.050 m (positive = submerged 5 cm on average).
  Given trim is 7 cm above ventilation and waves push the tip 25 cm,
  30 breaches in 50 s (ss-window) is consistent. ✓

NED-sign cross-check on pid_deeper:
- pos_d mean = −0.993 m. Target = −1.169 m. −0.993 is *less* negative
  → boat is even *lower* than target (further from trim). This is the
  theta-shift residual: equilibrium pitch at −1.169 m is ~3° vs trim
  0.82°; the inversion interprets the wand angle as indicating ~−1.169 m
  while actual pos_d is ~−0.993 m (14 cm shallower).
- Despite the bias, the foil-tip depth mean = 0.444 m (44 cm below
  surface, excellent margin). Breach = 0 for seed 0. ✓

`dashboard_pid_deeper.png` is visually distinctive: pos_d sits ~30 cm
lower than the other two controllers; flap activity is elevated (more
lift needed to hold the deeper position against gravity); wave-induced
fluctuations are present but don't breach because the foil has clearance.

### Comparison overlays

`compare_ride_height.png` shows three pos_d traces side by side:
- pid_natural: tight band around −1.40 m, low amplitude.
- mechanical: oscillates around −1.11 m with larger amplitude.
- pid_deeper: oscillates around −1.00 m (theta-shift bias) with
  intermediate amplitude; stays far from the ventilation threshold.

`compare_flap_command.png`: pid_natural has the smoothest trace
(lowest std 5.06 deg). Mechanical and pid_deeper both show larger
excursions (~7.6 and ~7.2 deg std respectively).

`compare_wand_angle.png`: all three controllers share the same
`WandSensor`, so the wand-angle signals are nearly identical.
The control law is entirely responsible for the pos_d differences.

## Monte Carlo across 50 seeds

Source: `metrics.json["monte_carlo"]` aggregates. All values are means
over the 10–60 s steady-state window.

| Metric | mechanical | pid_natural | pid_deeper |
|---|---|---|---|
| n_seeds | 50 | 50 | 50 |
| target_pos_d (m) | −1.400 | −1.400 | −1.169 |
| ride_height_mean (m) | −1.194 ± 0.091 | −1.398 ± 0.014 | −1.031 ± 0.076 |
| rms_vs_target (m) | 0.252 ± 0.111 | **0.091 ± 0.009** | 0.189 ± 0.080 |
| breach_count (per ss-window) | 25.2 ± 8.5 | 85.9 ± 10.9 | **1.70 ± 1.25** |
| breach_fraction | 0.25 | 0.76 | 0.02 |
| flap_rms (rad, dev. from trim) | 0.145 ± 0.025 | **0.099 ± 0.008** | 0.141 ± 0.023 |
| flap_saturation_fraction | 0.170 ± 0.10 | 0.012 ± 0.008 | 0.153 ± 0.092 |
| pitch_rms_error (rad) | 0.059 ± 0.027 | **0.037 ± 0.003** | 0.047 ± 0.022 |
| speed_loss_mean (m/s) | 3.92 ± 1.07 | **1.42 ± 0.53** | 3.49 ± 0.98 |
| depth_factor_mean | 0.982 ± 0.012 | 0.815 ± 0.034 | **0.997 ± 0.003** |

`mc_ride_height_rms_around_target.png` (RMS vs own setpoint) is the
appropriate cross-setpoint comparison. It shows pid_natural tightest
(0.091 m), pid_deeper intermediate (0.189 m, theta-shift limited), and
mechanical widest (0.252 m). `mc_ride_height_rms.png` (RMS vs trim)
shows mechanical at 0.252, pid_natural at 0.091, pid_deeper at 0.420 —
the last is high because pid_deeper's setpoint is 23 cm from trim, so
`rms_vs_trim` is dominated by the setpoint offset, not the wave tracking
error. Do not use `mc_ride_height_rms.png` for cross-setpoint comparison.

`mc_breach_distribution.png` is the safety story: pid_deeper's
distribution (median 2, max 5) vs mechanical (median 24.5, max 50) vs
pid_natural (median 85.5, max 107). The separation is unambiguous.

`mc_flap_activity.png`: pid_natural has least flap variation. mechanical
and pid_deeper are similar. mechanical's higher saturation fraction
(17% vs 15%) reflects its wider operating range.

`mc_pitch_speed.png`: pid_natural's tighter pos_d tracking propagates
into lower pitch rms and less speed loss. pid_deeper and mechanical are
similar on pitch/speed (both riding lower, more drag).

## Mechanism

**Why pid_deeper dominates on safety**

The foil-tip ventilation threshold is −1.47 m (pos_d). Natural trim
sits 7 cm above this, leaving almost no headroom for wave excursion
(peak ±25 cm under WAVE_SF_BAY_MODERATE). The deeper target (−1.169 m)
places the trim 30 cm below the threshold. Even with the ~14 cm
theta-shift bias, the actual mean pos_d (−1.031 m) still provides
~44 cm of headroom — more than enough to absorb most wave excursions.
Result: 1.7 breaches / 60 s vs 85.9 for pid_natural.

**Why the mechanical wand has fewer breaches than pid_natural**

The mechanical linkage inherits a steady-state offset: the boat
equilibrates at −1.19 m (21 cm below trim) rather than −1.40 m. This
is not active wave anticipation — it's a geometry artefact of the
WandLinkage under wave forcing. But it functions as a safety margin,
pushing the foil tip ~21 cm further from the surface. The PID's
integrator correctly removes that bias, which is its job — but at
this trim point, the "correct" answer exposes the foil tip.

**Why pid_deeper rms_vs_target > pid_natural rms_vs_target**

Algebraically, the inversion formula (`pos_d_est = -z_p*cos(heel)
- L*cos(wand_angle) + offset`) is calibrated at the natural trim theta
(0.82°). The deeper setpoint shifts the pitch equilibrium to ~3°. The
offset cannot absorb a theta different from the calibration theta, leaving
an ~8 cm calm-water residual and ~14 cm under waves. This is documented
as the "theta-shift bias" in the scratchpad. It is not a code bug; the
`cos(heel)` fix (commit `1dc9305`) correctly makes the inversion
pos_d-agnostic for *fixed* theta. To eliminate the theta-shift residual:
re-solve the trim at the override target pos_d and use that theta for
calibration (out of scope for this report, user-approved to defer).

**Why both PIDs have lower flap activity than the mechanical wand**

The mechanical linkage maps wand angle to flap command nearly linearly
near the operating point. Wave-orbital motion causes large wand swings
(tens of degrees) that pass through directly, saturating the flap 17%
of the time. The PIDs at Kp=0.6 attenuate this: a given pos_d error
produces a smaller flap command than the linkage geometry dictates.
With Kd=0 (required to avoid wave-noise amplification), the derivative
term is inactive.

## Tuning suggestions

**Recommended starting point**: pid_deeper — it achieves the safety
margin the natural-trim PID lacks, with acceptable tracking quality.

```python
from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.moth_scenarios import create_pid_wand_config

# NED: + 0.30 makes pos_d less negative = boat rides lower = more margin
target = compute_tip_at_surface_pos_d() + 0.30   # = −1.469 + 0.30 = −1.169 m
sensor, estimator, controller = create_pid_wand_config(
    lqr, heel_angle=heel, target_pos_d=float(target)
)
```

Other tuning directions:

- **Eliminate theta-shift bias**: re-solve trim at `target_pos_d`
  (`design_moth_lqr` with a custom theta initialisation at the new
  depth), then pass `lqr_deep` to `create_pid_wand_config`. This makes
  the inversion calibration exact at the operating point.
- **Lighter waves** (`WAVE_SF_BAY_LIGHT`): at Hs~0.3 m, the natural-
  trim PID no longer breaches; pid_deeper is overly conservative. The
  pid_natural ordering may recover in that regime.
- **Kp/Ki tuning**: Kp=0.6, Ki=0.1 is conservative. Raising toward
  Kp=1.0 reduces rms_vs_target but requires validating breach count
  (instability at Kp=1.5 with Ki>0 on some seeds).
- **Derivative action**: low-pass the wand signal at ~1 Hz (well below
  1/Tp = 0.33 Hz of the dominant wave period) before computing the
  D term, or use an EKF to recover w (vertical velocity) directly.

## Caveats

- Single trim point (10 m/s, 30° heel, head seas). Beam or quartering
  seas may change the breach ordering.
- Single wave preset. Shorter-period chop (higher ka) may interact
  differently with the wand kinematics.
- The theta-shift bias of pid_deeper (~8 cm calm, ~14 cm wavy) is a
  known limitation of the closed-form inversion calibrated at the
  natural trim theta. The breach metric is the robust safety indicator;
  rms_vs_target should be interpreted accordingly.
- All RMS / breach numbers are over the 10–60 s steady-state window
  (first 10 s excluded for transient).
- `flap_rms` is the RMS of the flap *deviation from trim flap* (not the
  absolute command), so it measures variation about the operating point.
- The breach metric is **wave-aware** (computed via
  `compute_leeward_tip_depth` which accounts for wave surface elevation
  at the foil position). Absolute counts depend on the wave preset.
- `depth_factor_mean` from the MC is the fraction of time the foil is
  submerged, not a depth ratio. 0.997 = foil submerged 99.7% of the
  time; 0.815 = submerged 81.5%.
