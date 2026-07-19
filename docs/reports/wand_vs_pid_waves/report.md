# wand_vs_pid_waves — report

**What this is.** A controlled comparison of four ride-height controllers for a
foiling Moth in waves: a passive mechanical wand-to-flap linkage and three
wand-only feedback laws. All four hold the same boatspeed (a P speed governor
models the sailor holding the sheet), see the same wave field per seed (paired
Monte Carlo), and are each calibrated at the pinned trim of their own ride-height
setpoint. The study answers three questions: what sets the foil-tip breach rate,
how the passive linkage compares to closed-loop feedback, and whether an
integrator helps on a wand-derived (relative) height sensor.

See [`model_setup.md`](model_setup.md) for what the simulation captures and
[`recipe.md`](recipe.md) for how to regenerate and tune. This narrative is
produced from the committed artifacts by following
[`interpretation_skill.md`](interpretation_skill.md).

**Provenance** (`metrics.json["provenance"]`): fmd commit
`9bc4cb9`, install mode `editable`, params hash `a669b19e5c68`. An `editable`
install mode means the artifacts were generated from a working tree rather than a
pinned release.

## Flagged findings (read first)

**Breach count is set by the ride-height setpoint, not the control law.** The
three controllers that hold the natural trim (−1.40 m) breach at practically
identical rates — 62.8 (mechanical), 64.0 (pid_natural), 63.1 (p_tuned) foil-tip
breaches per 50-s window — despite very different flap behaviour. The deeper-trim
controller, whose tip sits 30 cm below the ventilation threshold, breaches 28.4
times: a 2.2× reduction bought purely by geometry (crest amplitude vs tip margin),
not by the feedback law.

**The integrator does not help tracking on this relative sensor.** Both
integrator-free controllers — the mechanical linkage and the proportional-only
p_tuned — track their setpoint *better* than the default PID (which carries
Ki = 0.1), and the softest of the three, p_tuned (Kp = 0.4, Ki = 0), tracks best
of the natural-setpoint group while using the least flap effort. Ranking the RMS
about each controller's own setpoint (paired 50-seed means): pid_deeper 0.0792 m <
**p_tuned 0.0896 m** < mechanical 0.0926 m < pid_natural 0.1029 m. The soft P law
beats the passive linkage by a paired 3.1 mm [2.5, 3.7] at 0.1 % flap saturation
vs the linkage's 8.0 %, and beats the default PID by 13.3 mm [12.5, 14.1]. The
mechanism (see § Mechanism) is that the wand's height estimate is biased under
waves, so an integrator servoes the true height toward a wandering wave-rectified
reference and *injects* variance rather than removing it.

**Riding deeper wins on safety and tracking, and pays in thrust, not speed.**
pid_deeper has the lowest breach count, the lowest RMS about its own setpoint, and
the lowest pitch RMS of the PIDs — because at depth the foil's lift is insensitive
to surface proximity. Its cost is +16.4 N of calm thrust (deeper strut immersion),
not lost boatspeed: the governor holds all four controllers within 0.23 m/s of
target.

## Summary

- **Safety (foil-tip breaches, 50-s window):** pid_deeper 28.4 ± 3.1, then a
  three-way tie at the natural setpoint — mechanical 62.8 ± 2.5, p_tuned
  63.1 ± 2.8, pid_natural 64.0 ± 2.9. Breach rate tracks the setpoint, not the
  controller.
- **Tracking (RMS about own setpoint):** pid_deeper 0.0792 m best; among the
  natural-setpoint three, p_tuned 0.0896 m beats mechanical 0.0926 m beats
  pid_natural 0.1029 m. The two integrator-free laws lead; the softest gain wins.
- **Effort:** flap saturation p_tuned 0.1 % ≪ pid_deeper 0.9 % < pid_natural
  1.9 % ≪ mechanical 8.0 %; flap RMS follows the same order. p_tuned's flap-rate
  RMS at seed 0 is 80 °/s vs 154 °/s for the mechanical linkage.
- **Speed / thrust:** under the governor (Kp = 40 N/(m/s), u_target = 10 m/s) all
  four hold speed (worst mean offset 0.23 m/s). The differentiator is mean added
  thrust to do so: p_tuned ≈ 0 N, pid_natural 0.9 N, pid_deeper 4.6 N (deeper
  strut), mechanical 9.4 N (largest flap activity → largest wave added
  resistance). Added resistance and speed offset are the same measurement in
  different units (see § Metric definitions).
- **All 200 runs stationary** (50 seeds × 4 controllers; drift test on u and pos_d
  passes; the governor never saturates).
- **Recommendation:** pid_deeper for the best breach margin and tracking when the
  extra strut drag is acceptable; otherwise **p_tuned** (soft proportional-only)
  as the natural-setpoint controller — it tracks best of that group at the lowest
  effort and essentially zero added resistance.

## Setup

Source: `metrics.json["setup"]`.

| Parameter | Value |
|---|---|
| Moth preset | `MOTH_BIEKER_V3` |
| Forward speed | 10.0 m/s (governed, u_target = 10) |
| Heel angle | 30.0 deg |
| Timestep | 0.005 s |
| Duration | 60.0 s |
| Steady-state window | 10–60 s (all RMS/breach metrics) |
| Monte Carlo seeds | 50, paired across controllers |
| Wave preset | `WAVE_SF_BAY_MODERATE` (Hs = 0.5 m, Tp = 3.0 s, JONSWAP γ = 4.0, Stokes 2nd order) |
| Wave direction | π rad (head seas); encounter frequency ≈ 1.0 Hz at 10 m/s |
| Thrust law | speed governor `F = max(T0 + Kp·(u_target − u), 0)`, Kp = 40 N/(m/s) |
| Encounter distance | integrated plant state (`enable_encounter_distance = True`) |

Natural trim (NED: pos_d negative = CG above the still-water surface):

| State | Value |
|---|---|
| pos_d | −1.3998 m (foil-tip ventilation threshold −1.469 m → 7 cm tip margin) |
| theta | +0.836 deg (nose-up) |
| w | +0.146 m/s |
| u | 10.0 m/s |
| flap / elevator | −0.132 / +0.128 deg |
| thrust (governor T0) | 75.51 N |

Controllers — **each calibrated and initialized at its own pinned trim**
(`metrics.json["setup"]["setpoint_trims"]`):

- **mechanical** (`create_mechanical_wand_config`): passive `WandLinkage`;
  `pullrod_offset` auto-tuned closed-form at the trim so the linkage outputs the
  trim flap at the trim wand angle (the natural trim is its exact calm
  equilibrium; calm-water bias −0.05 mm). Setpoint = natural trim −1.400 m.
- **pid_natural** (`create_pid_wand_config`): Kp = 0.6, Ki = 0.1, Kd = 0;
  setpoint = natural trim −1.400 m; inversion θ_ref = trim pitch.
- **pid_deeper** (`create_pid_wand_config`, `target_pos_d = −1.169 m` =
  `compute_tip_at_surface_pos_d() + 0.30`; NED: +0.30 makes pos_d less negative →
  boat rides 30 cm lower → foil tip 30 cm below the ventilation threshold).
  Calibrated at its own pinned trim: θ_ref = 0.824°, flap = −0.064°, elevator =
  +0.061°, T0 = 91.88 N.
- **p_tuned** (`create_pid_wand_config`, Kp = 0.4, Ki = 0, Kd = 0): same wand
  sensor, inversion, natural setpoint (−1.400 m) and own-trim calibration as
  pid_natural — the only difference is the gains (softer proportional, no
  integrator). T0 = 75.51 N.

### Metric definitions and reference frames (read before comparing columns)

- `ride_height_rms_around_target` — RMS of pos_d error vs the controller's **own
  setpoint**. The correct cross-setpoint tracking comparison.
- `ride_height_rms` — RMS vs the **natural trim** (−1.3998 m) for all controllers.
  For pid_deeper this is dominated by its deliberate 23 cm setpoint offset; never
  use it cross-setpoint.
- `pitch_rms_error`, `speed_loss_mean` — referenced to the **natural trim**
  (θ = 0.836°, u = 10 m/s) for all controllers, for cross-controller
  comparability.
- `flap_rms` — RMS of flap deviation from the controller's **own trim flap**
  (pid_deeper: −0.064°, the others −0.132°).
- `breach_count` / `breach_fraction` (Monte Carlo) — **wave-aware**: leeward tip
  depth vs the instantaneous wave surface at the foil's integrated encounter
  position, over 10–60 s. Counts are breach *onsets*; fraction is time-above-
  surface.
- The **single-seed dashboard** tip-depth stats are still-water-referenced
  (`wave_aware: false`) — do not mix them with the MC breach columns.
- `added_resistance_mean` = mean(F_sail − T0). With the governor unsaturated (0 %
  of steps, all seeds) this equals **Kp·mean_u_offset algebraically** — the same
  measurement as the speed columns in newtons, **not** an independent cross-check.
- `depth_factor_mean` — time-average of the force model's effective submerged-span
  fraction (1.0 = fully submerged).

## Single-seed time series (seed = 0)

### Per-controller dashboards

`dashboard_mechanical.png`, `dashboard_pid_natural.png`, `dashboard_pid_deeper.png`,
`dashboard_p_tuned.png` — six panels each: wave elevation at the foils, pos_d,
pitch, forward speed, flap command, leeward-tip depth (still-water reference in
this table).

| Metric (seed 0, 10–60 s) | mechanical | pid_natural | pid_deeper | p_tuned |
|---|---|---|---|---|
| pos_d mean (m) | −1.383 | −1.418 | −1.177 | −1.377 |
| pos_d target (m) | −1.400 | −1.400 | −1.169 | −1.400 |
| pos_d min / max (m) | −1.638 / −1.152 | −1.699 / −1.174 | −1.392 / −0.989 | −1.575 / −1.149 |
| Tip depth mean (m, still-water) | 0.076 | 0.039 | 0.282 | 0.082 |
| Tip depth min (m, still-water) | −0.219 | −0.268 | **+0.032** | −0.136 |
| Speed mean ± std (m/s) | 9.77 ± 0.07 | 10.00 ± 0.09 | 9.89 ± 0.06 | 10.00 ± 0.07 |
| Pitch mean ± std (deg) | 0.82 ± 2.11 | 0.96 ± 2.09 | 0.79 ± 1.73 | 0.85 ± 1.58 |
| Flap rate RMS (deg/s) | 154 | 110 | 153 | **80** |

NED-sign cross-checks (pos_d more negative = boat rising):

- **mechanical**: mean −1.383 m vs target −1.400 m → 1.7 cm less negative → rides
  ~1.7 cm **lower** than target under waves (wave rectification; its calm
  equilibrium is exact by construction). ✓
- **pid_natural**: mean −1.418 m → 1.8 cm more negative → rides ~1.8 cm **higher**
  than target (its integrator rectifies upward). Tip-depth mean is only 0.039 m,
  so wave excursions (±0.25 m) put the tip above the surface roughly once per
  encounter. ✓
- **pid_deeper**: mean −1.177 m vs target −1.169 m → 0.8 cm low of target. Tip
  depth never crosses the still-water surface for seed 0 (min +0.032 m); its
  wave-aware breaches come from crests reaching up to the tip, not the tip
  reaching the mean surface. ✓
- **p_tuned**: mean −1.377 m → 2.3 cm less negative → rides ~2.3 cm **lower** than
  target — the largest natural-setpoint rectification offset, because the soft
  proportional gain and absent integrator leave the wave-rectified DC bias
  uncorrected (still within the 3 cm calibration tolerance). Its flap-rate RMS is
  the lowest of all four (80 °/s), and its pitch std the smallest (1.58°). ✓

### Comparison overlays

`compare_ride_height.png`: the three natural-setpoint controllers overlay each
other around −1.40 m; pid_deeper holds a parallel band around −1.17 m. All respond
at the ~1 Hz encounter frequency; no controller drifts.

`compare_flap_command.png`: mechanical (blue) and pid_natural (red) repeatedly clip
the +15° flap limit; p_tuned (orange) stays well inside a ±10° envelope and rarely
saturates — the visual signature of its 0.1 % saturation vs 8.0 % (mechanical) and
1.9 % (pid_natural), 50-seed means. pid_deeper is intermediate at 0.9 %.

`compare_wand_angle.png`: mechanical, pid_natural and p_tuned see nearly identical
wand angles (same natural setpoint, same sensor); pid_deeper's wand rides ~20°
larger because the boat flies lower, the wand trailing more horizontal.

## Monte Carlo across 50 seeds

Source: `metrics.json["monte_carlo"]` (mean ± std over 50 paired seeds, 10–60 s).

| Metric | mechanical | pid_natural | pid_deeper | p_tuned |
|---|---|---|---|---|
| target_pos_d (m) | −1.400 | −1.400 | −1.169 | −1.400 |
| ride_height_mean (m) | −1.3824 ± 0.0011 | −1.4131 ± 0.0050 | −1.1766 ± 0.0011 | −1.3763 ± 0.0020 |
| rms vs own setpoint (m) | 0.0926 ± 0.0021 | 0.1029 ± 0.0037 | **0.0792 ± 0.0014** | 0.0896 ± 0.0028 |
| rms vs natural trim (m) | 0.0926 | 0.1029 | 0.2367 (offset-dominated) | 0.0896 |
| breach_count (wave-aware, /50 s) | 62.8 ± 2.5 | 64.0 ± 2.9 | **28.4 ± 3.1** | 63.1 ± 2.8 |
| breach_fraction (time tip exposed) | 0.375 | 0.429 | **0.093** | 0.353 |
| flap_rms (rad, vs own trim) | 0.1064 ± 0.0014 | 0.0922 ± 0.0025 | 0.0827 ± 0.0009 | **0.0634 ± 0.0014** |
| flap_saturation_fraction | 0.080 ± 0.007 | 0.019 ± 0.005 | 0.009 ± 0.003 | **0.001 ± 0.001** |
| pitch_rms_error (rad) | 0.0363 ± 0.0007 | 0.0362 ± 0.0012 | 0.0299 ± 0.0005 | **0.0279 ± 0.0007** |
| mean_u_offset (m/s) | 0.234 ± 0.006 | 0.022 ± 0.014 | 0.114 ± 0.003 | **−0.001 ± 0.004** |
| added_resistance_mean (N) (= Kp·row above) | 9.36 ± 0.22 | 0.87 ± 0.55 | 4.55 ± 0.11 | **−0.05 ± 0.16** |
| governor_saturation_fraction | 0.0 | 0.0 | 0.0 | 0.0 |
| stationarity pass fraction | 1.00 | 1.00 | 1.00 | 1.00 |
| depth_factor_mean | 0.859 ± 0.009 | 0.757 ± 0.024 | **0.9975 ± 0.0000** | 0.903 ± 0.008 |

Per-plot notes:

- `mc_ride_height_rms_around_target.png` (RMS vs own setpoint — the correct
  cross-setpoint comparison): four cleanly separated boxes, **pid_deeper lowest**
  (0.0792), then p_tuned (0.0896), mechanical (0.0926), pid_natural highest
  (0.1029). Per-seed std is 1.6–3.6 % of the mean — the entrained wave response
  gives tight CIs. Paired deltas (same wave seeds): p_tuned − mechanical =
  −3.1 mm [−3.7, −2.5] (t = 9.9); p_tuned − pid_natural = −13.3 mm [−14.1, −12.5];
  mechanical − pid_natural = −10.2 mm [−11.0, −9.4].
- `mc_ride_height_rms.png` (RMS vs natural trim): pid_deeper reads 0.2367 m because
  its setpoint is 23.1 cm from natural trim — the column measures the offset, not
  tracking. Kept for continuity; not a cross-setpoint metric.
- `mc_breach_distribution.png`: the three natural-setpoint boxes (62.8 / 64.0 /
  63.1) overlap almost completely; pid_deeper (28.4) sits far below. At u = 10 m/s
  with encounter frequency ≈ 1 Hz the natural-setpoint controllers breach on
  essentially every encounter; the deeper setpoint drops that to ≈ 0.55/encounter.
- `mc_flap_activity.png`: mechanical highest (0.1064 rad), then pid_natural
  (0.0922), pid_deeper (0.0827), p_tuned lowest (0.0634). The passive linkage
  passes wave-orbital wand motion straight through; the soft P law attenuates it
  most.
- `mc_pitch_speed.png`: pitch — p_tuned lowest (0.0279 rad), pid_deeper next
  (0.0299), mechanical ≈ pid_natural (0.0363/0.0362). Speed loss — mechanical
  worst (0.234 m/s), pid_deeper 0.114, pid_natural 0.022, p_tuned ≈ 0; these are
  the governor's steady droop offsets (added resistance ÷ Kp), all < 2.5 % of
  u_target.
- `surge_psd.png` (seed 0): all four spectra peak at the ≈ 1 Hz encounter band;
  the governor pole (~0.05 Hz, dashed) sits two decades below the wave peak (1/Tp,
  dotted). The clean separation confirms the governor closes the DC surge loop
  without shaping the wave-band response.

## Mechanism

**Why the deeper setpoint wins on breaches — and why the margin is 2.2×.** The
ventilation threshold is pos_d = −1.469 m. The natural setpoint leaves 7 cm of tip
margin; wave crests locally raise the surface by 0.25–0.4 m, so at −1.40 m the tip
is exposed roughly once per encounter regardless of controller. The deeper setpoint
puts the tip 30 cm under; only the larger crests reach it (breach fraction 9.3 % of
the window vs 35–43 %). This is a geometric, setpoint-driven result.

**Why the three natural-setpoint controllers tie on breaches.** They hold mean
heights within ~4 cm of each other (−1.376 to −1.418) at the same speed. Breach
onset at this setpoint is driven by the wave field, not by the ±1–2 cm differences
in mean ride height; the marginal breach distributions overlap almost completely.

**Why the soft proportional law tracks best of the natural-setpoint group.** The
wand's closed-form height estimate is biased under waves: the inversion assumes the
boat sits at its trim pitch, but pitch oscillates ±2° at the encounter frequency,
and the wand's trig nonlinearity rectifies the orbital motion. An integrator
(pid_natural, Ki = 0.1) drives the *true* height to null the error in that biased
*estimate*, so it servoes to a wandering, wave-group-scale reference — inflating
ride-height std (the DC job of the integrator is already done by the own-trim
calibration, so all it adds here is wave-band variance). Removing the integrator
and softening the proportional gain (p_tuned) leaves the loop tracking the genuine
disturbance only: it lands 13.3 mm tighter than pid_natural and 3.1 mm tighter than
the passive linkage, at a fraction of the flap effort. The mechanical linkage,
being a stiff proportional map with zero lag, tracks between the two PIDs but pays
for its stiffness in saturation (8.0 %) and added resistance (9.4 N) as it drives
the full wave-orbital wand motion into the flap.

**Why p_tuned rides slightly low.** With no integrator and the softest gain, the
wave-rectified DC bias is not nulled, so p_tuned's mean sits 2.3 cm below its
target (vs 1.7 cm for mechanical and 1.8 cm above for pid_natural). This is a
tracking-vs-DC-accuracy trade: the soft P law minimizes wave-band variance at the
cost of a small steady offset, still within the calibration tolerance and far from
any breach consequence at this setpoint.

**Why pid_deeper tracks and pitches best of the PIDs.** At depth factor ≈ 0.998 the
free-surface-lift model is saturated: heave forcing from surface proximity is
minimal, so the closed loop fights mostly the (weaker, at depth) orbital-velocity
AoA. Less disturbance in → less residual out. This is the same physics that makes
the deep foil expensive: 23 cm more strut in the water plus higher trim thrust =
+16.4 N calm.

**What the governor does and does not do.** It supplies the surge stiffness the
calibrated thrust table lacks (the table is a required-thrust curve — dF/du = 0
along the trim manifold, so u has no restoring force and drifts). It does **not**
shape the ~1 Hz wave-band dynamics (`surge_psd.png`). Its droop offsets (u low by
ΔT/Kp) are the newton-metered added resistance of each controller expressed in
m/s.

## Tuning suggestions

**Recommended starting points:**

1. **pid_deeper** — best breach margin (2.2×) and best tracking, when the extra
   +4.6 N mean added thrust (deeper strut) is acceptable.

```python
from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.moth_scenarios import create_pid_wand_config, apply_speed_governor

# NED: +0.30 makes pos_d less negative = boat rides lower = more tip margin
target = compute_tip_at_surface_pos_d() + 0.30   # = -1.469 + 0.30 = -1.169 m
sensor, estimator, controller = create_pid_wand_config(
    lqr, heel_angle=heel, target_pos_d=float(target),
)
# The factory calibrates theta_ref / flap / integrator state at the pinned
# trim for target_pos_d automatically; pair with the speed governor.
```

2. **p_tuned** — the best natural-setpoint controller: softest proportional gain,
   no integrator, lowest effort and essentially zero added resistance.

```python
sensor, estimator, controller = create_pid_wand_config(
    lqr, heel_angle=heel, Kp=0.4, Ki=0.0, Kd=0.0,
)
```

Directions worth exploring:

- **Setpoint sweep (margin vs drag):** the setpoint is a pure racing trade-off —
  each cm of extra depth costs strut drag (added resistance) and buys tip margin.
  Sweep `target_pos_d` between −1.40 and −1.17 to find the drag-optimal ride height
  under a given sea state.
- **Kp / Ki:** on this relative (wand) sensor the tracking-optimal answer is a
  *soft* proportional-only law. Stiffening or adding integral action costs flap
  saturation, added resistance, and (for Ki > 0) wave-band tracking variance. A
  Kp ≈ 0.3–0.5 plateau is a good default; the integrator is not needed because the
  own-trim calibration already zeroes the calm bias.
- **Derivative action:** low-pass the wand (~1 Hz cutoff is now *at* the encounter
  band; pick per sea state) or use an EKF for vertical velocity before enabling Kd.
- **Lighter seas** (`WAVE_SF_BAY_LIGHT`): at the natural setpoint the breach rate
  should fall once crest amplitude < 7 cm tip margin; the deeper setpoint's added
  resistance may then not pay for itself.

## Caveats

- Single operating point (10 m/s, 30° heel, head seas) and a single wave preset.
  Beam/quartering seas and shorter chop will change breach rates and possibly the
  natural-setpoint tracking order.
- The governor is a study-scoped "sailor holds boatspeed" model (P-law on u). Real
  sail thrust dynamics (gusts, sheeting lag) are not modeled; `--captive` runs a
  towing-tank-style fixed-speed diagnostic.
- Wave rectification (mean offsets of −1.8 to +2.3 cm under waves; sign and size
  differ by controller) is a real nonlinear-response effect, present for all
  controllers despite mm-exact calm calibration, and is reported, not calibrated
  away.
- All metric reference frames are as defined in § Metric definitions — in
  particular `ride_height_rms` (vs natural trim) is not a cross-setpoint metric.
- The dashboards' tip-depth panels and single-seed table use the still-water
  surface; the MC breach metric is wave-aware. Both are labeled in `metrics.json`;
  keep them separate.
