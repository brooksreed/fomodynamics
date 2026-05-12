# wand_vs_pid_waves — report (regenerated 2026-05-11)

## Flagged caveats (read first)

This run intentionally surfaces a finding that the plan did not
anticipate. **Two of the four physical-sanity orderings in the plan
are reversed** for the default trim point — the mechanical wand has
**fewer** breaches and **higher** flap activity than the PID, not
the other way around. The PID is functioning correctly (lowest RMS,
lowest flap saturation, lowest pitch RMS, lowest speed loss); the
inversion is mathematically consistent with the wand-angle forward
kinematics; the unit tests pass. The mechanism is explained in
§ Mechanism below.

If you intend to use this as a tuning starting point: **the trim
pos_d at 10 m/s (-1.40 m) sits only 7 cm above the foil-tip
ventilation depth (-1.47 m)**. SF Bay moderate waves have peak
excursion ~0.25 m, more than the available safety margin. The
mechanical wand sidesteps this through a 25 cm steady-state offset
(boat flies deeper than commanded); the PID eliminates that bias
and is consequently more exposed to ventilation events. A natural
follow-up is to re-run with `target_pos_d = surface_breach - 0.30 m`
to give the PID the same hidden margin the mechanical linkage gets
for free.

## Summary

- **PID has 2.8x lower ride-height RMS than mechanical wand**
  (50-seed mean 0.089 m vs 0.252 m). Single-seed gap is even larger
  (0.089 m vs 0.371 m at seed 0).
- **PID has 5.7x more leeward-tip breach events** (36.4 vs 6.3 per
  60s steady-state window, 50-seed mean). The cause is geometric,
  not controller failure: PID parks the boat exactly at trim
  (-1.40 m) while the mechanical wand drifts ~25 cm deeper
  (-1.19 m), giving the mechanical wand more headroom against the
  ventilation point at pos_d ~ -1.47 m.
- **PID has 33% less flap RMS and 14x less saturation**
  (0.097 vs 0.144 rad RMS; 1.2% vs 17.0% saturation fraction).
  The integrator does most of the work; the proportional term sees
  small errors. This contradicts the plan's expectation that PID
  would have *higher* flap activity.
- **PID has 36% less pitch RMS error and 2.8x less speed loss**
  (0.037 vs 0.059 rad; 1.42 vs 3.92 m/s). Pitch and speed are
  open-loop in the wand-only setup, so the difference is downstream
  of the better ride-height tracking (lower induced drag, more
  consistent attitude).
- **Both controllers stay foiling on average** (mean depth factor
  0.98 mechanical, 0.80 PID). The PID is closer to ventilation but
  still nets out submerged.

## Setup

Pulled verbatim from `metrics.json["setup"]`.

| Parameter | Value |
|---|---|
| Moth preset | `MOTH_BIEKER_V3` |
| Forward speed | 10.0 m/s |
| Heel angle | 30.0 deg |
| Timestep | 0.005 s |
| Duration | 60.0 s |
| Steady-state window | 10.0 s onwards |
| Monte Carlo seeds | 50 |
| Wave preset | `WAVE_SF_BAY_MODERATE` (Hs=0.5 m, Tp=3.0 s, JONSWAP gamma=4.0, Stokes 2nd order) |
| Wave direction | pi rad (head seas) |

Trim point (`design_moth_lqr(u_forward=10, heel=30deg)`):

| State | Value |
|---|---|
| pos_d | -1.400 m (boat 1.40 m above water; foil-tip-at-surface depth = -1.47 m, leaving 7 cm margin) |
| theta | +0.82 deg (nose-up) |
| w | +0.143 m/s |
| q | 0.0 deg/s |
| u | 10.000 m/s |

Trim control:

| Control | Value |
|---|---|
| Main flap | -0.140 deg |
| Rudder elevator | +0.139 deg |

Control bounds (informs saturation metric):

| Bound | Value |
|---|---|
| Flap min | -10.0 deg |
| Flap max | +15.0 deg |

Controllers (both in `fmd.simulator.moth_scenarios`):

- **Mechanical** (`create_mechanical_wand_config`): WandSensor +
  PassthroughEstimator + MechanicalWandController, default
  `WandLinkage` with `pullrod_offset=0.005` m.
- **PID** (`create_pid_wand_config`): WandSensor +
  PassthroughEstimator + PIDController with `Kp=0.6`, `Ki=0.1`,
  `Kd=0.0`. Closed-form wand-to-height inversion assumes trim
  attitude; a per-construction `wand_angle_offset` calibrates the
  inversion to reproduce `pos_d_target` at the trim wand angle.

## Single-seed time series (seed = 0)

### Per-controller dashboards

`plots/dashboard_mechanical.png` and `plots/dashboard_pid.png` each
show six panels: wave elevation (not populated in this report -
no aux dictionary is passed), pos_d, pitch, forward speed, control
effort, and leeward-tip depth. For seed = 0:

| Metric | Mechanical | PID |
|---|---|---|
| Ride-height RMS error | 0.371 m | **0.089 m** |
| Ride-height mean | -1.111 m | -1.406 m |
| Pitch RMS error | 0.057 rad (3.27 deg) | **0.038 rad** (2.18 deg) |
| Forward speed mean | 5.35 m/s | **8.55 m/s** |
| Mean tip depth | **0.320 m** | 0.047 m |
| Breach count (60 s window) | **5** | 37 |
| Breach fraction | **3.3%** | 32.1% |
| Flap std | 7.61 deg | **4.91 deg** |

Notes:

- The PID **tracks the trim point**: mean pos_d = -1.406 m
  approx trim pos_d. The mechanical wand sits at -1.111 m, i.e.
  nearly 30 cm **deeper** than commanded.
- That extra depth costs the mechanical wand on speed: it loses
  4.65 m/s (from 10 -> 5.35), versus PID's 1.45 m/s loss (10 ->
  8.55). The mechanical wand's deeper, more variable ride spends
  more time generating excess lift via flap motion (saturating)
  and more time grinding through hull and strut drag while pos_d
  wanders.
- Breach exposure is reversed: 5 events for mechanical, 37 for
  PID. The mechanical wand's mean tip depth (0.32 m below
  surface) is 7x the PID's (0.05 m). When waves push the boat
  ~0.25 m up from trim, the PID - which is **at** trim - easily
  surfaces the tip; the mechanical wand has a margin.

### Comparison overlays

- `plots/compare_ride_height.png` shows the two pos_d traces side
  by side over the full 60 s. The PID trace is a tight band
  around -1.40 m; the mechanical trace orbits -1.11 m with
  visibly larger amplitude.
- `plots/compare_flap_command.png` shows the flap commands. The
  PID is bounded inside roughly +-5 deg of trim with smooth
  trajectories. The mechanical wand alternates between long
  excursions to one bound and the other - its linkage geometry
  passes large wand-angle variation through nearly linearly.
- `plots/compare_wand_angle.png` shows the wand-angle
  measurements. Both controllers see roughly the same signal
  (they share `WandSensor` configuration), confirming the
  difference comes from the control law, not the sensor.

## Monte Carlo across 50 seeds

Each metric below is the cross-seed aggregate over the
10-to-60 s steady-state window. Numbers come from
`metrics.json["monte_carlo"]`; the box+strip plots in `plots/`
visualise the distributions.

| Metric (units) | Mechanical (mean +- std) | PID (mean +- std) | Plot |
|---|---|---|---|
| Ride-height RMS (m) | 0.252 +- 0.111 | **0.089 +- 0.005** | `mc_ride_height_rms.png` |
| Ride-height mean (m) | -1.194 +- 0.091 | **-1.406 +- 0.014** | (in metrics.json) |
| Mean main-foil depth factor | **0.982 +- 0.012** | 0.800 +- 0.034 | (in metrics.json) |
| Breach count (per 60 s ss-window) | **6.3 +- 2.5** (median 6) | 36.4 +- 4.6 (median 37) | `mc_breach_distribution.png` |
| Flap RMS (rad) | 0.144 +- 0.025 | **0.097 +- 0.008** | `mc_flap_activity.png` |
| Flap saturation fraction | 0.170 +- 0.101 | **0.012 +- 0.008** | (in metrics.json) |
| Pitch RMS error (rad) | 0.059 +- 0.027 | **0.037 +- 0.003** | `mc_pitch_speed.png` |
| Speed loss vs trim (m/s) | 3.92 +- 1.07 | **1.42 +- 0.53** | `mc_pitch_speed.png` |

Distribution shape:

- The PID's ride-height RMS distribution is **tight**: std/mean
  ~ 0.054, with min/max of 0.078/0.100 m across 50 seeds. The
  controller's response is essentially deterministic with respect
  to wave realisation.
- The mechanical wand's ride-height RMS has a wider tail (max
  0.540 m, std/mean ~ 0.44). Some seeds drive the system close to
  divergence - the same seeds also have the highest flap
  saturation (max 45.9% vs PID max 3.7%).
- The PID's breach count is **higher and more variable** in
  absolute terms (max 47 vs 13), but tighter in relative terms
  (std/mean ~ 0.13 vs 0.40).

## Mechanism

Why does the PID win on tracking and lose on breaches?

1. **The integrator removes steady bias.** The mechanical wand
   inherits its operating-point bias from the WandLinkage geometry
   (`pullrod_offset = 0.005` m, etc.). Even with the linkage
   tuned to zero-bias-at-trim in calm water, wave forcing
   introduces a mean lift deficit (asymmetric exposure of the
   foil to wave orbital motion) that the passive linkage cannot
   integrate away. The PID's `Ki` term explicitly accumulates
   the height error and biases the flap command to cancel that
   mean deficit. Result: PID `ride_height_mean` approx trim;
   mechanical `ride_height_mean` is 21 cm shallower.

2. **The proportional term suppresses high-frequency excursion.**
   The mechanical linkage is roughly linear across most of its
   range (the bellcrank geometry is L_v/L_p ~ 1 at the trim
   wand angle), so wave-induced wand motion passes through
   nearly 1:1 into flap motion. With no anti-windup or rate
   limit, the flap saturates 17% of the time (mean across
   seeds). The PID at Kp=0.6 maps the same wand variation to a
   smaller flap excursion, and with the operating point right
   at trim there is also less to correct in the first place.

3. **Trim margin is the silent variable.** The geometric
   advantage the mechanical wand has on breach count comes from
   its 21 cm steady offset, not from any active wave anticipation.
   In effect, the mechanical wand operates a **deeper trim**,
   which moves the foil tip further from the ventilation point.
   If the user dialled in a deeper `target_pos_d` for the PID
   (e.g. `target_pos_d = compute_tip_at_surface_pos_d() - 0.30`),
   the PID would match or beat the mechanical wand on breaches
   too, at the cost of more speed loss (deeper trim -> more
   drag).

4. **Speed loss and pitch follow ride-height quality.** Neither
   controller actuates the rudder elevator (both hold it at trim)
   or directly regulates `u`, so improvements in pitch and forward
   speed are downstream of improvements in pos_d tracking: less
   ride-height oscillation -> less flap reversal -> less induced
   drag -> less speed bleed -> less pitch perturbation.

5. **Why Kd = 0?** The wand-only signal carries the wave orbital
   motion directly - in `WAVE_SF_BAY_MODERATE` (Tp = 3 s, ~0.5 m
   peak crest-to-trough) the wand-angle measurement varies by
   tens of degrees over a wave period. A finite Kd amplifies that
   into large flap rates and the boat goes unstable (verified
   during tuning: `Kp=1.5, Ki=0.5, Kd=0.15` produces pitch RMS
   of ~0.5 rad - divergence). To use derivative action, low-pass
   the wand signal first or recover vertical velocity from an
   EKF (which is what `create_wand_only_config()` already does
   via LQG).

## Tuning suggestions

- **For shallower waves** (`WAVE_SF_BAY_LIGHT`, Hs ~ 0.3 m): the
  PID's safety margin is no longer the bottleneck. Expect the
  expected ordering to recover - PID beats mechanical on every
  metric, including breaches.
- **For deeper trim**: pass `target_pos_d =
  compute_tip_at_surface_pos_d() - 0.30` into `design_moth_lqr`.
  This gives both controllers an explicit 30 cm safety margin.
- **For more aggressive PID** (lower RMS): raise Kp toward ~1.0.
  Validate breach count stays bounded - at Kp=1.5 with Ki>0 the
  system destabilises on some seeds.
- **For a derivative term**: introduce a first-order low-pass on
  `pos_d_est` with cutoff ~1 Hz (well below 1/Tp ~ 0.33 Hz of
  the dominant wave forcing) before computing the derivative.
  Alternatively, swap `PassthroughEstimator` for an EKF that
  recovers w (vertical velocity) and feed that directly into a
  rate term - this becomes structurally similar to LQG.
- **For lower-bandwidth wand sensors** (e.g. coarse encoder
  resolution): the PID's effective gain margin is much smaller.
  Halve Kp and Ki as a starting point.

## Caveats

- Single trim point (10 m/s, 30 deg heel, head seas). Quartering or
  beam seas may produce different orderings - the wave-induced
  bias depends on wave direction.
- Single wave preset (SF Bay moderate). Short steep chop
  (`WAVE_SF_BAY_LIGHT`) is closer to the wand sensor's
  fixed-point convergence limit; longer-period swell would
  reduce the impact of wand-orbital coupling.
- PID inversion assumes `theta = 0`, `heel = 0`. Under large
  pitch or heel excursions the inverted pos_d will be biased
  by O(L_wand * (1 - cos theta)). At trim theta ~ 0.8 deg the bias
  is ~0.1 mm; at 5 deg pitch excursion it is ~4 mm - still negligible
  versus 0.25 m wave amplitude. For aggressive manoeuvres the
  assumption breaks; the mechanical wand has no such assumption.
- The wand sensor is wave-aware (fixed-point iteration in
  `wand_angle_from_state_waves`); the controller is **not**. A
  wave-aware PID (e.g. one that predicts wand-angle one period
  ahead and shapes the flap to phase-anticipate) is an obvious
  extension.
- All RMS / breach numbers are computed on the 10-to-60 s
  steady-state window - the initial transient is excluded.
- The metric `flap_rms` is in radians and includes the trim flap
  in its mean (the controller does not subtract `flap_trim` from
  the saved control trace), so it reads roughly as L2-norm of
  the flap command rather than of (u_flap - flap_trim). The
  ordering between controllers is still meaningful.
