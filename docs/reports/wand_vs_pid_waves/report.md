# wand_vs_pid_waves — report (regenerated 2026-07-18)

**Vintage**: post physics-correctness batch (wave-orbital/eta sign fixes, hull-frame
wand angle, free-surface-lift σ(h/c), integrated encounter distance), under the P
speed governor, with each controller calibrated and initialized at its **own**
pinned trim. Provenance (from `metrics.json["provenance"]`): fmd commit `701893e`
(branch `wand_vs_pid_waves`, **editable install** — this is a pre-merge vintage;
it becomes a pinned vintage when the branch merges), params hash `a669b19e5c68`.
Artifacts regenerated at commit `2b1e157`.

The previous report (2026-05-13) was generated on physics with an inverted
wave-orbital forcing sign and no surge equilibrium; its numbers are superseded.
§ Reconciliation below states exactly which of its claims survived, which numbers
moved, and which orderings flipped.

## Flagged findings (read first)

**Breach count is set by the setpoint, not the control law.** Mechanical wand and
natural-trim PID — same −1.40 m target — now breach at statistically equal rates
(62.8 ± 2.5 vs 64.0 ± 2.9 per 50-s window, paired seeds). The pre-fix result that
made the mechanical wand look 3.4× safer than pid_natural (25.2 vs 85.9) was an
artifact: the mechanical wand rode ~21 cm low (≈90% wave rectification plus a
speed transient — halved by the physics fix, mostly eliminated by the speed
governor, with the last 1.6 cm DC offset removed by the pullrod auto-tune), and
the inverted orbital forcing pushed the boat down at crests — accidentally
protective exactly when the tip was exposed. Neither mechanism exists in the
corrected model.

**pid_natural no longer wins on tracking.** The original physics guidance expected
the PID to beat the mechanical wand on ride-height RMS; pre-fix it did (0.091 vs
0.252 m). Post-fix the ordering is **reversed**: mechanical 0.0926 m vs pid_natural
0.1029 m about the same setpoint. This is not a tuning bug: with disturbance and
wand feedback now phase-cooperating, the passive linkage's stiffer wand→flap gain
attenuates wave-band height error slightly better than the deliberately soft PID
(Kp=0.6), at the cost of ~15% more flap activity and 8% vs 1.9% flap saturation.
The integrator's DC advantage is intact but no longer decisive, because the
mechanical wand's steady-state bias is now −0.05 mm in calm water (auto-tuned
pullrod) and only +1.7 cm under waves (rectification).

**The pid_deeper "theta-shift" limitation is gone — the old flagged ordering
reversed.** Pre-fix, pid_deeper tracked its own setpoint *worse* than pid_natural
(0.189 vs 0.091 m) due to an ~14 cm bias, then explained as an inversion
calibrated at the wrong pitch. Re-measured on the governed plant, that cascade
(deeper → drag → slow → nose-up → bias) was a **speed** effect: at equal speed the
trim pitch is depth-invariant (0.824° at −1.169 m vs 0.836° at −1.400 m). With
per-setpoint thrust (91.9 N vs 75.5 N) and per-setpoint controller calibration,
pid_deeper now tracks **best of all three** (0.0792 m vs own setpoint) and its
wave-mean sits 0.8 cm high of target (calm bias −0.25 mm). The real cost of the
deeper setpoint is thrust, not tracking.

## Summary

- **pid_deeper still wins on safety, by 2.2× not 15–51×**: 28.4 breaches per 50-s
  window (50-seed mean) vs 62.8 (mechanical) and 64.0 (pid_natural). The pre-fix
  15×/51× separations were substantially physics-bug artifacts; 2.2× at equal
  speed is the honest margin bought by the extra 23 cm of ride-height margin.
- **pid_deeper also wins tracking and pitch**: RMS vs own setpoint 0.0792 m
  (vs 0.0926 mechanical, 0.1029 pid_natural); pitch RMS 0.0299 rad (vs ~0.0363
  for both natural-trim controllers). Riding deeper, the foil sits where lift is
  insensitive to surface proximity (depth factor 0.998), so wave forcing couples
  less into heave/pitch.
- **The three controllers now cost different amounts of thrust, not speed.** Under
  the governor (Kp=40 N/(m/s), u_target=10 m/s) all three hold speed (worst mean
  offset 0.23 m/s); the differentiator is mean added thrust to do so: pid_natural
  0.9 N, pid_deeper 4.6 N (mostly the deeper strut), mechanical 9.4 N (largest
  flap activity → largest wave-added resistance). Note the added-resistance and
  mean-u-offset columns are the same measurement (see § Metric definitions).
- **All 50 seeds stationary for every controller** (drift test on u and pos_d
  passes 150/150 runs; governor never saturates). The pre-fix study had no surge
  equilibrium for mechanical/pid_deeper — their "steady-state" statistics averaged
  a decaying-speed transient (u falling all 60 s, some pre-fix seeds through zero).
- **Recommendation unchanged in direction, honest in size**: pid_deeper for SF Bay
  moderate, now for the right reason — it buys its 2.2× breach margin at +16.4 N
  calm thrust and +3.7 N wave added resistance vs pid_natural, with the best
  tracking and pitch numbers, not despite a bias.

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
| Wave preset | `WAVE_SF_BAY_MODERATE` (Hs=0.5 m, Tp=3.0 s, JONSWAP γ=4.0, Stokes 2nd order) |
| Wave direction | π rad (head seas); encounter frequency ≈ 1.0 Hz at 10 m/s |
| Thrust law | speed governor `F = max(T0 + Kp·(u_target − u), 0)`, Kp = 40 N/(m/s) |
| Encounter distance | integrated plant state (`enable_encounter_distance=True`) |

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
  `pullrod_offset` **auto-tuned closed-form at the trim** (0.005508 m — the linkage
  outputs the trim flap at the trim wand angle, so the natural trim is its exact
  calm equilibrium; calm-water bias −0.05 mm). Target = natural trim −1.400 m.
- **pid_natural** (`create_pid_wand_config`): Kp=0.6, Ki=0.1, Kd=0.0, target =
  natural trim −1.400 m; inversion θ_ref = trim theta.
- **pid_deeper** (`create_pid_wand_config`, `target_pos_d = −1.169 m` =
  `compute_tip_at_surface_pos_d() + 0.30`; NED: +0.30 makes pos_d less negative →
  boat rides 30 cm lower → foil tip 30 cm below the ventilation threshold).
  Calibrated at its **own** pinned trim: θ_ref = 0.824°, flap = −0.064°, elevator
  = +0.061°, T0 = 91.88 N.

### Metric definitions and reference frames (read before comparing columns)

- `ride_height_rms` — RMS of pos_d error **vs the natural trim** (−1.3998 m) for
  *all* controllers. For pid_deeper this is dominated by its deliberate 23 cm
  setpoint offset; never use it cross-setpoint.
- `ride_height_rms_around_target` — RMS vs the controller's **own setpoint**. The
  correct cross-setpoint tracking comparison.
- `pitch_rms_error`, `speed_loss_mean` — referenced to the **natural trim**
  (θ = 0.836°, u = 10 m/s) for all controllers, deliberately, for
  cross-controller comparability.
- `flap_rms` — RMS of flap deviation from the controller's **own trim flap**
  (pid_deeper: −0.064°, not the natural −0.132°). This reference changed in this
  vintage; pid_deeper's `flap_rms` is **not comparable** to the 2026-05 report
  (~0.07° reference shift).
- `breach_count` / `breach_fraction` (Monte Carlo) — **wave-aware**: leeward tip
  depth measured against the instantaneous wave surface at the foil's integrated
  encounter position, over the 10–60 s window. Counts are breach *onsets*
  (crossings into tip-above-surface); fraction is time-above-surface.
- The **single-seed dashboard table** tip-depth stats are still-water-referenced
  (`wave_aware: false` in `metrics.json["single_seed"]`) — do not mix them with
  the MC breach columns (seed 0 mechanical: 47 still-water vs 62 wave-aware).
- `added_resistance_mean` = mean(F_sail − T0). With the governor unsaturated
  (0% of steps, all seeds) this is **algebraically Kp·mean_u_offset** — it is the
  same measurement as the speed columns expressed in newtons, **not** an
  independent cross-check. It is still the physically meaningful "mean added
  thrust to hold 10 m/s in these waves".
- `depth_factor_mean` — time-average of the force model's depth factor
  (effective submerged-span fraction; 1.0 = fully submerged). The free-surface-
  lift model change redefined this curve, so pre/post-fix depth factors are
  **not comparable**.

## Single-seed time series (seed = 0)

### Per-controller dashboards

`dashboard_mechanical.png`, `dashboard_pid_natural.png`, `dashboard_pid_deeper.png`
— six panels each: wave elevation at the foils, pos_d, pitch, forward speed, flap
command, leeward-tip depth (still-water reference in this table):

| Metric (seed 0, 10–60 s) | mechanical | pid_natural | pid_deeper |
|---|---|---|---|
| pos_d mean (m) | −1.383 | −1.418 | −1.177 |
| pos_d target (m) | −1.400 | −1.400 | −1.169 |
| pos_d min / max (m) | −1.638 / −1.152 | −1.699 / −1.174 | −1.392 / −0.989 |
| Tip depth mean (m, still-water) | 0.076 | 0.039 | 0.282 |
| Tip depth min (m, still-water) | −0.219 | −0.268 | **+0.032** |
| Speed mean ± std (m/s) | 9.77 ± 0.07 | 10.00 ± 0.09 | 9.89 ± 0.06 |
| Pitch mean ± std (deg) | 0.82 ± 2.11 | 0.96 ± 2.09 | 0.79 ± 1.73 |
| Flap mean ± std (deg) | 0.62 ± 6.08 | 1.28 ± 4.95 | 0.29 ± 4.71 |
| Flap rate RMS (deg/s) | 154 | 110 | 153 |

NED-sign cross-checks:

- **mechanical**: pos_d mean −1.383 m vs target −1.400 m → 1.7 cm *less* negative
  → the boat rides ~1.7 cm **lower** than target under waves (wave rectification;
  its calm-water equilibrium is exact by construction). Compare the pre-fix
  −1.111 m (29 cm low for seed 0). The forward-speed panel shows u settling to
  ≈9.77 m/s — the governor holds it there; pre-fix u decayed to ~5.4 m/s with no
  equilibrium. ✓
- **pid_natural**: pos_d mean −1.418 m vs target −1.400 m → 1.8 cm *more*
  negative → rides ~1.8 cm **higher** than target under waves (rectification acts
  upward on this controller — sign differs from mechanical). Tip depth mean is
  only 0.039 m, so wave excursions (±0.25 m) put the tip above the surface
  roughly once per encounter. ✓
- **pid_deeper**: pos_d mean −1.177 m vs target −1.169 m → 0.8 cm high of target.
  The pre-fix vintage sat at −0.993 m, 17.6 cm *below* target — that bias is gone.
  Tip depth **never crosses the still-water surface for seed 0** (min +0.032 m);
  its wave-aware MC count (~25–30) comes from wave crests reaching up to the tip,
  not from the tip reaching the mean surface. ✓

All three pitch panels oscillate ±2° about ≈0.8°; pre-fix, mechanical and
pid_deeper carried ~3° mean pitch — that was the low-speed (high-CL) signature of
the missing surge equilibrium, not a property of the controllers.

### Comparison overlays

`compare_ride_height.png`: mechanical and pid_natural now overlay each other
around −1.40 m with similar amplitude (pre-fix, mechanical sat visibly low with
larger amplitude); pid_deeper holds a parallel band around −1.17 m. All three
respond at the ~1 Hz encounter frequency; no controller shows drift.

`compare_flap_command.png`: all three traverse a similar ±10–15° envelope at the
encounter frequency. Mechanical clips the +15° limit most often (8.0% saturation
vs 1.9% pid_natural, 0.9% pid_deeper, 50-seed means). The pre-fix picture — PID
dramatically calmer than mechanical — has compressed to a ~15% RMS difference.

`compare_wand_angle.png`: mechanical and pid_natural see nearly identical wand
angles (same setpoint, same sensor); **pid_deeper's wand rides ~20° larger**
(≈45–50° vs ≈25–30°) because the boat flies lower, the wand trailing more
horizontal. (The 2026-05 report's claim that all three wand signals are "nearly
identical" was true only while pid_deeper failed to reach its setpoint.)

## Monte Carlo across 50 seeds

Source: `metrics.json["monte_carlo"]` (mean ± std over 50 paired seeds,
10–60 s window).

| Metric | mechanical | pid_natural | pid_deeper |
|---|---|---|---|
| target_pos_d (m) | −1.400 | −1.400 | −1.169 |
| ride_height_mean (m) | −1.3824 ± 0.0011 | −1.4131 ± 0.0050 | −1.1766 ± 0.0011 |
| rms vs own setpoint (m) | 0.0926 ± 0.0021 | 0.1029 ± 0.0037 | **0.0792 ± 0.0014** |
| rms vs natural trim (m) | 0.0926 | 0.1029 | 0.2367 (setpoint-offset dominated) |
| breach_count (wave-aware, /50 s) | 62.8 ± 2.5 | 64.0 ± 2.9 | **28.4 ± 3.1** |
| breach_fraction (time tip exposed) | 0.375 | 0.429 | **0.093** |
| flap_rms (rad, vs own trim) | 0.1064 ± 0.0014 | 0.0922 ± 0.0025 | 0.0827 ± 0.0009 |
| flap_saturation_fraction | 0.080 ± 0.007 | 0.019 ± 0.005 | **0.009 ± 0.003** |
| pitch_rms_error (rad) | 0.0363 ± 0.0007 | 0.0362 ± 0.0012 | **0.0299 ± 0.0005** |
| mean_u_offset (m/s) | 0.234 ± 0.005 | **0.022 ± 0.014** | 0.114 ± 0.003 |
| added_resistance_mean (N) (= Kp·row above) | 9.36 ± 0.22 | **0.87 ± 0.55** | 4.55 ± 0.11 |
| governor_saturation_fraction | 0.0 | 0.0 | 0.0 |
| stationarity pass fraction | 1.00 | 1.00 | 1.00 |
| depth_factor_mean | 0.859 ± 0.009 | 0.757 ± 0.024 | **0.9975 ± 0.0000** |

Per-plot notes:

- `mc_ride_height_rms_around_target.png` (RMS vs own setpoint — the correct
  cross-setpoint comparison): three cleanly separated boxes, **pid_deeper lowest**
  (0.0792), mechanical middle (0.0926), pid_natural highest (0.1029). Both
  orderings involving a PID flipped vs the 2026-05 vintage (see Reconciliation).
  Note the spreads: per-seed std is ~2% of the mean — the post-fix wave response
  is an entrained, regular oscillation, so 50 seeds give very tight CIs.
- `mc_ride_height_rms.png` (RMS vs natural trim): pid_deeper reads 0.2367 m
  because its setpoint is 23.1 cm from natural trim — the column measures the
  offset, not tracking. Kept for continuity; do not use cross-setpoint.
- `mc_breach_distribution.png`: mechanical (62.8 ± 2.5) and pid_natural
  (64.0 ± 2.9) boxes overlap almost completely; pid_deeper (28.4 ± 3.1) is far
  below both. Contrast pre-fix medians 24.5 / 85.5 / 2. The safety story is now a
  **setpoint** story: at u = 10 m/s with encounter frequency ≈ 1 Hz, the two
  natural-trim controllers breach on essentially every encounter (≈1.26/s over
  50 s ≈ 1.2 per encounter); the deeper setpoint drops that to ≈0.55/encounter.
- `mc_flap_activity.png`: mechanical highest (0.1064 rad), pid_natural middle
  (0.0922), pid_deeper lowest (0.0827, own-trim reference). The passive linkage
  passes wave-orbital wand motion straight through; the PIDs' Kp=0.6 attenuates
  it. pid_deeper's lower activity is physical, not just the reference shift: at
  depth factor ≈1 the plant's lift is insensitive to surface proximity, so the
  wand sees smoother effective forcing.
- `mc_pitch_speed.png`: pitch — mechanical and pid_natural indistinguishable
  (0.0363 vs 0.0362 rad), pid_deeper clearly better (0.0299). Speed loss —
  mechanical worst (0.234 m/s), pid_deeper 0.114, pid_natural 0.022; these are
  the governor's steady P-droop offsets, i.e. added resistance ÷ Kp, and all are
  <2.5% of u_target (pre-fix losses were 1.5–4.6 m/s with no equilibrium).
- `surge_psd.png` (seed 0): all three spectra peak at the ≈1 Hz encounter band
  with harmonics; the governor pole (~0.05 Hz, dashed) sits two decades below the
  wave peak (1/Tp, dotted). The clean separation confirms the governor closes the
  DC surge loop without shaping the wave-band response — the wave-band metrics
  above are governor-invariant (verified by a Kp = 25–75 sweep during tuning).

## Reconciliation vs the 2026-05-13 vintage and the review's predictions

Baseline: the pre-fix 50-seed `metrics.json`
(`git show 701893e:docs/reports/wand_vs_pid_waves/metrics.json`). Physics deltas
between vintages: wave-orbital AoA sign, eta-depth sign, hull-frame wand angle,
free-surface lift σ(h/c), recalibrated thrust table, integrated encounter
distance, plus the study-side speed governor and per-setpoint trim calibration.
Seeds are paired (same wave realizations).

**Predictions made by the physics review (before the rerun) — all three held:**

| Prediction | Post-fix 50-seed verdict |
|---|---|
| Integral action removes the mechanical droop; PID rows survive by mechanism | HELD, then mooted: pid_natural parks 1.3 cm from target; the mechanical droop itself is now −0.05 mm calm (auto-tuned pullrod) + 1.7 cm rectification |
| Breach ordering pid_deeper < mechanical < pid_natural survives | HELD in every vintage (28.4 < 62.8 < 64.0) |
| pid_natural's setpoint is unsafe at Hs = 0.5 m | HELD: ~1.2 breaches per wave encounter — and equally true of *any* controller at the −1.40 m setpoint |

**Numbers that moved (same claim, different size):**

| Quantity | pre-fix | post-fix | Why |
|---|---|---|---|
| pid_deeper breach margin | 15× / 51× | **2.2×** | Two accidental protections removed: inverted orbital forcing pushed the boat down at crests; and (for pre-fix mech/deeper) u-decay parked them deeper than intended |
| mechanical ride bias | 21 cm low | 1.7 cm low | ~90% of the pre-fix droop was wave rectification under bugged forcing (halved by the fix); DC part removed by pullrod auto-tune |
| speed loss | 1.5–4.6 m/s | 0.02–0.23 m/s | Pre-fix runs had no surge equilibrium (thrust table gives zero surge stiffness); governor holds u by construction |
| mech-vs-PID flap-activity gap | 1.6× RMS, 17% vs 1.2% sat | 1.15× RMS, 8.0% vs 1.9% sat | Opposing-phase forcing exaggerated the passive linkage's wildness |
| per-seed scatter | RMS spread 0.12–0.54 m | ±2% of mean | Cooperating-phase response is entrained; pre-fix opposing phase drove intermittent near-instability |

**Orderings that flipped:**

| Ordering | pre-fix | post-fix |
|---|---|---|
| Tracking (RMS vs own setpoint) | natural 0.091 < deeper 0.189 < mech 0.252 | **deeper 0.079 < mech 0.093 < natural 0.103** |
| Breaches, mechanical vs pid_natural | mech 25.2 ≪ natural 85.9 | statistical tie (62.8 vs 64.0) |
| Pitch RMS | natural best (0.038) | deeper best (0.030); mech = natural |
| Wand-angle signals | "nearly identical" all three | pid_deeper distinct (~20° larger) |

**Claims from the 2026-05 report now known to be wrong (not merely stale):**

1. *"The mechanism is geometric: the mechanical wand drifts 21 cm lower... giving
   it 21 cm more safety margin"* — the 21 cm was ~90% wave rectification under
   sign-inverted forcing plus a speed transient, not linkage geometry; the
   pre-fix breach advantage it "bought" was an artifact of the same bugs.
2. *"The theta-shift bias is a known limitation... ~8 cm calm, ~14 cm under
   waves"* — it was a symptom of operating 4.6 m/s off-design with fixed thrust.
   At the governed equal-speed point the pitch shift is 0.012° and the calm bias
   is −0.25 mm. The old report's own Option-D suggestion (re-solve trim at the
   setpoint) was the fix, and is now the default behavior of the factories.
3. *"pid_deeper... at the cost of slightly higher rms_vs_target"* — reversed;
   pid_deeper has the **lowest** rms vs its own setpoint.
4. The pre-fix single-seed table (u ≈ 5.2–5.4 m/s for mechanical/pid_deeper,
   pitch ~3°, flap means 5–6°) described trajectories with no surge equilibrium;
   every derived "steady-state" number in it averaged a decaying transient.

## Mechanism

**Why the deeper setpoint still wins on breaches — and why the margin is 2.2×.**
The ventilation threshold is pos_d = −1.469 m. The natural setpoint leaves 7 cm
of tip margin; wave crests locally raise the surface by 0.25–0.4 m, so at
−1.40 m the tip is exposed roughly once per encounter regardless of controller.
The deeper setpoint puts the tip 30 cm under; only the larger crests reach it
(breach fraction 9.3% of the window vs 38–43%). The pre-fix 15–51× margins
required the bugged physics to actively press the deep-riding boats down at
crests; corrected physics gives the honest geometric answer.

**Why mechanical ≈ pid_natural on breaches now.** Both hold the same mean height
to within 3 cm of each other (−1.382 vs −1.413) at the same speed. Breach onset
at this setpoint is driven by the wave field, not by the ±1 cm differences in
tracking. The two controllers' breach counts differ by less than half a std.

**Why mechanical now beats pid_natural on wave-band tracking.** The linkage's
effective height-to-flap gain (set by geometry) is stiffer than Kp=0.6, and it
acts with zero controller lag; with forcing and feedback in phase it shaves
~10% off the height RMS. The price shows up exactly where the physics guidance
says it should: highest flap RMS (0.106 rad), highest saturation (8%), and the
highest added resistance (9.4 N — flap motion is drag). pid_natural buys a
calmer flap and 0.9 N added resistance with slightly looser height tracking.
Neither difference approaches the breach-count significance.

**Why pid_deeper tracks best.** At depth factor ≈ 0.998 the free-surface lift
model is saturated (σ ≈ 1): heave forcing from surface proximity is minimal, so
the closed loop fights mostly orbital-velocity AoA, which is weaker at depth.
Less disturbance in → less residual out (pitch RMS 0.0299 rad, flap RMS 0.0827,
saturation 0.9%). This is the same physics that makes the deep foil expensive:
23 cm more strut in the water plus higher trim thrust = +16.4 N calm.

**What the governor does and does not do.** It closes the surge DC loop (pole
~0.05 Hz) that the calibrated thrust table leaves open (the table is a
required-thrust curve, not a control law — as the dynamic law it has zero surge
stiffness, which is why pre-fix runs decayed). It does **not** shape the ~1 Hz
wave-band dynamics (`surge_psd.png`; Kp-sweep flat). Its droop offsets (u low by
ΔT/Kp: 0.23 / 0.02 / 0.11 m/s) are the newton-metered added resistance of each
controller expressed in m/s.

## Tuning suggestions

**Recommended starting point**: pid_deeper — best breach margin, best tracking,
best pitch, at +4.6 N mean added thrust vs +0.9 N for pid_natural.

```python
from fmd.simulator.components.moth_forces import compute_tip_at_surface_pos_d
from fmd.simulator.moth_scenarios import create_pid_wand_config

# NED: +0.30 makes pos_d less negative = boat rides lower = more tip margin
target = compute_tip_at_surface_pos_d() + 0.30   # = -1.469 + 0.30 = -1.169 m
sensor, estimator, controller = create_pid_wand_config(
    lqr, heel_angle=heel, target_pos_d=float(target),
)
# The factory calibrates theta_ref / flap / integrator state at the pinned
# trim for target_pos_d automatically; pair it with the speed governor
# (apply_speed_governor) so the deeper point has its thrust equilibrium.
```

Directions worth exploring:

- **Setpoint sweep (margin vs drag)**: with the theta-shift artifact gone, the
  setpoint is a pure racing tradeoff — each cm of extra depth costs strut drag
  (added resistance) and buys tip margin. The drag-optimal ride height under
  waves is now a well-posed question; sweep target_pos_d between −1.40 and −1.17.
- **Kp/Ki on the honest plant**: the pre-fix conclusion "PID needs to be soft"
  was tuned against exaggerated disturbance. With the corrected phase, a stiffer
  Kp may close the ~10% tracking gap to the mechanical wand before hitting the
  saturation/added-resistance penalty the linkage pays.
- **Derivative action**: unchanged advice — low-pass the wand (~1 Hz cutoff is
  now *at* the encounter band; pick per sea state) or use an EKF for w before
  enabling Kd.
- **Lighter seas** (`WAVE_SF_BAY_LIGHT`): at the natural setpoint the breach rate
  should fall dramatically once crest amplitude < 7 cm tip margin; the deeper
  setpoint's added resistance may then not pay for itself.

## Caveats

- Single operating point (10 m/s, 30° heel, head seas) and a single wave preset.
  Beam/quartering seas and shorter chop will change breach rates and possibly
  the mech-vs-natural tie.
- **Pre-merge editable vintage**: generated from the `wand_vs_pid_waves` branch
  (editable install, commit `701893e`); numbers become the pinned reference only
  after the branch merges.
- The governor is a study-scoped "sailor holds boatspeed" model (P-law on u).
  Real sail thrust dynamics (gusts, sheeting lag) are not modeled; captive mode
  (`--captive`) exists for towing-tank-style checks.
- Wave rectification (mean offsets of −1.3 to +1.7 cm under waves despite
  mm-exact calm calibration; sign differs by controller) is a real
  nonlinear-response effect, present for all controllers, and is reported, not
  calibrated away.
- All metric reference frames are as defined in § Metric definitions — in
  particular pid_deeper's `flap_rms` is not comparable to the 2026-05 vintage,
  and `ride_height_rms` (vs natural trim) is not a cross-setpoint metric.
- `settling_time` reads 60 s for all controllers by construction (pos_d never
  stays within 1 cm of target in a seaway); it is not a meaningful column here.
- The dashboards' tip-depth panels and single-seed table use the still-water
  surface; the MC breach metric is wave-aware. Both are labeled in
  `metrics.json`; keep them separate.
