# Model & test setup — wand_vs_pid_waves

A short description of what this study's simulation captures and what it
abstracts away, so the report numbers can be read with the right model in mind.
For the how-to-run and how-to-tune material see [`recipe.md`](recipe.md); for the
full model, see the deeper docs linked at the end.

## Vehicle dynamics: 3-DOF longitudinal, static heel

The Moth (`MOTH_BIEKER_V3`) is simulated in a reduced longitudinal set of
freedoms — heave (vertical position `pos_d`), pitch (`theta`), and surge
(forward speed `u`) — with roll held at a **static heel angle** (30° here). This
is the plane where the wand controls ride height, so it is the relevant subspace
for a wand-vs-flap-controller comparison; lateral/directional dynamics and dynamic
roll are out of scope. The state vector is `[pos_d, theta, w, q, u]` in the NED
frame (see [Frame conventions](../../frame_conventions.md)).

**NED sign convention (used throughout the report):** `pos_d` is positive *down*.
A foiling Moth flies with `pos_d` **negative** (CG about 1.4 m above the
still-water surface). A `pos_d` that becomes *more negative* means the boat is
**rising**.

## Foil forces and free-surface lift

Lift and drag come from the main foil, the rudder elevator, and the strut, plus
hull buoyancy when the hull touches down. The main foil's lift is modulated by a
**free-surface-lift factor** σ(h/c): as the foil approaches the surface its
effective lift falls off, and once part of the span emerges it ventilates. Two
depth-related quantities appear in the report:

- **depth_factor** — the effective submerged-span fraction (1.0 = fully
  submerged). It only moves once a meaningful fraction of the span is exposed.
- **leeward foil-tip depth** — the NED depth of the lowest (leeward) foil tip
  under heel. This is the earlier ventilation warning: the tip breaches the
  surface (depth < 0) well before the whole foil does. A **breach** in the report
  is a tip-depth zero-crossing into negative.

## Waves: spectrum, orbital kinematics, and encounter distance

Waves are an irregular sea built from the `WAVE_SF_BAY_MODERATE` preset — a
JONSWAP spectrum (Hs = 0.5 m, Tp = 3.0 s, γ = 4.0) with 2nd-order Stokes
kinematics, running as head seas (`mean_direction = π`). Each Monte Carlo seed is
a distinct wave realization; all controllers are driven by the **same** field per
seed, so comparisons are paired.

The waves affect the boat two ways: the **surface elevation** at the foil changes
its submersion (hence σ and the tip-depth breach metric), and the **orbital
velocity** field changes the local angle of attack. Because the encounter surface
the foil sees depends on *where along the wave* the foil currently is, the plant
integrates an **encounter-distance state** `x_n = ∫(u·cosθ + w·sinθ) dt` rather
than the naive `u·t` approximation, and the wave-aware breach metric queries the
surface at that integrated position. This is enabled explicitly
(`enable_encounter_distance = True`); the study fails loud if it is ever off.

## The wand: geometry and measurement model

A wand pivots at the bowsprit and trails down to the water surface; its angle is a
proxy for ride height. Boat **high** (pos_d more negative) → wand near vertical
(small angle) → less commanded lift; boat **low** (foil tip nearer the surface) →
wand more horizontal (larger angle) → more commanded lift. The wand angle is taken
in the **hull frame** (it rotates with the boat), so pitch enters the measurement.

Two ways to use the wand are compared:

- **Mechanical linkage** (`WandLinkage`): a pure trig/lever map from wand angle to
  flap command — fast, fixed-gain, no integrator. Its pullrod offset is auto-tuned
  in closed form so the trim point is the linkage's exact calm equilibrium.
- **Wand-only feedback** (`create_pid_wand_config`): the wand angle is inverted to
  a closed-form ride-height estimate under the trim-attitude assumption
  (`pos_d_est = −z_pivot·cos(heel) − L_wand·cos(wand_angle) + offset`), then a
  PID acts on the height error. The inversion assumes the boat sits at its trim
  pitch; under waves pitch oscillates, so the estimate carries a wave-rectified
  bias — the central mechanism behind the report's integrator finding.

Both are calibrated at the **pinned trim of their own ride-height setpoint**
(θ_ref, flap, elevator, thrust solved there), so the calm bias is mm-level at any
setpoint.

## Thrust: a P speed governor as a "sailor model"

Sail thrust is modeled as a proportional speed governor,
`F = max(T0 + Kp·(u_target − u), 0)`, standing in for a sailor holding boatspeed.
T0 is the pinned-trim thrust at each controller's setpoint.

Why not just use the calibrated thrust *table* T(u)? That table is a
**required-thrust curve** — the calm drag at each speed — so along the trim
manifold `dF/du = 0`: it supplies exactly the thrust to hold any speed but no
*restoring* force, so any persistent wave-added drag makes the boat slide down the
speed manifold with nothing to arrest it. The governor supplies the missing surge
stiffness, giving every configuration a genuine surge equilibrium. Its pole
(~0.05 Hz) sits two decades below the wave encounter band (~1 Hz), so it closes
the DC speed loop without shaping the wave-band ride-height dynamics the study
cares about (confirmed by `surge_psd.png`). A `--captive` mode fixes u instead
(surge disabled) for towing-tank-style checks.

## What is abstracted away

- No lateral/directional dynamics or dynamic roll (static heel only).
- No sail aerodynamics beyond the scalar governor (no gusts, sheeting lag, or
  apparent-wind coupling).
- No actuator dynamics beyond the flap/elevator command limits; sensor noise is a
  simple additive model.

## Deeper references (public docs)

- [`docs/simulator_models.md`](../../simulator_models.md) — state vectors, model
  presets, and the simulator infrastructure.
- [`docs/dev/moth_reference.md`](../../dev/moth_reference.md) — the Moth model,
  trim, and force components in detail.
- [`docs/frame_conventions.md`](../../frame_conventions.md) — NED/FRD frames and
  transforms.
- [`docs/physical_intuition_guide.md`](../../physical_intuition_guide.md) —
  interpreting foiling dynamics and force/trim behaviour.
