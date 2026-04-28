# Moth 3DOF Equations of Motion

This document is the living reference for the Moth 3DOF longitudinal dynamics model implemented in BLUR. It covers coordinate frames, state/control definitions, equations of motion, force components, the ventilation model, and trim/equilibrium conditions.

When this document and implementation disagree, treat implementation as source of truth and update this file.

## 0. Model Snapshot and Scope

**Audited against code snapshot:** commit `1ee36cc` (foil force decomposition fix + measured geometry)
**Date:** 2026-03-13
**Intended scope:** Longitudinal Moth3D model (`pos_d, theta, w, q, u`) and trim workflow.

### 0.1 Source Hierarchy

1. `src/fmd/simulator/moth_3d.py` (state derivative assembly, kinematics, mass/inertia handling)
2. `src/fmd/simulator/components/moth_forces.py` (component-level force and moment models)
3. `src/fmd/simulator/params/moth.py` and `src/fmd/simulator/params/presets.py` (parameter definitions/default values)
4. `src/fmd/simulator/trim.py` (trim objective/constraints/solver details)

### 0.2 What Changed Recently (Important)

- Hull model now includes **two-point buoyancy** (nonzero `F_z` and `M_y`), not drag-only.
- `pos_d_dot` now includes **CG-velocity terms** from `sailor_position_schedule`.
- Composite pitch inertia is computed **per-timestep** from sailor position schedule.
- Sail force supports **lookup-table thrust vs speed** (with affine fallback).
- Strut drag currently uses a **fixed configured strut depth** (not dynamic immersion).
- Foil force decomposition now separates **alpha_geo** (geometric AoA for force rotation) from **alpha_eff** (effective AoA including control surface deflection for polar lookup). The full lift+drag rotation matrix is applied using alpha_geo.

**Source files:**
- `src/fmd/simulator/moth_3d.py` -- `Moth3D` dynamics class
- `src/fmd/simulator/components/moth_forces.py` -- Force components (`MothMainFoil`, `MothRudderElevator`, `MothSailForce`, `MothHullDrag`, `MothStrutDrag`, `compute_depth_factor`)
- `src/fmd/simulator/params/moth.py` -- `MothParams` parameter class
- `src/fmd/simulator/params/presets.py` -- `MOTH_BIEKER_V3` preset
- `src/fmd/simulator/trim.py` -- trim solver objective/constraints

---

## 1. Coordinate Frames and Sign Conventions

**Body frame:** FRD (Forward-Right-Down)
- $+x$: forward
- $+y$: starboard (right)
- $+z$: down

**Positive pitch** ($\theta > 0$): nose up (rotation about $+y$ axis, right-hand rule produces nose-down, so positive $\theta$ = nose up follows the convention that pitch-up is positive in aerospace usage).

**Positive pitch moment** ($M_y > 0$): nose-up torque.

**Moment from offset force:**
$$\mathbf{M} = \mathbf{r} \times \mathbf{F}$$

The pitch component is:
$$M_y = r_z \cdot F_x - r_x \cdot F_z$$

where $\mathbf{r} = [r_x, r_y, r_z]$ is the position vector from CG to the force application point, and $\mathbf{F} = [F_x, F_y, F_z]$ is the force in body frame.

**Lever arm convention:**
- `position_x` positive forward (foil forward of CG has `position_x > 0`)
- `position_z` positive down (foil below CG has `position_z > 0`)

**Depth convention:**
- `pos_d` is the CG depth in NED frame (positive down, so `pos_d > 0` means CG is below the waterline)
- `foil_depth = pos_d + position_z * cos(heel) * cos(theta) - position_x * sin(theta)` is the heel- and pitch-corrected depth of the foil center below the waterline. This is the z-row of the rotation matrix $R = R_y(\theta) R_x(\text{heel})$ applied to the body-frame point. The `cos(heel) * cos(theta)` factor on `position_z` accounts for both heel and pitch reducing the vertical projection of the body z-axis. The `-position_x * sin(theta)` term accounts for pitch rotating foils vertically: nose-up pitch makes the bow shallower and the stern deeper. See `compute_foil_ned_depth()` in `moth_forces.py` for the canonical implementation.

**Sign verification examples (see `MothMainFoil.compute_moth()`):**
- Forward foil (`position_x > 0`) producing upward lift ($F_z < 0$): $M_y = r_z \cdot F_x - r_x \cdot F_z$. The $-r_x \cdot F_z$ term is $-(+)(-)= +$, giving a nose-up moment. Physically correct: lifting force forward of CG pitches nose up.
- Aft rudder (`position_x < 0`) producing upward lift ($F_z < 0$): $-r_x \cdot F_z = -(-)(-) = -$, giving a nose-down moment. Physically correct: lifting force aft of CG pitches nose down.
- Sail CE above CG (`ce_position_z < 0`) with forward thrust ($F_x > 0$): $M_y = r_z \cdot F_x = (-)(+) = -$, giving a nose-down moment. Physically correct: forward force above CG pitches nose down.

---

## 2. State and Control Vectors

### 2.1 State Vector

$$\mathbf{x} = \begin{bmatrix} \text{pos\_d} \\ \theta \\ w \\ q \\ u \end{bmatrix}$$

| Index | Symbol | Name | Units | Convention |
|-------|--------|------|-------|------------|
| 0 | $\text{pos\_d}$ | Vertical position (heave) | m | Positive down in NED |
| 1 | $\theta$ | Pitch angle | rad | Positive nose-up |
| 2 | $w$ | Body-frame vertical velocity | m/s | Positive down |
| 3 | $q$ | Body-frame pitch rate | rad/s | Positive nose-up about $+y$ |
| 4 | $u$ | Body-frame surge velocity | m/s | Positive forward |

Default state: `[pos_d=-1.3, theta=0, w=0, q=0, u=10.0]` (normal foiling trim, hull clear of water, level, zero velocities, 10 m/s forward speed).

**Surge dynamics flag (`surge_enabled`):** When `surge_enabled=True` (**default**), $u$ is a dynamic state with $\dot{u} = F_{x,\text{total}} / m_{\text{eff,surge}} - q \cdot w$ (see Section 3.2). When `surge_enabled=False`, $\dot{u} = 0$ and forward speed comes from the exogenous `u_forward_schedule(t)`.

### 2.2 Control Vector

$$\mathbf{u} = \begin{bmatrix} \delta_{\text{flap}} \\ \delta_{\text{elev}} \end{bmatrix}$$

| Index | Symbol | Name | Units | Bounds |
|-------|--------|------|-------|--------|
| 0 | $\delta_{\text{flap}}$ | Main foil flap deflection | rad | $[-10°, +15°]$ = $[-0.1745, +0.2618]$ |
| 1 | $\delta_{\text{elev}}$ | Rudder elevator deflection | rad | $[-3°, +6°]$ = $[-0.0524, +0.1047]$ |

Bounds are defined in `moth_3d.py` as `MAIN_FLAP_MIN`, `MAIN_FLAP_MAX`, `RUDDER_ELEVATOR_MIN`, `RUDDER_ELEVATOR_MAX`.

### 2.3 Exogenous Input

- $u_{\text{fwd}}(t)$: Forward speed (m/s), treated as a time-varying exogenous input. Default: constant 10.0 m/s (~20 kt).
- Set via `u_forward` argument to `Moth3D.__init__()`. When not provided, defaults to `lambda t: 10.0`.
- When `surge_enabled=True`, the forward speed comes from state $u$ instead of the schedule.
- A minimum speed clamp $u_{\text{safe}} = \max(u_{\text{fwd}}, 0.1)$ prevents division by zero in AoA calculations.

### 2.4 Symbol Table (Canonical)

| Symbol | Code meaning | Notes |
|-------|--------------|-------|
| $u_{\text{fwd}}$ | Forward speed used this step | Schedule value when `surge_enabled=False`, state `u` when `surge_enabled=True` |
| $u_{\text{safe}}$ | Clamped forward speed | $u_{\text{safe}}=\max(u_{\text{fwd}},0.1)$ for AoA divisions |
| $\mathbf{r}_{\text{cg}}$ | CG offset from hull CG due to sailor | Computed from current `sailor_position_schedule(t)` |
| $\dot{\mathbf{r}}_{\text{cg}}$ | CG velocity due to sailor schedule | Finite-difference estimate in `Moth3D._compute_step_terms()` |
| $I_{yy,\text{composite}}(t)$ | Composite pitch inertia about system CG | Recomputed each timestep from sailor position |

---

## 3. Equations of Motion

The state derivative $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, t)$ is computed in `Moth3D.forward_dynamics()`.

### 3.1 Kinematics

$$\dot{\text{pos\_d}} = -u_{\text{fwd}} \sin\theta + w \cos\theta - \dot{r}_{\text{cg},x}\sin\theta + \dot{r}_{\text{cg},z}\cos\theta$$

$$\dot{\theta} = q$$

The first equation projects body-frame velocities into the NED down-axis and includes CG-motion terms from a time-varying sailor position schedule. If the sailor schedule is static, $\dot{\mathbf{r}}_{\text{cg}}=\mathbf{0}$ and the equation reduces to the simpler form:

$$\dot{\text{pos\_d}} = -u_{\text{fwd}} \sin\theta + w \cos\theta$$

### 3.2 Dynamics

$$\dot{w} = \frac{F_{z,\text{total}}}{m_{\text{eff,heave}}} + q \cdot u_{\text{fwd}}$$

$$\dot{q} = \frac{M_{y,\text{total}}}{I_{\text{eff}}}$$

$$\dot{u} = \begin{cases} \frac{F_{x,\text{total}}}{m_{\text{eff,surge}}} - q \cdot w & \text{if surge\_enabled} \\ 0 & \text{otherwise} \end{cases}$$

where the effective masses and inertia include added mass effects:

$$m_{\text{eff,heave}} = m_{\text{total}} + m_{\text{added,heave}}$$

$$m_{\text{eff,surge}} = m_{\text{total}} + m_{\text{added,surge}}$$

$$I_{\text{eff}} = I_{yy,\text{composite}}(t) + I_{\text{added,pitch}}$$

The $q \cdot u_{\text{fwd}}$ term in $\dot{w}$ and $-q \cdot w$ term in $\dot{u}$ are Coriolis coupling terms from the body-frame equations of motion.

The composite pitch inertia $I_{yy,\text{composite}}(t)$ accounts for both the hull inertia and the sailor mass distribution about the system CG using the reduced-mass parallel axis theorem:

$$I_{yy,\text{composite}} = I_{yy,\text{hull}} + \mu (x_s^2 + z_s^2)$$

where $\mu = m_{\text{hull}} \cdot m_{\text{sailor}} / m_{\text{total}}$ is the reduced mass and $(x_s, z_s)$ is the sailor position relative to the hull CG.

Default values: $m_{\text{added,heave}} = 10$ kg, $I_{\text{added,pitch}} = 8.75$ kg·m², nominal $I_{yy,\text{composite}} \approx 14.24$ kg·m² at default sailor position. See `damping_mechanisms_research.md` for added mass derivation.

The total forces and moments are:

$$F_{z,\text{total}} = F_{z,\text{foil}} + F_{z,\text{rudder}} + F_{z,\text{hull}} + F_{z,\text{struts}} + m_{\text{total}} \cdot g \cdot \cos\theta$$

$$F_{x,\text{total}} = F_{x,\text{foil}} + F_{x,\text{rudder}} + F_{x,\text{sail}} + F_{x,\text{hull}} + F_{x,\text{struts}} - m_{\text{total}} \cdot g \cdot \sin\theta$$

$$M_{y,\text{total}} = M_{y,\text{foil}} + M_{y,\text{rudder}} + M_{y,\text{sail}} + M_{y,\text{hull}} + M_{y,\text{struts}}$$

Key observations from the code:
- The sail contributes to both $F_{x,\text{total}}$ and $F_{z,\text{total}}$ through NED→body rotation: $F_x = F_{\text{sail}} \cos\theta$, $F_z = F_{\text{sail}} \sin\theta$.
- Hull **drag** itself is purely $F_x$, but hull component includes buoyancy, so $F_{z,\text{hull}}$ and $M_{y,\text{hull}}$ are generally nonzero when immersed.
- Gravity contributes both $F_z = mg\cos\theta$ and $F_x = -mg\sin\theta$ (the x-component enters the surge dynamics when `surge_enabled=True`).
- Strut drag contributes to $F_{x,\text{total}}$ (see Section 4.5).

### 3.3 Post-Step Processing

After each integration step, $\theta$ is wrapped to $[-\pi, \pi]$ via:

$$\theta \leftarrow \text{atan2}(\sin\theta, \cos\theta)$$

This is implemented in `Moth3D.post_step()` for numerical stability.

---

## 4. Force Components

All force components are Equinox modules implementing `compute_moth(state, control, u_forward, t)` and returning `(force_b, moment_b)` as 3D vectors in body frame.

### 4.0 Contribution Matrix (Current Implementation)

| Component | $F_x$ | $F_z$ | $M_y$ |
|----------|-------|-------|-------|
| Main foil | Yes | Yes | Yes |
| Rudder elevator | Yes | Yes | Yes |
| Sail | Yes | No | Yes |
| Hull component (`MothHullDrag`) | Yes (drag + buoyancy projection) | Yes (buoyancy) | Yes (two-point buoyancy) |
| Main strut drag | Yes | No | Yes (via strut centroid z-offset) |
| Rudder strut drag | Yes | No | Yes (via strut centroid z-offset) |
| Gravity | Yes | Yes | No |

**Force rotation (main foil + rudder):** Both lift and drag are fully rotated by the geometric angle of attack $\alpha_{\text{geo}}$ (the flow angle, not including control surface deflections):

$$F_x = -D\cos\alpha_{\text{geo}} + L\sin\alpha_{\text{geo}}, \qquad F_z = -D\sin\alpha_{\text{geo}} - L\cos\alpha_{\text{geo}}$$

Note the sign convention: positive $\alpha_{\text{geo}}$ (flow from below) tilts the lift vector forward ($+L\sin\alpha_{\text{geo}}$ on $F_x$), which is a thrust-producing effect. Drag is resolved into both body axes via $\cos/\sin$ of $\alpha_{\text{geo}}$.

### 4.1 Main Foil (`MothMainFoil`)

**Angle of attack decomposition:**

The local vertical flow velocity at the foil, including pitch rate coupling and wave orbital velocity:

$$w_{\text{local}} = w - q \cdot x_{\text{foil}} + w_{\text{orbital}}$$

Two distinct angles are computed:

1. **Geometric AoA** (for force rotation into body axes):
$$\alpha_{\text{geo}} = \arctan\!\left(\frac{w_{\text{local}}}{u_{\text{safe}}}\right)$$

2. **Effective AoA** (for lift/drag polar, includes control deflection):
$$\alpha_{\text{eff}} = \eta \cdot \delta_{\text{flap}} + \frac{w_{\text{local}}}{u_{\text{safe}}}$$

where $\eta$ is `flap_effectiveness` (default 0.5), $u_{\text{safe}} = \max(u_{\text{fwd}}, 0.1)$, and $x_{\text{foil}}$ is the foil x-position relative to CG (positive forward).

The physical distinction: $\alpha_{\text{geo}}$ represents the actual flow direction (arctan of vertical/horizontal flow), which determines how lift and drag project into body axes. $\alpha_{\text{eff}}$ represents the angle the foil "sees" including its control surface deflection, which determines the magnitude of lift and drag through the polar. Control surface deflections change lift magnitude but do not change the incoming flow direction.

The term $-q \cdot x_{\text{foil}} / u$ captures pitch-rate induced AoA change. For the forward main foil ($x > 0$), positive pitch rate (nose-up) reduces the effective AoA, providing pitch damping.

**Lift and drag coefficients:**

$$C_L = (C_{L_0} + C_{L_\alpha} \cdot \alpha_{\text{eff}}) \cdot f_{\text{depth}}$$

$$C_D = C_{D_0} + \frac{C_L^2}{\pi \cdot AR \cdot e} + C_{D,\text{flap}} \cdot \delta_{\text{flap}}^2$$

where $f_{\text{depth}}$ is the ventilation/depth factor (Section 5), $AR$ is the aspect ratio ($\text{span}^2 / S$), $e$ is the Oswald efficiency factor, and $C_{D,\text{flap}}$ is the flap deflection drag coefficient (default 0.15, based on McCormick formula for cf/c~0.25 reduced for hydrofoil).

**Dynamic pressure:**

$$q_{\text{dyn}} = \frac{1}{2} \rho \, u_{\text{fwd}}^2$$

Note: uses `u_forward` as received by the component (already clamped by the caller).

**Lift and drag forces:**

$$L = q_{\text{dyn}} \cdot S \cdot C_L$$

$$D = q_{\text{dyn}} \cdot S \cdot C_D$$

**Body-frame force decomposition (using $\alpha_{\text{geo}}$):**

$$F_x = -D \cos\alpha_{\text{geo}} + L \sin\alpha_{\text{geo}}$$

$$F_y = 0$$

$$F_z = -D \sin\alpha_{\text{geo}} - L \cos\alpha_{\text{geo}}$$

Lift and drag are both fully rotated by $\alpha_{\text{geo}}$. Positive $\alpha_{\text{geo}}$ (flow from below, typical during foiling) tilts the lift vector forward, reducing net surge drag. The $+L\sin\alpha_{\text{geo}}$ term on $F_x$ is the "lift-forward-tilt" thrust contribution. Drag projects into both body axes: mostly opposing surge ($-D\cos\alpha_{\text{geo}}$) with a small downward component ($-D\sin\alpha_{\text{geo}}$).

**Pitching moment (cross product $\mathbf{r} \times \mathbf{F}$):**

$$M_x = 0$$

$$M_y = r_z \cdot F_x - r_x \cdot F_z$$

$$M_z = 0$$

where $r_x$ = `position_x` (positive forward) and $r_z$ = `position_z` (positive down).

**Default parameter values** (from `MOTH_BIEKER_V3`):

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| `rho_water` | $\rho$ | 1025.0 | kg/m$^3$ |
| `main_foil_area` | $S$ | 0.08455 | m$^2$ |
| `main_foil_cl_alpha` | $C_{L_\alpha}$ | 5.7 | 1/rad |
| `main_foil_cl0` | $C_{L_0}$ | 0.15 | -- |
| `main_foil_cd0` | $C_{D_0}$ | 0.006 | -- |
| `main_foil_oswald` | $e$ | 0.85 | -- |
| `main_foil_aspect_ratio` | $AR$ | 10.7 | -- |
| `main_foil_flap_effectiveness` | $\eta$ | 0.5 | -- |
| `main_foil_cd_flap` | $C_{D,\text{flap}}$ | 0.15 | -- |
| `main_foil_position` | $[r_x, r_y, r_z]$ | $[0.42, 0, 1.85]$ | m |
| `main_foil_span` | -- | 0.95 | m |

### 4.2 Rudder Elevator (`MothRudderElevator`)

**Angle of attack decomposition (same structure as main foil):**

$$w_{\text{local}} = w - q \cdot r_x + w_{\text{orbital}}$$

$$\alpha_{\text{geo}} = \arctan\!\left(\frac{w_{\text{local}}}{u_{\text{safe}}}\right)$$

$$\alpha_{\text{eff}} = \delta_{\text{elev}} + \frac{w_{\text{local}}}{u_{\text{safe}}}$$

The pitch rate coupling term: for the rudder ($r_x < 0$, aft of CG), $w_{\text{local}} = (w - q \cdot r_x) = (w + q \cdot |r_x|)$. Positive pitch rate ($q > 0$, nose up) causes the tail to move downward, increasing the local flow angle and providing pitch damping.

**Lift coefficient with depth factor:**

$$C_{L,\text{rudder}} = C_{L_\alpha,\text{rudder}} \cdot \alpha_{\text{eff}} \cdot f_{\text{depth}}$$

Note: no $C_{L_0}$ for the rudder (unlike the main foil).

**Drag coefficient (full parabolic polar):**

$$C_{D,\text{rudder}} = C_{D_0,\text{rudder}} + \frac{C_{L,\text{rudder}}^2}{\pi \cdot AR_{\text{rudder}} \cdot e_{\text{rudder}}}$$

where $C_{D_0,\text{rudder}}$ is the rudder zero-lift drag, $AR_{\text{rudder}} = \text{span}^2 / S_{\text{rudder}}$, and $e_{\text{rudder}}$ is the rudder Oswald efficiency factor.

**Lift and drag forces:**

$$L_{\text{rudder}} = q_{\text{dyn}} \cdot S_{\text{rudder}} \cdot C_{L,\text{rudder}}$$

$$D_{\text{rudder}} = q_{\text{dyn}} \cdot S_{\text{rudder}} \cdot C_{D,\text{rudder}}$$

**Body-frame forces (using $\alpha_{\text{geo}}$):**

$$F_x = -D_{\text{rudder}} \cos\alpha_{\text{geo}} + L_{\text{rudder}} \sin\alpha_{\text{geo}}$$

$$F_z = -D_{\text{rudder}} \sin\alpha_{\text{geo}} - L_{\text{rudder}} \cos\alpha_{\text{geo}}$$

**Pitching moment (full cross product):**

$$M_y = r_z \cdot F_x - r_x \cdot F_z$$

**Sign verification:** Rudder at $r_x < 0$ (aft) with upward force ($F_z < 0$): $M_y = -r_x \cdot F_z = -(-)(-) = -(+) < 0$, producing a nose-down moment. This is physically correct.

**Default parameter values** (from `MOTH_BIEKER_V3`):

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| `rudder_area` | $S_{\text{rudder}}$ | 0.051 | m$^2$ |
| `rudder_cl_alpha` | $C_{L_\alpha,\text{rudder}}$ | 5.0 | 1/rad |
| `rudder_position` | $[r_x, r_y, r_z]$ | $[-1.865, 0, 1.77]$ | m |
| `rudder_span` | -- | 0.68 | m |

### 4.3 Sail Force (`MothSailForce`)

Forward thrust applied in the **NED horizontal plane**, then rotated to body frame using pitch angle $\theta$. The sail force direction is set by the wind, not the hull pitch attitude.

**Thrust model priority:**
1. **Lookup table** (if `sail_thrust_speeds` and `sail_thrust_values` are non-empty): $F_{\text{sail}} = \text{interp}(u, \text{speeds}, \text{values})$ with flat extrapolation outside the table range.
2. **Affine model** (if `thrust_slope != 0` and no table): $F_{\text{sail}} = T_{\text{coeff}} + T_{\text{slope}} \cdot u$
3. **Constant** (fallback): $F_{\text{sail}} = T_{\text{coeff}}$

**NED to body-frame rotation:**

$$F_{x,\text{body}} = F_{\text{sail}} \cos\theta, \quad F_{y,\text{body}} = 0, \quad F_{z,\text{body}} = F_{\text{sail}} \sin\theta$$

**Pitching moment (cross-term):**

$$M_y = r_{z,\text{CE}} \cdot F_{x,\text{body}} - r_{x,\text{CE}} \cdot F_{z,\text{body}}$$

Expanding:

$$M_y = r_{z,\text{CE}} \cdot F_{\text{sail}} \cos\theta - r_{x,\text{CE}} \cdot F_{\text{sail}} \sin\theta$$

The full cross product $\mathbf{r} \times \mathbf{F}$ is required because the NED-to-body rotation produces both $F_x$ and $F_z$ components. The `ce_position_x` (forward of CG) and `ce_position_z` (above CG, negative in FRD) together determine the moment. For the current Moth geometry, the $r_z \cdot F_x$ term dominates (forward force above CG gives nose-down moment), while the $r_x \cdot F_z$ cross-term provides a secondary correction.

**Default parameter values** (from `MOTH_BIEKER_V3`):

| Parameter | Value | Unit |
|-----------|-------|------|
| `sail_thrust_coeff` (fallback constant) | 107.6 | N |
| `sail_thrust_speeds` | (6.0, 7.0, ..., 20.0) | m/s |
| `sail_thrust_values` | (235.4, 154.8, ..., 267.9) | N |
| `sail_ce_position` | $[0.20, 0, -2.13]$ | m |

### 4.4 Hull Contact Drag + Buoyancy (`MothHullDrag`)

**Immersion model:**

$$\text{immersion} = \max(0, \; \text{pos\_d} + d_{\text{contact}})$$

where $d_{\text{contact}}$ = `contact_depth` (default 0.94 m, the system CG-to-hull-bottom distance in body frame, computed dynamically as `hull_cg_above_bottom - cg_offset[2]`). The hull bottom is at NED depth `pos_d + contact_depth`. Hull drag is active when this is positive (hull bottom below the water surface), i.e., when `pos_d > -contact_depth`.

**Hull-contact drag force:**

$$F_x = -k_{\text{drag}} \cdot \text{immersion}$$

This drag term opposes forward motion.

**Two-point buoyancy model:**

Two buoyancy points (forward/aft) each get half the buoyancy coefficient:

$$F_{\text{buoy,pt}} = \frac{k_{\text{buoy}}}{2} \cdot \max(0, d_{\text{pt}})$$

Each point's upward world-frame buoyancy is decomposed into body-frame forces:

$$F_{x,\text{pt}} = F_{\text{buoy,pt}} \sin\theta, \qquad F_{z,\text{pt}} = -F_{\text{buoy,pt}} \cos\theta$$

and pitch moment:

$$M_{y,\text{pt}} = r_z F_{x,\text{pt}} - r_x F_{z,\text{pt}}$$

Summed hull component output:

$$F_x = -k_{\text{drag}}\cdot \text{immersion} + \sum_{\text{pt}} F_{x,\text{pt}}, \quad
F_z = \sum_{\text{pt}} F_{z,\text{pt}}, \quad
M_y = \sum_{\text{pt}} M_{y,\text{pt}}$$

**Default parameter values** (from `MOTH_BIEKER_V3`):

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| `hull_drag_coeff` | 500.0 | N/m | |
| `hull_contact_depth` | 0.94 | m | Computed: `hull_cg_above_bottom - combined_cg_offset[2]` |
| `hull_buoyancy_coeff` | 5000.0 | N/m | |

`hull_contact_depth` is a computed `@property` derived from the hull geometry
and sailor position. It represents the distance from the system CG to the hull
bottom in the body z-axis. See `docs/moth_vertical_geometry.md` for the full
vertical layout.

### 4.5 Strut Drag (`MothStrutDrag`)

Hydrofoil struts (vertical members connecting the hull to the horizontal foils) produce drag from both pressure and skin friction forces. Two instances exist: main foil strut and rudder strut.

**Pressure drag (based on frontal area):**

$$D_{\text{pressure}} = \frac{1}{2} \rho \, u^2 \cdot C_{D,\text{pressure}} \cdot (t_{\text{strut}} \cdot d_{\text{strut}})$$

where $t_{\text{strut}}$ is the strut thickness, $d_{\text{strut}}$ is the configured strut depth parameter, and $C_{D,\text{pressure}}$ is the pressure drag coefficient.

**Skin friction drag (based on wetted area):**

$$D_{\text{skin}} = \frac{1}{2} \rho \, u^2 \cdot C_f \cdot (2 \cdot c_{\text{strut}} \cdot d_{\text{strut}})$$

where $c_{\text{strut}}$ is the strut chord, and the factor of 2 accounts for both sides of the strut being wetted.

> Implementation note: current `MothStrutDrag` uses fixed `strut_depth` from parameters (approximated from foil z-position), not dynamic immersed depth.

**Total strut drag:**

$$F_x = -(D_{\text{pressure}} + D_{\text{skin}})$$

$$F_y = 0, \quad F_z = 0$$

**Default parameter values** (from `MOTH_BIEKER_V3`):

| Parameter | Main Strut | Rudder Strut | Unit |
|-----------|-----------|-------------|------|
| Chord | 0.09 | 0.07 | m |
| Thickness | 0.013 | 0.010 | m |
| $C_{D,\text{pressure}}$ | 0.01 | 0.01 | -- |
| $C_f$ | 0.003 | 0.003 | -- |

Estimated total strut drag at 10 m/s remains order-of-magnitude tens of newtons for default preset.

### 4.6 Gravity

**Body-frame forces:**

$$F_{z,\text{gravity}} = m_{\text{total}} \cdot g \cdot \cos\theta$$

$$F_{x,\text{gravity}} = -m_{\text{total}} \cdot g \cdot \sin\theta$$

At level flight ($\theta = 0$), gravity contributes $+m g$ in the body $z$-direction (downward, as expected in FRD). At positive pitch ($\theta > 0$, nose up), the component along body $z$ decreases by $\cos\theta$.

The $x$-component ($-m g \sin\theta$) enters $F_{x,\text{total}}$ and affects surge dynamics when `surge_enabled=True`. At typical foiling pitch angles ($\theta < 3°$), this is small (~5 N) compared to total drag (~100+ N).

Gravity acts through the system CG, so it produces no pitching moment ($M_{y,\text{gravity}} = 0$). Component force models use effective positions relative to system CG; in current implementation this CG offset is recomputed each timestep from `sailor_position_schedule(t)`.

**Default values** (from `MOTH_BIEKER_V3`):

| Parameter | Value | Unit |
|-----------|-------|------|
| `total_mass` ($m_{\text{total}}$) | 92.0 | kg |
| `g` | 9.80665 | m/s$^2$ |
| `composite_pitch_inertia` ($I_{yy}$) | 14.24 | kg$\cdot$m$^2$ |

---

## 5. Ventilation / Depth Factor Model

The depth factor $f_{\text{depth}} \in [0, 1]$ models lift reduction as a foil approaches or breaches the water surface. It is computed by `compute_depth_factor()` in `moth_forces.py`.

### 5.1 Foil Depth

$$d_{\text{foil}} = \text{pos\_d} + r_z \cos(\phi) \cos(\theta) - r_x \sin\theta$$

where $r_z$ = `position_z` is the foil's z-offset below the CG, $r_x$ = `position_x` is the foil's x-offset forward of the CG, and $\phi$ is the static heel angle. This is the z-row of the rotation matrix $R = R_y(\theta) R_x(\phi)$ applied to the body-frame point $[r_x, 0, r_z]$. The $\cos(\phi)\cos(\theta)$ factor accounts for both heel and pitch reducing the vertical projection of the body z-axis. The $-r_x \sin\theta$ term is the **pitch-corrected depth**: when the body pitches, foils at different longitudinal positions move vertically. See `compute_foil_ned_depth()` in `moth_forces.py` for the canonical implementation.

**Physical interpretation:**
- Nose-up ($\theta > 0$): the bow rises and the stern drops
  - Forward foil ($r_x > 0$): correction $= -r_x \sin\theta < 0$ → shallower
  - Aft rudder ($r_x < 0$): correction $= -r_x \sin\theta > 0$ → deeper
- At $\theta = 0$: $\sin(0) = 0$, correction vanishes (backward-compatible)

**Numerical significance** (at `MOTH_BIEKER_V3` defaults):
- At $\theta = 7.5°$ ($0.131$ rad): rudder correction $= +1.5 \cdot \sin(0.131) \approx +0.20$ m, main foil correction $= -0.6 \cdot \sin(0.131) \approx -0.08$ m
- This correction resolves the rudder early-ventilation artifact identified in Wave 4C

### 5.2 Heel Geometry

When the boat is heeled at a static heel angle $\phi \geq 0$ (windward heel), the windward tip of the horizontal T-foil rises:

$$d_{\text{tip}} = d_{\text{foil}} - \frac{S}{2} \sin\phi$$

where $S$ is the foil span. The maximum submergence depth (foil depth at which the tip just reaches the surface) is:

$$d_{\text{max}} = \max\!\left(\frac{S}{2} \sin\phi, \; d_{\text{min}}\right)$$

where $d_{\text{min}} = 0.015$ m is a differentiability floor. At zero heel, $\sin(0) = 0$ and the floor provides a ~3 cm transition width. At typical foiling heel ($\phi = 30°$), the floor has no effect since $(S/2)\sin(30°) \gg d_{\text{min}}$.

The raw exposed fraction of the span above water is:

$$f_{\text{exposed,raw}} = \frac{d_{\text{max}} - d_{\text{foil}}}{d_{\text{max}}}$$

This fraction is:
- 0 when $d_{\text{foil}} = d_{\text{max}}$ (tip just at surface)
- 1 when $d_{\text{foil}} = 0$ (foil center at surface, entire windward half-span exposed)
- $>1$ when $d_{\text{foil}} < 0$ (foil center above water)

### 5.3 Smooth Mode (Default)

The smooth mode provides a differentiable depth factor suitable for gradient-based optimization and control. It has no hard cutoffs or discontinuities.

**Step 1: Smooth saturation of exposed fraction to $[0, 1)$:**

$$f_{\text{exposed,pos}} = \frac{\text{softplus}(k_{\text{sat}} \cdot f_{\text{exposed,raw}})}{k_{\text{sat}}} \quad (k_{\text{sat}} = 50)$$

$$f_{\text{exposed}} = 1 - e^{-f_{\text{exposed,pos}}}$$

This maps $f_{\text{exposed,raw}}$ to a smooth $[0, 1)$ range without hard clamping.

**Step 2: Ventilation taper via normalized threshold:**

$$f_{\text{norm}} = \frac{f_{\text{exposed}}}{\max(f_{\text{threshold}}, \epsilon)}$$

$$f_{\text{heel}} = \frac{1}{2}\left(1 - \tanh\left((f_{\text{norm}} - 0.5) \cdot 6.0\right)\right)$$

where $f_{\text{threshold}}$ is the `ventilation_threshold` parameter (default 0.30). The depth factor transitions from ~1 to ~0 as the exposed fraction passes through the threshold.

The result is returned directly as $f_{\text{depth}}$. There is no separate depth-only model or blend — the heel geometry model is the sole physics model at all heel angles.

### 5.4 Binary Mode

Hard cutoff at foil centerline depth:

$$f_{\text{depth}} = \begin{cases} 1 & \text{if } d_{\text{foil}} > 0 \\ 0 & \text{if } d_{\text{foil}} \leq 0 \end{cases}$$

This is for demonstration only and is not suitable for gradient-based optimization or control.

### 5.5 Edge Cases

- **Heel angle = 0:** At zero heel, a horizontal T-foil is flat relative to the water surface — the entire span breaches simultaneously. There is no progressive tip exposure, so the ventilation transition is physically a step function. The model handles this by flooring $d_{\text{max}}$ at $d_{\text{min}} = 0.015$ m, which produces a near-binary transition over ~3 cm. This preserves differentiability for gradient-based methods while correctly representing the sharp physics. The system has significantly less open-loop stability at zero heel.
- **Foil above water** ($d_{\text{foil}} < 0$): In smooth mode, $f_{\text{depth}} \to 0$ smoothly. In binary mode, $f_{\text{depth}} = 0$ exactly.
- **Ventilation non-recovery (not modeled):** In reality, once a foil breaches the surface and air enters the low-pressure side, the flow separates and lift drops catastrophically. The foil typically does not recover lift when re-submerged — the air cavity persists until the boat slows or the foil is fully flushed. The current model treats ventilation as reversible: if the foil re-submerges, $f_{\text{depth}}$ returns to ~1 and full lift is restored. This means the model allows "skipping" trajectories (repeated breach-and-recovery) that appear stable but would crash in reality. A trajectory where any foil's depth factor reaches ~0 should be considered a **soft instability** — the boat has likely ventilated and will not recover without active intervention.

### 5.6 Nominal Heel Angle

The `Moth3D` model defaults to a **30 deg** static heel angle (`heel_angle = np.deg2rad(30.0)`). This represents the typical windward heel of a Moth in foiling flight, where the sailor hikes out to balance the sail force.

**Rationale:**
- At 30 deg heel, the ventilation model correctly uses foil span to determine the transition region between full lift and ventilated lift.
- The windward T-foil tip rises by $(S/2) \sin(30°) = S/4$ above the foil center, creating a physically meaningful ventilation geometry.
- At zero heel (upright), a horizontal T-foil is either entirely in or entirely out of the water — the span does not matter. The min_submergence floor (Section 5.5) provides a near-binary transition.

**Configurable:** The heel angle can be set to any value via `Moth3D(params, heel_angle=...)`. The model works correctly at all heel angles from 0 to 90 deg.

**Foil ventilation ordering at 30 deg:**
- Main foil ($S = 1.0$ m, $r_z = 0.60$ m): tip breach at $\text{pos\_d} \approx -0.35$ m
- Rudder ($S = 0.7$ m, $r_z = 0.50$ m): tip breach at $\text{pos\_d} \approx -0.325$ m
- The main foil starts to ventilate just before the rudder, matching physical expectations for a Moth (the main foil's larger span causes its windward tip to breach first).

### 5.7 Application to Components

- **Main foil:** Depth factor multiplies the lift coefficient: $C_L = (C_{L_0} + C_{L_\alpha} \cdot \alpha_{\text{eff}}) \cdot f_{\text{depth}}$. The drag coefficient uses the factored $C_L$ in its induced drag term, so drag also reduces with ventilation.
- **Rudder:** Same depth factor model applied to rudder lift coefficient: $C_{L,\text{rudder}} = C_{L_\alpha} \cdot \alpha_{\text{eff}} \cdot f_{\text{depth}}$.
- Both foils use their own `position_x` and `position_z` for computing `foil_depth` (with pitch correction), and their own `foil_span` for the ventilation geometry.

### 5.8 Geometric Corrections Audit Checklist

The table below captures the current status of key geometry-dependent computations in `moth_forces.py`. It is intentionally line-number free to reduce maintenance churn.

| # | Computation | Location | Status | Notes |
|---|------------|----------|--------|-------|
| 1 | **Foil depth** | `MothMainFoil`, `MothRudderElevator` | **Correct** | Uses canonical `compute_foil_ned_depth()` with $-r_x \sin\theta$ term |
| 2 | **Moment arms** ($M_y = r_z F_x - r_x F_z$) | Main foil + rudder | **Correct** | Body-frame cross product is exact at all angles |
| 3 | **AoA pitch-rate coupling** ($w - q \cdot r_x$) | Main foil + rudder | **Correct** | Uses effective x-position (including CG offset) |
| 4 | **Sail CE moment** ($r_{z,\text{CE}} \cdot T$) | `MothSailForce` | **Correct for current model** | CE position fixed in body frame; thrust may come from lookup table |
| 5 | **Hull contact + buoyancy geometry** | `MothHullDrag` | **Correct for current simplification** | Uses contact-depth immersion + two-point buoyancy; detailed hull-shape immersion is deferred |

---

## 6. Trim / Equilibrium

**Definition:** Trim is the steady-state condition where all state derivatives are zero:

$$\dot{\mathbf{x}} = \mathbf{0} \implies \dot{\text{pos\_d}} = 0, \quad \dot{\theta} = 0, \quad \dot{w} = 0, \quad \dot{q} = 0, \quad \dot{u} = 0$$

### 6.1 Kinematic Constraint (Standard Static-Sailor Case)

From $\dot{\text{pos\_d}} = -u \sin\theta + w \cos\theta = 0$:

$$w = u \cdot \frac{\sin\theta}{\cos\theta} = u \cdot \tan\theta$$

From $\dot{\theta} = q = 0$:

$$q = 0$$

These are kinematic constraints that must hold at any trim point.

For trim workflows in `trim.py`, sailor position is typically treated as static over time, so $\dot{\mathbf{r}}_{\text{cg}}=\mathbf{0}$ and the above constraint remains valid. If a time-varying sailor schedule is used during trim evaluation, the full `pos_d_dot` expression in Section 3.1 should be used.

### 6.2 Dynamic Equilibrium

From $\dot{w} = 0$: total z-force must balance:

$$F_{z,\text{foil}} + F_{z,\text{rudder}} + F_{z,\text{hull}} + F_{z,\text{struts}} + m g \cos\theta = 0$$

From $\dot{q} = 0$: total pitch moment must balance:

$$M_{y,\text{foil}} + M_{y,\text{rudder}} + M_{y,\text{sail}} + M_{y,\text{hull}} + M_{y,\text{struts}} = 0$$

From $\dot{u} = 0$ (when `surge_enabled=True`): total x-force must balance:

$$F_{x,\text{foil}} + F_{x,\text{rudder}} + F_{x,\text{sail}} + F_{x,\text{hull}} + F_{x,\text{struts}} - m g \sin\theta = 0$$

This x-force balance (thrust = drag) is the key constraint for sail thrust calibration. With `surge_enabled=False`, $\dot{u} = 0$ is enforced trivially (speed is exogenous) and the x-force balance is NOT part of the optimization residual. This was the root cause of incorrect thrust calibrations in Phase 1.

### 6.3 Free and Fixed Variables

**Operating mode** (`optimize_thrust=False`):

| Variable | Role | Bounds |
|----------|------|--------|
| $\text{pos\_d}$ | Free | $[-0.6, 0.5]$ m |
| $\theta$ | Free | $[-0.3, 0.3]$ rad ($\approx \pm 17°$) |
| $u$ | Free (surge) | $[u_{\text{target}} \pm 0.5]$ m/s |
| $\delta_{\text{flap}}$ | Free | $[-0.1745, 0.2618]$ rad |
| $\delta_{\text{elev}}$ | Free | $[-0.0524, 0.1047]$ rad |
| $q$ | Fixed | 0 |
| $w$ | Derived | $u \cdot \tan\theta$ |

**Calibration mode** (`calibrate_thrust=True`): adds thrust as a 6th decision variable with bounds $[10, 500]$ N.

### 6.4 Solution Method

Trim is found by minimizing a scale-aware multi-term objective that combines the state derivative residuals (the primary physics target) with regularization terms that select among multiple possible equilibria.

See [`docs/trim_solver.md`](trim_solver.md) for full solver details: objective function, weights, continuation strategy, optimizer configuration, calibrated thrust curves, and known limitations (including multi-solution risk and continuation sweep artifacts).

**Key solver properties relevant to interpreting results:**

- The kinematic constraints ($q = 0$, $w = u \tan\theta$) are hard constraints enforced before every objective evaluation.
- Regularization terms prefer small deflections, which can bias the solver away from physically correct solutions at low speeds. See the sign chain reference in [`docs/physical_intuition_guide.md`](physical_intuition_guide.md) § "Control input → physical effect" for diagnosing suspicious trim results.

### 6.5 Open-Loop Stability

The system has a positive real eigenvalue at all speeds (~+0.33 to +0.58 rad/s), with instability increasing with speed. Time constants of 1.7-3.0 s are physically reasonable for an open-loop unstable foiling boat. See `tests/simulator/moth/test_damping.py` for current reference eigenvalues.

---

## 7. Damping Mechanisms

The model includes two primary physical damping mechanisms that improve open-loop stability:

### 7.1 Pitch Rate Coupling (Mq Derivative)

When the boat pitches at rate $q$, foils at distance $x$ from the CG experience an additional vertical velocity component, changing their effective angle of attack:

$$\alpha_{\text{induced}} = -\frac{q \cdot x}{u}$$

For the forward main foil ($x > 0$), positive pitch rate reduces AoA. For the aft rudder ($x < 0$), positive pitch rate increases AoA. Both effects produce restoring moments that damp pitch oscillations.

The pitch damping derivative $M_q$ is approximately:

$$M_q = -\frac{1}{2}\rho u \sum_i (S_i \cdot C_{L_\alpha,i} \cdot x_i^2)$$

This is always stabilizing (negative) regardless of foil position sign due to the $x^2$ term.

### 7.2 Added Mass Effects

Water acceleration around the foils creates added mass and inertia that slow the dynamic response:

| Parameter | Default | Description |
|-----------|---------|-------------|
| $m_{\text{added,heave}}$ | 10 kg | Added mass for heave motion |
| $I_{\text{added,pitch}}$ | 8.75 kg·m² | Added pitch inertia |

These values are approximately 70% of thin airfoil theory predictions, accounting for 3D effects and free surface proximity. See `damping_mechanisms_research.md` for detailed derivation.

### 7.3 Effect on Stability

With these damping mechanisms, the system is nearly marginally stable at trim:
- **Baseline (no damping):** Eigenvalue ~174 rad/s, time constant ~6 ms
- **With damping:** Eigenvalue <0.1 rad/s, time constant >10 s

The pitch rate coupling (especially the rudder's long moment arm) provides the dominant damping effect.

---

## 8. Open-Loop Characterization Test Matrix

This section documents the comprehensive open-loop test matrix used to validate the Moth 3DOF model behavior. The test matrix consists of 5 categories with approximately 50 individual tests, all run at the `MOTH_BIEKER_V3` parameter set with a default forward speed of 10 m/s and simulation duration of 5 seconds.

Results are stored as JSON metadata + NPZ trajectories for reproducible report and plot generation. The test runner is `scripts/moth_dynamics_sweep.py`.

### 8.1 Overview

| Category | Name | Tests | Type | Description |
|----------|------|-------|------|-------------|
| 1 | Perturbation Response | 18 | Trim-based | State perturbation from trim, controls held |
| 2 | Trim Sensitivity | 10 | Trim-based | One control fixed at offset, re-trim with other free |
| 3 | Off-Equilibrium Transient | 9 | Off-equilibrium | Control offset from trim, no re-trim |
| 4 | Damping Verification | 6 | Trim-based | Parameter variants with eigenvalue analysis |
| 5 | Speed Variation | 7 | Mixed | Trim at multiple speeds + speed-step transient |

**Key distinction:** Categories 1, 2, and 4 start from an equilibrium (trim) condition. Categories 3 and 5B start from a non-equilibrium condition and observe the transient response.

### 8.2 Category 1: Perturbation Response

**Setup:** Find trim at 10 m/s. Apply a state perturbation to the trim state. Hold controls at trim values throughout.

**Sweep parameters:**
- Pitch perturbation: $\pm 0.1°, \pm 0.5°, \pm 1°, \pm 2°, \pm 5°$ (10 tests)
- Heave velocity perturbation: $\pm 0.01, \pm 0.05, \pm 0.1, \pm 0.2$ m/s (8 tests)

**Metrics:** Stability (bounded or divergent), divergence time, maximum deviation from trim, response symmetry.

**Expected physics:**
- Small perturbations ($< 0.5°$): System remains bounded due to pitch rate coupling damping (Section 7.1). The post-Wave 4B damping mechanisms hold in the linear regime.
- Large perturbations ($> 1°$): System diverges on a 2--5 second timescale as nonlinear effects dominate.
- Symmetric response: $|\text{response}(+\delta)| \approx |\text{response}(-\delta)|$ within a factor of 2.
- Heave velocity perturbations couple to pitch through the kinematic relationship $\dot{w} = f(\theta, w, q)$.

**Pass criteria:**
- Small perturbations bounded for full duration
- Large perturbations diverge (pitch deviation exceeds 30°)
- Symmetric within factor of 2 for $\pm$ pairs

### 8.3 Category 2: Trim Sensitivity

**Setup:** Fix one control surface at an offset from its baseline trim value. Re-trim with the other control surface free to find the new equilibrium. Compare the new equilibrium to baseline.

**Sweep parameters:**
- Flap offset (elevator free): $\pm 2°, \pm 5°, \pm 8°$ (6 tests)
- Elevator offset (flap free): $\pm 1°, \pm 2°$ (4 tests)

**Metrics:** Change in trim state ($\Delta \theta$, $\Delta \text{pos\_d}$) and trim control.

**Expected physics:**
- **Flap positive** (more main foil lift): System needs less angle of attack $\to$ $\theta$ decreases. The additional lift also raises ride height $\to$ $\text{pos\_d}$ decreases.
- **Elevator positive** (more rudder lift): Aft lift increase creates nose-down moment $\to$ $\theta$ decreases.
- **Monotonic:** Larger offset $\to$ larger equilibrium shift.
- **Sailor position schedule effects:** The current model computes CG offset and composite inertia from sailor position, so sailor changes can affect trim through geometry and inertia coupling. (There is still no explicit separate gravity moment term about CG because gravity is applied at system CG by construction.)

**Pass criteria:**
- Flap+ $\to$ $\Delta\theta < 0$ (nose-down)
- Elevator+ $\to$ $\Delta\theta < 0$ (nose-down)
- Monotonic: $|\Delta\theta(\pm 5°)| > |\Delta\theta(\pm 2°)|$

### 8.4 Category 3: Off-Equilibrium Transient

**Setup:** Start at the baseline trim state. Apply a control surface offset directly to the trim control (no re-trim). Simulate with the offset control held constant.

**This differs from Category 2** in that no new equilibrium is found. The system starts at the old equilibrium with the new control input and we observe the transient departure.

**Sweep parameters:**
- Flap offset: $\pm 2°, \pm 5°$ (4 tests)
- Elevator offset: $\pm 1°, \pm 2°$ (4 tests)
- Combined (reinforcing): flap $+2°$, elevator $+1°$ (1 test)

**Metrics:** Initial acceleration vector ($\dot{x}$ at $t=0$), initial response direction, divergence time, maximum deviation.

**Expected physics:**
The initial acceleration is predicted by the dynamics evaluated at the trim state with the offset control:

$$\dot{x}(0) = f(x_{\text{trim}}, u_{\text{trim}} + \Delta u)$$

For small offsets, this approximates to $B \cdot \Delta u$ from the linearization.

- **Flap positive:** Additional lift $\to$ initial upward acceleration ($\dot{w} < 0$ in body frame)
- **Elevator positive:** Additional aft lift $\to$ initial nose-down pitch acceleration ($\dot{q} < 0$)
- **Combined (reinforcing):** Both effects present simultaneously
- **Larger offset $\to$ larger initial acceleration magnitude**

**Pass criteria:**
- Initial acceleration direction matches $B \cdot \Delta u$ prediction
- $|\text{accel}(5°)| > |\text{accel}(2°)|$
- Combined test shows both upward and nose-down acceleration

### 8.5 Category 4: Damping Verification

**Setup:** Create parameter variants that isolate specific damping mechanisms. For each variant, find trim, apply a pitch perturbation, simulate, and compute the linearized eigenvalues.

**Parameter variants:**

| Variant | $m_{\text{added,heave}}$ | $I_{\text{added,pitch}}$ | Description |
|---------|--------------------------|--------------------------|-------------|
| Full damping | 10 kg | 8.75 kg·m² | `MOTH_BIEKER_V3` defaults |
| No added mass | 0 kg | 0 kg·m² | Only pitch rate coupling |
| High added mass | 20 kg | 17.5 kg·m² | 2$\times$ default |

**Perturbation sizes:** $\theta + 0.5°$ (linear regime) and $\theta + 2°$ (moderate nonlinear)

**Metrics:** Eigenvalue structure (A matrix), divergence time from simulation, relative response speed.

**Expected eigenvalue structure** (see Section 7):
- Full damping: 2 fast stable modes ($\text{Re}(\lambda) < -10$) + 2 nearly marginal modes ($|\text{Re}(\lambda)| < 1$). Maximum real eigenvalue $\approx 0.005$ rad/s.
- No added mass: Stable eigenvalues have larger magnitude (faster dynamics, less inertia to overcome).
- High added mass: Stable eigenvalues have smaller magnitude (slower dynamics, more effective inertia).

**Pass criteria:**
- Full damping divergence time $\geq$ no-added-mass divergence time (added mass slows everything, including divergence)
- Eigenvalue structure matches expected pattern for each variant
- High added mass $\to$ slowest dynamics (smallest eigenvalue magnitudes)

### 8.6 Category 5: Speed Variation

#### 8.6.1 Sub-category 5A: Trim at Multiple Speeds

**Setup:** Find trim equilibrium at each of 5, 6, 7, 8 m/s. Compare trim states.

**Expected physics:** Lift is proportional to $u^2$ (dynamic pressure). At higher speed, less angle of attack is needed to generate the required lift force equal to weight:

$$L = \frac{1}{2} \rho u^2 S C_{L_\alpha} \alpha = mg$$

Therefore $\alpha \propto 1/u^2$, and since $\theta \approx \alpha$ at trim, trim pitch angle decreases monotonically with speed.

**Pass criteria:**
- Trim converges at all 4 speeds
- $\theta_{\text{trim}}$ monotonically decreasing: $\theta(5) > \theta(6) > \theta(7) > \theta(8)$

#### 8.6.2 Sub-category 5B: Speed Step from Trim

**Setup:** Find trim at 10 m/s. Create a new `Moth3D` at the step speed (8 or 12 m/s) but start from the 10 m/s trim state and controls.

**Sweep:** 10$\to$8 m/s, 10$\to$12 m/s, 10$\to$14 m/s (3 tests)

**Expected physics:**
- **Speed drop (10$\to$8):** Lift drops to $(8/10)^2 = 64\%$ of weight $\to$ net downward force $\to$ boat sinks ($\text{pos\_d}$ increases initially)
- **Speed increase (10$\to$12):** Lift rises to $(12/10)^2 = 144\%$ of weight $\to$ net upward force $\to$ boat rises ($\text{pos\_d}$ decreases initially)
- **Speed increase (10$\to$14):** Even larger effect, $(14/10)^2 = 196\%$

**Pass criteria:**
- Speed drop: $\text{pos\_d}$ increases in early response
- Speed increase: $\text{pos\_d}$ decreases in early response
- Larger speed change $\to$ larger initial response