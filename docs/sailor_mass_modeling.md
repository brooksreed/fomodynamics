# Sailor Mass Modeling: Design Analysis

This document analyzes approaches for correctly modeling the sailor's mass contribution to Moth dynamics. It covers the current model limitation, corrected equations of motion, comparison with analogous problems in other domains, and a recommended progression from static correction through time-varying sailor position.

For background, see [moth_modeling.md](moth_modeling.md) (inertia tensor section) and [moth_simulation_vision.md](moth_simulation_vision.md) (Phase 3 sailor movement).

---

## 0. Coordinate Frames, Reference Points, and What fomodynamics Assumes

fomodynamics’s Moth model uses:

- **Axes:** body frame is FRD (Forward-Right-Down): \(+x\) forward, \(+y\) starboard, \(+z\) down.
- **Pitch convention:** \(\theta > 0\) is **nose-up**, and \(M_y > 0\) is a **nose-up** pitch moment.
- **3DOF state meaning:** \([pos_d,\; \theta,\; w,\; q]\) where:
  - `pos_d` is the modeled **CG** depth in NED-down (positive down),
  - `w` is the modeled vertical velocity in body \(+z\) (positive down),
  - `q` is pitch rate about \(+y\) (positive nose-up).

### 0.1 The important implementation detail (and why the current model breaks)

In the current implementation:

- `MothParams` defines geometric positions (foil/rudder/sail CE/sailor) **relative to the hull CG**.
- Force components compute moments as \(M = r \times F\) assuming \(r\) is **relative to the dynamics CG**.

So the model is implicitly trying to be a **CG-referenced** dynamics model, but is currently being supplied **hull-CG-referenced** lever arms. When the sailor shifts the system CG, this mismatch can completely erase trim/dynamics sensitivity.

This document’s “Option A” fix below (write dynamics about the system CG) matches the force-component architecture fomodynamics already has.

---

## 1. Problem Statement

### The Current Bug

The Wave 4C configuration sweep ([retro](plans/moth_3dof_model_and_estimation/open_loop_configuration_sweep.md#sailor-position-has-no-effect-model-limitation)) identified that changing `sailor_position` in `MothParams` has **no effect on trim or dynamics**. The root cause is a split in the equations of motion:

**What sees the sailor position:**
- Inertia computation: `iyy = hull_inertia_matrix[1, 1]` (hull only — doesn't even include sailor contribution)

**What ignores the sailor position:**
- Gravity: applied as `m_total * g * cos(θ)` at the body origin, producing zero pitch moment
- Component moment arms: foil at `x = +0.3 m`, rudder at `x = -1.955 m`, both relative to hull CG rather than the system CG the dynamics intend to use

The body frame origin is fixed at the boat CG (non-sailor mass: hull + rig + foils + rigging = 50 kg). When the sailor sits at `x_s = -0.3 m` (aft of boat CG), the system CG shifts aft, but no term in the EOM reflects this. The model is physically inconsistent: it's a 125 kg system whose gravity acts through a point that isn't the center of mass.

### Why It Matters

The sailor is 60% of the system mass (75 kg sailor / 125 kg total). `hull_mass` = 50 kg represents the full non-sailor boat mass (hull shell + rig + foils + rigging), not just the hull shell. A 0.3 m fore/aft shift in sailor position moves the system CG by ~0.18 m — comparable to the foil-to-rudder baseline of 2.255 m. This should produce a measurable shift in equilibrium pitch angle, required flap/elevator trim, and stability margins.

For control design, getting the gravity moment right is essential: it determines how much control authority is needed to maintain trim, and how the equilibrium shifts as the sailor moves.

---

## 2. Analogous Problems in Other Domains

### 2.1 Motorcycle Rider Lean

**Problem similarity:** High. Rider mass is 25–35% of total system mass in motorcycles; in a Moth, the sailor is ~65%.

**Modeling approaches:**
- **Full multibody:** 15-segment rider model with 28 DOF (University of Padova, BikeSim). Used for detailed biomechanics studies — handles rider lean, head/torso movement, arm inputs.
- **Inverse pendulum:** Rider as a single rigid body with rotational DOF along the forward axis. Captures the dominant lean coupling without segment-level detail.
- **Point mass on linkage:** Rider CG position parameterized by lean angle. Simplest approach that still captures the CG shift physics.

**Key insight:** Even in high-fidelity motorcycle simulations, the rider-CG coupling (not rider segment dynamics) dominates the vehicle behavior. The point-mass-on-linkage model captures >90% of the effect for stability and control studies.

**Sources:**
- Cossalter & Lot, "A Motorcycle Multi-Body Model for Real Time Simulations," Vehicle System Dynamics, 2002
- Saccon et al., "Three-Dimensional Parametric Biomechanical Rider Model for Multibody Applications," Applied Sciences, 2020

### 2.2 Aircraft Fuel Slosh

**Problem similarity:** Medium. Fuel mass can be 30–50% of vehicle mass, and it moves continuously (not rigidly).

**Modeling approaches:**
- **Equivalent mechanical model:** Liquid mass split into fixed component (moves with tank) and sloshing component (spring-mass-damper). The spring frequency matches the lowest slosh mode; damping captures viscous dissipation.
- **Dynamic CG tracking:** CG position updated at each timestep from fuel mass distribution. Used in flight management systems.
- **Multi-tank allocation:** Each tank modeled as a point mass at its geometric center. CG computed from weighted sum. Moments of inertia from parallel axis theorem.

**Key insight:** The spring-mass approach is relevant if we later want to model *reaction dynamics* — the sailor's body responding to boat accelerations rather than moving as a rigid point. For now, the multi-tank (point mass) approach is the right analogy.

**Sources:**
- Abramson, "The Dynamic Behavior of Liquids in Moving Containers," NASA SP-106, 1966
- Dodge, "The New Dynamic Behavior of Liquids in Moving Containers," SwRI, 2000

### 2.3 Sailing VPPs (Velocity Prediction Programs)

**Problem similarity:** High. The ORC VPP explicitly models crew position as an optimization variable.

**Modeling approaches:**
- **Static balance:** Crew weight placement (windward/leeward, fore/aft) is an optimization variable. For each candidate speed, the VPP solves force/moment balance including crew gravity moment. The crew position that maximizes VMG is selected.
- **Discrete positions:** Some VPPs use 2–4 crew positions (e.g., centered, hiking, trapezing). Each position has pre-computed CG and righting moment contributions.
- **Dynamic VPPs (DVPPs):** 6-DOF time-domain simulations where crew position is a time-varying input. Used for America's Cup design studies.

**Key insight:** Even steady-state VPPs treat crew position as a first-class variable affecting force/moment balance. The Moth model must do at least this much.

**Sources:**
- Claughton & Oliver, "Developments in Hydrodynamic Force Models for VPPs," SNAME, 1998
- ORC VPP Documentation 2023, https://orc.org/uploads/files/ORC-VPP-Documentation-2023.pdf

### 2.4 Summary Table

| Domain | Mass Ratio | Approach | Fidelity | Complexity |
|--------|-----------|----------|----------|------------|
| Motorcycle (research) | 25–35% | Full multibody (15+ bodies) | Very high | Very high |
| Motorcycle (control) | 25–35% | Point mass on linkage | Medium | Low |
| Aircraft fuel | 30–50% | Spring-mass per tank | Medium-high | Medium |
| Aircraft FMS | 30–50% | Point mass per tank | Medium | Low |
| Sailing VPP (static) | 40–70% | CG optimization variable | Medium | Low |
| Sailing DVPP | 40–70% | Time-varying CG input | Medium-high | Medium |
| **Moth (proposed)** | **~65%** | **Point mass, static then scheduled** | **Medium** | **Low** |

---

## 3. Corrected Equations of Motion

### 3.1 Setup

The Moth 3DOF state vector is `[pos_d, θ, w, q]` (heave position, pitch angle, body-frame heave velocity, pitch rate). Geometric positions in `MothParams` are currently defined relative to the **hull CG**, but the *recommended* dynamics reference point is the **system CG**.

**Masses and positions (all in body frame, relative to hull CG):**
- Hull mass \(m_h\) at origin (by definition)
- Sailor mass \(m_s\) at position \(\mathbf{r}_s = [x_s, 0, z_s]\)
- Total mass \(m = m_h + m_s\)

**System CG offset from body origin:**

$$
\mathbf{r}_{cg} = \frac{m_s \cdot \mathbf{r}_s}{m} = \left[\frac{m_s x_s}{m},\; 0,\; \frac{m_s z_s}{m}\right] = [x_{cg},\; 0,\; z_{cg}]
$$

**Component positions (relative to hull CG = body origin):**
- Main foil: \(\mathbf{r}_f = [x_f, 0, z_f]\) (nominally [+0.55, 0, +1.82] m)
- Rudder: \(\mathbf{r}_r = [x_r, 0, z_r]\) (nominally [-1.755, 0, +1.77] m)
- Sail CE: \(\mathbf{r}_{sail} = [x_{sail}, 0, z_{sail}]\)

### 3.2 Two Correct Formulations

There are two equivalent ways to fix the model. Both produce identical dynamics; they differ in where the bookkeeping complexity lives.

#### Option A: Redefine Origin at System CG

Move the reference point for all moment computations to the system CG. This eliminates the gravity moment (gravity acts through the CG by definition) but requires adjusting all moment arms.

**Adjusted moment arms:**

$$
\mathbf{r}'_f = \mathbf{r}_f - \mathbf{r}_{cg}, \quad
\mathbf{r}'_r = \mathbf{r}_r - \mathbf{r}_{cg}, \quad
\mathbf{r}'_{sail} = \mathbf{r}_{sail} - \mathbf{r}_{cg}
$$

**Composite pitch inertia about system CG** (parallel axis theorem):

$$
I_{yy} = I_{yy,hull} + m_h \cdot (x_{cg}^2 + z_{cg}^2) + m_s \cdot \left[(x_s - x_{cg})^2 + (z_s - z_{cg})^2\right]
$$

The first transfer term shifts the hull inertia from hull CG to system CG. The second accounts for the sailor as a point mass at distance \(|\mathbf{r}_s - \mathbf{r}_{cg}|\) from the system CG.

This simplifies to:

$$
I_{yy} = I_{yy,hull} + \frac{m_h \cdot m_s}{m} \cdot (x_s^2 + z_s^2)
$$

(This is the *reduced mass* form of the parallel axis theorem for a two-body system.)

**Gravity:** No pitch moment (acts through CG).

**Force/moment equations (about system CG):**

$$
(m + m_{a,heave}) \cdot \dot{w} = F_{z,foil} + F_{z,rudder} + F_{z,hull} + m g \cos\theta
$$

$$
(I_{yy} + I_{a,pitch}) \cdot \dot{q} = \sum_i (\mathbf{r}'_i \times \mathbf{F}_i)_y
$$

where the moment arm vectors \(\mathbf{r}'_i\) are relative to the system CG.

**Kinematics** must also account for the offset: the position being integrated (`pos_d`) is for the *system CG*, not the hull CG. If sensors are at the hull CG, a correction is needed for output comparison. For the 3DOF longitudinal model this is minor.

#### Option B: Keep Origin at Hull CG, Add Gravity Moment

Keep all moment arms as defined (relative to hull CG) and add the missing gravity moment explicitly.

**Gravity force in body frame** at pitch angle \(\theta\):

$$
\mathbf{F}_{grav,body} = m g \begin{bmatrix} -\sin\theta \\ 0 \\ \cos\theta \end{bmatrix}
$$

**Gravity moment about body origin** (hull CG):

$$
\mathbf{M}_{grav} = \mathbf{r}_{cg} \times \mathbf{F}_{grav,body}
$$

The pitch component (y-axis):

$$
M_{y,grav} = z_{cg} \cdot (-mg\sin\theta) - x_{cg} \cdot (mg\cos\theta)
$$

$$
\boxed{M_{y,grav} = -mg\left(x_{cg}\cos\theta + z_{cg}\sin\theta\right)}
$$

**Physical check:** If the CG is forward of body origin (\(x_{cg} > 0\)), the gravity moment is nose-down (\(M_y < 0\)) at \(\theta = 0\). Correct — forward CG creates a nose-down tendency. ✓

**Composite pitch inertia about hull CG** (body origin):

$$
I_{yy} = I_{yy,hull} + m_s \cdot (x_s^2 + z_s^2)
$$

This is simpler than Option A because we only add the sailor's point-mass contribution about the hull CG. No hull transfer term is needed since the hull inertia is already about its own CG = body origin.

**Angular equation of motion (about body origin):**

When writing rotational EOM about a point O that is not the CG, an additional "transport term" appears:

$$
I_O \cdot \dot{q} = M_{components} + M_{grav} - \mathbf{r}_{cg} \times (m \cdot \mathbf{a}_O)
$$

where \(\mathbf{a}_O\) is the acceleration of the body origin. The transport term \(\mathbf{r}_{cg} \times m \cdot \mathbf{a}_O\) couples translational and rotational dynamics.

For the pitch component:

$$
M_{y,transport} = -(m \cdot x_{cg} \cdot \dot{w} - m \cdot z_{cg} \cdot \dot{u})
$$

In the current 3DOF model, \(\dot{u} = 0\) (forward speed is prescribed), so:

$$
M_{y,transport} = -m \cdot x_{cg} \cdot \dot{w}
$$

This term couples heave acceleration into the pitch equation. For small CG offsets and moderate accelerations, it may be small but is needed for formal correctness.

### 3.3 Recommendation: Option A (System CG Reference)

**Option A is recommended** for the following reasons:

1. **Cleaner EOM:** Standard Newton-Euler form with no transport terms. The gravity moment vanishes by construction.
2. **Extensibility:** When the sailor position becomes time-varying, the moment arm adjustments update naturally. Option B would require re-deriving the transport term for each new coupling.
3. **Existing infrastructure:** `MothParams.combined_cg_offset` already computes \(\mathbf{r}_{cg}\). The moment arm adjustment is a subtraction.
4. **Physical intuition:** Forces create moments about the CG. Component positions relative to CG have direct physical meaning (e.g., "the foil is 0.4 m forward of the system CG" is more useful than "0.6 m forward of hull CG minus 0.19 m CG offset").

**Important clarification:** For a **static** sailor position (v1), the system CG offset is constant, so the “CG frame” does **not** move during the simulation. You just build a consistent model once.

If the sailor position becomes **time-varying** (v2+), the system CG offset changes and additional modeling choices appear (Section 6).

### 3.4 Numerical Example

**Baseline configuration** (MOTH_BIEKER_V3 defaults):
- \(m_h = 50\) kg, \(m_s = 75\) kg, \(m = 125\) kg (includes rig/foil mass baked into hull)
- \(\mathbf{r}_s = [-0.15, 0, -0.2]\) m (sailor aft and above hull CG)
- \(I_{yy,hull} = 118.6\) kg·m²

**System CG offset:**

$$
\mathbf{r}_{cg} = \frac{75}{125} [-0.15, 0, -0.2] = [-0.09, 0, -0.12] \text{ m}
$$

The system CG is 9 cm aft and 12 cm above the hull CG.

**Adjusted moment arms:**

| Component | Original (from hull CG) | Adjusted (from system CG) | Δ |
|-----------|------------------------|--------------------------|---|
| Main foil x | +0.55 m | +0.64 m | +0.09 m |
| Rudder x | -1.755 m | -1.665 m | +0.09 m |

The foil arm *increases* and the rudder arm *decreases* (both shift forward relative to the aft-shifted CG). This means the foil has relatively more pitch authority — physically correct for a forward CG shift.

**Composite inertia:**

$$
I_{yy} = 118.6 + \frac{50 \times 75}{125} \times (0.15^2 + 0.2^2) = 118.6 + 30 \times 0.0625 = 118.6 + 1.88 = 120.5 \text{ kg·m}^2
$$

Compare to the current model using hull-only inertia: 118.6 kg·m². The sailor adds ~1.6% to pitch inertia at this modest offset. If the sailor moves forward to \(x_s = +0.3\) m (hiking forward), the composite inertia increases more significantly, and the moment arms change, altering the equilibrium.

**Gravity moment (Option B cross-check):**

$$
M_{y,grav} = -125 \times 9.81 \times (-0.09) \times \cos(0) = +110 \text{ N·m (nose-up)}
$$

The aft CG creates a nose-up gravity moment, requiring more nose-down control (elevator) to trim. This is the effect the current model completely misses.

---

## 4. Modeling Approaches: Options and Progression

### 4.1 Option 1: Static CG Correction (Fix Current Bug) — **IMPLEMENTED** (Wave 4E)

**What:** Compute system CG at model construction time. Adjust all moment arms and composite inertia. Sailor position remains a parameter in `MothParams`, not a state.

**Implementation (completed):**
- `MothParams.composite_pitch_inertia` property: reduced-mass parallel axis theorem
- `create_moth_components()` accepts `cg_offset`, subtracts from all component positions
- `Moth3D.__init__()` computes `cg_offset = params.combined_cg_offset`, uses composite inertia
- Fixed `MOTH_BIEKER_V3.sailor_position[2]`: `0.2` → `-0.6` (above hull CG)

**Verified effects:**
- 10cm sailor offset produces ~3 rad/s² instantaneous pitch acceleration
- Sailor forward → nose-down, sailor aft → nose-up (monotonic)
- Composite I_yy: 14.24 kg·m² (was 8.0 hull-only, +78%)

**Pros:**
- Minimal code change (~15 lines in `__init__` + component factory)
- No new states, no performance impact
- Fixes the identified bug completely for static configurations
- Configuration sweep tests produce real changes

**Cons:**
- Cannot vary sailor position during simulation
- Requires rebuilding `Moth3D` object for each sailor configuration

### 4.2 Option 2: Sailor Position as Time-Varying Schedule — **IMPLEMENTED** (Wave 4F)

**What:** Sailor position is a function of time `r_s(t)`, analogous to the existing `u_forward_schedule`. System CG, moment arms, and inertia recomputed at each timestep.

**Implementation:**
- Add `sailor_position_schedule: Callable[[float], Array]` to `Moth3D`
- In `forward_dynamics()`, call schedule to get `r_s(t)`, compute CG offset, adjust arms
- Use Option A formulation (system CG reference)
- Be explicit about the **approximation level** (Section 6). In particular: deciding what point `pos_d` represents matters once internal mass motion is introduced.

**Kinematic note (only if you choose to model the CG motion explicitly):**

When the sailor position changes, the system CG velocity includes a term from the sailor's movement relative to the hull:

$$
\dot{\mathbf{r}}_{cg} = \frac{m_s}{m} \dot{\mathbf{r}}_s
$$

The NED-down kinematics project body-frame velocities onto the down axis. If `pos_d` is defined as the **system CG depth**, then an additional term appears from the CG motion in the body frame:

$$
\dot{pos}_d
= -u \sin\theta + w \cos\theta
  - \dot{x}_{cg} \sin\theta + \dot{z}_{cg} \cos\theta
$$

where \(\dot{x}_{cg} = \frac{m_s}{m}\dot{x}_s\) and \(\dot{z}_{cg} = \frac{m_s}{m}\dot{z}_s\).

**Correctness warning:** if you are prescribing \(r_s(t)\) with nontrivial accelerations and want momentum-consistent dynamics, you are in internal-motion/multibody territory (Section 6). For many “move sailor slowly” studies, it is often better to treat \(r_s(t)\) as a **quasi-static configuration schedule** (update CG/levers/inertia, omit explicit internal-motion kinematics/dynamics) and be explicit that this is an approximation.

**Pros:**
- Captures fore/aft weight shifts during sailing (hiking forward in lulls, aft in gusts)
- No new states — sailor movement is prescribed, not dynamic
- Natural extension of the existing `u_forward_schedule` pattern
- JAX-compatible (schedule is a pure function)

**Cons:**
- Requires knowing the sailor trajectory a priori (not reactive)
- Recomputing CG/arms every timestep has minor performance cost (JAX JIT will optimize)
- The kinematic correction term adds complexity

**Verdict:** Good for Phase 2 studies. Lets you simulate "what if the sailor moves forward at t=5s" without the complexity of sailor dynamics.

**Implementation (Wave 4F):**
- `Moth3D` accepts `sailor_position_schedule: Callable[[float], Array]` returning `[x, y, z]`
- Default: constant schedule from `params.sailor_position` (identical to Wave 4E)
- Components store raw hull-CG-relative positions; CG offset passed per-call via `cg_offset` parameter on `compute_moth()`
- `_compute_cg_offset()` and `_compute_composite_iyy()` are module-level JAX-compatible helpers
- Kinematic correction uses central finite differences: `ṙ_s ≈ (r_s(t+ε) - r_s(t-ε)) / 2ε` with `ε = 1e-4`
- 10 new tests validate: constant-schedule equivalence, step/ramp behavior, JIT compatibility, full simulation

### 4.3 Option 3: Discrete Sailor Modes

**What:** Define N sailor positions (e.g., centered, hiking, full-hike) with pre-computed CG offsets and inertias. Switch between them.

**Implementation:**
- Define enum or dict of sailor positions with associated `r_s` vectors
- Pre-compute `cg_offset`, adjusted moment arms, and `iyy` for each mode
- Mode selection via schedule or control input

**Pros:**
- Fast (no per-timestep CG computation — just a lookup)
- Easy to understand and validate
- Natural for control studies ("if sailor hikes at t=3s, what happens?")

**Cons:**
- Discrete transitions are non-smooth (problematic for gradient-based optimization)
- Limited positions may miss important configurations
- Doesn't naturally extend to 6DOF (hiking angle is continuous)

**Verdict:** Viable but less flexible than Option 2. Better suited for game-like simulations or real-time applications where lookup speed matters. Not recommended for fomodynamics's JAX/optimization focus.

### 4.4 Option 4: Sailor Position as State Variable

**What:** Add sailor fore/aft position \(x_s\) (and later lateral \(y_s\)) as a dynamic state. Requires a model for what drives sailor movement.

**State extension:**
- 3DOF: `[pos_d, θ, w, q]` → `[pos_d, θ, w, q, x_s, ẋ_s]` (6 states)
- 6DOF: add `y_s, ẏ_s` for lateral (8 extra states total)

**Sailor dynamics options:**
- **Direct control:** \(\ddot{x}_s = u_{sailor}\) — sailor movement is a control input. Simple but requires specifying the control.
- **Spring-damper to target:** \(\ddot{x}_s = -k(x_s - x_{target}) - c \dot{x}_s\) — sailor seeks a target position with physical dynamics (inertia, damping). Target is the control input.
- **Reactive:** \(\ddot{x}_s = f(\theta, \dot{\theta}, ...)\) — sailor reacts to boat state. Models the human balance response. Requires biomechanics assumptions.

**Pros:**
- Captures sailor reaction dynamics (oscillation, overshoot)
- Enables optimal control of sailor position (MPC deciding when to hike)
- Most physically complete

**Cons:**
- Doubles state dimension for 3DOF model
- Sailor dynamics are speculative — we don't have data to validate
- Spring-damper parameters are hard to calibrate
- Adds complexity without clear near-term benefit

**Verdict:** Future consideration (Phase 3+). Only needed when studying sailor-boat coupling dynamics or optimizing sailor movement patterns. The schedule approach (Option 2) handles most use cases.

### 4.5 Option 5: Spring-Mass Sailor (Fuel Slosh Analogy)

**What:** Model the sailor as a mass connected to the hull by a spring-damper, analogous to fuel slosh equivalent mechanical models. The sailor oscillates relative to the hull in response to accelerations.

**Dynamics:**

$$
m_s \ddot{x}_s = -k_s x_s - c_s \dot{x}_s + m_s a_{hull,x}
$$

where \(a_{hull,x}\) is the hull's forward acceleration felt by the sailor.

**Pros:**
- Captures oscillatory coupling (sailor weight oscillation amplifying pitch oscillation)
- Well-understood theory from spacecraft fuel slosh community
- Single additional DOF with two parameters (k, c)

**Cons:**
- Sailor is not a liquid — voluntary muscle control breaks the spring-mass model at low frequencies
- Appropriate only for high-frequency perturbation response, not quasi-static positioning
- Requires calibration data we don't have

**Verdict:** Interesting for stability analysis (does sailor oscillation destabilize pitch?). Not appropriate as the primary sailor model. Could be added as a perturbation study on top of Option 2.

### 4.6 Recommended Progression

| Phase | Approach | What It Enables |
|-------|----------|----------------|
| **v1** | **Static CG correction (Option 1 / CG-referenced model)** — ✅ Wave 4E | Fix configuration sweep, make geometry + moments consistent, correct trim sensitivities |
| **v2** | **Time-varying configuration schedule (Option 2)** — ✅ Wave 4F | Smooth "what-if" scenarios (fore/aft shifts, hiking) without adding state dimension |
| **v3 (future)** | **Sailor as state with rate/accel limits (Option 4)** | MPC/trajectory optimization can choose hiking/fore-aft strategy |
| **v4 (optional)** | **Multibody-lite / internal momentum terms** | Pumping / high-frequency sailor-boat coupling studies |

---

## 5. Extension to 6DOF

The 3DOF model only has pitch and heave. In 6DOF, the sailor's lateral position (hiking out) becomes the dominant effect. Here's how each approach extends.

### 5.1 CG and Inertia in 6DOF

**System CG (3D):**

$$
\mathbf{r}_{cg} = \frac{m_s}{m} [x_s, y_s, z_s]
$$

**Full composite inertia tensor about system CG:**

$$
\mathbf{I}_{total} = \mathbf{I}_{hull,cg'} + \mathbf{I}_{sailor,cg'}
$$

where \(\mathbf{I}_{hull,cg'}\) is the hull diagonal inertia transferred to system CG via the parallel axis theorem (generalized):

$$
I_{ij,transfer} = m \cdot (\|\mathbf{d}\|^2 \delta_{ij} - d_i d_j)
$$

with \(\mathbf{d}\) being the displacement vector from component CG to system CG and \(\delta_{ij}\) the Kronecker delta.

**For a point mass sailor at \(\mathbf{d}_s = \mathbf{r}_s - \mathbf{r}_{cg}\) from system CG:**

$$
I_{xx,sailor} = m_s(d_{sy}^2 + d_{sz}^2), \quad
I_{yy,sailor} = m_s(d_{sx}^2 + d_{sz}^2), \quad
I_{zz,sailor} = m_s(d_{sx}^2 + d_{sy}^2)
$$

$$
I_{xy,sailor} = -m_s \cdot d_{sx} \cdot d_{sy}, \quad
I_{xz,sailor} = -m_s \cdot d_{sx} \cdot d_{sz}, \quad
I_{yz,sailor} = -m_s \cdot d_{sy} \cdot d_{sz}
$$

**Key point from moth_modeling.md:** The off-diagonal products of inertia (\(I_{xy}\), \(I_{xz}\), \(I_{yz}\)) arise *naturally* from the point mass geometry when the sailor hikes out (\(y_s \neq 0\)). This is exactly the "multibody lite" approach — no need to maintain a full time-varying tensor; just recompute from the point mass position.

### 5.2 Hiking as Roll Control

In a Moth at 6DOF, the sailor's lateral position is the **primary roll control input**. The roll righting moment from hiking is:

$$
M_{x,sailor} \approx m_s \cdot g \cdot y_s \cdot \cos\phi
$$

where \(y_s\) is the sailor's lateral offset (positive to windward) and \(\phi\) is the heel angle. This moment balances the heeling moment from sail force.

For the 6DOF extension, the sailor's lateral position should be either:
- A time-varying input (schedule), or
- A slow state with dynamics (spring-damper to target hiking angle)

The approach is the same as the longitudinal case — the math generalizes cleanly.

### 5.3 Utility Function Design

For both 3DOF and 6DOF, the following utility function handles the CG/inertia computation:

```python
def composite_mass_properties(
    m_hull: float,
    I_hull: NDArray,           # [3,] diagonal or [3,3] tensor
    m_sailor: float,
    r_sailor: NDArray,         # [3,] position relative to hull CG
) -> tuple[float, NDArray, NDArray, NDArray]:
    """Compute composite mass properties for hull + point-mass sailor.

    Returns:
        total_mass: Combined mass
        cg_offset: System CG position relative to hull CG [3,]
        I_total: Composite inertia tensor about system CG [3,3]
        moment_arm_correction: Vector to subtract from component positions [3,]
    """
```

This function is pure numpy (no JAX dependency), lives in `fmd.simulator.params` or `fmd.core`, and is called once at construction (Option 1) or per-timestep (Option 2, as a JAX-compatible version).

---

## 6. Time-Varying Sailor Motion: What Changes Physically

Once sailor position varies during the simulation, you are no longer strictly in “single rigid body with fixed inertia” mechanics. It is helpful to separate three modeling levels:

### 6.1 Level 1: Quasi-static schedule (recommended v2)

Treat sailor position \(r_s(t)\) as a **slowly varying configuration input** that updates:

- CG offset \(r_{cg}(t)\),
- inertia about CG \(I_{cg}(t)\),
- lever arms \(r'_i(t) = r_i - r_{cg}(t)\).

This captures the dominant effects for trim/control studies (righting moments, changed pitch/roll authority, changed inertia). It intentionally omits internal momentum exchange terms.

### 6.2 Level 2: Sailor as a controlled internal DOF (recommended v3)

Make sailor position (e.g. \(x_s, y_s\)) a **state** with rate/acceleration limits (or 2nd-order dynamics). This is the natural bridge to optimization/MPC where hiking/fore-aft becomes a first-class decision variable.

### 6.3 Level 3: Pumping / high-frequency coupling (optional v4)

If you want to study pumping (e.g. shoulders/torso motion to inject energy), you may need multibody-lite dynamics and/or unsteady external-force models. “Moving a point mass inside a rigid body” alone will not reproduce real pumping power without modeling how that motion changes external forces (foil unsteady lift/ventilation/waves).

---

## 7. Design Patterns to Adopt (Validated by AC38-Style MPC Implementations)

From the architecture study in `ac38_mpc_research_findings.md`, several patterns align with and reinforce the recommendations here:

1. **Write Newton–Euler equations about the system CG** (Option A / CG-referenced dynamics).
2. **Crew as point mass**: crew/sailor inertia is treated as zero; effects enter via CG shift + parallel axis theorem.
3. **Recompute CG and inertia when geometry changes** (canting foils / hiking / sailor movement).
4. Add a small **moment transport utility** (compute moments about CG and optionally about hull origin for diagnostics).
5. For MPC, consider **rate-based actuation** (actuator angles are states; controls are rates), and build a **steady-state trim solver** before full MPC.

---

## 8. Open Questions

1. **Sailor z-position convention:** `sailor_position` uses body \(+z\) down (FRD). The sailor CG is typically *above* the hull CG, so \(z_s < 0\) is expected. Verify the sign in presets (e.g. `MOTH_BIEKER_V3`) and any downstream assumptions.

2. **Added mass interaction:** When the CG shifts, do the added mass coefficients (heave, pitch) need adjustment? Formally, added mass depends on body geometry relative to the free surface, not CG position. So the added mass stays the same, but the *effective* added inertia about the new CG reference may need a parallel axis correction if the added mass has a spatial distribution. For v1, assume added mass is unchanged.

3. **Sensor location vs. CG:** If `pos_d` tracks the system CG but sensors are at a fixed hull location, the measurement equation needs a geometric correction. This matters for state estimation (EKF/UKF) but not for open-loop simulation.

4. **Sailor position data:** We don't have direct measurements of sailor position. Validation will rely on indirect evidence: does the model predict trim shifts in the right direction and approximate magnitude when sailor position changes? Comparison with VPP data or sailing intuition.

---

*Created: January 2026*
*Related: [moth_modeling.md](moth_modeling.md), [moth_simulation_vision.md](moth_simulation_vision.md), [Wave 4C retro](plans/moth_3dof_model_and_estimation/open_loop_configuration_sweep.md)*
