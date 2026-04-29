# Moth Sailboat Simulation Vision

This document captures the long-term vision for the Moth sailboat simulation project to guide incremental development.

---

## Summary

Build a dynamic simulation of a Mackay Bieker Moth V3 hydrofoiling sailboat for:
- **Control prototyping** - Design and test ride height controllers (electronic and mechanical)
- **Data analysis** - Improve state estimation from sailing telemetry
- **Design studies** - Analyze effects of configuration changes (weight distribution, gearing, etc.)

The model should be accurate enough for control design and frequency response analysis, not necessarily a full VPP.

---

## The Boat

**Platform:** Mackay Bieker Moth V3 ([mackayboats.com](https://mackayboats.com/index.cfm/boats/bieker-moth/))

Key characteristics:
- Single-handed hydrofoiling dinghy (~3.4m LOA)
- T-foil main foil with flap for ride height control
- T-foil rudder with elevator (angle of attack controlled by sailor)
- Mechanical wand system for automatic flap control
- Sailor position is primary balance input (heel/pitch)

---

## Goals & Use Cases

### Primary: Electronic Control Prototyping
- Abstract away mechanical wand complexity
- Direct flap angle control as input
- Test different control algorithms (PID, LQR, MPC)
- Evaluate sensor requirements and placement
- Learning platform for control methods applicable to other systems

### Secondary: Data Analysis & State Estimation
- Improve state estimation from Vakaros Atlas 2 telemetry (2Hz GPS/IMU)
- Model-based filtering (EKF/UKF) for smoother state estimates
- Validate model against real sailing data
- Future: Higher-fidelity ArduPilot-based instrumentation

### Tertiary: Mechanical Wand System Analysis
- Understand wand dynamics and stability margins
- Optimize gearing ratios and set points
- Compare electronic vs mechanical control performance

---

## Modeling Approach

### Rigid Body Dynamics
- **6-DOF rigid body** (extend from existing `RigidBody6DOF`)
- Ignore structural flexibility (boat, mast, foils)
- Accurate geometry and mass/inertia from Moth specs

### Sailor Body
- **Initial (v1):** Sailor as a **point mass** at a fixed position relative to the hull.
  - Dynamics are written about the **system CG** (CG-referenced Newton–Euler form).
  - Geometry for force application points (foils, sail CE) is stored in a hull reference frame, but is **shifted to CG-relative lever arms** for moment computation.
- **Future (v2+):** Introduce sailor motion in increasing fidelity:
  - **Schedule / scenario input:** \(r_s(t)\) prescribed (smooth hiking/fore-aft changes).
  - **State + rate limits:** \(r_s\) becomes a state with rate/accel constraints so MPC can decide hiking/fore-aft.
  - **Optional multibody-lite:** only if needed for pumping / high-frequency internal-motion coupling.

### Sail Forces
- Simplified force model: lift/drag as function of mainsheet setting
- Not modeling detailed sail shape inputs
- Good enough for control studies, not for VPP accuracy

### Foil Hydrodynamics
- **Approach:** Analytical models from literature, empirically tuned
  - Thin airfoil theory + finite span corrections
  - CL/CD vs angle of attack curves
- **Accuracy target:** Correct trends and approximate magnitudes
- **Not doing (yet):** Custom CFD or neural network response surfaces

### Ventilation & Tip Piercing
- **Critical area** - most crashes relate to ventilation
- Model as gradual lift degradation near surface (narrow transition region)
- Not binary ventilated/non-ventilated (too abrupt)
- Research area: behavior when tip pierces slightly vs stays just below

### Wave Disturbance
- **Architecture:** Design for sinusoidal and spectrum-based waves from the start
- **Initial implementation:** Flat water until model is validated
- **Future:** Add regular waves, then realistic sea states

### Speed & Wind Handling

**Primary focus:** Pitch/heave and roll/yaw dynamics, not surge.

The control system behavior varies significantly with speed. Rather than modeling full surge dynamics from the start, we use a **hybrid approach**:

**Initial phases (quasi-steady speed):**
- Boat speed is a slowly-varying parameter or equilibrium state
- Force coefficients parameterized by speed (and apparent wind)
- Focus simulation effort on pitch/heave/roll dynamics
- Useful for studying control system behavior at different operating points

**True wind coupling:**
- Specify true wind (TWA/TWS), compute apparent wind from boat speed
- Captures the sail force ↔ boat speed feedback loop
- Important for understanding stability as speed changes

**Later phases (full surge dynamics):**
- Speed becomes a dynamic state
- Sail thrust vs foil/hull drag determines acceleration
- Required for acceleration/deceleration studies and maneuvers

**Transition path:**
- Start with quasi-steady: validate pitch/heave/roll control at fixed speeds
- Add surge dynamics when studying acceleration phases
- Eventually converges toward full dynamic VPP capability

**Key question:** How far can quasi-steady approaches take us before full surge dynamics are needed? Empirical validation will guide this.

---

## Foil Control System Options

### Option A: Mechanical Wand System
Models the standard Moth ride height control:
- Wand extending from bowsprit skims water surface
- Wand angle drives main foil flap via linkage
- Nonlinear gearing ratio and adjustable set point

**Control inputs:**
- Wand length
- Gearing ratio
- Set point (trim)

**Initial model:** Quasi-static (flap angle = f(ride height))
**Future:** Add wand inertia, water resistance, sensor lag

### Option B: Electronic Control System
Direct electronic actuation of flap:
- Ride height sensor (e.g., ultrasonic) at one or two locations
- Electronic actuator for flap angle

**Control input:** Main foil flap angle (direct)

**Note:** Option B is simpler to start with in simulation; Option A can be added later for comparative studies.

---

## Control Inputs & Constraints

### Input Vector (Electronic Control Mode)
| Input | Description | Range | Rate Limit |
|-------|-------------|-------|------------|
| Mainsheet | Boom angle from centerline | 0° to 45° | TBD |
| Tiller | Rudder angle | ~±30° | TBD |
| Rudder AoA | Rudder foil elevator | -1.5° to +4.5° | TBD |
| Main flap | Main foil flap angle | TBD | TBD |

### Input Vector (Mechanical Wand Mode)
Replace main flap with: wand length, gearing ratio, set point

### State Constraints (Model Validity)
- Foil angles within stall limits
- Heel angle within recovery range (see below)
- Ride height within foil span (ventilation boundary)
- Speed within modeled regime

### Heel Angle Notes
Moths sail upwind with significant **windward heel** (like a windsurfer):
- Light air: 20-30° windward heel typical
- Windier conditions: 25-35° windward heel typical
- Recovery possible from 40-45° unless very light wind
- Leeward capsize is more common failure mode than windward

### Soft Targets (Control Objectives)
- Ride height within target band
- Pitch angle near zero
- Heel angle at target windward heel for conditions

---

## Phased Implementation

### Phase 1: Straight-Line Upwind (Quasi-Steady) **[IMPLEMENTED]**
- Flat water ✓
- Narrow true wind range (e.g., 8-12 kts) ✓
- Quasi-steady speed (parameterized, not dynamic) ✓
- Electronic control (Option B) ✓
- Validate against known target speeds and attitudes ✓
- **Implementation:** `Moth3D` class with component-based force model, `WaveField` for wave disturbance, trim solver, 3D visualization via Rerun

### Phase 2: Broader Conditions
- Add wave disturbance (sinusoidal)
- Wider wind range
- Add mechanical wand model (Option A)
- Frequency response analysis

### Phase 2.5: Straight-Line Acceleration/Deceleration
- Add surge dynamics (speed as state)
- True wind → apparent wind coupling with speed feedback
- Study control behavior during speed changes
- Still straight-line (simpler than maneuvers)

### Phase 3: Maneuvers
- Tacking/gybing dynamics
- Sailor movement model (at least schedule- or state-based hiking/fore-aft)
- Transient behavior validation
- Full dynamic VPP territory

---

## Data Sources

### Available Now
- **Vakaros Atlas 2:** GPS/IMU at 2Hz, extensive logs
- **GoPro video:** Visual validation for model sanity checks
- **Sailor knowledge:** Approximate ranges for heel, pitch, target speeds

### Not Available (Use Literature/Estimates)
- Flap angles during sailing
- Tiller angle, rudder AoA
- Direct force measurements

### Future
- ArduPilot-based system with higher sample rate
- Potentially instrumented control surfaces
- Integration with fmd.analysis for processing

---

## Key Unknowns & Research Areas

1. **Ventilation behavior** - How lift degrades near surface, hysteresis effects
2. **Foil parameter identification** - Tuning CL/CD curves to match real behavior
3. **Sail force modeling** - What level of fidelity is needed for control studies?
4. **Wand dynamics** - Does quasi-static model capture important behavior?
5. **Rider coupling** - When does sailor movement matter for dynamics, and at what fidelity (schedule vs state vs multibody-lite)?
6. **Speed modeling threshold** - How far can quasi-steady speed take us before full surge dynamics are required?

---

## References

### Moth-Specific Research
- [Miguel Brito - Gomboc Moth VPP (2019)](https://sumtozero.com/wp-content/uploads/2021/02/2019.-Miguel-Brito.-Use-of-Gomboc-to-Predict-the-Performance-of-a-Hydro-foiled-Moth.pdf)
- [Flight Dynamics of a Hydrofoiling Moth - DVPP](https://www.researchgate.net/publication/328571438_Flight_Dynamics_and_Stability_of_a_Hydrofoiling_International_Moth_with_a_Dynamic_Velocity_Prediction_Program_DVPP)
- [Moth Electronic Wand System](https://www.boatdesign.net/attachments/moth-electronic-wand-system-pdf.89499/)
- [Investigating Sailing Styles on Moth Performance](https://www.researchgate.net/publication/228831085_Investigating_sailing_styles_and_boat_set-up_on_the_performance_of_a_hydrofoiling_Moth_dinghy)

### Related Work
- [Speed Sailing VPP - Dane Hull (2014)](https://foils.org/wp-content/uploads/2019/04/2014-Winner-Speed-Sailing-Design-Velocity-Prediction-Program-Dane-Hull-2014.pdf)

---

## Relationship to fomodynamics Architecture

This model will build on existing fomodynamics infrastructure:

- **Base class:** Extend `RigidBody6DOF` for 13-state quaternion dynamics
- **Components:** Create `JaxForceElement` subclasses for sail, foil, rudder forces
- **Parameters:** Use `attrs`-based parameter classes for Moth configuration
- **Integration:** Use existing RK4 integrator with `jax.lax.scan`
- **Analysis:** Output to `DataStream` for comparison with Vakaros telemetry

---

*Last updated: January 2026*
*Version: 0.3 - Added speed handling approach, heel angle notes; open-loop characterization formalized (Wave 4C)*
