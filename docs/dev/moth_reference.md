# Moth Model Reference

Moth-specific pitfalls, trim solver behavior, and timestep conventions.

**Related docs:**
- [../physical_intuition_guide.md](../physical_intuition_guide.md) — physics intuition, debugging dynamics
- [../timestep_guide.md](../timestep_guide.md) — eigenvalue tables, RK4 stability
- [../moth_vertical_geometry.md](../moth_vertical_geometry.md) — hull-datum coordinate system

---

## Moth-Specific Pitfalls

1. **Aux/Intermediate Extraction**: When extracting intermediates from a model for auxiliary logging, ensure ALL inputs (including environmental effects like waves) are replicated. It's easy to miss conditional inputs buried inside component methods.

2. **SLSQP ill-conditioning**: SLSQP stalls at ~5 iterations when objective gradients are O(1e5), which happens with scale-aware objectives. Use L-BFGS-B as primary optimizer and SLSQP only for polish. The trim solver uses this two-phase approach.

3. **JAX vs SciPy FD for small problems**: For optimization with ~4 decision variables, SciPy finite-difference gradients are faster than JAX eager-mode gradients (~2.5x). JAX overhead dominates at small problem sizes.

4. **NED sign errors in report writing**: When interpreting simulation plots, pos_d becoming more negative means the boat is RISING (higher altitude), not sinking. This is counterintuitive and has caused systematic sign errors in past reports. Always use `/generate-plot-interpretation-report` for plot interpretation — it has NED sign checks built in. Never write plot interpretations without it. Cross-check: if depth factors are dropping, the boat is rising (foils emerging); if hull buoyancy activates, the boat is sinking (hull entering water).

5. **Timestep mismatch**: Controllers designed at one dt may not work at another. Use `validate_simulation_dt()` before simulation. Default Moth dt is 5ms (`MOTH_DEFAULT_DT`). The Moth model has speed-dependent eigenvalues that constrain the maximum stable RK4 dt — at higher speeds you may need dt well below 5ms. Always check `docs/timestep_guide.md` for eigenvalue tables before choosing a dt for Moth simulations.

6. **Hull-datum vs body-frame positions**: MothParams stores component positions in hull-datum coordinates (x aft from bow, z up from hull bottom) and derives body-frame positions via `hull_datum_to_body()`. Never pass hull-datum values where body-frame is expected or vice versa. The conversion is: `body_x = hull_cg_from_bow - hull_x`, `body_z = hull_cg_above_bottom - hull_z`. See `docs/moth_vertical_geometry.md`.

7. **Mutable numpy array defaults in attrs**: Never use `default=np.array(...)` for attrs fields — `np.asarray` on matching dtype returns the same object, so all instances share one array. Use `factory=lambda: np.array(...)` instead.

8. **Moth3D doesn't store MothParams**: The Moth3D equinox module unpacks MothParams fields during `__init__` and does not retain the original object. If you need the params downstream (e.g., for geometry plotting), pass the MothParams separately rather than trying to retrieve it from the model.

9. **Moth eigenvalue structure**: The linearized Moth3D has 2 fast stable real eigenvalues, 1 stable real (heave), 1 near-zero real (frozen surge), and 1 unstable real (pitch divergence). The unstable eigenvalue grows monotonically with speed. Added mass/inertia effect on eigenvalues is <1%. See `docs/physical_intuition_guide.md` and `tests/simulator/moth/test_damping.py` for current reference values.

10. **Moth foiling envelope**: With current geometry (MOTH_BIEKER_V3), stable foiling range is ~8-18 m/s. 6 m/s is below takeoff (elevator saturates). 20 m/s diverges in open-loop. Trim thrust is monotonically increasing across the full 6-20 m/s range (post AoA decomposition fix).

## Current Trim Values (MOTH_BIEKER_V3, 2026-03-13)

| Speed (m/s) | theta (°) | pos_d (m) | thrust (N) | RK4 margin |
|-------------|-----------|-----------|------------|------------|
| 8 | +1.80 | -1.446 | 54.1 | 19.2 |
| 10 | +0.82 | -1.409 | 74.8 | 16.0 |
| 12 | +0.28 | -1.427 | 99.3 | 13.6 |
| 14 | -0.04 | -1.454 | 127.5 | 11.6 |
| 16 | -0.24 | -1.465 | 161.9 | 10.2 |

Trim theta decreases monotonically with speed (less nose-up). pos_d stays in -1.41 to -1.47m range (consistent foiling depth). Thrust is monotonically increasing.

## LQR Gain Schedule

- Design points: 8, 10, 12, 14, 16 m/s (5-point schedule)
- All controllable closed-loop eigenvalues have |eig| < 1.0 (stable)
- Surge mode (state[4]) is uncontrollable: K[:,4] = [0,0], |eig| = 1.0
- Max closed-loop eigenvalue magnitude: 0.979 (at 8 m/s)
- `MothGainScheduledController` linearly interpolates K, x_trim, u_trim by speed
- Clamped to nearest design point outside [8, 16] m/s range

## Report Writing

When writing interpretation reports for Moth simulations:

- **Use hull height** (bow/stern height above water, positive = above) instead of pos_d. The `moth_open_loop.py` script plots this automatically. At trim (10 m/s): bow ≈ 0.48m, stern ≈ 0.46m above water.
- **Focus on foiling-regime dynamics** — the first 1-2s of perturbation response (pitch-heave coupling, non-minimum-phase behavior, force balance) is where the model physics are most accurate. This is the physically important part.
- **Ventilation and hull contact are rough boundary models** — note them briefly (e.g., "the ventilation boundary arrests the rise at bow height ~0.55m") but don't analyze those regimes in detail.
- **Surge speed changes** are dominated by gravity body-frame projection (`gravity_fx = -mg·sin(θ)`), not aerodynamic frontal area changes. Strut immersion drag (more strut in water → more drag) is secondary. Never say "reduced drag at deeper submergence" — strut drag increases with immersion.

## Trim Solver Behavior

- Trim residuals are speed-dependent: < 0.01 at 12+ m/s, 0.01-0.08 at 8-10 m/s, > 0.1 at 6-7 m/s. Low-speed residuals are inherently higher due to less dynamic pressure and higher AoA requirements.
- At steady trim, body-frame flow AoA at the rudder is approximately `2*theta + elevator` (not `theta + elevator`) because `w = u*tan(theta) ~ u*theta` contributes an additional theta term.
- Multistart should always include the default guess as one of the starts, not just alternative seeds.
- The CG-to-main-foil offset is the critical geometry variable for pitch balance. Current preset has 0.42m offset (hull_cg_from_bow=1.99, main_foil_from_bow=1.57).
- hull_contact_depth is computed dynamically at runtime in the JAX model, tracking the actual system CG as the sailor moves. The CasADi model still uses a static baked-in value (pending migration).

## Moth Timestep Conventions

- Default Moth simulation/control dt: **5ms** (`MOTH_DEFAULT_DT = 0.005`)
- At 10 m/s, max stable RK4 dt is 9.6ms; 5ms provides 50% margin
- At speeds >17 m/s, use dt < 4.7ms (e.g., `dt=0.002`)
- LQR, EKF, LQG all default to 5ms; MPC uses 20ms control interval with M=4 internal sub-steps (5ms effective)
- Trim solver is dt-independent (continuous-time equilibria)
- See [docs/timestep_guide.md](../timestep_guide.md) for eigenvalue tables and timing architecture
