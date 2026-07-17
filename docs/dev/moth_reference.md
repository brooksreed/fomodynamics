# Moth Model Reference

Moth-specific pitfalls, trim solver behavior, and timestep conventions.

**Related docs:**
- [../physical_intuition_guide.md](../physical_intuition_guide.md) — physics intuition, debugging dynamics
- [../timestep_guide.md](../timestep_guide.md) — eigenvalue tables, RK4 stability
- [../moth_vertical_geometry.md](../moth_vertical_geometry.md) — hull-datum coordinate system

---

## Moth-Specific Pitfalls

1. **Aux/Intermediate Extraction**: When extracting intermediates from a model for auxiliary logging, ensure ALL inputs (including environmental effects like waves) are replicated. It's easy to miss conditional inputs buried inside component methods.

2. **(Historical, retired SciPy solver)** SLSQP ill-conditioning: SLSQP stalls at ~5 iterations when objective gradients are O(1e5), which happens with scale-aware objectives. This applied to the old SciPy-based trim solver (L-BFGS-B primary, SLSQP polish); the current trim solver (`fmd.simulator.trim_casadi`) uses CasADi/IPOPT with a different two-phase strategy (penalty phase → hard-constraint phase) — see `docs/trim_solver.md`.

3. **JAX vs SciPy FD for small problems**: For optimization with ~4 decision variables, SciPy finite-difference gradients are faster than JAX eager-mode gradients (~2.5x). JAX overhead dominates at small problem sizes.

4. **NED sign errors in report writing**: When interpreting simulation plots, pos_d becoming more negative means the boat is RISING (higher altitude), not sinking. This is counterintuitive and has caused systematic sign errors in past reports. Always use `/generate-plot-interpretation-report` for plot interpretation — it has NED sign checks built in. Never write plot interpretations without it. Cross-check: if depth factors are dropping, the boat is rising (foils emerging); if hull buoyancy activates, the boat is sinking (hull entering water).

5. **Timestep mismatch**: Controllers designed at one dt may not work at another. Use `validate_simulation_dt()` before simulation. Default Moth dt is 5ms (`MOTH_DEFAULT_DT`). The Moth model has speed-dependent eigenvalues that constrain the maximum stable RK4 dt — at higher speeds you may need dt well below 5ms. Always check `docs/timestep_guide.md` for eigenvalue tables before choosing a dt for Moth simulations.

6. **Hull-datum vs body-frame positions**: MothParams stores component positions in hull-datum coordinates (x aft from bow, z up from hull bottom) and derives body-frame positions via `hull_datum_to_body()`. Never pass hull-datum values where body-frame is expected or vice versa. The conversion is: `body_x = hull_cg_from_bow - hull_x`, `body_z = hull_cg_above_bottom - hull_z`. See `docs/moth_vertical_geometry.md`.

7. **Mutable numpy array defaults in attrs**: Never use `default=np.array(...)` for attrs fields — `np.asarray` on matching dtype returns the same object, so all instances share one array. Use `factory=lambda: np.array(...)` instead.

8. **Moth3D doesn't store MothParams**: The Moth3D equinox module unpacks MothParams fields during `__init__` and does not retain the original object. If you need the params downstream (e.g., for geometry plotting), pass the MothParams separately rather than trying to retrieve it from the model.

9. **Moth eigenvalue structure**: The linearized Moth3D has 2 fast stable real eigenvalues, 1 stable real (heave), 1 near-zero real (frozen surge), and 1 unstable real (pitch divergence). The unstable eigenvalue grows monotonically with speed. Added mass/inertia effect on eigenvalues is <1%. See `docs/physical_intuition_guide.md` and `tests/simulator/moth/test_damping.py` for current reference values.

10. **Moth foiling envelope**: With current geometry (MOTH_BIEKER_V3), stable foiling range is ~8-18 m/s. 6 m/s is below takeoff (elevator saturates). 20 m/s diverges in open-loop. Trim thrust is monotonically increasing across the full 6-20 m/s range (post AoA decomposition fix).

## Current Trim Values (MOTH_BIEKER_V3, post physics-correctness batch, 2026-07-16)

| Speed (m/s) | theta (°) | pos_d (m) | thrust (N) |
|-------------|-----------|-----------|------------|
| 8 | +1.836 | -1.400 | 56.3 |
| 10 | +0.836 | -1.400 | 75.5 |
| 12 | +0.297 | -1.400 | 102.1 |
| 14 | -0.026 | -1.400 | 134.9 |
| 16 | -0.225 | -1.398 | 173.8 |

Trim theta decreases monotonically with speed (less nose-up). pos_d now stays essentially pinned at `DEFAULT_POS_D_REF` (-1.40m) — the C1.E trim-null regularization plus C1.G recalibration replaced the old free-solve pos_d spread (-1.41 to -1.47m). Thrust is monotonically increasing (RK4-margin column dropped: it went stale with the trim/timestep changes above and needs regeneration via `docs/timestep_guide.md`'s methodology, not hand-copied here).

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

- **(Historical, retired SciPy solver)** Trim residuals were speed-dependent: < 0.01 at 12+ m/s, 0.01-0.08 at 8-10 m/s, > 0.1 at 6-7 m/s, with low-speed residuals inherently higher due to less dynamic pressure and higher AoA requirements. The current CasADi/IPOPT solver's hard-constraint phase converges to `max(|xdot|) < 1e-8` at all 15 calibration speeds (6-20 m/s) — see `docs/trim_solver.md` "Typical Results".
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
