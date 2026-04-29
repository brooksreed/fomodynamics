# Moth Sailboat Modeling Notes

This document collects technical notes and decisions for the Moth sailboat model. It serves as a working document for modeling choices - not a complete specification.

For the overall project vision and phased implementation plan, see [moth_simulation_vision.md](moth_simulation_vision.md).

---

## Topics

### Inertia Tensor Simplifications

For a "start simple" model of a Moth: **No, you do not need a full 3×3 inertia tensor**.

Assume a **diagonal** inertia tensor \((I_{xx}, I_{yy}, I_{zz})\) and set the off-diagonal products of inertia to zero. Below is the breakdown of why, plus the one exception/gotcha to watch out for.

#### 1. Symmetry kills the off-diagonals

In rigid-body dynamics, the off-diagonal terms \((I_{xy}, I_{xz}, I_{yz})\) represent mass asymmetry.

- **\(I_{xy}\) and \(I_{yz}\)**: because the boat is symmetric port/starboard (about the \(x\!-\!z\) plane), these terms are **zero by definition** (assuming the sailor is centered or handled separately).
- **\(I_{xz}\)**: this is the only term that can plausibly exist. It represents asymmetry in the vertical/longitudinal plane. Since the Moth has a high rig and a deep foil, the principal axis of inertia may be tilted slightly relative to the waterline.

#### 2. Why you can ignore \(I_{xz}\) (for now)

While \(I_{xz}\) may be non-zero, it is typically negligible compared to the dominant couplings in the problem:

- **Inertial coupling**: a non-zero \(I_{xz}\) creates torque coupling where a pure yaw acceleration \((\dot{r})\) induces a roll torque, and vice versa.
- **Force coupling (dominant)**: in a Moth, the physical coupling is massive. The rudder is ~1.5 m underwater and the sail force acts ~4 m in the air. A rudder input creates a large roll moment simply due to the lever arm of the force—usually orders of magnitude larger than the subtle inertial coupling from \(I_{xz}\).

#### 3. The "gotcha": the sailor is ~2/3 of the mass

The most important reason to avoid a complex 3×3 tensor for the whole system is that the system isn't rigid.

The sailor (~80 kg) moves relative to the hull (~45 kg). If you try to bake the sailor into a single static inertia tensor, your \(I_{xx}\) and \(I_{zz}\) will constantly change, and products of inertia like \(I_{xy}\) can swing wildly as the sailor hikes.

**Better approach ("multibody lite")**

Instead of a complex tensor, use superposition:

- **Boat**: fixed, diagonal tensor (constant).
- **Sailor**: point mass at position \((x_s, y_s, z_s)\) (time-varying).

This naturally captures the "off-diagonal" effects dynamically: when the sailor hikes out, the point-mass geometry automatically produces the correct coupling terms without you having to maintain a full time-varying 3×3 inertia matrix.

#### Summary

Use this matrix:

$$
\mathbf{I}_{boat} =
\begin{bmatrix}
I_{xx} & 0 & 0 \\
0 & I_{yy} & 0 \\
0 & 0 & I_{zz}
\end{bmatrix}
$$

- **\(I_{xx}\) (roll)**: low (hull is narrow).
- **\(I_{yy}\) (pitch)**: high (hull is long).
- **\(I_{zz}\) (yaw)**: high (hull is long).

Then add the sailor separately (e.g., as forces/moments about the CG, or as a moving point mass contributing inertia via geometry). This keeps the model simple but physically meaningful for a hiking boat.

---

### Sailor Body Modeling

See [sailor_mass_modeling.md](sailor_mass_modeling.md) for full analysis of modeling approaches, corrected EOM derivations, and recommended progression.

Key summary:
- Sailor mass dominates system (~65% of total: 80 kg vs 45 kg hull)
- Body frame origin at hull CG; system CG shifts with sailor position
- Recommended approach: "multibody lite" — point mass sailor with composite inertia via parallel axis theorem
- v1 fix: static CG correction (adjust moment arms and inertia at construction)
- Future: time-varying sailor position via schedule (analogous to `u_forward_schedule`)

---

### Trim Solver

See [trim_solver.md](trim_solver.md) for the trim solver technical reference: objective function structure, weight tuning, kinematic constraints, thrust calibration architecture, and typical results.

---

### Foil Force Decomposition (alpha_geo vs alpha_eff)

The foil force model separates two physically distinct angles of attack:

- **alpha_geo** (geometric AoA): the actual flow direction at the foil, computed as `arctan2(w_local, u_safe)`. Used to rotate the lift and drag vectors into body frame. Does NOT include control surface deflections.
- **alpha_eff** (effective AoA): the angle the foil "sees" for polar lookup, computed as `control_deflection + w_local / u_safe`. Includes flap/elevator deflection. Determines lift and drag magnitudes through the CL/CD polar.

The key physical insight: a control surface deflection (flap or elevator) changes how much force the foil produces, but does not change the direction of the incoming flow. The old model conflated these, using a single `aoa` for both polar lookup and force rotation. This created artificial coupling where control inputs changed force direction.

The full body-frame force decomposition:
```
fx = -drag * cos(alpha_geo) + lift * sin(alpha_geo)
fz = -drag * sin(alpha_geo) - lift * cos(alpha_geo)
```

Both lift and drag are rotated by alpha_geo. The `+lift * sin(alpha_geo)` term on fx is the "lift-forward-tilt" thrust contribution during foiling (positive alpha_geo from positive theta/w). This is the dominant mechanism that reduces required sail thrust — the foils partially propel the boat forward through lift-forward-tilt.

See `docs/plans/foil_force_decomposition_fix_20260311/foil_force_decomposition_design_doc.md` for the detailed derivation and `docs/moth_3dof_equations.md` Section 4.1-4.2 for the full equations.

---

### Future Topics

Topics to document as modeling progresses:

- Sail force model choices
- Ventilation behavior near surface
- Wand system dynamics
