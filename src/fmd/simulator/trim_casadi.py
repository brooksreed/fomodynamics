"""CasADi/IPOPT trim solver for Moth 3DOF longitudinal dynamics.

Two-phase solver:
  Phase 1 (penalty): min sum((xdot/scale)^2) + regularization  (robust, relaxed tol)
  Phase 2 (hard constraint): min regularization s.t. xdot=0     (warm start, exact feasibility)

Decision variables (8-vector z):
    z = [pos_d, theta, w, q, u, main_flap, rudder_elevator, thrust]

Fixed via bounds: q = 0 (steady state), u = u_target (speed pin).

Phase 1 finds a good initial point via penalty formulation with characteristic-
value scaling, then Phase 2 polishes it with hard equality constraints for exact
feasibility. Phase 2 drops the theta_dot constraint (which equals q, pinned to 0
by bounds) to avoid a rank-deficient Jacobian.

The quadratic thrust penalty avoids the zero-thrust attractor that plagues
linear min(thrust) formulations. At thrust=0 the gradient is zero, so there
is no force pulling the optimizer toward infeasible zero-thrust.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import attrs
import casadi as cs
import numpy as np

from fmd.simulator.casadi.moth_3d import Moth3DCasadiExact
from fmd.simulator.moth_3d import MAIN_FLAP_MAX, MAIN_FLAP_MIN
from fmd.simulator.params.moth import MothParams

_STATE_NAMES = ["pos_d", "theta", "w", "q", "u"]
_XDOT_NAMES = ["pos_d_dot", "theta_dot", "w_dot", "q_dot", "u_dot"]
_Z_NAMES = ["pos_d", "theta", "w", "q", "u", "main_flap", "rudder_elevator", "thrust"]
_Z_UNITS = ["m", "rad", "m/s", "rad/s", "m/s", "rad", "rad", "N"]
_ANGLE_VARS = {1, 5, 6}  # indices that are angles (for deg display)
_VIOLATION_THRESHOLD = 1e-4  # constraint violation threshold for flagging

# Scale-independent objective weights (encode relative importance, not scaling).
# These must be small enough that the residual term dominates at convergence.
# At a good trim point, the scaled residual is O(1e-12). The regularization
# terms are O(w * (z/scale)^2) ~ w. With w = 1e-8, the regularization is
# negligible compared to any residual above 1e-6, ensuring physical accuracy.
_W_THRUST = 1e-8  # Phase 1: gentle thrust preference (must not dominate residual)
_W_CTRL = 1e-8    # Phase 1: light control uniqueness

# Phase 2 weights: regularization in the null space of xdot=0.
# These only affect which feasible point IPOPT chooses, not feasibility itself.
_W_HARD_THRUST = 1e-2   # reference-centered: min (thrust - T_ref)^2 / scale^2
_W_HARD_CTRL = 1e-3     # light control regularization


@dataclass(frozen=True)
class CharacteristicScales:
    """Characteristic values for scaling the trim NLP.

    Defined in human-friendly units, converted to SI via properties.
    The optimizer treats a 1-characteristic-value deviation as O(1).
    """

    # Decision variable scales
    theta_deg: float = 1.0        # pitch
    flap_deg: float = 3.0         # main flap
    elev_deg: float = 2.0         # rudder elevator
    pos_d_m: float = 0.05         # ride height
    thrust_N: float = 100.0       # thrust
    w_ms: float = 0.1             # heave velocity

    # State derivative scales (xdot)
    pos_d_dot_ms: float = 0.05
    theta_dot_rads: float = 0.035
    w_dot_ms2: float = 0.5
    q_dot_rads2: float = 0.35
    u_dot_ms2: float = 0.2

    @property
    def theta_rad(self) -> float:
        return np.deg2rad(self.theta_deg)

    @property
    def flap_rad(self) -> float:
        return np.deg2rad(self.flap_deg)

    @property
    def elev_rad(self) -> float:
        return np.deg2rad(self.elev_deg)

    @property
    def xdot_scale(self) -> np.ndarray:
        """Scale for each of the 5 xdot components [pos_d_dot, theta_dot, w_dot, q_dot, u_dot]."""
        return np.array([
            self.pos_d_dot_ms,
            self.theta_dot_rads,
            self.w_dot_ms2,
            self.q_dot_rads2,
            self.u_dot_ms2,
        ])

    @property
    def z_scale(self) -> np.ndarray:
        """Scale for each of the 8 decision variables [pos_d, theta, w, q, u, flap, elev, thrust].

        Public API for external callers (diagnostics, analysis scripts).
        The internal solver uses individual scale fields directly in _build_nlp
        (Phase 1 objective scaling) and _build_nlp_hard (Phase 2 control regularization).
        """
        return np.array([
            self.pos_d_m,
            self.theta_rad,
            self.w_ms,
            1.0,       # q is pinned to 0
            1.0,       # u is pinned to target
            self.flap_rad,
            self.elev_rad,
            self.thrust_N,
        ])


DEFAULT_SCALES = CharacteristicScales()


@dataclass
class PhaseInfo:
    """Per-phase solver statistics."""

    phase: str          # "penalty" or "hard_constraint"
    status: str         # IPOPT return_status
    iterations: int
    wall_time_s: float
    objective: float
    max_residual: float


@dataclass
class CasadiTrimResult:
    """Result from CasADi trim solver."""

    state: np.ndarray  # [pos_d, theta, w, q, u]
    control: np.ndarray  # [main_flap, rudder_elevator]
    thrust: float
    residual: float  # max(|xdot|) at solution
    solve_time: float  # wall-clock seconds for entire solve
    success: bool
    iter_count: int = -1  # IPOPT iteration count
    ipopt_stats: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    phases: list[PhaseInfo] = field(default_factory=list)

    def format_diagnostics(self) -> str:
        """Format infeasibility diagnostics as a human-readable string."""
        if not self.diagnostics:
            return "No diagnostics available."

        lines = []
        status = self.ipopt_stats.get("return_status", "Unknown")
        u_target = self.diagnostics.get("u_target", "?")
        lines.append(
            f"Trim solver: {status} at u={u_target} m/s"
            f" (iter={self.iter_count}, {self.solve_time:.3f}s)"
        )
        lines.append("")

        # Constraint violations (from xdot values)
        g_vals = self.diagnostics.get("constraint_values")
        if g_vals is not None:
            lines.append("  State derivative residuals (xdot):")
            for i, name in enumerate(_XDOT_NAMES):
                val = g_vals[i]
                flag = " ✓" if abs(val) < _VIOLATION_THRESHOLD else " ← LARGE"
                lines.append(f"    {name:>12s}: {val:+12.6f}{flag}")
            lines.append("")

        # Active bounds
        active = self.diagnostics.get("active_bounds")
        if active:
            lines.append("  Active bounds:")
            for entry in active:
                name = entry["variable"]
                val = entry["value"]
                bound_type = entry["bound"]
                bound_val = entry["bound_value"]
                lam = entry["multiplier"]
                idx = _Z_NAMES.index(name) if name in _Z_NAMES else -1
                if idx in _ANGLE_VARS:
                    val_str = f"{np.rad2deg(val):.1f}°"
                    bnd_str = f"{np.rad2deg(bound_val):.1f}°"
                elif _Z_UNITS[idx] == "N" if idx >= 0 else False:
                    val_str = f"{val:.1f} N"
                    bnd_str = f"{bound_val:.1f} N"
                else:
                    val_str = f"{val:.4f}"
                    bnd_str = f"{bound_val:.4f}"
                lines.append(
                    f"    {name:>18s}: {val_str} at {bound_type} bound"
                    f" ({bnd_str}), λ={lam:+.2e}"
                )
            lines.append("")

        # Final iterate
        z_final = self.diagnostics.get("z_final")
        if z_final is not None:
            lines.append("  Final iterate:")
            for i, (name, unit) in enumerate(zip(_Z_NAMES, _Z_UNITS)):
                val = z_final[i]
                if i in _ANGLE_VARS:
                    lines.append(f"    {name:>18s}: {np.rad2deg(val):+8.3f}° ({val:+.6f} rad)")
                elif unit == "N":
                    lines.append(f"    {name:>18s}: {val:+10.3f} N")
                else:
                    lines.append(f"    {name:>18s}: {val:+10.6f} {unit}")

        # Bounds summary
        lbz = self.diagnostics.get("lbz")
        ubz = self.diagnostics.get("ubz")
        if lbz is not None and ubz is not None:
            lines.append("")
            lines.append("  Variable bounds:")
            for i, name in enumerate(_Z_NAMES):
                lb, ub = lbz[i], ubz[i]
                if i in _ANGLE_VARS:
                    lines.append(
                        f"    {name:>18s}: [{np.rad2deg(lb):+8.2f}°, {np.rad2deg(ub):+8.2f}°]"
                    )
                else:
                    lines.append(
                        f"    {name:>18s}: [{lb:+10.4f}, {ub:+10.4f}] {_Z_UNITS[i]}"
                    )

        return "\n".join(lines)


def _create_model(
    params: MothParams,
    heel_angle: float,
    ventilation_mode: str,
    ventilation_threshold: float,
) -> Moth3DCasadiExact:
    """Create zero-thrust CasADi model for trim solving."""
    p0 = attrs.evolve(
        params,
        sail_thrust_coeff=1e-10,
        sail_thrust_speeds=(),
        sail_thrust_values=(),
    )
    return Moth3DCasadiExact(
        p0,
        heel_angle=heel_angle,
        ventilation_mode=ventilation_mode,
        ventilation_threshold=ventilation_threshold,
        surge_enabled=True,
    )


def _build_xdot_expr(model: Moth3DCasadiExact, z: cs.SX) -> cs.SX:
    """Build thrust-corrected xdot symbolic expression from decision variables z.

    The thrust is applied in the NED horizontal plane and rotated to body frame
    by pitch angle theta, producing both body-x and body-z components:
        F_bx = thrust * cos(theta)
        F_bz = thrust * sin(theta)

    The moment uses the full cross product with the sail CE position:
        M_y = ce_z * F_bx - ce_x * F_bz
    """
    pos_d, theta, w, q, u = z[0], z[1], z[2], z[3], z[4]
    main_flap, rudder_elevator, thrust = z[5], z[6], z[7]

    x = cs.vertcat(pos_d, theta, w, q, u)
    ctrl = cs.vertcat(main_flap, rudder_elevator)
    xdot = model.forward_dynamics(x, ctrl)

    ce_x_eff = model.sail_ce_position_x - model.cg_offset_x
    ce_z_eff = model.sail_ce_position_z - model.cg_offset_z
    i_eff = model.iyy + model.added_inertia_pitch
    m_eff_heave = model.total_mass + model.added_mass_heave
    m_eff_surge = model.total_mass + model.added_mass_surge

    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)

    # NED-consistent thrust corrections:
    # Body-frame force components from NED horizontal thrust
    f_bx = thrust * cos_theta
    f_bz = thrust * sin_theta

    return xdot + cs.vertcat(
        0, 0,
        f_bz / m_eff_heave,                           # w_dot
        (ce_z_eff * f_bx - ce_x_eff * f_bz) / i_eff,  # q_dot
        f_bx / m_eff_surge,                             # u_dot
    )


def _variable_bounds(
    params: MothParams,
    u_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute variable bounds for the 8-vector z. Thrust lower bound is 0 (no positive minimum floor)."""
    lbz = np.array([
        -3.0,                           # pos_d (overwritten below)
        np.deg2rad(-20.0),              # theta
        -5.0,                           # w
        0.0,                            # q: pinned to 0
        u_target,                       # u: pinned to target
        float(MAIN_FLAP_MIN),          # main_flap
        float(params.rudder_elevator_min),  # rudder_elevator
        0.0,                            # thrust: non-negative
    ])
    ubz = np.array([
        0.5,                            # pos_d (overwritten below)
        np.deg2rad(20.0),               # theta
        5.0,                            # w
        0.0,                            # q: pinned to 0
        u_target,                       # u: pinned to target
        float(MAIN_FLAP_MAX),          # main_flap
        float(params.rudder_elevator_max),  # rudder_elevator
        500.0,                          # thrust
    ])

    # Foiling regime bounds on pos_d
    hull_contact_depth = float(params.hull_contact_depth)
    ubz[0] = -(hull_contact_depth + 0.05)  # hull above water + 5cm margin

    main_foil_depth_from_sys_cg = (
        float(params.main_foil_position[2]) - float(params.combined_cg_offset[2])
    )
    lbz[0] = -(main_foil_depth_from_sys_cg + 0.5)  # 50cm beyond foil depth

    return lbz, ubz


def _geometry_initial_guess(
    params: MothParams,
    u_target: float,
    heel_angle: float,
    target_theta: float | None = None,
    target_pos_d: float | None = None,
) -> np.ndarray:
    """Compute initial guess from foil geometry.

    Sets pos_d where the main foil's leeward tip is at the water surface,
    then lowers the boat 5cm for more submergence. When targets are set,
    uses them instead of defaults.
    """
    eff_z = float(params.main_foil_position[2])
    tip_rise = (float(params.main_foil_span) / 2) * np.sin(heel_angle)
    pos_d_guess = tip_rise - eff_z * np.cos(heel_angle) + 0.05

    if target_pos_d is not None:
        pos_d_guess = target_pos_d

    theta_guess = 0.0
    if target_theta is not None:
        theta_guess = target_theta

    w_guess = u_target * np.tan(theta_guess)

    return np.array([
        pos_d_guess,
        theta_guess,
        w_guess,
        0.0,            # q = 0 (steady state)
        u_target,       # u = target speed
        0.05,           # flap = 0.05 rad (~3 deg)
        0.02,           # elevator = 0.02 rad (~1 deg)
        100.0,          # thrust = 100 N
    ])


def _build_nlp(
    model: Moth3DCasadiExact,
    params: MothParams,
    u_target: float,
    scales: CharacteristicScales = DEFAULT_SCALES,
) -> tuple:
    """Build Phase 1 penalty NLP (relaxed tol, robust convergence).

    Objective (all terms O(1) at characteristic deviations):
        J = sum((xdot_i / xdot_scale_i)^2)                           # residual
          + W_THRUST * (thrust / thrust_scale)^2                      # quadratic thrust reg
          + W_CTRL * ((flap / flap_scale)^2 + (elev / elev_scale)^2)  # control reg

    Returns (solver, lbz, ubz, f_xdot).
    """
    z = cs.SX.sym("z", 8)
    xdot = _build_xdot_expr(model, z)

    xdot_sc = cs.DM(scales.xdot_scale)
    scaled_xdot = xdot / xdot_sc
    residual_obj = cs.dot(scaled_xdot, scaled_xdot)

    thrust_obj = _W_THRUST * (z[7] / scales.thrust_N) ** 2
    ctrl_obj = _W_CTRL * (
        (z[5] / scales.flap_rad) ** 2 + (z[6] / scales.elev_rad) ** 2
    )

    objective = residual_obj + thrust_obj + ctrl_obj

    lbz, ubz = _variable_bounds(params, u_target)

    nlp = {"x": z, "f": objective}
    opts = {
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 300,
        "ipopt.print_level": 0,
        "print_time": 0,
    }
    solver = cs.nlpsol("trim", "ipopt", nlp, opts)
    f_xdot = cs.Function("f_xdot", [z], [xdot], ["z"], ["xdot"])

    return solver, lbz, ubz, f_xdot


def _build_nlp_hard(
    model: Moth3DCasadiExact,
    params: MothParams,
    u_target: float,
    scales: CharacteristicScales = DEFAULT_SCALES,
) -> tuple:
    """Build Phase 2 hard-constraint NLP with reference-centered thrust.

    Constraints: xdot[0,2,3,4] = 0  (4 equality, drop theta_dot = q which is pinned)
    Objective: W * (thrust - T_ref)^2 / scale^2 + W_ctrl * controls^2 / scale^2

    The thrust term is centered on Phase 1's thrust (T_ref), not on zero.
    This avoids the zero-thrust attractor that occurs with absolute min(thrust^2),
    while keeping Phase 2 near Phase 1's min-thrust operating point.

    T_ref is a CasADi parameter so the NLP only needs to be built once.

    Returns (solver, f_xdot).
    """
    z = cs.SX.sym("z", 8)
    T_ref = cs.SX.sym("T_ref")  # Phase 1 thrust (parameter, not optimized)
    xdot = _build_xdot_expr(model, z)

    # Drop theta_dot (index 1) — it equals q which is already pinned to 0 by bounds.
    # Including it makes the Jacobian rank-deficient.
    # Unscaled constraints: let IPOPT handle scaling internally via nlp_scaling_method.
    g = cs.vertcat(xdot[0], xdot[2], xdot[3], xdot[4])

    # Phase 2 objective: reference-centered thrust + control regularization.
    # Using (thrust - T_ref)^2 instead of thrust^2 avoids the zero-thrust attractor.
    # At T_ref (the Phase 1 solution), the gradient is zero, so IPOPT has no incentive
    # to move thrust away from the Phase 1 value. It only adjusts thrust as needed
    # to satisfy the hard constraints exactly.
    objective = (
        _W_HARD_THRUST * ((z[7] - T_ref) / scales.thrust_N) ** 2
        + _W_HARD_CTRL * (
            (z[5] / scales.flap_rad) ** 2 + (z[6] / scales.elev_rad) ** 2
        )
    )

    nlp = {"x": z, "f": objective, "g": g, "p": T_ref}
    opts = {
        "ipopt.tol": 1e-8,
        "ipopt.max_iter": 500,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-8,
        "ipopt.warm_start_mult_bound_push": 1e-8,
        "ipopt.mu_init": 1e-6,
    }
    solver = cs.nlpsol("trim_hard", "ipopt", nlp, opts)
    f_xdot = cs.Function("f_xdot_hard", [z], [xdot], ["z"], ["xdot"])

    return solver, f_xdot


def _extract_diagnostics(
    z_opt: np.ndarray,
    xdot_val: np.ndarray,
    lbz: np.ndarray,
    ubz: np.ndarray,
    u_target: float,
    lam_x: np.ndarray | None = None,
    bound_tol: float = 1e-6,
) -> dict:
    """Extract structured diagnostics from IPOPT solution."""
    active_bounds = []
    for i, name in enumerate(_Z_NAMES):
        at_lower = abs(z_opt[i] - lbz[i]) < bound_tol
        at_upper = abs(z_opt[i] - ubz[i]) < bound_tol
        multiplier = lam_x[i] if lam_x is not None else 0.0
        if at_lower and abs(multiplier) > 1e-8:
            active_bounds.append({
                "variable": name,
                "value": z_opt[i],
                "bound": "lower",
                "bound_value": lbz[i],
                "multiplier": multiplier,
            })
        if at_upper and abs(multiplier) > 1e-8:
            active_bounds.append({
                "variable": name,
                "value": z_opt[i],
                "bound": "upper",
                "bound_value": ubz[i],
                "multiplier": multiplier,
            })

    active_bounds.sort(key=lambda e: abs(e["multiplier"]), reverse=True)

    return {
        "u_target": u_target,
        "constraint_values": xdot_val,
        "z_final": z_opt,
        "lbz": lbz,
        "ubz": ubz,
        "active_bounds": active_bounds,
    }


def _sanity_check(
    z_opt: np.ndarray, lbz: np.ndarray, ubz: np.ndarray, u_target: float,
) -> list[str]:
    """Post-solve sanity checks. Returns warnings (not errors)."""
    warnings = []
    pos_d, theta, _w, _q, u = z_opt[0], z_opt[1], z_opt[2], z_opt[3], z_opt[4]
    flap, elev, thrust = z_opt[5], z_opt[6], z_opt[7]

    if abs(u - u_target) > 1e-6:
        warnings.append(f"u deviates from target: u={u:.6f}, u_target={u_target:.6f}")
    if thrust < 10.0 or thrust > 400.0:
        warnings.append(f"Thrust outside [10, 400] N: {thrust:.1f} N")
    if abs(theta) > np.deg2rad(10.0):
        warnings.append(f"|theta| > 10 deg: {np.rad2deg(theta):.2f} deg")
    if pos_d > -0.1 or pos_d < -2.0:
        warnings.append(f"pos_d outside [-2.0, -0.1] m: {pos_d:.4f} m")
    if abs(flap - lbz[5]) < 1e-4 or abs(flap - ubz[5]) < 1e-4:
        warnings.append(f"Flap at bound: {np.rad2deg(flap):.2f} deg")
    if abs(elev - lbz[6]) < 1e-4 or abs(elev - ubz[6]) < 1e-4:
        warnings.append(f"Elevator at bound: {np.rad2deg(elev):.2f} deg")
    # Kinematic consistency: w = u * tan(theta)
    w_expected = u * np.tan(theta)
    if abs(_w - w_expected) > 1e-4:
        warnings.append(
            f"Kinematic inconsistency: |w - u*tan(theta)| = {abs(_w - w_expected):.6f}"
        )

    return warnings


def find_casadi_trim(
    params: MothParams,
    u_target: float,
    heel_angle: float = np.deg2rad(30.0),
    ventilation_mode: str = "smooth",
    ventilation_threshold: float = 0.30,
    scales: CharacteristicScales = DEFAULT_SCALES,
    target_theta: float | None = None,
    target_pos_d: float | None = None,
    z0: np.ndarray | None = None,
    fixed_controls: dict[str, float] | None = None,
) -> CasadiTrimResult:
    """Find trim equilibrium using two-phase CasADi/IPOPT solver.

    Phase 1 (penalty): Robust convergence with relaxed tolerance.
    Phase 2 (hard constraint): Warm-started from Phase 1 for exact feasibility.

    Args:
        params: Moth parameters (thrust table will be zeroed internally).
        u_target: Target forward speed (m/s).
        heel_angle: Static heel angle (rad). Default 30 degrees.
        ventilation_mode: "smooth" or "binary".
        ventilation_threshold: Ventilation onset threshold.
        scales: Characteristic scales for NLP normalization.
        target_theta: If set, pin theta to this value (rad).
        target_pos_d: If set, pin pos_d to this value (m).
        z0: Initial guess for decision variables (8-vector). If provided and
            correct length, used instead of geometry-based initial guess.
            Will be clipped to variable bounds.
        fixed_controls: Dict mapping control names to fixed values (rad).
            Supported keys: "main_flap", "rudder_elevator".
            Fixed controls are pinned via bounds (lb=ub=value).

    Returns:
        CasadiTrimResult with state, control, thrust, residual, and warnings.
    """
    t0 = time.monotonic()

    model = _create_model(params, heel_angle, ventilation_mode, ventilation_threshold)

    # Phase 1: penalty formulation (robust, relaxed tol)
    solver1, lbz, ubz, f_xdot1 = _build_nlp(model, params, u_target, scales)

    # Apply target pins to bounds
    if target_theta is not None:
        lbz[1] = target_theta
        ubz[1] = target_theta
    if target_pos_d is not None:
        lbz[0] = target_pos_d
        ubz[0] = target_pos_d

    # Apply fixed control pins
    _CONTROL_INDEX = {"main_flap": 5, "rudder_elevator": 6}
    if fixed_controls is not None:
        for name, value in fixed_controls.items():
            if name not in _CONTROL_INDEX:
                raise ValueError(f"Unknown control name: {name!r}. "
                                 f"Valid: {list(_CONTROL_INDEX.keys())}")
            idx = _CONTROL_INDEX[name]
            lbz[idx] = value
            ubz[idx] = value

    if z0 is not None and len(z0) == 8:
        z0_clipped = np.clip(np.asarray(z0, dtype=float), lbz, ubz)
    else:
        if z0 is not None:
            warnings.warn(
                f"z0 has length {len(z0)}, expected 8. Falling back to geometry guess.",
                stacklevel=2,
            )
        z0_init = _geometry_initial_guess(
            params, u_target, heel_angle,
            target_theta=target_theta, target_pos_d=target_pos_d,
        )
        z0_clipped = np.clip(z0_init, lbz, ubz)

    try:
        sol1 = solver1(x0=z0_clipped, lbx=lbz, ubx=ubz)
        stats1 = solver1.stats()

        z1_opt = np.array(sol1["x"]).flatten()
        xdot1 = np.array(f_xdot1(z1_opt)).flatten()
        residual1 = float(np.max(np.abs(xdot1)))
        status1 = stats1["return_status"]
        iters1 = stats1.get("iter_count", -1)
        obj1 = float(sol1["f"])

    except Exception as e:
        total_time = time.monotonic() - t0
        return CasadiTrimResult(
            state=np.zeros(5),
            control=np.zeros(2),
            thrust=0.0,
            residual=float("inf"),
            solve_time=total_time,
            success=False,
            ipopt_stats={"return_status": f"Exception: {e}"},
            warnings=[f"Solver exception: {e}"],
            phases=[],
        )

    t1 = time.monotonic()
    phase1 = PhaseInfo(
        phase="penalty",
        status=status1,
        iterations=iters1,
        wall_time_s=t1 - t0,
        objective=obj1,
        max_residual=residual1,
    )

    # Phase 2: hard constraints (warm start from Phase 1)
    solver2, f_xdot2 = _build_nlp_hard(model, params, u_target, scales)

    # Default Phase 2 outputs (used if Phase 2 throws)
    z_final = z1_opt
    xdot_final = xdot1
    residual_final = residual1
    status2 = "Phase2NotAttempted"
    iters2 = -1
    obj2 = float("nan")
    lam_x_final = np.array(sol1["lam_x"]).flatten()

    try:
        # Phase 1 has no constraints (pure penalty), so there are no constraint
        # multipliers (lam_g) to warm-start Phase 2. Only bound multipliers (lam_x)
        # are passed. T_ref = Phase 1 thrust for reference-centered objective.
        sol2 = solver2(
            x0=z1_opt, lbx=lbz, ubx=ubz,
            lbg=np.zeros(4), ubg=np.zeros(4),
            lam_x0=np.array(sol1["lam_x"]).flatten(),
            p=float(z1_opt[7]),  # T_ref = Phase 1 thrust
        )
        stats2 = solver2.stats()

        z_final = np.array(sol2["x"]).flatten()
        xdot_final = np.array(f_xdot2(z_final)).flatten()
        residual_final = float(np.max(np.abs(xdot_final)))
        status2 = stats2["return_status"]
        iters2 = stats2.get("iter_count", -1)
        obj2 = float(sol2["f"])
        lam_x_final = np.array(sol2["lam_x"]).flatten()

    except Exception as e:
        status2 = f"Phase2Exception: {e}"

    t2 = time.monotonic()
    phase2 = PhaseInfo(
        phase="hard_constraint",
        status=status2,
        iterations=iters2,
        wall_time_s=t2 - t1,
        objective=obj2,
        max_residual=residual_final,
    )

    # Success: Phase 2 IPOPT converged AND residual acceptable
    success = (
        status2 in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        and residual_final < 1e-6
    )

    # Diagnostics and sanity checks use final (Phase 2) result
    diagnostics = _extract_diagnostics(
        z_final, xdot_final, lbz, ubz, u_target, lam_x_final
    )
    warns = _sanity_check(z_final, lbz, ubz, u_target)

    return CasadiTrimResult(
        state=z_final[:5],
        control=z_final[5:7],
        thrust=float(z_final[7]),
        residual=residual_final,
        solve_time=t2 - t0,
        success=success,
        iter_count=iters1 + max(iters2, 0),
        ipopt_stats={"return_status": status2},
        diagnostics=diagnostics,
        warnings=warns,
        phases=[phase1, phase2],
    )


def find_casadi_trim_sweep(
    params: MothParams,
    speeds: list[float] | np.ndarray,
    **kwargs,
) -> list[CasadiTrimResult]:
    """Run trim solver at multiple speeds.

    Args:
        params: Moth parameters.
        speeds: List of target speeds (m/s).
        **kwargs: Additional arguments passed to find_casadi_trim.

    Returns:
        List of CasadiTrimResult, one per speed.
    """
    return [find_casadi_trim(params, float(u), **kwargs) for u in speeds]


# ---------------------------------------------------------------------------
# SciPy-compatible kwargs that CasADi ignores silently
# ---------------------------------------------------------------------------
_IGNORED_SCIPY_KWARGS = frozenset({
    "prev_trim", "pos_d_guess", "theta_guess", "main_flap_guess",
    "rudder_elevator_guess", "tol", "calibrate_thrust", "thrust_guess",
    "diagnostics", "use_jax_grad", "regularization_weights",
    "u_bounds_margin",
})


def find_moth_trim(
    params: MothParams,
    u_forward: float = 10.0,
    target_theta: float | None = None,
    target_pos_d: float | None = None,
    heel_angle: float | None = None,
    z0: np.ndarray | None = None,
    fixed_controls: dict[str, float] | None = None,
    **kwargs,
) -> CasadiTrimResult:
    """Find moth trim using the CasADi two-phase solver.

    Drop-in replacement for the SciPy ``find_moth_trim``.

    SciPy-specific kwargs (``prev_trim``, ``pos_d_guess``, etc.) are
    accepted but silently ignored since CasADi is robust without them.

    Args:
        params: MothParams instance.
        u_forward: Target forward speed (m/s).
        target_theta: If set, pin theta to this value (rad).
        target_pos_d: If set, pin pos_d to this value (m).
        heel_angle: Static heel angle (rad). If None, uses 30 deg default.
        z0: Initial guess for decision variables (8-vector). If provided,
            used instead of the geometry-based initial guess.
        fixed_controls: Dict mapping control names to fixed values (rad).
            Supported keys: "main_flap", "rudder_elevator".
        **kwargs: Additional arguments. SciPy-specific ones are ignored;
            CasADi-specific ones (``scales``, ``ventilation_mode``, etc.)
            are passed through.

    Returns:
        CasadiTrimResult with state, control, thrust, residual, and warnings.
    """
    if not isinstance(params, MothParams):
        raise TypeError(
            f"Expected MothParams, got {type(params).__name__}. "
            "If you have a Moth3D instance, pass moth.params (a MothParams) instead."
        )

    if heel_angle is None:
        heel_angle = np.deg2rad(30.0)

    # Filter out SciPy-specific kwargs
    casadi_kwargs = {k: v for k, v in kwargs.items() if k not in _IGNORED_SCIPY_KWARGS}

    return find_casadi_trim(
        params,
        u_target=u_forward,
        heel_angle=heel_angle,
        target_theta=target_theta,
        target_pos_d=target_pos_d,
        z0=z0,
        fixed_controls=fixed_controls,
        **casadi_kwargs,
    )


# ---------------------------------------------------------------------------
# Calibration API
# ---------------------------------------------------------------------------

@dataclass
class CalibrationTrimResult:
    """Result from a single-speed calibration solve.

    Attributes:
        speed: Target speed (m/s).
        thrust: Calibrated thrust (N).
        trim: Full CasADi trim result.
        max_xdot_residual: Maximum absolute state derivative at the solution,
            i.e. max(|xdot|). Worst-case constraint violation across all
            state derivatives (pos_d_dot, theta_dot, w_dot, q_dot, u_dot).
        warnings: Validation warnings from physical plausibility checks.
    """

    speed: float
    thrust: float
    trim: CasadiTrimResult
    max_xdot_residual: float
    warnings: list[str] = field(default_factory=list)


def validate_trim_result(
    result: CasadiTrimResult,
    u_target: float,
) -> list[str]:
    """Physical plausibility checks on a CasADi trim result.

    Args:
        result: CasadiTrimResult to validate.
        u_target: Target speed (m/s).

    Returns:
        List of warning strings (empty if all checks pass).
    """
    warns = list(result.warnings)  # start with solver warnings

    theta = result.state[1]
    pos_d = result.state[0]

    # Pitch angle check
    theta_limit = 3.0 if u_target >= 8.0 else 5.0
    if abs(np.degrees(theta)) > theta_limit:
        warns.append(f"|theta|={abs(np.degrees(theta)):.2f}deg > {theta_limit}deg")

    # Depth check
    if pos_d < -2.0 or pos_d > -0.5:
        warns.append(f"pos_d={pos_d:.3f}m outside [-2.0, -0.5]")

    # Thrust check
    if result.thrust < 10.0:
        warns.append(f"thrust={result.thrust:.1f}N near lower bound (10N)")
    if result.thrust > 490.0:
        warns.append(f"thrust={result.thrust:.1f}N near upper bound (500N)")

    # Controls at bounds
    if result.diagnostics:
        active = result.diagnostics.get("active_bounds", [])
        for entry in active:
            name = entry["variable"]
            if name in ("main_flap", "rudder_elevator"):
                warns.append(
                    f"{name} at {entry['bound']} bound "
                    f"({np.rad2deg(entry['bound_value']):.1f} deg)"
                )

    return warns


def validate_thrust_sweep(
    speeds: np.ndarray | tuple | list,
    thrusts: np.ndarray | tuple | list,
    monotonic_tol: float = 2.0,
    jump_fraction: float = 0.5,
) -> list[str]:
    """Check calibrated thrust values across a speed sweep.

    Args:
        speeds: Array of speeds (m/s).
        thrusts: Array of calibrated thrusts (N).
        monotonic_tol: Tolerance for monotonicity check (N).
        jump_fraction: Maximum fractional change between adjacent speeds.

    Returns:
        List of warning strings.
    """
    speeds = np.asarray(speeds)
    thrusts = np.asarray(thrusts)
    warn_list = []

    for i in range(1, len(thrusts)):
        if thrusts[i] < thrusts[i - 1] - monotonic_tol:
            warn_list.append(
                f"Non-monotonic thrust: {thrusts[i]:.1f}N at {speeds[i]:.1f} m/s "
                f"< {thrusts[i-1]:.1f}N at {speeds[i-1]:.1f} m/s"
            )

    for i in range(1, len(thrusts)):
        if thrusts[i - 1] > 0:
            frac_change = abs(thrusts[i] - thrusts[i - 1]) / thrusts[i - 1]
            if frac_change > jump_fraction:
                warn_list.append(
                    f"Sharp thrust jump ({frac_change:.0%}) between "
                    f"{speeds[i-1]:.1f} and {speeds[i]:.1f} m/s"
                )

    return warn_list


def calibrate_moth_thrust(
    params: MothParams,
    target_u: float,
    heel_angle: float = np.deg2rad(30.0),
    z0: np.ndarray | None = None,
    **kwargs,
) -> CalibrationTrimResult:
    """Calibrate sail thrust at a single speed.

    Since CasADi always solves for thrust as a free variable, this just
    validates and wraps the trim result.

    Args:
        params: Moth parameters.
        target_u: Target forward speed (m/s).
        heel_angle: Static heel angle (rad).
        z0: Initial guess for decision variables (8-vector). Passed to
            find_casadi_trim for warm-starting from a previous solution.
        **kwargs: Additional args passed to find_casadi_trim.

    Returns:
        CalibrationTrimResult with speed, thrust, trim result, and warnings.
    """
    result = find_casadi_trim(params, target_u, heel_angle=heel_angle, z0=z0, **kwargs)
    warns = validate_trim_result(result, target_u)

    return CalibrationTrimResult(
        speed=target_u,
        thrust=result.thrust,
        trim=result,
        max_xdot_residual=result.residual,
        warnings=warns,
    )


