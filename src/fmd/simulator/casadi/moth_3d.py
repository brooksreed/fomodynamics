"""CasADi Moth3D longitudinal dynamics (pitch + heave + optional surge).

This is the CasADi implementation that exactly matches Moth3D (JAX) for
MPC applications and equivalence testing.

State Vector:
    [pos_d, theta, w, q, u]  # 5 states
    - pos_d: Vertical position (heave), positive down in NED (m)
    - theta: Pitch angle, positive nose-up (rad)
    - w: Body-frame vertical velocity, positive down (m/s)
    - q: Body-frame pitch rate, positive nose-up about +y (rad/s)
    - u: Body-frame surge velocity, positive forward (m/s)

Control Vector:
    [main_flap_angle, rudder_elevator_angle]  # 2 controls
    - main_flap_angle: Main foil flap deflection (rad)
    - rudder_elevator_angle: Rudder elevator deflection (rad)
"""

from __future__ import annotations

import casadi as cs
import numpy as np

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.params import MothParams


# Minimum forward speed (m/s) to avoid division by zero
_MIN_FORWARD_SPEED = 0.1


def _casadi_softplus(x):
    """Overflow-safe softplus: log(1 + exp(x)).

    Matches jax.nn.softplus numerically using the identity:
        softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    """
    return cs.fmax(x, 0) + cs.log(1 + cs.exp(-cs.fabs(x)))


class Moth3DCasadiExact(CasadiDynamicSystem):
    """Moth 3DOF longitudinal dynamics - CasADi implementation.

    Exactly matches Moth3D (JAX) for equivalence testing.
    All force components are computed inline (no separate component objects).

    State vector (5 elements):
        [0] pos_d - Vertical position, positive down (m)
        [1] theta - Pitch angle, positive nose-up (rad)
        [2] w     - Body-frame vertical velocity, positive down (m/s)
        [3] q     - Body-frame pitch rate, positive nose-up (rad/s)
        [4] u     - Body-frame surge velocity, positive forward (m/s)

    Control vector (2 elements):
        [0] main_flap_angle       - Main foil flap deflection (rad)
        [1] rudder_elevator_angle - Rudder elevator deflection (rad)
    """

    state_names: tuple[str, ...] = ("pos_d", "theta", "w", "q", "u")
    control_names: tuple[str, ...] = ("main_flap_angle", "rudder_elevator_angle")

    def __init__(
        self,
        params: MothParams,
        heel_angle: float = np.deg2rad(30.0),
        ventilation_mode: str = "smooth",
        ventilation_threshold: float = 0.30,
        surge_enabled: bool = True,
        u_forward: float = 10.0,
    ):
        """Initialize Moth3D CasADi model from parameters.

        Args:
            params: MothParams instance with validated model parameters.
            heel_angle: Static heel angle (rad), >= 0. Default 30 deg.
            ventilation_mode: "smooth" (default) or "binary".
            ventilation_threshold: Exposed span fraction for ventilation onset.
            surge_enabled: If True, surge velocity is a dynamic state.
            u_forward: Constant forward speed when surge_enabled=False (m/s).
        """
        # Physical parameters
        self.total_mass = float(params.total_mass)
        self.g = float(params.g)
        self.rho = float(params.rho_water)
        self.added_mass_heave = float(params.added_mass_heave)
        self.added_inertia_pitch = float(params.added_inertia_pitch)
        self.added_mass_surge = float(params.added_mass_surge)
        self.hull_mass = float(params.hull_mass)
        self.sailor_mass = float(params.sailor_mass)
        self.hull_iyy = float(params.hull_inertia_matrix[1, 1])

        # CG offset (fixed, from sailor_position)
        cg_offset = params.combined_cg_offset
        self.cg_offset_x = float(cg_offset[0])
        self.cg_offset_z = float(cg_offset[2])

        # Composite Iyy (fixed position, reduced-mass parallel axis theorem)
        r_sailor = params.sailor_position
        reduced_mass = self.hull_mass * self.sailor_mass / self.total_mass
        self.iyy = self.hull_iyy + reduced_mass * (
            float(r_sailor[0]) ** 2 + float(r_sailor[2]) ** 2
        )

        # Main foil parameters
        self.main_foil_area = float(params.main_foil_area)
        self.main_foil_cl_alpha = float(params.main_foil_cl_alpha)
        self.main_foil_cl0 = float(params.main_foil_cl0)
        self.main_foil_cd0 = float(params.main_foil_cd0)
        self.main_foil_oswald = float(params.main_foil_oswald)
        self.main_foil_ar = float(params.main_foil_aspect_ratio)
        self.main_foil_flap_effectiveness = float(params.main_foil_flap_effectiveness)
        self.main_foil_cd_flap = float(params.main_foil_cd_flap)
        self.main_foil_position_x = float(params.main_foil_position[0])
        self.main_foil_position_z = float(params.main_foil_position[2])
        self.main_foil_span = float(params.main_foil_span)

        # Rudder parameters
        self.rudder_area = float(params.rudder_area)
        self.rudder_cl_alpha = float(params.rudder_cl_alpha)
        self.rudder_cd0 = float(params.rudder_cd0)
        self.rudder_oswald = float(params.rudder_oswald)
        self.rudder_ar = float(params.rudder_aspect_ratio)
        self.rudder_position_x = float(params.rudder_position[0])
        self.rudder_position_z = float(params.rudder_position[2])
        self.rudder_span = float(params.rudder_span)

        # Sail parameters
        self.sail_thrust_coeff = float(params.sail_thrust_coeff)
        self.sail_thrust_slope = float(params.sail_thrust_slope)
        self.sail_ce_position_x = float(params.sail_ce_position[0])
        self.sail_ce_position_z = float(params.sail_ce_position[2])
        self.sail_thrust_speeds = tuple(float(s) for s in params.sail_thrust_speeds)
        self.sail_thrust_values = tuple(float(v) for v in params.sail_thrust_values)

        # Hull drag parameters
        self.hull_drag_coeff = float(params.hull_drag_coeff)
        self.hull_cg_above_bottom = float(params.hull_cg_above_bottom)
        self.hull_buoyancy_coeff = float(params.hull_buoyancy_coeff)
        self.buoyancy_fwd_x = float(params.hull_length) / 4.0
        self.buoyancy_aft_x = -float(params.hull_length) / 4.0

        # Strut parameters
        self.main_strut_chord = float(params.main_strut_chord)
        self.main_strut_thickness = float(params.main_strut_thickness)
        self.main_strut_cd_pressure = float(params.main_strut_cd_pressure)
        self.main_strut_cf_skin = float(params.main_strut_cf_skin)
        # Strut geometry: extends from hull bottom (top) down to the foil (bottom).
        # Hull bottom is at z=hull_cg_above_bottom in body FRD (below CG).
        self.main_strut_bottom_z = float(params.main_foil_position[2])
        self.main_strut_top_z = float(params.hull_cg_above_bottom)
        self.main_strut_max_depth = float(params.main_foil_strut_depth)
        self.rudder_strut_chord = float(params.rudder_strut_chord)
        self.rudder_strut_thickness = float(params.rudder_strut_thickness)
        self.rudder_strut_cd_pressure = float(params.rudder_strut_cd_pressure)
        self.rudder_strut_cf_skin = float(params.rudder_strut_cf_skin)
        self.rudder_strut_bottom_z = float(params.rudder_position[2])
        self.rudder_strut_top_z = float(params.hull_cg_above_bottom)
        self.rudder_strut_max_depth = float(params.rudder_strut_depth)

        # Ventilation parameters
        self.heel_angle = float(heel_angle)
        self.ventilation_mode = ventilation_mode
        self.ventilation_threshold = float(ventilation_threshold)

        # Surge dynamics
        self.surge_enabled = surge_enabled
        self.u_forward = float(u_forward)

    def _compute_depth_factor(self, foil_depth, foil_span):
        """Compute depth factor for foil ventilation.

        Matches compute_depth_factor() in moth_forces.py exactly.
        """
        if self.ventilation_mode == "binary":
            return cs.if_else(foil_depth > 0.0, 1.0, 0.0)

        # Smooth mode
        min_submergence = 0.015
        max_submergence_raw = (foil_span / 2.0) * cs.sin(self.heel_angle)
        max_submergence = cs.fmax(max_submergence_raw, min_submergence)

        eps = 1e-6

        # Exposed fraction
        exposed_raw = (max_submergence - foil_depth) / max_submergence

        # Smooth saturation: softplus then 1-exp(-x)
        k_sat = 50.0
        exposed_pos = _casadi_softplus(k_sat * exposed_raw) / k_sat
        exposed = 1.0 - cs.exp(-exposed_pos)

        # Normalize and tanh transition
        normalized = exposed / cs.fmax(self.ventilation_threshold, eps)
        return 0.5 * (1.0 - cs.tanh((normalized - 0.5) * 6.0))

    def _compute_main_foil(self, x, u, u_fwd):
        """Compute main foil force and moment in body frame.

        Assumes forward flight (u > 0). u_safe = max(u, 0.1) clamps the
        AoA denominator at low speeds.

        Returns (force_x, force_z, moment_y).
        """
        pos_d = x[0]
        theta = x[1]
        w = x[2]
        q = x[3]
        main_flap = u[0]

        # Effective positions adjusted for CG offset
        eff_pos_x = self.main_foil_position_x - self.cg_offset_x
        eff_pos_z = self.main_foil_position_z - self.cg_offset_z

        u_safe = cs.fmax(u_fwd, 0.1)

        # Local vertical flow and angle-of-attack separation
        w_local = w - q * eff_pos_x
        alpha_geo = cs.atan2(w_local, u_safe)
        alpha_eff = self.main_foil_flap_effectiveness * main_flap + w_local / u_safe

        # Heel-corrected foil depth (matches compute_foil_ned_depth in moth_forces.py)
        foil_depth = (pos_d
                      + eff_pos_z * cs.cos(self.heel_angle) * cs.cos(theta)
                      - eff_pos_x * cs.sin(theta))
        depth_factor = self._compute_depth_factor(foil_depth, self.main_foil_span)

        # Lift and drag coefficients
        cl = (self.main_foil_cl0 + self.main_foil_cl_alpha * alpha_eff) * depth_factor
        cd = self.main_foil_cd0 * depth_factor + cl ** 2 / (cs.pi * self.main_foil_ar * self.main_foil_oswald)
        cd = cd + self.main_foil_cd_flap * main_flap ** 2

        # Dynamic pressure
        q_dyn = 0.5 * self.rho * u_fwd ** 2

        # Forces
        lift = q_dyn * self.main_foil_area * cl
        drag = q_dyn * self.main_foil_area * cd

        # Body frame forces
        fx = -drag * cs.cos(alpha_geo) + lift * cs.sin(alpha_geo)
        fz = -drag * cs.sin(alpha_geo) - lift * cs.cos(alpha_geo)

        # Moment about CG: M_y = r_z * F_x - r_x * F_z
        my = eff_pos_z * fx - eff_pos_x * fz

        return fx, fz, my

    def _compute_rudder(self, x, u, u_fwd):
        """Compute rudder elevator force and moment in body frame.

        Assumes forward flight (u > 0). u_safe = max(u, 0.1) clamps the
        AoA denominator at low speeds.

        Returns (force_x, force_z, moment_y).
        """
        pos_d = x[0]
        theta = x[1]
        w = x[2]
        q = x[3]
        rudder_elevator_angle = u[1]

        # Effective positions adjusted for CG offset
        eff_pos_x = self.rudder_position_x - self.cg_offset_x
        eff_pos_z = self.rudder_position_z - self.cg_offset_z

        u_safe = cs.fmax(u_fwd, 0.1)

        # Local vertical flow and angle-of-attack separation
        w_local = w - q * eff_pos_x
        alpha_geo = cs.atan2(w_local, u_safe)
        alpha_eff = rudder_elevator_angle + w_local / u_safe

        # Heel-corrected depth factor (matches compute_foil_ned_depth in moth_forces.py)
        foil_depth = (pos_d
                      + eff_pos_z * cs.cos(self.heel_angle) * cs.cos(theta)
                      - eff_pos_x * cs.sin(theta))
        depth_factor = self._compute_depth_factor(foil_depth, self.rudder_span)

        # Lift coefficient with depth factor
        rudder_cl = self.rudder_cl_alpha * alpha_eff * depth_factor

        # Drag coefficient: parabolic polar (same as main foil)
        rudder_cd = self.rudder_cd0 * depth_factor + rudder_cl ** 2 / (cs.pi * self.rudder_ar * self.rudder_oswald)

        # Dynamic pressure
        q_dyn = 0.5 * self.rho * u_fwd ** 2

        # Lift and drag forces
        rudder_lift = q_dyn * self.rudder_area * rudder_cl
        rudder_drag = q_dyn * self.rudder_area * rudder_cd

        # Body frame forces (same decomposition as main foil)
        rudder_fx = -rudder_drag * cs.cos(alpha_geo) + rudder_lift * cs.sin(alpha_geo)
        rudder_fz = -rudder_drag * cs.sin(alpha_geo) - rudder_lift * cs.cos(alpha_geo)

        # Full moment: M_y = r_z * F_x - r_x * F_z
        rudder_my = eff_pos_z * rudder_fx - eff_pos_x * rudder_fz

        return rudder_fx, rudder_fz, rudder_my

    def _casadi_interp(self, u_fwd, speeds, values):
        """Piecewise linear interpolation matching jnp.interp behavior.

        Extrapolates flat beyond the endpoints (clamps to first/last value).
        Uses CasADi if_else for symbolic compatibility.

        Args:
            u_fwd: CasADi symbolic or float forward speed.
            speeds: Tuple of floats, sorted ascending.
            values: Tuple of floats, same length as speeds.

        Returns:
            CasADi symbolic interpolated thrust.
        """
        n = len(speeds)
        if n == 0:
            return self.sail_thrust_coeff + self.sail_thrust_slope * u_fwd
        if n == 1:
            return values[0]

        # Start from last segment and work backward with nested if_else.
        # This builds: if u >= speeds[-1]: values[-1]
        #              elif u >= speeds[-2]: lerp(...)
        #              ...
        #              else: values[0]
        result = values[-1]  # extrapolate right = last value
        for i in range(n - 2, -1, -1):
            s0, s1 = speeds[i], speeds[i + 1]
            v0, v1 = values[i], values[i + 1]
            slope = (v1 - v0) / (s1 - s0) if s1 != s0 else 0.0
            segment_val = v0 + slope * (u_fwd - s0)
            result = cs.if_else(u_fwd < s1, segment_val, result)
        # Extrapolate left = first value
        result = cs.if_else(u_fwd < speeds[0], values[0], result)
        return result

    def _compute_sail(self, u_fwd=None, theta=None):
        """Compute sail force and moment in body frame.

        Sail thrust is applied in the NED horizontal plane, then rotated
        to body frame using pitch angle theta:
            force_bx = F_sail * cos(theta)
            force_bz = F_sail * sin(theta)
            moment_y = ce_z * force_bx - ce_x * force_bz

        Args:
            u_fwd: Forward speed for speed-dependent sail model. If None,
                uses self.u_forward.
            theta: Pitch angle (rad) for NED→body rotation.

        Returns (force_x, force_z, moment_y).
        """
        # Effective CE positions adjusted for CG offset
        eff_ce_pos_x = self.sail_ce_position_x - self.cg_offset_x
        eff_ce_pos_z = self.sail_ce_position_z - self.cg_offset_z

        if u_fwd is None:
            u_fwd = self.u_forward

        # Use lookup table if available, otherwise affine model
        if self.sail_thrust_speeds:
            f_x = self._casadi_interp(u_fwd, self.sail_thrust_speeds, self.sail_thrust_values)
        else:
            f_x = self.sail_thrust_coeff + self.sail_thrust_slope * u_fwd

        # NED→body rotation: thrust is horizontal in NED, rotate by pitch
        cos_theta = cs.cos(theta) if theta is not None else 1.0
        sin_theta = cs.sin(theta) if theta is not None else 0.0
        force_bx = f_x * cos_theta
        force_bz = f_x * sin_theta

        # Moment: M_y = r_z * F_x - r_x * F_z (cross product r × F)
        moment_y = eff_ce_pos_z * force_bx - eff_ce_pos_x * force_bz

        return force_bx, force_bz, moment_y

    def _compute_hull_drag(self, x):
        """Compute hull contact drag + buoyancy force.

        Returns (force_x, force_z, moment_y).
        """
        pos_d = x[0]
        theta = x[1]

        # Dynamic contact depth: hull_cg_above_bottom - cg_offset_z
        # Tracks the runtime system CG (matches JAX MothHullDrag)
        runtime_contact_depth = self.hull_cg_above_bottom - self.cg_offset_z

        # Immersion: hull bottom NED depth = pos_d + contact_depth
        # Positive means hull bottom is below the water surface
        immersion = cs.fmax(0.0, runtime_contact_depth + pos_d)

        # Drag opposes forward motion
        hull_drag = self.hull_drag_coeff * immersion

        # --- Two-point buoyancy ---
        eff_fwd_x = self.buoyancy_fwd_x - self.cg_offset_x
        eff_aft_x = self.buoyancy_aft_x - self.cg_offset_x
        eff_z = runtime_contact_depth  # hull-bottom z below system CG

        # NED depth at each buoyancy point (heel_angle=0 for hull)
        # depth = pos_d + eff_z * cos(theta) - eff_x * sin(theta)
        depth_fwd = pos_d + eff_z * cs.cos(theta) - eff_fwd_x * cs.sin(theta)
        depth_aft = pos_d + eff_z * cs.cos(theta) - eff_aft_x * cs.sin(theta)

        imm_fwd = cs.fmax(0.0, depth_fwd)
        imm_aft = cs.fmax(0.0, depth_aft)

        half_coeff = self.hull_buoyancy_coeff / 2.0
        f_buoy_fwd = half_coeff * imm_fwd
        f_buoy_aft = half_coeff * imm_aft

        sin_theta = cs.sin(theta)
        cos_theta = cs.cos(theta)

        # Forward point body forces
        fwd_fx = f_buoy_fwd * sin_theta
        fwd_fz = -f_buoy_fwd * cos_theta
        fwd_my = eff_z * fwd_fx - eff_fwd_x * fwd_fz

        # Aft point body forces
        aft_fx = f_buoy_aft * sin_theta
        aft_fz = -f_buoy_aft * cos_theta
        aft_my = eff_z * aft_fx - eff_aft_x * aft_fz

        buoy_fx = fwd_fx + aft_fx
        buoy_fz = fwd_fz + aft_fz
        buoy_my = fwd_my + aft_my

        return -hull_drag + buoy_fx, buoy_fz, buoy_my

    def _compute_strut_drag(self, x, u_fwd, strut_chord, strut_thickness,
                              strut_cd_pressure, strut_cf_skin,
                              strut_max_depth, strut_top_z, strut_bottom_z,
                              strut_position_x):
        """Compute strut drag force and moment in body frame.

        Submerged depth is computed dynamically from state (pos_d, theta)
        using the same NED depth formula as compute_foil_ned_depth.

        Returns (force_x, force_z, moment_y).
        """
        pos_d = x[0]
        theta = x[1]

        # Effective positions adjusted for CG offset
        eff_pos_x = strut_position_x - self.cg_offset_x
        eff_top_z = strut_top_z - self.cg_offset_z
        eff_bottom_z = strut_bottom_z - self.cg_offset_z

        # NED depth of strut top and bottom
        # depth = pos_d + eff_z * cos(heel) * cos(theta) - eff_x * sin(theta)
        top_depth = (pos_d
                     + eff_top_z * cs.cos(self.heel_angle) * cs.cos(theta)
                     - eff_pos_x * cs.sin(theta))
        bottom_depth = (pos_d
                        + eff_bottom_z * cs.cos(self.heel_angle) * cs.cos(theta)
                        - eff_pos_x * cs.sin(theta))

        # NED depth span of the full strut (top to bottom)
        ned_span = cs.fmax(bottom_depth - top_depth, 1e-10)

        # NED submerged span
        ned_submerged = cs.fmin(
            cs.fmax(bottom_depth - cs.fmax(0.0, top_depth), 0.0),
            ned_span,
        )

        # Immersion fraction [0, 1] and physical submerged length
        immersion = ned_submerged / ned_span
        submerged_length = immersion * strut_max_depth

        # Pressure drag: frontal area = thickness * submerged length
        frontal_area = submerged_length * strut_thickness
        drag_pressure = 0.5 * self.rho * strut_cd_pressure * frontal_area * u_fwd ** 2

        # Skin friction drag: wetted area = 2 * chord * submerged length
        wetted_area = 2.0 * strut_chord * submerged_length
        drag_skin = 0.5 * self.rho * strut_cf_skin * wetted_area * u_fwd ** 2

        drag_total = drag_pressure + drag_skin
        fx = -drag_total

        # Moment arm: centroid of submerged segment in body frame
        centroid_z = eff_bottom_z - submerged_length / 2.0
        my = centroid_z * fx

        return fx, 0.0, my

    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute state derivative from component forces.

        Args:
            x: Current state [pos_d, theta, w, q, u] as CasADi symbolic
            u: Control input [main_flap_angle, rudder_elevator_angle] as CasADi symbolic
            t: Current simulation time (unused - CG offset is fixed)

        Returns:
            State derivative [pos_d_dot, theta_dot, w_dot, q_dot, u_dot] as cs.SX
        """
        # Forward speed: from state when surge enabled, else constant
        if self.surge_enabled:
            u_fwd = x[4]
        else:
            u_fwd = self.u_forward

        u_safe = cs.fmax(u_fwd, _MIN_FORWARD_SPEED)

        # Component forces
        foil_fx, foil_fz, foil_my = self._compute_main_foil(x, u, u_safe)
        rudder_fx, rudder_fz, rudder_my = self._compute_rudder(x, u, u_safe)
        sail_fx, sail_fz, sail_my = self._compute_sail(u_safe, theta=x[1])
        hull_fx, hull_fz, hull_my = self._compute_hull_drag(x)

        # Strut drag (main foil strut and rudder strut) - dynamic immersion
        ms_fx, ms_fz, ms_my = self._compute_strut_drag(
            x, u_safe, self.main_strut_chord, self.main_strut_thickness,
            self.main_strut_cd_pressure, self.main_strut_cf_skin,
            self.main_strut_max_depth, self.main_strut_top_z,
            self.main_strut_bottom_z, self.main_foil_position_x)
        rs_fx, rs_fz, rs_my = self._compute_strut_drag(
            x, u_safe, self.rudder_strut_chord, self.rudder_strut_thickness,
            self.rudder_strut_cd_pressure, self.rudder_strut_cf_skin,
            self.rudder_strut_max_depth, self.rudder_strut_top_z,
            self.rudder_strut_bottom_z, self.rudder_position_x)

        # Gravity in body frame
        theta = x[1]
        gravity_fz = self.total_mass * self.g * cs.cos(theta)
        gravity_fx = -self.total_mass * self.g * cs.sin(theta)

        # Sum forces and pitch moments
        total_fz = foil_fz + rudder_fz + sail_fz + hull_fz + ms_fz + rs_fz + gravity_fz
        total_fx = foil_fx + rudder_fx + sail_fx + hull_fx + ms_fx + rs_fx + gravity_fx
        total_my = foil_my + rudder_my + sail_my + hull_my + ms_my + rs_my

        # Kinematics (no sailor velocity correction - CG offset is fixed)
        pos_d_dot = -u_fwd * cs.sin(theta) + x[2] * cs.cos(theta)
        theta_dot = x[3]

        # Effective mass/inertia including added mass
        m_eff_heave = self.total_mass + self.added_mass_heave
        m_eff_surge = self.total_mass + self.added_mass_surge
        i_eff = self.iyy + self.added_inertia_pitch

        # Coriolis coupling
        w_dot = total_fz / m_eff_heave + x[3] * u_fwd
        q_dot = total_my / i_eff

        # Surge dynamics
        if self.surge_enabled:
            u_dot = total_fx / m_eff_surge - x[3] * x[2]
        else:
            u_dot = 0.0

        return cs.vertcat(pos_d_dot, theta_dot, w_dot, q_dot, u_dot)

    def post_step(self, x: cs.SX) -> cs.SX:
        """Post-process state: wrap theta (pitch angle) to [-pi, pi].

        Args:
            x: State after integration step

        Returns:
            State with wrapped theta
        """
        theta = x[1]
        theta_wrapped = cs.arctan2(cs.sin(theta), cs.cos(theta))
        return cs.vertcat(x[0], theta_wrapped, x[2], x[3], x[4])
