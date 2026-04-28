"""Moth 3DOF force and moment components.

Implements hydrodynamic and aerodynamic forces for the Moth sailboat
longitudinal (pitch + heave) model.

Components:
    - MothMainFoil: Main T-foil lift and drag with ventilation model
    - MothRudderElevator: Rudder T-foil with elevator for pitch control
    - MothSailForce: Simplified constant forward thrust from sail
    - MothHullDrag: Hull contact drag penalty when not foiling

All components are Equinox modules (PyTrees) that can be JIT-compiled
and differentiated. Forces and moments are returned as 3D vectors in
body frame coordinates.

State convention:
    [pos_d, theta, w, q, u] - Moth3D 5-state vector
    pos_d: Vertical position (m), positive down (NED)
    theta: Pitch angle (rad), positive nose-up
    w: Body-frame vertical velocity (m/s), positive down
    q: Pitch rate (rad/s), positive nose-up
    u: Surge velocity (m/s), positive forward

Control convention:
    [main_flap_angle, rudder_elevator_angle] - 2-control vector

Ventilation Model:
    Physics-based depth factor using foil span and heel angle geometry.
    Two modes: "smooth" (default, differentiable) and "binary" (hard cutoff).
    See compute_depth_factor() for details.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional

import numpy as np
from fmd.simulator.components.base import JaxForceElement

# Moth3D state/control indices (defined locally to avoid circular import)
POS_D = 0   # Vertical position (m)
THETA = 1   # Pitch angle (rad)
W = 2       # Body-frame vertical velocity (m/s)
Q = 3       # Body-frame pitch rate (rad/s)
U = 4       # Body-frame surge velocity (m/s)
MAIN_FLAP = 0         # Main foil flap angle (rad)
RUDDER_ELEVATOR = 1   # Rudder elevator angle (rad)


# ===========================================================================
# Ventilation / Depth Factor
# ===========================================================================

def compute_foil_ned_depth(
    pos_d: Array,
    eff_pos_x: float,
    eff_pos_z: float,
    theta: Array,
    heel_angle: float,
    eta: float = 0.0,
) -> Array:
    """Compute NED depth of a body-frame point accounting for pitch and heel.

    Applies the z-row of the rotation matrix R = Ry(theta) @ Rx(heel)
    to a body-frame point [x, 0, z] (y=0, centerline foil):

        depth_ned = pos_d + z * cos(heel) * cos(theta) - x * sin(theta) - eta

    This is the single source of truth for foil depth in the Moth 3DOF model.
    The viz geometry code (geometry.py compute_surface_waterline) uses the
    equivalent full rotation matrix.

    Args:
        pos_d: CG vertical position in NED (m), positive = deeper.
        eff_pos_x: Foil x-position relative to system CG (m), FRD body frame.
        eff_pos_z: Foil z-position relative to system CG (m), FRD body frame.
        theta: Pitch angle (rad).
        heel_angle: Static heel angle (rad).
        eta: Wave surface elevation at foil position (m). Default 0.

    Returns:
        NED depth of foil center (m). Positive = submerged.
    """
    return (pos_d
            + eff_pos_z * jnp.cos(heel_angle) * jnp.cos(theta)
            - eff_pos_x * jnp.sin(theta)
            - eta)


def compute_leeward_tip_depth(
    pos_d: Array,
    eff_pos_x: float,
    eff_pos_z: float,
    theta: Array,
    heel_angle: float,
    foil_span: float,
    eta: float = 0.0,
) -> Array:
    """Compute NED depth of the leeward foil tip (shallowest point).

    On a heeled boat, the leeward tip rises toward the surface by
    ``(foil_span / 2) * sin(heel_angle)`` relative to the foil center.

    Args:
        pos_d: CG vertical position in NED (m), positive = deeper.
        eff_pos_x: Foil x-position relative to system CG (m), FRD body frame.
        eff_pos_z: Foil z-position relative to system CG (m), FRD body frame.
        theta: Pitch angle (rad).
        heel_angle: Static heel angle (rad).
        foil_span: Foil wingspan (m).
        eta: Wave surface elevation at foil position (m). Default 0.

    Returns:
        NED depth of the leeward foil tip (m). Positive = submerged.
    """
    foil_center_depth = compute_foil_ned_depth(
        pos_d, eff_pos_x, eff_pos_z, theta, heel_angle, eta,
    )
    half_span_rise = (foil_span / 2.0) * jnp.sin(heel_angle)
    return foil_center_depth - half_span_rise


def compute_tip_at_surface_pos_d(
    params=None,
    heel_angle: float = None,
    theta: float = 0.0,
) -> float:
    """Compute pos_d that places the leeward foil tip at the water surface.

    On a moth heeled to windward, the leeward tip of the main T-foil rises
    toward the surface. This function returns the CG depth (pos_d) at which
    that tip is exactly at depth = 0 (waterline).

    Uses compute_foil_ned_depth() for the foil center depth, then subtracts
    the half-span rise. Solves algebraically for pos_d (linear in pos_d).

    Args:
        params: Moth parameter set (needs main_foil_position, main_foil_span,
            hull_mass, sailor_mass, sailor_position). Default: MOTH_BIEKER_V3.
        heel_angle: Static heel angle (rad). Default: 30 degrees.
        theta: Pitch angle (rad). Default 0 (level trim).

    Returns:
        pos_d value (m) that places the leeward tip at the surface.
    """
    import numpy as np

    if params is None:
        from fmd.simulator.params import MOTH_BIEKER_V3
        params = MOTH_BIEKER_V3
    if heel_angle is None:
        heel_angle = np.deg2rad(30.0)

    total_mass = params.hull_mass + params.sailor_mass
    cg_offset = params.sailor_mass * params.sailor_position / total_mass
    foil_pos = params.main_foil_position - cg_offset
    half_span_rise = (params.main_foil_span / 2.0) * np.sin(heel_angle)
    depth_at_zero_pos_d = float(compute_foil_ned_depth(
        0.0, foil_pos[0], foil_pos[2], theta, heel_angle))
    return half_span_rise - depth_at_zero_pos_d


def compute_depth_factor(
    foil_depth: Array,
    foil_span: float,
    heel_angle: float,
    ventilation_threshold: float = 0.30,
    mode: str = "smooth",
) -> Array:
    """Compute depth factor for foil ventilation.

    Models the fraction of lift retained as a foil approaches the surface.
    When the boat is heeled, the leeward tip of the horizontal T-foil rises
    and can breach the surface, causing ventilation (air ingestion) and
    lift loss.

    Geometry:
        tip_depth = foil_depth - (span/2) * sin(heel_angle)
        max_submergence = (span/2) * sin(heel_angle)
        exposed_fraction = (max_submergence - foil_depth) / max_submergence

    When heel_angle=0 (upright), there is no partial ventilation from heel.
    The model degenerates to a smooth depth-based factor: ~1 when submerged,
    smoothly tapering toward ~0 at/above the surface.

    Args:
        foil_depth: Depth of foil center below waterline (m). Positive = submerged.
        foil_span: Full wingspan of T-foil (m).
        heel_angle: Static heel angle (rad), always >= 0 by convention.
        ventilation_threshold: Fraction of exposed span at which ventilation
            begins to reduce lift (default 0.30 = 30%).
        mode: "smooth" (default, differentiable) or "binary" (hard cutoff).

    Returns:
        depth_factor: Scalar in approximately [0, 1]. 1.0 = fully submerged,
            ~0 = ventilated / above surface.
    """
    if mode == "binary":
        return jnp.where(foil_depth > 0.0, 1.0, 0.0)

    # --- Smooth mode (span-based heel geometry) ---
    # The leeward tip of a heeled T-foil rises by (span/2)*sin(heel).
    # As the foil approaches the surface, the exposed fraction of the span
    # determines the lift reduction. At zero heel the entire span breaches
    # simultaneously (binary), so we floor max_submergence at a small value
    # (~1.5 cm) to maintain differentiability with a ~3 cm transition.
    min_submergence = 0.015  # differentiability floor (m)
    max_submergence_raw = (foil_span / 2.0) * jnp.sin(heel_angle)
    max_submergence = jnp.maximum(max_submergence_raw, min_submergence)
    eps = 1e-6

    # Exposed fraction: how much of the span is above water
    # = 0 when foil_depth = max_submergence (tip just at surface)
    # = 1 when foil_depth = 0 (entire half-span exposed)
    # > 1 when foil_depth < 0 (foil center above water)
    exposed_raw = (max_submergence - foil_depth) / max_submergence

    # Smooth saturation to [0, 1) without hard clip
    k_sat = 50.0
    exposed_pos = jax.nn.softplus(k_sat * exposed_raw) / k_sat
    exposed = 1.0 - jnp.exp(-exposed_pos)  # smooth in [0, 1)

    # Normalize so ventilation_threshold maps to 0.5 transition point
    normalized = exposed / jnp.maximum(ventilation_threshold, eps)
    return 0.5 * (1.0 - jnp.tanh((normalized - 0.5) * 6.0))


class MothMainFoil(JaxForceElement):
    """Main T-foil lift and drag with ventilation model.

    Assumes forward flight (u > 0). At u <= 0.1 m/s, ``u_safe = max(u, 0.1)``
    clamps the denominator used for AoA computation, preventing division by zero
    but producing non-physical results. No runtime guard is used (would break JIT).

    Computes hydrodynamic lift and drag from the main foil, including:
    - Angle of attack from pitch, flap deflection, and heave rate
    - Physics-based ventilation model (heel angle + depth)
    - Induced drag via Oswald span efficiency model
    - Pitching moment from foil position offset

    Attributes:
        rho: Water density (kg/m^3)
        area: Foil planform area (m^2)
        cl_alpha: Lift curve slope (1/rad)
        cl0: Zero-AoA lift coefficient (-)
        cd0: Zero-lift drag coefficient (-)
        oswald: Oswald span efficiency factor (-)
        ar: Aspect ratio (-)
        flap_effectiveness: Flap lift effectiveness factor (-)
        position_x: Foil x-position relative to hull CG (m), positive forward (FRD body frame)
        position_z: Foil z-position relative to hull CG (m), positive down (FRD body frame)
        foil_span: Full wingspan of T-foil (m)
        heel_angle: Static heel angle (rad), >= 0
        ventilation_threshold: Exposed span fraction for ventilation onset
        ventilation_mode: "smooth" (default) or "binary"
    """

    rho: float
    area: float
    cl_alpha: float
    cl0: float
    cd0: float
    oswald: float
    ar: float
    flap_effectiveness: float
    cd_flap: float
    position_x: float
    position_z: float
    foil_span: float
    heel_angle: float = eqx.field(default=0.0)
    ventilation_threshold: float = eqx.field(default=0.30)
    ventilation_mode: str = eqx.field(static=True, default="smooth")

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute main foil force/moment with default u_forward=10.0."""
        return self.compute_moth(state, control, 10.0, t)

    def compute_moth(
        self,
        state: Array,
        control: Array,
        u_forward: float,
        t: float = 0.0,
        cg_offset: Optional[Array] = None,
        env=None,
        eta: float = 0.0,
        u_orbital_body: float = 0.0,
        w_orbital_body: float = 0.0,
        alpha_override=None,
    ) -> tuple[Array, Array]:
        """Compute main foil force/moment with specified forward speed.

        Args:
            state: Moth3D state [pos_d, theta, w, q]
            control: Control [main_flap_angle, rudder_elevator_angle]
            u_forward: Forward speed (m/s)
            t: Current simulation time
            cg_offset: System CG offset [x, y, z] to subtract from stored
                positions. When provided, effective positions become
                (position - cg_offset), making moments relative to system CG.
            env: Deprecated, ignored. Wave effects are passed via eta/u_orbital_body/w_orbital_body.
            eta: Wave surface elevation at this foil's position (m).
            u_orbital_body: Horizontal orbital velocity in body frame (m/s).
            w_orbital_body: Vertical orbital velocity in body frame (m/s).
            alpha_override: If provided, use this as effective AoA for CL
                calculation instead of the instantaneous value. Used by the
                lift lag filter (Phase 3).

        Returns:
            Tuple of (force, moment) in body frame, each shape (3,)
        """
        pos_d = state[POS_D]
        theta = state[THETA]
        w = state[W]
        main_flap = control[MAIN_FLAP]

        # Effective positions adjusted for CG offset
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        eff_pos_x = self.position_x - _cg[0]
        eff_pos_z = self.position_z - _cg[2]

        u_safe = jnp.maximum(u_forward, 0.1)

        # Effective flow speed including horizontal orbital velocity
        u_eff = jnp.maximum(u_safe + u_orbital_body, 0.1)

        # Local vertical flow and angle-of-attack separation
        # alpha_geo: geometric flow angle (for force rotation)
        # alpha_eff: effective AoA including control (for polar)
        q = state[Q]
        w_local = w - q * eff_pos_x + w_orbital_body
        alpha_geo = jnp.arctan2(w_local, u_eff)
        alpha_eff_inst = self.flap_effectiveness * main_flap + w_local / u_eff

        # Use override (filtered) alpha for CL if provided
        alpha_for_cl = alpha_override if alpha_override is not None else alpha_eff_inst

        # Ventilation / depth factor (canonical heel-corrected depth)
        foil_depth = compute_foil_ned_depth(pos_d, eff_pos_x, eff_pos_z, theta, self.heel_angle, eta=eta)
        depth_factor = compute_depth_factor(
            foil_depth,
            self.foil_span,
            self.heel_angle,
            self.ventilation_threshold,
            self.ventilation_mode,
        )

        # Lift and drag coefficients
        cl = (self.cl0 + self.cl_alpha * alpha_for_cl) * depth_factor
        cd = self.cd0 * depth_factor + cl**2 / (jnp.pi * self.ar * self.oswald)
        cd = cd + self.cd_flap * main_flap ** 2

        # Dynamic pressure using effective flow speed
        q_dyn = 0.5 * self.rho * u_eff**2

        # Forces
        lift = q_dyn * self.area * cl
        drag = q_dyn * self.area * cd

        # Body frame forces
        fx = -drag * jnp.cos(alpha_geo) + lift * jnp.sin(alpha_geo)
        fy = 0.0
        fz = -drag * jnp.sin(alpha_geo) - lift * jnp.cos(alpha_geo)

        force_b = jnp.array([fx, fy, fz])

        # Moment about CG from foil position: M = r × F
        # M_y = r_z * F_x - r_x * F_z
        mx = 0.0
        my = eff_pos_z * fx - eff_pos_x * fz
        mz = 0.0

        moment_b = jnp.array([mx, my, mz])

        return force_b, moment_b


class MothRudderElevator(JaxForceElement):
    """Rudder T-foil with elevator for pitch control.

    Assumes forward flight (u > 0). At u <= 0.1 m/s, ``u_safe = max(u, 0.1)``
    clamps the denominator used for AoA computation, preventing division by zero
    but producing non-physical results. No runtime guard is used (would break JIT).

    Computes hydrodynamic lift and drag from the rudder elevator, including
    pitch rate coupling that affects the local angle of attack.
    Includes depth factor model (same as main foil).

    Drag model: parabolic polar cd = cd0 + cl^2 / (pi * ar * oswald),
    same structure as the main foil.

    Attributes:
        rho: Water density (kg/m^3)
        area: Rudder planform area (m^2)
        cl_alpha: Lift curve slope (1/rad)
        cd0: Zero-lift drag coefficient (-)
        oswald: Oswald span efficiency factor (-)
        ar: Aspect ratio (-)
        position_x: Rudder x-position relative to hull CG (m), positive forward (FRD body frame)
        position_z: Rudder z-position relative to hull CG (m), positive down (FRD body frame)
        foil_span: Full wingspan of rudder T-foil (m)
        heel_angle: Static heel angle (rad), >= 0
        ventilation_threshold: Exposed span fraction for ventilation onset
        ventilation_mode: "smooth" (default) or "binary"
    """

    rho: float
    area: float
    cl_alpha: float
    cd0: float = eqx.field(default=0.0)
    oswald: float = eqx.field(default=0.85)
    ar: float = eqx.field(default=7.0)
    position_x: float = eqx.field(default=0.0)
    position_z: float = eqx.field(default=0.0)
    foil_span: float = eqx.field(default=0.5)
    heel_angle: float = eqx.field(default=0.0)
    ventilation_threshold: float = eqx.field(default=0.30)
    ventilation_mode: str = eqx.field(static=True, default="smooth")

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute rudder force/moment with default u_forward=10.0."""
        return self.compute_moth(state, control, 10.0, t)

    def compute_moth(
        self,
        state: Array,
        control: Array,
        u_forward: float,
        t: float = 0.0,
        cg_offset: Optional[Array] = None,
        env=None,
        eta: float = 0.0,
        u_orbital_body: float = 0.0,
        w_orbital_body: float = 0.0,
        alpha_override=None,
    ) -> tuple[Array, Array]:
        """Compute rudder force/moment with specified forward speed.

        Args:
            state: Moth3D state [pos_d, theta, w, q]
            control: Control [main_flap_angle, rudder_elevator_angle]
            u_forward: Forward speed (m/s)
            t: Current simulation time
            cg_offset: System CG offset [x, y, z] to subtract from stored
                positions. When provided, effective positions become
                (position - cg_offset), making moments relative to system CG.
            env: Deprecated, ignored. Wave effects are passed via eta/u_orbital_body/w_orbital_body.
            eta: Wave surface elevation at this foil's position (m).
            u_orbital_body: Horizontal orbital velocity in body frame (m/s).
            w_orbital_body: Vertical orbital velocity in body frame (m/s).
            alpha_override: If provided, use this as effective AoA for CL
                calculation instead of the instantaneous value.

        Returns:
            Tuple of (force, moment) in body frame, each shape (3,)
        """
        pos_d = state[POS_D]
        theta = state[THETA]
        w = state[W]
        q = state[Q]
        rudder_elevator_angle = control[RUDDER_ELEVATOR]

        # Effective positions adjusted for CG offset
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        eff_pos_x = self.position_x - _cg[0]
        eff_pos_z = self.position_z - _cg[2]

        u_safe = jnp.maximum(u_forward, 0.1)

        # Effective flow speed including horizontal orbital velocity
        u_eff = jnp.maximum(u_safe + u_orbital_body, 0.1)

        # Local vertical flow and angle-of-attack separation
        w_local = w - q * eff_pos_x + w_orbital_body
        alpha_geo = jnp.arctan2(w_local, u_eff)
        alpha_eff_inst = rudder_elevator_angle + w_local / u_eff

        # Use override (filtered) alpha for CL if provided
        alpha_for_cl = alpha_override if alpha_override is not None else alpha_eff_inst

        # Depth factor with canonical heel-corrected depth
        foil_depth = compute_foil_ned_depth(pos_d, eff_pos_x, eff_pos_z, theta, self.heel_angle, eta=eta)
        depth_factor = compute_depth_factor(
            foil_depth,
            self.foil_span,
            self.heel_angle,
            self.ventilation_threshold,
            self.ventilation_mode,
        )

        # Lift coefficient with depth factor
        rudder_cl = self.cl_alpha * alpha_for_cl * depth_factor

        # Drag coefficient: parabolic polar (same as main foil)
        rudder_cd = self.cd0 * depth_factor + rudder_cl**2 / (jnp.pi * self.ar * self.oswald)

        # Dynamic pressure using effective flow speed
        q_dyn = 0.5 * self.rho * u_eff**2

        # Lift and drag forces
        rudder_lift = q_dyn * self.area * rudder_cl
        rudder_drag = q_dyn * self.area * rudder_cd

        # Body frame forces (same decomposition as main foil)
        rudder_fx = -rudder_drag * jnp.cos(alpha_geo) + rudder_lift * jnp.sin(alpha_geo)
        rudder_fz = -rudder_drag * jnp.sin(alpha_geo) - rudder_lift * jnp.cos(alpha_geo)

        force_b = jnp.array([rudder_fx, 0.0, rudder_fz])

        # Full pitching moment from rudder position: M = r × F
        # M_y = r_z * F_x - r_x * F_z (both lift AND drag contribute)
        rudder_my = eff_pos_z * rudder_fx - eff_pos_x * rudder_fz

        moment_b = jnp.array([0.0, rudder_my, 0.0])

        return force_b, moment_b


class MothSailForce(JaxForceElement):
    """Sail force model with optional speed-dependent thrust.

    The sail provides forward thrust (surge direction) and a pitching
    moment from the sail center of effort height above the hull CG.

    Three modes:
    - **Lookup table** (preferred): When thrust_speeds and thrust_values
      are non-empty, uses ``jnp.interp(u, speeds, values)`` to get thrust.
      Extrapolates flat beyond the table endpoints.
    - **Affine**: F_sail(u) = thrust_coeff + thrust_slope * u
    - **Constant**: When thrust_slope=0 (default), reduces to constant thrust.

    The lookup table takes priority when present.

    Attributes:
        thrust_coeff: Base forward thrust / intercept (N)
        thrust_slope: Speed-dependent thrust slope (N/(m/s)), default 0
        ce_position_z: Sail CE z-position relative to hull CG (m),
                       negative = above hull CG in body frame
        thrust_speeds: Speed breakpoints for lookup table (m/s), sorted ascending
        thrust_values: Thrust values at each breakpoint (N)
    """

    thrust_coeff: float
    thrust_slope: float = eqx.field(default=0.0)
    ce_position_x: float = eqx.field(default=0.0)
    ce_position_z: float = eqx.field(default=0.0)
    thrust_speeds: Array = eqx.field(default_factory=lambda: jnp.array([]))
    thrust_values: Array = eqx.field(default_factory=lambda: jnp.array([]))

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute sail force/moment with default u_forward=10.0."""
        return self.compute_moth(state, control, 10.0, t)

    def compute_moth(
        self,
        state: Array,
        control: Array,
        u_forward: float,
        t: float = 0.0,
        cg_offset: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Compute sail force/moment with specified forward speed.

        The sail produces forward thrust in the NED horizontal plane,
        then rotated to body frame using pitch angle theta:
            force_b = [F_sail * cos(theta), 0, F_sail * sin(theta)]

        This models the physical reality that sail force direction is set
        by the wind, not the hull pitch attitude.

        Thrust model priority:
        1. Lookup table (if thrust_speeds/thrust_values non-empty):
           F_sail = interp(u_forward, thrust_speeds, thrust_values)
        2. Affine: F_sail = thrust_coeff + thrust_slope * u_forward
        3. Constant: when thrust_slope=0, reduces to F_sail = thrust_coeff

        Moment uses full cross product: M_y = ce_z * F_bx - ce_x * F_bz.
        The sail CE is fixed to the hull, so moment arm uses body-frame force.

        Args:
            state: Moth3D state [pos_d, theta, w, q]
            control: Control [main_flap_angle, rudder_elevator_angle]
            u_forward: Forward speed (m/s)
            t: Current simulation time
            cg_offset: System CG offset [x, y, z] to subtract from stored
                positions. When provided, effective positions become
                (position - cg_offset), making moments relative to system CG.
        """
        theta = state[THETA]

        # Effective positions adjusted for CG offset
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        eff_ce_pos_x = self.ce_position_x - _cg[0]
        eff_ce_pos_z = self.ce_position_z - _cg[2]

        # Thrust: use lookup table if available, otherwise affine model.
        # The if/else is on a concrete value (array shape), so only one
        # branch is traced during JIT compilation.
        if self.thrust_speeds.shape[0] > 0:
            f_x = jnp.interp(u_forward, self.thrust_speeds, self.thrust_values)
        else:
            f_x = self.thrust_coeff + self.thrust_slope * u_forward

        # NED→body rotation: thrust is horizontal in NED, rotate by pitch
        force_bx = f_x * jnp.cos(theta)
        force_bz = f_x * jnp.sin(theta)
        force_b = jnp.array([force_bx, 0.0, force_bz])

        # Moment: M_y = r_z * F_x - r_x * F_z (cross product r × F)
        moment_y = eff_ce_pos_z * force_bx - eff_ce_pos_x * force_bz

        moment_b = jnp.array([0.0, moment_y, 0.0])

        return force_b, moment_b


class MothHullDrag(JaxForceElement):
    """Hull contact drag and buoyancy when not foiling.

    Applies a drag force proportional to hull immersion depth, plus a
    two-point buoyancy restoring force that prevents catastrophic sinking.
    When the boat is foiling (pos_d < -contact_depth), the hull is
    above water and there is no hull drag or buoyancy. As the boat sinks
    (pos_d increases toward zero and beyond), the hull contacts water
    and drag/buoyancy increase linearly with immersion.

    The hull bottom is at NED depth ``pos_d + contact_depth``.
    Hull drag is active when this is positive (hull bottom below surface).

    Buoyancy is modeled as two points (forward and aft), each providing
    half the total buoyancy coefficient. This creates both a vertical
    restoring force and a pitch restoring moment.

    Attributes:
        drag_coeff: Drag penalty per meter of immersion (N/m)
        contact_depth: CG-to-hull-bottom distance in body frame (m)
        buoyancy_coeff: Total buoyancy restoring force per meter of immersion (N/m)
        buoyancy_fwd_x: Forward buoyancy point x-position relative to hull CG (m)
        buoyancy_aft_x: Aft buoyancy point x-position relative to hull CG (m)
    """

    drag_coeff: float
    contact_depth: float  # Static default (from params.hull_contact_depth)
    hull_cg_above_bottom: float = eqx.field(default=0.0)  # For dynamic contact depth
    buoyancy_coeff: float = eqx.field(default=0.0)
    buoyancy_fwd_x: float = eqx.field(default=0.0)
    buoyancy_aft_x: float = eqx.field(default=0.0)

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute hull drag force/moment with default u_forward=10.0."""
        return self.compute_moth(state, control, 10.0, t)

    def compute_moth(
        self,
        state: Array,
        control: Array,
        u_forward: float,
        t: float = 0.0,
        cg_offset: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Compute hull drag + buoyancy force/moment with specified forward speed."""
        pos_d = state[POS_D]
        theta = state[THETA]

        # Dynamic contact depth: hull_cg_above_bottom - cg_offset[2]
        # Always uses the runtime CG offset to track the moving system CG.
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        runtime_contact_depth = self.hull_cg_above_bottom - _cg[2]

        # Immersion: hull bottom NED depth = pos_d + contact_depth
        # Positive means hull bottom is below the water surface
        immersion = jnp.maximum(0.0, pos_d + runtime_contact_depth)

        # Drag opposes forward motion
        hull_drag = self.drag_coeff * immersion

        # --- Two-point buoyancy ---
        # CG offset adjusts x-positions for moment arms
        eff_fwd_x = self.buoyancy_fwd_x - _cg[0]
        eff_aft_x = self.buoyancy_aft_x - _cg[0]
        eff_z = runtime_contact_depth  # hull-bottom z below system CG

        # NED depth at each buoyancy point (heel_angle=0 for hull)
        depth_fwd = compute_foil_ned_depth(pos_d, eff_fwd_x, eff_z, theta, 0.0)
        depth_aft = compute_foil_ned_depth(pos_d, eff_aft_x, eff_z, theta, 0.0)

        # Immersion at each point (positive = submerged)
        imm_fwd = jnp.maximum(0.0, depth_fwd)
        imm_aft = jnp.maximum(0.0, depth_aft)

        # Buoyancy force per point (upward in world frame)
        half_coeff = self.buoyancy_coeff / 2.0
        f_buoy_fwd = half_coeff * imm_fwd
        f_buoy_aft = half_coeff * imm_aft

        # Decompose to body frame for each point
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        # Forward point body forces
        fwd_fx = f_buoy_fwd * sin_theta
        fwd_fz = -f_buoy_fwd * cos_theta
        # Moment: M_y = r_z * F_x - r_x * F_z
        fwd_my = eff_z * fwd_fx - eff_fwd_x * fwd_fz

        # Aft point body forces
        aft_fx = f_buoy_aft * sin_theta
        aft_fz = -f_buoy_aft * cos_theta
        aft_my = eff_z * aft_fx - eff_aft_x * aft_fz

        # Total buoyancy
        buoy_fx = fwd_fx + aft_fx
        buoy_fz = fwd_fz + aft_fz
        buoy_my = fwd_my + aft_my

        force_b = jnp.array([-hull_drag + buoy_fx, 0.0, buoy_fz])
        moment_b = jnp.array([0.0, buoy_my, 0.0])

        return force_b, moment_b


class MothStrutDrag(JaxForceElement):
    """Strut drag from a vertical NACA-section strut.

    Computes pressure drag (frontal area) and skin friction drag (wetted area)
    for a streamlined vertical strut. Can be instantiated once per strut
    (main foil strut, rudder strut) with different parameters.

    Drag depends on the dynamically computed submerged length of the strut,
    which varies with ride height (pos_d) and pitch angle (theta). The strut
    extends vertically in body frame from ``strut_top_z`` (hull bottom)
    to ``strut_bottom_z`` (at the foil). The NED depth of each
    end is computed via ``compute_foil_ned_depth``, and the submerged length
    is clipped to ``[0, strut_max_depth]``.

    The drag moment arm uses the centroid of the submerged segment rather
    than a fixed midpoint. When the strut is partially emerged, the centroid
    shifts downward toward the still-submerged portion.

    Drag acts in the body x-direction only (2D longitudinal model).

    Attributes:
        strut_chord: Strut chord length in streamwise direction (m)
        strut_thickness: Strut thickness facing the flow (m)
        strut_cd_pressure: Pressure drag coefficient based on frontal area (-)
        strut_cf_skin: Skin friction coefficient based on wetted area (-)
        strut_position_x: Strut x-position relative to hull CG (m), body frame
        strut_max_depth: Physical strut length from top to bottom (m)
        strut_top_z: Body-frame z of strut top (hull bottom) (m)
        strut_bottom_z: Body-frame z of strut bottom (at foil) (m)
        heel_angle: Static heel angle for NED depth calculation (rad)
        rho: Water density (kg/m^3)
    """

    strut_chord: float
    strut_thickness: float
    strut_cd_pressure: float
    strut_cf_skin: float
    strut_position_x: float
    strut_max_depth: float
    strut_top_z: float
    strut_bottom_z: float
    heel_angle: float
    rho: float

    def compute(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute strut drag force/moment with default u_forward=10.0."""
        return self.compute_moth(state, control, 10.0, t)

    def compute_moth(
        self,
        state: Array,
        control: Array,
        u_forward: float,
        t: float = 0.0,
        cg_offset: Optional[Array] = None,
        env=None,
    ) -> tuple[Array, Array]:
        """Compute strut drag force/moment with specified forward speed.

        Submerged depth is computed dynamically from state (pos_d, theta):
        1. NED depth of strut bottom and top via compute_foil_ned_depth
        2. Submerged length = clip(bottom_depth - max(0, top_depth), 0, max_depth)

        The drag moment arm is at the centroid of the submerged segment.
        When fully submerged, the centroid is at max_depth/2 below the top.
        When partially emerged, the centroid shifts toward the submerged
        portion.

        Args:
            state: Moth3D state [pos_d, theta, w, q, u]
            control: Control [main_flap_angle, rudder_elevator_angle]
            u_forward: Forward speed (m/s)
            t: Current simulation time
            cg_offset: System CG offset [x, y, z] to subtract from stored
                positions.

        Returns:
            Tuple of (force, moment) in body frame, each shape (3,)
        """
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        eff_pos_x = self.strut_position_x - _cg[0]
        eff_top_z = self.strut_top_z - _cg[2]
        eff_bottom_z = self.strut_bottom_z - _cg[2]

        pos_d = state[POS_D]
        theta = state[THETA]

        # NED depth of strut top and bottom
        top_depth = compute_foil_ned_depth(
            pos_d, eff_pos_x, eff_top_z, theta, self.heel_angle)
        bottom_depth = compute_foil_ned_depth(
            pos_d, eff_pos_x, eff_bottom_z, theta, self.heel_angle)

        # NED depth span of the full strut (top to bottom).
        # This differs from the physical length (strut_max_depth) due to
        # heel and pitch: ned_span = max_depth * cos(heel) * cos(theta).
        ned_span = jnp.maximum(bottom_depth - top_depth, 1e-10)

        # NED submerged span: how much of the NED depth range is below water
        ned_submerged = jnp.clip(
            bottom_depth - jnp.maximum(0.0, top_depth),
            0.0,
            ned_span,
        )

        # Immersion fraction [0, 1] — fraction of physical strut in water
        immersion = ned_submerged / ned_span

        # Physical submerged length along the strut
        submerged_length = immersion * self.strut_max_depth

        # Pressure drag: frontal area = thickness * submerged length
        frontal_area = submerged_length * self.strut_thickness
        drag_pressure = 0.5 * self.rho * self.strut_cd_pressure * frontal_area * u_forward ** 2

        # Skin friction drag: wetted area = 2 * chord * submerged length (both sides)
        wetted_area = 2.0 * self.strut_chord * submerged_length
        drag_skin = 0.5 * self.rho * self.strut_cf_skin * wetted_area * u_forward ** 2

        drag_total = drag_pressure + drag_skin

        # Force opposes forward motion
        fx = -drag_total
        force_b = jnp.array([fx, 0.0, 0.0])

        # Moment arm: centroid of submerged segment in body frame.
        # The submerged segment spans the bottom fraction of the strut.
        # Its body-frame centroid z is at:
        #   eff_bottom_z - submerged_length/2
        centroid_z = eff_bottom_z - submerged_length / 2.0

        # M_y = r_z * F_x - r_x * F_z
        my = centroid_z * fx

        moment_b = jnp.array([0.0, my, 0.0])

        return force_b, moment_b

    def compute_immersion(
        self,
        state: Array,
        cg_offset: Optional[Array] = None,
    ) -> Array:
        """Compute strut immersion fraction (0=dry, 1=fully submerged).

        Utility method for aux output logging.
        """
        _cg = cg_offset if cg_offset is not None else jnp.zeros(3)
        eff_pos_x = self.strut_position_x - _cg[0]
        eff_top_z = self.strut_top_z - _cg[2]
        eff_bottom_z = self.strut_bottom_z - _cg[2]

        pos_d = state[POS_D]
        theta = state[THETA]

        top_depth = compute_foil_ned_depth(
            pos_d, eff_pos_x, eff_top_z, theta, self.heel_angle)
        bottom_depth = compute_foil_ned_depth(
            pos_d, eff_pos_x, eff_bottom_z, theta, self.heel_angle)

        ned_span = jnp.maximum(bottom_depth - top_depth, 1e-10)
        ned_submerged = jnp.clip(
            bottom_depth - jnp.maximum(0.0, top_depth),
            0.0,
            ned_span,
        )
        return ned_submerged / ned_span


def create_moth_components(
    params: "MothParams",
    heel_angle: float = np.deg2rad(30.0),
    ventilation_mode: str = "smooth",
    ventilation_threshold: float = 0.30,
    cg_offset: Optional[np.ndarray] = None,
) -> tuple[MothMainFoil, MothRudderElevator, MothSailForce, MothHullDrag,
           MothStrutDrag, MothStrutDrag]:
    """Create all Moth force components from MothParams.

    Factory function that constructs each force component using
    the appropriate parameters from a MothParams instance.

    Args:
        params: MothParams instance with validated model parameters.
        heel_angle: Static heel angle (rad), >= 0. Default 30 deg
            (nominal foiling heel). Override for other conditions.
        ventilation_mode: "smooth" (default) or "binary".
        ventilation_threshold: Exposed span fraction for ventilation onset.
        cg_offset: System CG offset from hull CG [x, y, z]. When provided,
            all component positions are adjusted so they are relative to
            the system CG rather than the hull CG.

    Returns:
        Tuple of (main_foil, rudder, sail, hull_drag, main_strut, rudder_strut).
    """
    # Adjust positions relative to system CG
    if cg_offset is not None:
        foil_pos = params.main_foil_position - cg_offset
        rudder_pos = params.rudder_position - cg_offset
        sail_pos = params.sail_ce_position - cg_offset
    else:
        foil_pos = params.main_foil_position
        rudder_pos = params.rudder_position
        sail_pos = params.sail_ce_position

    main_foil = MothMainFoil(
        rho=params.rho_water,
        area=params.main_foil_area,
        cl_alpha=params.main_foil_cl_alpha,
        cl0=params.main_foil_cl0,
        cd0=params.main_foil_cd0,
        oswald=params.main_foil_oswald,
        ar=params.main_foil_aspect_ratio,
        flap_effectiveness=params.main_foil_flap_effectiveness,
        cd_flap=params.main_foil_cd_flap,
        position_x=float(foil_pos[0]),
        position_z=float(foil_pos[2]),
        foil_span=params.main_foil_span,
        heel_angle=heel_angle,
        ventilation_threshold=ventilation_threshold,
        ventilation_mode=ventilation_mode,
    )

    rudder = MothRudderElevator(
        rho=params.rho_water,
        area=params.rudder_area,
        cl_alpha=params.rudder_cl_alpha,
        cd0=params.rudder_cd0,
        oswald=params.rudder_oswald,
        ar=params.rudder_aspect_ratio,
        position_x=float(rudder_pos[0]),
        position_z=float(rudder_pos[2]),
        foil_span=params.rudder_span,
        heel_angle=heel_angle,
        ventilation_threshold=ventilation_threshold,
        ventilation_mode=ventilation_mode,
    )

    # Build lookup table arrays (empty arrays if no table configured)
    if params.sail_thrust_speeds:
        thrust_speeds_arr = jnp.array(params.sail_thrust_speeds)
        thrust_values_arr = jnp.array(params.sail_thrust_values)
    else:
        thrust_speeds_arr = jnp.array([])
        thrust_values_arr = jnp.array([])

    sail = MothSailForce(
        thrust_coeff=params.sail_thrust_coeff,
        thrust_slope=params.sail_thrust_slope,
        ce_position_x=float(sail_pos[0]),
        ce_position_z=float(sail_pos[2]),
        thrust_speeds=thrust_speeds_arr,
        thrust_values=thrust_values_arr,
    )

    hull_drag = MothHullDrag(
        drag_coeff=params.hull_drag_coeff,
        contact_depth=params.hull_contact_depth,
        hull_cg_above_bottom=params.hull_cg_above_bottom,
        buoyancy_coeff=params.hull_buoyancy_coeff,
        buoyancy_fwd_x=params.hull_length / 4,
        buoyancy_aft_x=-params.hull_length / 4,
    )

    # Strut drag components
    # Strut x-position matches the foil it supports.
    # Strut extends from hull bottom (top) down to the foil (bottom).
    # Hull bottom is at z=hull_cg_above_bottom in body FRD (below CG).
    # strut_max_depth = physical strut depth parameter (main_foil_strut_depth).
    hull_bottom_z = params.hull_cg_above_bottom  # body FRD: below CG
    main_strut_bottom_z = float(foil_pos[2])
    main_strut_max_depth = params.main_foil_strut_depth
    main_strut = MothStrutDrag(
        strut_chord=params.main_strut_chord,
        strut_thickness=params.main_strut_thickness,
        strut_cd_pressure=params.main_strut_cd_pressure,
        strut_cf_skin=params.main_strut_cf_skin,
        strut_position_x=float(foil_pos[0]),
        strut_max_depth=main_strut_max_depth,
        strut_top_z=hull_bottom_z,
        strut_bottom_z=main_strut_bottom_z,
        heel_angle=heel_angle,
        rho=params.rho_water,
    )

    rudder_strut_bottom_z = float(rudder_pos[2])
    rudder_strut_max_depth = params.rudder_strut_depth
    rudder_strut = MothStrutDrag(
        strut_chord=params.rudder_strut_chord,
        strut_thickness=params.rudder_strut_thickness,
        strut_cd_pressure=params.rudder_strut_cd_pressure,
        strut_cf_skin=params.rudder_strut_cf_skin,
        strut_position_x=float(rudder_pos[0]),
        strut_max_depth=rudder_strut_max_depth,
        strut_top_z=hull_bottom_z,
        strut_bottom_z=rudder_strut_bottom_z,
        heel_angle=heel_angle,
        rho=params.rho_water,
    )

    return main_foil, rudder, sail, hull_drag, main_strut, rudder_strut
