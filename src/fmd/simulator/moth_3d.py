"""Moth 3DOF longitudinal dynamics (pitch + heave).

This module provides a reduced-order model for Moth sailboat foiling,
focusing on longitudinal dynamics only (no lateral/yaw motion).

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

Forward Speed:
    u_forward is treated as a time-varying input for future surge
    dynamics extension. Default is constant 10 m/s (~20 kt).

Frame Convention:
    - Body frame: +x forward, +y starboard, +z down
    - NED world frame at theta=0
    - Pitch positive nose-up (rotation about +y)

Example:
    from fmd.simulator import Moth3D, simulate
    from fmd.simulator.params import MOTH_BIEKER_V3
    import jax.numpy as jnp

    moth = Moth3D(MOTH_BIEKER_V3)
    result = simulate(moth, moth.default_state(), dt=0.005, duration=10.0)
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing import NamedTuple, Optional, Tuple

from dataclasses import dataclass as py_dataclass

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.params import MothParams
from fmd.simulator.components.moth_forces import (
    MothMainFoil,
    MothRudderElevator,
    MothSailForce,
    MothHullDrag,
    MothStrutDrag,
    compute_foil_ned_depth,
    compute_depth_factor,
    create_moth_components,
)


# State vector indices
POS_D = 0   # Vertical position (m)
THETA = 1   # Pitch angle (rad)
W = 2       # Body-frame vertical velocity (m/s)
Q = 3       # Body-frame pitch rate (rad/s)
U = 4       # Body-frame surge velocity (m/s)

# Control vector indices
MAIN_FLAP = 0         # Main foil flap angle (rad)
RUDDER_ELEVATOR = 1   # Rudder elevator angle (rad)

# Default control bounds (radians) - used when not overridden by params
# Main flap: -10° to +15°
MAIN_FLAP_MIN = np.deg2rad(-10)   # -0.1745
MAIN_FLAP_MAX = np.deg2rad(15)    # +0.2618

# Default rudder elevator bounds (used by legacy code and as defaults)
RUDDER_ELEVATOR_MIN = np.deg2rad(-3)   # -0.0524
RUDDER_ELEVATOR_MAX = np.deg2rad(6)    # +0.1047

# Finite difference epsilon for sailor velocity (seconds)
_SAILOR_VEL_EPS = 1e-4

# Minimum forward speed (m/s) to avoid division by zero in aero/hydro calculations
_MIN_FORWARD_SPEED = 0.1


class ConstantSchedule(eqx.Module):
    """Time-invariant scalar schedule (e.g., forward speed).

    The value is stored as a JAX scalar array so that eqx.filter_jit
    treats it as a traced leaf (not static). This is the key to avoiding
    retraces when the speed changes.
    """
    value: Array

    def __init__(self, value):
        self.value = jnp.asarray(float(value))

    def __call__(self, t):
        return self.value


class ConstantArraySchedule(eqx.Module):
    """Time-invariant array schedule (e.g., sailor position)."""
    value: Array

    def __init__(self, value):
        self.value = jnp.asarray(value)

    def __call__(self, t):
        return self.value


def _compute_cg_offset(r_sailor, sailor_mass, total_mass):
    """Compute system CG offset from hull CG due to sailor position."""
    return sailor_mass * r_sailor / total_mass


def _compute_composite_iyy(hull_iyy, hull_mass, sailor_mass, total_mass, r_sailor):
    """Compute composite pitch inertia about system CG (reduced-mass parallel axis theorem)."""
    reduced_mass = hull_mass * sailor_mass / total_mass
    return hull_iyy + reduced_mass * (r_sailor[0] ** 2 + r_sailor[2] ** 2)


@py_dataclass(frozen=True)
class MothGeometry:
    """Effective geometry of Moth3D accounting for CG offset from sailor mass.

    Positions are relative to the composite system CG, not the hull CG.
    """
    cg_offset: np.ndarray
    main_foil_position: np.ndarray
    rudder_position: np.ndarray
    main_foil_raw_position: np.ndarray
    rudder_raw_position: np.ndarray
    composite_iyy: float


class StepTerms(NamedTuple):
    """Per-step intermediates shared by forward_dynamics and compute_aux."""
    u_fwd: Array
    u_safe: Array
    cg_offset: Array
    iyy: Array
    f_foil: Array
    m_foil: Array
    f_rudder: Array
    m_rudder: Array
    f_sail: Array
    m_sail: Array
    f_hull: Array
    m_hull: Array
    f_main_strut: Array
    m_main_strut: Array
    f_rudder_strut: Array
    m_rudder_strut: Array
    total_fz: Array
    total_fx: Array
    total_my: Array
    pos_d_dot: Array
    main_df: Array
    rudder_df: Array
    main_foil_depth: Array
    rudder_foil_depth: Array
    main_alpha_geo: Array
    main_alpha_eff: Array
    rudder_alpha_geo: Array
    rudder_alpha_eff: Array
    main_lift_aero: Array
    main_drag_aero: Array
    rudder_lift_aero: Array
    rudder_drag_aero: Array
    # Wave aux outputs
    wave_eta_main: Array
    wave_eta_rudder: Array
    wave_u_orbital_main: Array
    wave_w_orbital_main: Array
    wave_u_orbital_rudder: Array
    wave_w_orbital_rudder: Array


# Aux output names for Moth3D (stable order)
_MOTH_AUX_NAMES = (
    "main_df", "rudder_df",
    "main_foil_depth", "rudder_foil_depth",
    "pos_d_dot",
    "main_lift_aero", "main_drag_aero",
    "rudder_lift_aero", "rudder_drag_aero",
    "sail_force", "hull_drag", "hull_buoyancy",
    "cg_offset_x", "cg_offset_z",
    "main_alpha_geo", "main_alpha_eff",
    "rudder_alpha_geo", "rudder_alpha_eff",
    "total_fx", "total_fz", "total_my",
    "u_fwd",
    "main_strut_drag", "rudder_strut_drag",
    "main_strut_immersion", "rudder_strut_immersion",
    "wave_eta_main", "wave_eta_rudder",
    "wave_u_orbital_main", "wave_w_orbital_main",
    "wave_u_orbital_rudder", "wave_w_orbital_rudder",
)


class Moth3D(JaxDynamicSystem):
    """Moth 3DOF longitudinal dynamics (pitch + heave + optional surge).

    Implements longitudinal (pitch + heave) dynamics with component-based
    force model including main foil, rudder elevator, sail, and hull drag.
    Optionally includes surge dynamics when surge_enabled=True.

    State vector (5 elements):
        [0] pos_d - Vertical position, positive down (m)
        [1] theta - Pitch angle, positive nose-up (rad)
        [2] w     - Body-frame vertical velocity, positive down (m/s)
        [3] q     - Body-frame pitch rate, positive nose-up (rad/s)
        [4] u     - Body-frame surge velocity, positive forward (m/s)

    Control vector (2 elements):
        [0] main_flap_angle       - Main foil flap deflection (rad)
        [1] rudder_elevator_angle - Rudder elevator deflection (rad)

    Attributes:
        total_mass: Combined hull + sailor mass (kg)
        iyy: Pitch moment of inertia (kg*m^2)
        g: Gravitational acceleration (m/s^2)
        main_foil: Main T-foil force component
        rudder: Rudder elevator force component
        sail: Sail force component
        hull_drag: Hull contact drag component
        u_forward_schedule: Time-varying forward speed function (m/s)
        control_bounds: Tuple of (min, max) for each control

    Example:
        from fmd.simulator.params import MOTH_BIEKER_V3

        moth = Moth3D(MOTH_BIEKER_V3)
        state = moth.default_state()  # [-1.3, 0.0, 0.0, 0.0, 10.0]
        deriv = moth.forward_dynamics(state, moth.default_control())
    """

    # Physical parameters
    total_mass: float
    iyy: float
    g: float
    rho_water: float
    added_mass_heave: float
    added_inertia_pitch: float
    added_mass_surge: float
    hull_mass: float
    sailor_mass: float
    hull_iyy: float

    # Surge dynamics flag
    surge_enabled: bool = eqx.field(static=True, default=True)

    # Lift lag (Wagner-type first-order filter)
    enable_lift_lag: bool = eqx.field(static=True, default=False)
    main_foil_chord: float = eqx.field(default=0.089)
    rudder_chord: float = eqx.field(default=0.075)

    # Force components (Equinox modules)
    main_foil: MothMainFoil
    rudder: MothRudderElevator
    sail: MothSailForce
    hull_drag: MothHullDrag
    main_strut: MothStrutDrag
    rudder_strut: MothStrutDrag

    # Forward speed schedule (time-varying) — eqx.Module pytree leaf, not static
    u_forward_schedule: ConstantSchedule  # or any eqx.Module with __call__(t) -> float

    # Sailor position schedule (time-varying) — eqx.Module pytree leaf, not static
    sailor_position_schedule: ConstantArraySchedule  # or any eqx.Module with __call__(t) -> Array

    # Control bounds (set from params in __init__)
    control_bounds: Tuple[Tuple[float, float], ...] = eqx.field(static=True)

    # Static metadata
    state_names: Tuple[str, ...] = eqx.field(
        static=True, default=("pos_d", "theta", "w", "q", "u")
    )
    control_names: Tuple[str, ...] = eqx.field(
        static=True, default=("main_flap_angle", "rudder_elevator_angle")
    )
    aux_names: Tuple[str, ...] = eqx.field(
        static=True, default=_MOTH_AUX_NAMES
    )

    def __init__(
        self,
        params: MothParams,
        u_forward: Optional[eqx.Module] = None,
        heel_angle: float = np.deg2rad(30.0),
        ventilation_mode: str = "smooth",
        ventilation_threshold: float = 0.30,
        sailor_position_schedule: Optional[eqx.Module] = None,
        surge_enabled: bool = True,
        enable_lift_lag: bool = False,
    ):
        """Initialize Moth 3DOF model from parameters.

        Args:
            params: MothParams instance with validated model parameters.
            u_forward: Optional forward speed schedule (eqx.Module).
                       Default: ConstantSchedule(10.0) (~20 kt).
            heel_angle: Static heel angle (rad), >= 0. Default 30 deg
                       (nominal foiling heel for Moth class boats).
            ventilation_mode: "smooth" (default) or "binary".
            ventilation_threshold: Exposed span fraction for ventilation onset.
            sailor_position_schedule: Optional sailor position schedule
                (eqx.Module) returning [x, y, z] array. Default: constant at
                params.sailor_position. Enables "what-if" studies without
                adding state dimensions.
            surge_enabled: If True, surge velocity u is a dynamic state
                with forces accumulated from all components. If False,
                u_dot=0 and forward speed comes from u_forward_schedule.
                Default: True.
            enable_lift_lag: If True, add Wagner-type first-order lift lag
                filter with 2 extra states (alpha_filt_main, alpha_filt_rudder).
                State dimension becomes 7 instead of 5.
        """
        self.surge_enabled = surge_enabled
        self.enable_lift_lag = enable_lift_lag
        # Read control bounds from params
        self.control_bounds = (
            (float(MAIN_FLAP_MIN), float(MAIN_FLAP_MAX)),
            (float(params.rudder_elevator_min), float(params.rudder_elevator_max)),
        )
        self.rho_water = params.rho_water
        self.total_mass = params.total_mass
        # Nominal composite Iyy for inspection/tests. Not used in dynamics —
        # forward_dynamics() recomputes Iyy per-timestep from sailor schedule.
        self.iyy = params.composite_pitch_inertia
        self.g = params.g
        self.added_mass_heave = params.added_mass_heave
        self.added_inertia_pitch = params.added_inertia_pitch
        self.added_mass_surge = params.added_mass_surge
        self.hull_mass = params.hull_mass
        self.sailor_mass = params.sailor_mass
        self.hull_iyy = float(params.hull_inertia_matrix[1, 1])

        # Components store raw hull-CG-relative positions (no CG adjustment).
        # CG offset is applied per-timestep in forward_dynamics().
        foil, rudder, sail, hull, main_strut, rudder_strut = create_moth_components(
            params,
            heel_angle=heel_angle,
            ventilation_mode=ventilation_mode,
            ventilation_threshold=ventilation_threshold,
            cg_offset=None,
        )
        self.main_foil = foil
        self.rudder = rudder
        self.sail = sail
        self.hull_drag = hull
        self.main_strut = main_strut
        self.rudder_strut = rudder_strut

        # Chord values for lift lag time constant
        self.main_foil_chord = params.main_foil_chord
        self.rudder_chord = params.rudder_chord

        if u_forward is None:
            self.u_forward_schedule = ConstantSchedule(10.0)  # Default ~20 kt
        else:
            self.u_forward_schedule = u_forward

        if sailor_position_schedule is None:
            self.sailor_position_schedule = ConstantArraySchedule(
                jnp.array([float(params.sailor_position[0]),
                            float(params.sailor_position[1]),
                            float(params.sailor_position[2])]))
        else:
            self.sailor_position_schedule = sailor_position_schedule

        # State and aux names depend on lift lag
        if enable_lift_lag:
            self.state_names = ("pos_d", "theta", "w", "q", "u", "alpha_filt_main", "alpha_filt_rudder")
            self.aux_names = _MOTH_AUX_NAMES + ("main_alpha_filt", "rudder_alpha_filt", "main_lift_filtered", "rudder_lift_filtered")
        else:
            self.state_names = ("pos_d", "theta", "w", "q", "u")
            self.aux_names = _MOTH_AUX_NAMES

    @property
    def num_states(self) -> int:
        """Number of state variables (5 or 7 with lift lag)."""
        return 7 if self.enable_lift_lag else 5

    @property
    def num_controls(self) -> int:
        """Number of control inputs (2)."""
        return 2

    @property
    def control_lower_bounds(self) -> Array:
        """Lower bounds for control inputs."""
        return jnp.array([self.control_bounds[0][0], self.control_bounds[1][0]])

    @property
    def control_upper_bounds(self) -> Array:
        """Upper bounds for control inputs."""
        return jnp.array([self.control_bounds[0][1], self.control_bounds[1][1]])

    def default_state(self) -> Array:
        """Default state: CG above water, level, zero velocities, nominal speed.

        Returns:
            Initial state [pos_d=-1.3, theta=0, w=0, q=0, u=u_forward(0)]
            With lift lag: appends [alpha_filt_main=0, alpha_filt_rudder=0]
            pos_d=-1.3 places hull bottom above water in the new geometry
            (hull_contact_depth ~0.94, so hull bottom at -1.3 + 0.94 = -0.36 < 0).
        """
        base = jnp.array([-1.3, 0.0, 0.0, 0.0, self.u_forward_schedule(0.0)])
        if self.enable_lift_lag:
            return jnp.concatenate([base, jnp.zeros(2)])
        return base

    def default_control(self) -> Array:
        """Default control: zero deflection.

        Returns:
            Zero control [0.0, 0.0]
        """
        return jnp.zeros(2)

    def get_geometry(self, t: float = 0.0) -> MothGeometry:
        """Return effective geometry at time t, accounting for CG offset from sailor mass.

        Positions are relative to composite system CG (not hull CG).
        This is an analysis/debug API (Python/NumPy), not for JIT hot paths.
        """
        r_sailor = self.sailor_position_schedule(t)
        cg_offset = _compute_cg_offset(r_sailor, self.sailor_mass, self.total_mass)
        iyy = _compute_composite_iyy(
            self.hull_iyy, self.hull_mass, self.sailor_mass,
            self.total_mass, r_sailor,
        )
        return MothGeometry(
            cg_offset=np.asarray(cg_offset),
            main_foil_position=np.array([
                self.main_foil.position_x - float(cg_offset[0]),
                0.0,
                self.main_foil.position_z - float(cg_offset[2]),
            ]),
            rudder_position=np.array([
                self.rudder.position_x - float(cg_offset[0]),
                0.0,
                self.rudder.position_z - float(cg_offset[2]),
            ]),
            main_foil_raw_position=np.array([
                self.main_foil.position_x, 0.0, self.main_foil.position_z,
            ]),
            rudder_raw_position=np.array([
                self.rudder.position_x, 0.0, self.rudder.position_z,
            ]),
            composite_iyy=float(iyy),
        )

    def _compute_step_terms(self, state, control, t, env=None):
        """Compute all per-step intermediates used by both dynamics and aux.

        All wave queries are centralized here. Per-foil NED positions are
        computed from body-frame offsets and pitch angle. Wave elevation
        and full orbital velocity (horizontal + vertical) are queried at
        each foil, rotated to body frame, and passed to force components.

        Returns a StepTerms NamedTuple. Both forward_dynamics and
        compute_aux consume this to ensure exact consistency.
        """
        u_fwd = jnp.where(self.surge_enabled, state[U], self.u_forward_schedule(t))
        u_safe = jnp.maximum(u_fwd, _MIN_FORWARD_SPEED)

        r_sailor = self.sailor_position_schedule(t)
        cg_offset = _compute_cg_offset(r_sailor, self.sailor_mass, self.total_mass)
        iyy = _compute_composite_iyy(
            self.hull_iyy, self.hull_mass, self.sailor_mass,
            self.total_mass, r_sailor,
        )

        theta = state[THETA]

        # Effective foil positions relative to system CG
        eff_main_x = self.main_foil.position_x - cg_offset[0]
        eff_main_z = self.main_foil.position_z - cg_offset[2]
        eff_rudder_x = self.rudder.position_x - cg_offset[0]
        eff_rudder_z = self.rudder.position_z - cg_offset[2]

        # ------------------------------------------------------------------
        # Centralized wave queries: per-foil NED positions
        # ------------------------------------------------------------------
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        if env is not None and env.wave_field is not None:
            # Per-foil NED-north positions (body offset projected to NED)
            x_main_ned = u_safe * t + eff_main_x * cos_theta + eff_main_z * sin_theta
            x_rudder_ned = u_safe * t + eff_rudder_x * cos_theta + eff_rudder_z * sin_theta

            # Wave elevation at each foil
            eta_main = env.wave_field.elevation(x_main_ned, 0.0, t)
            eta_rudder = env.wave_field.elevation(x_rudder_ned, 0.0, t)

            # NED depth of each foil (without eta, for orbital velocity query)
            main_foil_z_ned = compute_foil_ned_depth(
                state[POS_D], eff_main_x, eff_main_z, theta, self.main_foil.heel_angle)
            rudder_foil_z_ned = compute_foil_ned_depth(
                state[POS_D], eff_rudder_x, eff_rudder_z, theta, self.rudder.heel_angle)

            # Full orbital velocity in NED at each foil
            orb_main_ned = env.wave_field.orbital_velocity(x_main_ned, 0.0, main_foil_z_ned, t)
            orb_rudder_ned = env.wave_field.orbital_velocity(x_rudder_ned, 0.0, rudder_foil_z_ned, t)

            # NED -> body frame rotation: Ry(theta)
            # body_x =  ned_n * cos(theta) + ned_d * sin(theta)
            # body_z = -ned_n * sin(theta) + ned_d * cos(theta)
            u_orbital_body_main = orb_main_ned[0] * cos_theta + orb_main_ned[2] * sin_theta
            w_orbital_body_main = -orb_main_ned[0] * sin_theta + orb_main_ned[2] * cos_theta

            u_orbital_body_rudder = orb_rudder_ned[0] * cos_theta + orb_rudder_ned[2] * sin_theta
            w_orbital_body_rudder = -orb_rudder_ned[0] * sin_theta + orb_rudder_ned[2] * cos_theta
        else:
            eta_main = 0.0
            eta_rudder = 0.0
            u_orbital_body_main = 0.0
            w_orbital_body_main = 0.0
            u_orbital_body_rudder = 0.0
            w_orbital_body_rudder = 0.0

        # ------------------------------------------------------------------
        # Force components (wave data passed as parameters)
        # ------------------------------------------------------------------
        # Lift lag: pass filtered alpha as alpha_override when enabled
        if self.enable_lift_lag:
            alpha_filt_main = state[5]
            alpha_filt_rudder = state[6]
        else:
            alpha_filt_main = None
            alpha_filt_rudder = None

        f_foil, m_foil = self.main_foil.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset,
            eta=eta_main, u_orbital_body=u_orbital_body_main, w_orbital_body=w_orbital_body_main,
            alpha_override=alpha_filt_main)
        f_rudder, m_rudder = self.rudder.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset,
            eta=eta_rudder, u_orbital_body=u_orbital_body_rudder, w_orbital_body=w_orbital_body_rudder,
            alpha_override=alpha_filt_rudder)
        f_sail, m_sail = self.sail.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset)
        f_hull, m_hull = self.hull_drag.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset)
        f_main_strut, m_main_strut = self.main_strut.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset)
        f_rudder_strut, m_rudder_strut = self.rudder_strut.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset)

        gravity_fz = self.total_mass * self.g * cos_theta
        gravity_fx = -self.total_mass * self.g * sin_theta

        total_fz = (f_foil[2] + f_rudder[2] + f_sail[2] + f_hull[2]
                     + f_main_strut[2] + f_rudder_strut[2] + gravity_fz)
        total_fx = (f_foil[0] + f_rudder[0] + f_sail[0] + f_hull[0]
                     + f_main_strut[0] + f_rudder_strut[0] + gravity_fx)
        total_my = (m_foil[1] + m_rudder[1] + m_sail[1] + m_hull[1]
                     + m_main_strut[1] + m_rudder_strut[1])

        r_s_plus = self.sailor_position_schedule(t + _SAILOR_VEL_EPS)
        r_s_minus = self.sailor_position_schedule(t - _SAILOR_VEL_EPS)
        r_dot_sailor = (r_s_plus - r_s_minus) / (2.0 * _SAILOR_VEL_EPS)
        cg_dot = (self.sailor_mass / self.total_mass) * r_dot_sailor

        pos_d_dot = (
            -u_fwd * sin_theta + state[W] * cos_theta
            - cg_dot[0] * sin_theta + cg_dot[2] * cos_theta
        )

        # ------------------------------------------------------------------
        # Aux: foil depths and depth factors
        # ------------------------------------------------------------------
        main_foil_depth = compute_foil_ned_depth(
            state[POS_D], eff_main_x, eff_main_z, theta,
            self.main_foil.heel_angle, eta=eta_main)
        rudder_foil_depth = compute_foil_ned_depth(
            state[POS_D], eff_rudder_x, eff_rudder_z, theta,
            self.rudder.heel_angle, eta=eta_rudder)

        main_df = compute_depth_factor(
            main_foil_depth, self.main_foil.foil_span,
            self.main_foil.heel_angle, self.main_foil.ventilation_threshold,
            self.main_foil.ventilation_mode)
        rudder_df = compute_depth_factor(
            rudder_foil_depth, self.rudder.foil_span,
            self.rudder.heel_angle, self.rudder.ventilation_threshold,
            self.rudder.ventilation_mode)

        # ------------------------------------------------------------------
        # Aux: AoA and aero coefficients (uses effective flow speed)
        # ------------------------------------------------------------------
        u_eff_main = u_safe + u_orbital_body_main
        u_eff_rudder = u_safe + u_orbital_body_rudder

        w_local_main = state[W] - state[Q] * eff_main_x + w_orbital_body_main
        main_alpha_geo = jnp.arctan2(w_local_main, u_eff_main)
        main_alpha_eff = self.main_foil.flap_effectiveness * control[MAIN_FLAP] + w_local_main / u_eff_main

        w_local_rudder = state[W] - state[Q] * eff_rudder_x + w_orbital_body_rudder
        rudder_alpha_geo = jnp.arctan2(w_local_rudder, u_eff_rudder)
        rudder_alpha_eff = control[RUDDER_ELEVATOR] + w_local_rudder / u_eff_rudder

        q_dyn_main = 0.5 * self.rho_water * u_eff_main**2
        q_dyn_rudder = 0.5 * self.rho_water * u_eff_rudder**2

        main_cl = (self.main_foil.cl0 + self.main_foil.cl_alpha * main_alpha_eff) * main_df
        main_cd = (self.main_foil.cd0
                   + main_cl**2 / (jnp.pi * self.main_foil.ar * self.main_foil.oswald)
                   + self.main_foil.cd_flap * control[MAIN_FLAP]**2)
        main_lift_aero = q_dyn_main * self.main_foil.area * main_cl
        main_drag_aero = q_dyn_main * self.main_foil.area * main_cd

        rudder_cl = self.rudder.cl_alpha * rudder_alpha_eff * rudder_df
        rudder_cd = self.rudder.cd0 + rudder_cl**2 / (jnp.pi * self.rudder.ar * self.rudder.oswald)
        rudder_lift_aero = q_dyn_rudder * self.rudder.area * rudder_cl
        rudder_drag_aero = q_dyn_rudder * self.rudder.area * rudder_cd

        return StepTerms(
            u_fwd=u_fwd,
            u_safe=u_safe,
            cg_offset=cg_offset,
            iyy=iyy,
            f_foil=f_foil, m_foil=m_foil,
            f_rudder=f_rudder, m_rudder=m_rudder,
            f_sail=f_sail, m_sail=m_sail,
            f_hull=f_hull, m_hull=m_hull,
            f_main_strut=f_main_strut, m_main_strut=m_main_strut,
            f_rudder_strut=f_rudder_strut, m_rudder_strut=m_rudder_strut,
            total_fz=total_fz,
            total_fx=total_fx,
            total_my=total_my,
            pos_d_dot=pos_d_dot,
            main_df=main_df,
            rudder_df=rudder_df,
            main_foil_depth=main_foil_depth,
            rudder_foil_depth=rudder_foil_depth,
            main_alpha_geo=main_alpha_geo,
            main_alpha_eff=main_alpha_eff,
            rudder_alpha_geo=rudder_alpha_geo,
            rudder_alpha_eff=rudder_alpha_eff,
            main_lift_aero=main_lift_aero,
            main_drag_aero=main_drag_aero,
            rudder_lift_aero=rudder_lift_aero,
            rudder_drag_aero=rudder_drag_aero,
            wave_eta_main=eta_main,
            wave_eta_rudder=eta_rudder,
            wave_u_orbital_main=u_orbital_body_main,
            wave_w_orbital_main=w_orbital_body_main,
            wave_u_orbital_rudder=u_orbital_body_rudder,
            wave_w_orbital_rudder=w_orbital_body_rudder,
        )

    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative from component forces.

        Uses _compute_step_terms for all intermediate calculations,
        shared with compute_aux for exactness.

        Args:
            state: Current state [pos_d, theta, w, q, u] (5 states)
                   or [pos_d, theta, w, q, u, alpha_filt_main, alpha_filt_rudder] (7 states)
            control: Control input [main_flap_angle, rudder_elevator_angle]
            t: Current simulation time
            env: Optional Environment with wave/wind/current fields.

        Returns:
            State derivative, shape (5,) or (7,) matching input.
        """
        terms = self._compute_step_terms(state, control, t, env=env)

        theta_dot = state[Q]
        m_eff_heave = self.total_mass + self.added_mass_heave
        m_eff_surge = self.total_mass + self.added_mass_surge
        i_eff = terms.iyy + self.added_inertia_pitch

        w_dot = terms.total_fz / m_eff_heave + state[Q] * terms.u_fwd
        q_dot = terms.total_my / i_eff

        u_dot = jnp.where(
            self.surge_enabled,
            terms.total_fx / m_eff_surge - state[Q] * state[W],
            0.0,
        )

        base_deriv = jnp.array([terms.pos_d_dot, theta_dot, w_dot, q_dot, u_dot])

        if self.enable_lift_lag:
            # Wagner-type first-order filter: d(alpha_filt)/dt = (alpha_inst - alpha_filt) / tau
            # tau = 4 * chord / (pi * V)
            u_safe = terms.u_safe
            tau_main = 4.0 * self.main_foil_chord / (jnp.pi * u_safe)
            tau_rudder = 4.0 * self.rudder_chord / (jnp.pi * u_safe)

            alpha_filt_main = state[5]
            alpha_filt_rudder = state[6]

            # Instantaneous alpha_eff from step terms (same as what force components compute)
            alpha_inst_main = terms.main_alpha_eff
            alpha_inst_rudder = terms.rudder_alpha_eff

            d_alpha_main = (alpha_inst_main - alpha_filt_main) / tau_main
            d_alpha_rudder = (alpha_inst_rudder - alpha_filt_rudder) / tau_rudder

            return jnp.concatenate([base_deriv, jnp.array([d_alpha_main, d_alpha_rudder])])

        return base_deriv

    def compute_aux(self, state, control, t=0.0, env=None):
        """Compute auxiliary outputs for logging.

        Returns 32 quantities (or 34 with lift lag) in stable order matching aux_names.
        Uses _compute_step_terms to ensure values are exact with dynamics.
        """
        terms = self._compute_step_terms(state, control, t, env=env)

        sail_force = terms.f_sail[0]
        hull_drag = -terms.f_hull[0]
        hull_buoyancy = -terms.f_hull[2]  # positive = upward lift

        main_strut_drag = -terms.f_main_strut[0]
        rudder_strut_drag = -terms.f_rudder_strut[0]

        # Strut immersion fractions (0=dry, 1=fully submerged)
        main_strut_immersion = self.main_strut.compute_immersion(
            state, cg_offset=terms.cg_offset)
        rudder_strut_immersion = self.rudder_strut.compute_immersion(
            state, cg_offset=terms.cg_offset)

        aux = jnp.array([
            terms.main_df,
            terms.rudder_df,
            terms.main_foil_depth,
            terms.rudder_foil_depth,
            terms.pos_d_dot,
            terms.main_lift_aero,
            terms.main_drag_aero,
            terms.rudder_lift_aero,
            terms.rudder_drag_aero,
            sail_force,
            hull_drag,
            hull_buoyancy,
            terms.cg_offset[0],
            terms.cg_offset[2],
            terms.main_alpha_geo,
            terms.main_alpha_eff,
            terms.rudder_alpha_geo,
            terms.rudder_alpha_eff,
            terms.total_fx,
            terms.total_fz,
            terms.total_my,
            terms.u_fwd,
            main_strut_drag,
            rudder_strut_drag,
            main_strut_immersion,
            rudder_strut_immersion,
            terms.wave_eta_main,
            terms.wave_eta_rudder,
            terms.wave_u_orbital_main,
            terms.wave_w_orbital_main,
            terms.wave_u_orbital_rudder,
            terms.wave_w_orbital_rudder,
        ])

        if self.enable_lift_lag:
            # Filtered lift: CL from filtered alpha (state[5], state[6]),
            # same q_dyn and depth_factor as instantaneous lift_aero.
            alpha_filt_main = state[5]
            alpha_filt_rudder = state[6]

            main_cl_filt = (self.main_foil.cl0 + self.main_foil.cl_alpha * alpha_filt_main) * terms.main_df
            main_lift_filtered = (0.5 * self.rho_water * (terms.u_safe + terms.wave_u_orbital_main) ** 2
                                  * self.main_foil.area * main_cl_filt)

            rudder_cl_filt = self.rudder.cl_alpha * alpha_filt_rudder * terms.rudder_df
            rudder_lift_filtered = (0.5 * self.rho_water * (terms.u_safe + terms.wave_u_orbital_rudder) ** 2
                                    * self.rudder.area * rudder_cl_filt)

            aux = jnp.concatenate([aux, jnp.array([
                alpha_filt_main, alpha_filt_rudder,
                main_lift_filtered, rudder_lift_filtered,
            ])])

        assert aux.shape == (self.num_aux,), f"aux length {aux.shape[0]} != num_aux {self.num_aux}"
        return aux

    def post_step(self, state: Array) -> Array:
        """Post-process state after integration step.

        Wraps theta (pitch angle) to [-pi, pi] for numerical stability.

        Args:
            state: State vector after integration step

        Returns:
            State with wrapped theta
        """
        theta = state[THETA]
        theta_wrapped = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
        return state.at[THETA].set(theta_wrapped)

    def get_forward_speed(self, t: float) -> float:
        """Get forward speed at time t.

        Args:
            t: Simulation time

        Returns:
            Forward speed u (m/s)
        """
        return self.u_forward_schedule(t)
