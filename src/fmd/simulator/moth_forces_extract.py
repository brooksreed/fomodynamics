"""Post-hoc force extraction for Moth 3DOF simulations.

Recomputes per-component forces from saved (state, control, t) triples.
Forces are deterministic pure functions of state/control, so recomputation
is mathematically identical to inline logging during simulation.

Example:
    from fmd.simulator import Moth3D, simulate
    from fmd.simulator.params import MOTH_BIEKER_V3
    from fmd.simulator.moth_forces_extract import extract_forces

    moth = Moth3D(MOTH_BIEKER_V3)
    result = simulate(moth, moth.default_state(), dt=0.005, duration=5.0)
    forces = extract_forces(moth, result)
    # forces.main_foil_force has shape (num_steps, 3)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from fmd.simulator.moth_3d import Moth3D, THETA, U, POS_D, _compute_cg_offset
from fmd.simulator.components.moth_forces import compute_foil_ned_depth
from fmd.simulator.integrator import SimulationResult


@dataclass
class MothForceLog:
    """Per-component force/moment arrays from a Moth 3DOF simulation.

    All arrays have shape (num_steps, 3) with body-frame [Fx, Fy, Fz]
    or [Mx, My, Mz] components.
    """
    times: np.ndarray            # (num_steps,)
    main_foil_force: np.ndarray  # (num_steps, 3)
    main_foil_moment: np.ndarray
    rudder_force: np.ndarray
    rudder_moment: np.ndarray
    sail_force: np.ndarray
    sail_moment: np.ndarray
    hull_drag_force: np.ndarray
    hull_drag_moment: np.ndarray
    gravity_force: np.ndarray    # (num_steps, 3) body-frame gravity
    strut_main_force: np.ndarray  # (num_steps, 3) main strut drag
    strut_main_moment: np.ndarray
    strut_rudder_force: np.ndarray  # (num_steps, 3) rudder strut drag
    strut_rudder_moment: np.ndarray


def extract_forces(moth: Moth3D, result: SimulationResult, env=None) -> MothForceLog:
    """Recompute per-component forces from simulation results.

    Mirrors the force computation in Moth3D.forward_dynamics(), calling
    each component's compute_moth() method with the saved state/control/time.
    Uses jax.vmap for efficient batch computation.

    Args:
        moth: The Moth3D model instance used for the simulation.
        result: SimulationResult with times, states, controls arrays.
        env: Optional Environment with wave/wind/current fields. When
            provided, wave elevation and orbital velocities are computed
            at each foil position (same logic as Moth3D._compute_step_terms).
            When None, zeros are passed (calm water).

    Returns:
        MothForceLog with per-component force/moment arrays.
    """
    times = result.times
    states = result.states
    controls = result.controls

    def _forces_at_step(state, control, t):
        """Compute all component forces at a single timestep."""
        u_fwd = jnp.where(moth.surge_enabled, state[U], moth.u_forward_schedule(t))
        u_safe = jnp.maximum(u_fwd, 0.1)

        r_sailor = moth.sailor_position_schedule(t)
        cg_offset = _compute_cg_offset(r_sailor, moth.sailor_mass, moth.total_mass)

        # Wave data: compute per-foil elevation and orbital velocities
        theta = state[THETA]
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        eff_main_x = moth.main_foil.position_x - cg_offset[0]
        eff_main_z = moth.main_foil.position_z - cg_offset[2]
        eff_rudder_x = moth.rudder.position_x - cg_offset[0]
        eff_rudder_z = moth.rudder.position_z - cg_offset[2]

        if env is not None and env.wave_field is not None:
            x_main_ned = u_safe * t + eff_main_x * cos_theta + eff_main_z * sin_theta
            x_rudder_ned = u_safe * t + eff_rudder_x * cos_theta + eff_rudder_z * sin_theta

            eta_main = env.wave_field.elevation(x_main_ned, 0.0, t)
            eta_rudder = env.wave_field.elevation(x_rudder_ned, 0.0, t)

            main_foil_z_ned = compute_foil_ned_depth(
                state[POS_D], eff_main_x, eff_main_z, theta, moth.main_foil.heel_angle)
            rudder_foil_z_ned = compute_foil_ned_depth(
                state[POS_D], eff_rudder_x, eff_rudder_z, theta, moth.rudder.heel_angle)

            orb_main_ned = env.wave_field.orbital_velocity(x_main_ned, 0.0, main_foil_z_ned, t)
            orb_rudder_ned = env.wave_field.orbital_velocity(x_rudder_ned, 0.0, rudder_foil_z_ned, t)

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

        f_foil, m_foil = moth.main_foil.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset,
            eta=eta_main, u_orbital_body=u_orbital_body_main, w_orbital_body=w_orbital_body_main,
        )
        f_rudder, m_rudder = moth.rudder.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset,
            eta=eta_rudder, u_orbital_body=u_orbital_body_rudder, w_orbital_body=w_orbital_body_rudder,
        )
        f_sail, m_sail = moth.sail.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset
        )
        f_hull, m_hull = moth.hull_drag.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset
        )

        # Strut drag forces
        f_strut_main, m_strut_main = moth.main_strut.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset
        )
        f_strut_rudder, m_strut_rudder = moth.rudder_strut.compute_moth(
            state, control, u_safe, t, cg_offset=cg_offset
        )

        # Gravity in body frame (same as moth_3d.py forward_dynamics)
        gravity_fx = -moth.total_mass * moth.g * jnp.sin(theta)
        gravity_fy = 0.0
        gravity_fz = moth.total_mass * moth.g * jnp.cos(theta)
        f_gravity = jnp.array([gravity_fx, gravity_fy, gravity_fz])

        return (f_foil, m_foil, f_rudder, m_rudder,
                f_sail, m_sail, f_hull, m_hull, f_gravity,
                f_strut_main, m_strut_main, f_strut_rudder, m_strut_rudder)

    # vmap over time axis
    all_forces = jax.vmap(_forces_at_step)(states, controls, times)

    # all_forces is a tuple of (num_steps, 3) arrays
    (f_foil, m_foil, f_rudder, m_rudder,
     f_sail, m_sail, f_hull, m_hull, f_gravity,
     f_strut_main, m_strut_main, f_strut_rudder, m_strut_rudder) = all_forces

    return MothForceLog(
        times=np.asarray(times),
        main_foil_force=np.asarray(f_foil),
        main_foil_moment=np.asarray(m_foil),
        rudder_force=np.asarray(f_rudder),
        rudder_moment=np.asarray(m_rudder),
        sail_force=np.asarray(f_sail),
        sail_moment=np.asarray(m_sail),
        hull_drag_force=np.asarray(f_hull),
        hull_drag_moment=np.asarray(m_hull),
        gravity_force=np.asarray(f_gravity),
        strut_main_force=np.asarray(f_strut_main),
        strut_main_moment=np.asarray(m_strut_main),
        strut_rudder_force=np.asarray(f_strut_rudder),
        strut_rudder_moment=np.asarray(m_strut_rudder),
    )
