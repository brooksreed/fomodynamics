"""Rerun .rrd writer for Moth 3DOF simulation playback.

Writes a Moth 3DOF simulation to a .rrd file for viewing in the
Rerun Viewer. The viewer supports timeline scrubbing, 3D camera
control, and scalar time-series plots.

Requires: rerun-sdk (included in default `uv sync`)

Example:
    from fmd.simulator import Moth3D, simulate
    from fmd.simulator.params import MOTH_BIEKER_V3
    from fmd.simulator.moth_forces_extract import extract_forces
    from fmd.analysis.viz3d import write_moth_rrd

    moth = Moth3D(MOTH_BIEKER_V3)
    result = simulate(moth, moth.default_state(), dt=0.01, duration=5.0)
    forces = extract_forces(moth, result)
    write_moth_rrd(MOTH_BIEKER_V3, result, "moth_test.rrd", forces=forces)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fmd.analysis.viz3d._rerun_check import require_rerun
from fmd.analysis.viz3d.geometry import build_moth_wireframe, compute_surface_waterline
from fmd.analysis.viz3d.coordinates import frd_to_rerun, moth_3dof_to_rerun_quat
from fmd.simulator.moth_forces_extract import MothForceLog
from fmd.simulator.integrator import SimulationResult
from fmd.simulator.params import MothParams

# Moth 3DOF state indices
_POS_D = 0
_THETA = 1
_W = 2
_Q = 3
_U = 4
_MAIN_FLAP = 0
_RUDDER_ELEVATOR = 1

# Force colors (RGBA)
_COLOR_MAIN_FOIL = [0, 120, 255, 255]   # blue
_COLOR_RUDDER = [255, 165, 0, 255]      # orange
_COLOR_SAIL = [0, 200, 100, 255]        # green
_COLOR_GRAVITY = [200, 50, 50, 255]     # red
_COLOR_HULL = [128, 128, 128, 255]      # gray
_COLOR_WIREFRAME = [180, 180, 180, 255] # light gray
_COLOR_STRUT = [140, 140, 140, 255]     # medium gray
_COLOR_WATER = [50, 100, 200, 30]       # translucent blue
_COLOR_WATERLINE = [0, 255, 255, 255]   # cyan



def write_moth_rrd(
    params: MothParams,
    result: SimulationResult,
    output_path: str | Path,
    forces: MothForceLog | None = None,
    *,
    force_scale: float = 0.001,
    blueprint: bool = True,
    heel_angle: float = np.deg2rad(30.0),
    water_elevation_fn=None,
) -> Path:
    """Write a Moth 3DOF simulation to a .rrd file.

    Args:
        params: MothParams instance with boat geometry.
        result: SimulationResult with times, states, controls.
        output_path: Path for the output .rrd file.
        forces: Optional MothForceLog with per-component force arrays.
            If not provided, force arrows will not be rendered.
        force_scale: Arrow length per Newton (m/N). Default 0.001 = 1m per 1000N.
        blueprint: Whether to send a default blueprint layout. Default True.
        heel_angle: Heel (roll) angle in radians. Default 30 degrees.
        water_elevation_fn: Optional callable(n, e, t) -> eta for wave fields.

    Returns:
        Path to the written .rrd file.
    """
    require_rerun()
    import rerun as rr
    import rerun.blueprint as rrb

    output_path = Path(output_path)

    times = np.asarray(result.times)
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    num_steps = len(times)

    # Initialize recording
    rr.init("moth_3dof_viz", spawn=False)
    rr.save(str(output_path))

    # Set global coordinate system to Z-up
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- Static geometry ---

    # Water mesh (translucent surface at z=0), child of world/water
    # Per-timestep Transform3D on world/water moves it with the boat
    water_size = 20.0
    water_vertices = np.array([
        [-water_size, -water_size, 0.0],
        [water_size, -water_size, 0.0],
        [water_size, water_size, 0.0],
        [-water_size, water_size, 0.0],
    ])
    water_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    rr.log(
        "world/water/surface",
        rr.Mesh3D(
            vertex_positions=water_vertices,
            triangle_indices=water_indices,
            vertex_colors=[_COLOR_WATER] * 4,
        ),
        static=True,
    )

    # Build wireframe from params (body frame FRD, relative to hull CG)
    wireframe = build_moth_wireframe(params)

    # Shift wireframe from hull-CG-relative to system-CG-relative
    cg_offset = params.combined_cg_offset
    for name in wireframe:
        wireframe[name] = wireframe[name] - cg_offset

    # Convert wireframe vertices from FRD to Rerun frame and log as static
    wireframe_components = {
        "hull_profile": (_COLOR_WIREFRAME, 0.012),
        "hull_deck": (_COLOR_WIREFRAME, 0.008),
        "main_foil": (_COLOR_MAIN_FOIL, 0.008),
        "main_foil_strut": (_COLOR_STRUT, 0.005),
        "rudder": (_COLOR_RUDDER, 0.008),
        "rudder_strut": (_COLOR_STRUT, 0.005),
        "wing_rack_port": (_COLOR_WIREFRAME, 0.008),
        "wing_rack_stbd": (_COLOR_WIREFRAME, 0.008),
        "sail_mast": (_COLOR_SAIL, 0.012),
    }

    for name, (color, radius) in wireframe_components.items():
        verts_rerun = frd_to_rerun(wireframe[name])
        rr.log(
            f"world/boat/{name}",
            rr.LineStrips3D([verts_rerun], colors=[color], radii=[radius]),
            static=True,
        )

    # --- Integrate forward position from surge velocity ---
    # 3DOF doesn't track pos_n directly; integrate u*cos(theta) via forward Euler
    u = states[:, _U]
    theta = states[:, _THETA]
    pos_n_dot = u * np.cos(theta)
    pos_n = np.zeros(num_steps)
    if num_steps > 1:
        pos_n[1:] = np.cumsum(pos_n_dot[:-1] * np.diff(times))

    # --- Position trail (static, logged once with all positions) ---
    # pos_d is NED depth (negative = above water). Rerun Z-up: negate pos_d.
    z_rerun = -states[:, _POS_D]
    positions_rerun = np.column_stack([
        np.zeros(num_steps),  # X_rerun = East = 0
        pos_n,                 # Y_rerun = North = integrated forward position
        z_rerun,               # Z_rerun = up
    ])
    rr.log(
        "world/trail",
        rr.LineStrips3D(
            [positions_rerun],
            colors=[[255, 255, 255, 128]],
            radii=[0.02],
        ),
        static=True,
    )

    # --- Force arrow origins in body frame (FRD -> Rerun) ---
    # Shift from hull-CG-relative to system-CG-relative
    foil_origin_frd = np.array([
        params.main_foil_position[0], 0.0, params.main_foil_position[2]
    ]) - cg_offset
    rudder_origin_frd = np.array([
        params.rudder_position[0], 0.0, params.rudder_position[2]
    ]) - cg_offset
    sail_origin_frd = np.array([
        params.sail_ce_position[0], params.sail_ce_position[1],
        params.sail_ce_position[2]
    ]) - cg_offset
    cg_origin_frd = np.array([0.0, 0.0, 0.0])  # gravity acts at system CG

    foil_origin_rr = frd_to_rerun(foil_origin_frd)
    rudder_origin_rr = frd_to_rerun(rudder_origin_frd)
    sail_origin_rr = frd_to_rerun(sail_origin_frd)
    cg_origin_rr = frd_to_rerun(cg_origin_frd)

    # Extract rectangle vertices (first 4 points) for waterline intersection
    surface_rects = {
        name: wireframe[name][:4]
        for name in ["main_foil", "main_foil_strut", "rudder", "rudder_strut"]
    }

    # --- Per-timestep logging for transforms and forces ---
    for i in range(num_steps):
        rr.set_time("sim_time", duration=float(times[i]))

        # Boat position in NED -> Rerun
        rerun_pos = positions_rerun[i]

        # Boat orientation (pitch + heel) as Rerun xyzw quaternion
        rerun_quat = moth_3dof_to_rerun_quat(states[i, _THETA], heel_angle)

        # Transform for the boat entity
        rr.log(
            "world/boat",
            rr.Transform3D(
                translation=rerun_pos.tolist(),
                quaternion=rr.Quaternion(xyzw=rerun_quat.tolist()),
            ),
        )

        # Water follows boat (XY only, Z stays at 0)
        rr.log(
            "world/water",
            rr.Transform3D(
                translation=[rerun_pos[0], rerun_pos[1], 0.0],
            ),
        )

        # Waterline indicators on all 4 surfaces (foils + struts)
        for surface_name, verts in surface_rects.items():
            wl = compute_surface_waterline(
                vertices_body=verts,
                pos_d=states[i, _POS_D],
                theta=states[i, _THETA],
                heel_angle=heel_angle,
                water_elevation_fn=water_elevation_fn,
                boat_n=pos_n[i],
                boat_e=0.0,
                t=float(times[i]),
            )
            entity = f"world/boat/waterline/{surface_name}"
            if wl is not None:
                rr.log(
                    entity,
                    rr.LineStrips3D(
                        [frd_to_rerun(wl)],
                        colors=[_COLOR_WATERLINE],
                        radii=[0.004],
                    ),
                )
            else:
                rr.log(entity, rr.Clear(recursive=False))

        # Force arrows (only if forces provided)
        if forces is not None:
            # Main foil
            f_foil_rr = frd_to_rerun(forces.main_foil_force[i]) * force_scale
            f_foil_mag = float(np.linalg.norm(forces.main_foil_force[i]))
            rr.log(
                "world/boat/forces/main_foil",
                rr.Arrows3D(
                    origins=[foil_origin_rr],
                    vectors=[f_foil_rr],
                    colors=[_COLOR_MAIN_FOIL],
                    labels=[f"{f_foil_mag:.0f}N"],
                ),
            )

            # Rudder
            f_rudder_rr = frd_to_rerun(forces.rudder_force[i]) * force_scale
            f_rudder_mag = float(np.linalg.norm(forces.rudder_force[i]))
            rr.log(
                "world/boat/forces/rudder",
                rr.Arrows3D(
                    origins=[rudder_origin_rr],
                    vectors=[f_rudder_rr],
                    colors=[_COLOR_RUDDER],
                    labels=[f"{f_rudder_mag:.0f}N"],
                ),
            )

            # Sail
            f_sail_rr = frd_to_rerun(forces.sail_force[i]) * force_scale
            f_sail_mag = float(np.linalg.norm(forces.sail_force[i]))
            rr.log(
                "world/boat/forces/sail",
                rr.Arrows3D(
                    origins=[sail_origin_rr],
                    vectors=[f_sail_rr],
                    colors=[_COLOR_SAIL],
                    labels=[f"{f_sail_mag:.0f}N"],
                ),
            )

            # Gravity
            f_grav_rr = frd_to_rerun(forces.gravity_force[i]) * force_scale
            f_grav_mag = float(np.linalg.norm(forces.gravity_force[i]))
            rr.log(
                "world/boat/forces/gravity",
                rr.Arrows3D(
                    origins=[cg_origin_rr],
                    vectors=[f_grav_rr],
                    colors=[_COLOR_GRAVITY],
                    labels=[f"{f_grav_mag:.0f}N"],
                ),
            )

            # Hull drag
            f_hull_rr = frd_to_rerun(forces.hull_drag_force[i]) * force_scale
            f_hull_mag = float(np.linalg.norm(forces.hull_drag_force[i]))
            rr.log(
                "world/boat/forces/hull_drag",
                rr.Arrows3D(
                    origins=[cg_origin_rr],
                    vectors=[f_hull_rr],
                    colors=[_COLOR_HULL],
                    labels=[f"{f_hull_mag:.0f}N"],
                ),
            )

        # Flap angle indicator (small line at main foil trailing edge)
        flap_rad = controls[i, _MAIN_FLAP]
        flap_len = 0.15  # indicator length in meters
        # Trailing edge is aft of foil position
        te_x = foil_origin_frd[0] - params.main_foil_chord / 2
        te_z = foil_origin_frd[2]
        flap_end_x = te_x - flap_len * np.cos(flap_rad)
        flap_end_z = te_z + flap_len * np.sin(flap_rad)
        flap_line_frd = np.array([
            [te_x, 0.0, te_z],
            [flap_end_x, 0.0, flap_end_z],
        ])
        rr.log(
            "world/boat/flap_indicator",
            rr.LineStrips3D(
                [frd_to_rerun(flap_line_frd)],
                colors=[[255, 255, 0, 255]],
                radii=[0.01],
            ),
        )

    # --- Scalar channels with batch logging ---
    # Use rr.Scalars.columns() for proper time series visualization
    time_column = rr.TimeColumn("sim_time", duration=times)

    rr.send_columns(
        "boat/controls/main_flap_deg",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=np.degrees(controls[:, _MAIN_FLAP])),
    )
    rr.send_columns(
        "boat/controls/rudder_elev_deg",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=np.degrees(controls[:, _RUDDER_ELEVATOR])),
    )
    rr.send_columns(
        "boat/state/pos_d",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=states[:, _POS_D]),
    )
    rr.send_columns(
        "boat/state/theta_deg",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=np.degrees(states[:, _THETA])),
    )
    rr.send_columns(
        "boat/state/u_mps",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=states[:, _U]),
    )
    rr.send_columns(
        "boat/state/w_mps",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=states[:, _W]),
    )
    rr.send_columns(
        "boat/state/q_degps",
        indexes=[time_column],
        columns=rr.Scalars.columns(scalars=np.degrees(states[:, _Q])),
    )

    # --- Default blueprint ---
    if blueprint:
        spatial_3d_kwargs = {"name": "3D View", "origin": "world"}
        try:
            spatial_view = rrb.Spatial3DView(
                **spatial_3d_kwargs,
                eye_controls=rrb.EyeControls3D(
                    tracking_entity="world/boat",
                ),
            )
        except (TypeError, AttributeError):
            # Older rerun-sdk without eye_controls support
            spatial_view = rrb.Spatial3DView(**spatial_3d_kwargs)

        bp = rrb.Blueprint(
            rrb.Horizontal(
                spatial_view,
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Position", origin="boat/state"),
                    rrb.TimeSeriesView(name="Controls", origin="boat/controls"),
                ),
                column_shares=[0.6, 0.4],
            ),
        )
        rr.send_blueprint(bp)

    return output_path
