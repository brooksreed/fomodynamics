"""Rerun .rrd writer for Moth 3DOF LQG closed-loop visualization.

Writes a Moth 3DOF LQG simulation to a .rrd file, showing the 3D scene
(true state), estimation vs truth time series, error with 2-sigma bounds,
and control inputs.

Requires: rerun-sdk (included in default `uv sync`)

Example:
    from fmd.simulator.moth_scenarios import ScenarioConfig, run_scenario
    from fmd.analysis.viz3d import write_moth_lqg_rrd

    result = run_scenario(ScenarioConfig(name="demo", duration=10.0))
    write_moth_lqg_rrd(MOTH_BIEKER_V3, result, "moth_lqg.rrd")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fmd.analysis.viz3d._rerun_check import require_rerun
from fmd.analysis.viz3d.geometry import build_moth_wireframe, compute_surface_waterline
from fmd.analysis.viz3d.coordinates import frd_to_rerun, moth_3dof_to_rerun_quat
from fmd.simulator.moth_forces_extract import MothForceLog
from fmd.simulator.closed_loop_pipeline import ClosedLoopResult
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


def _validate_force_lengths(forces: MothForceLog, num_steps: int) -> None:
    """Ensure all force arrays align with the LQG timebase."""
    force_fields = {
        "times": forces.times,
        "main_foil_force": forces.main_foil_force,
        "main_foil_moment": forces.main_foil_moment,
        "rudder_force": forces.rudder_force,
        "rudder_moment": forces.rudder_moment,
        "sail_force": forces.sail_force,
        "sail_moment": forces.sail_moment,
        "hull_drag_force": forces.hull_drag_force,
        "hull_drag_moment": forces.hull_drag_moment,
        "gravity_force": forces.gravity_force,
    }
    for name, values in force_fields.items():
        if len(values) != num_steps:
            raise ValueError(
                f"forces.{name} must have length {num_steps} "
                f"to match result.times, got {len(values)}"
            )


def write_moth_lqg_rrd(
    params: MothParams | None,
    result: ClosedLoopResult,
    output_path: str | Path,
    forces: MothForceLog | None = None,
    *,
    force_scale: float = 0.001,
    blueprint: bool = True,
    heel_angle: float | None = None,
    water_elevation_fn=None,
) -> Path:
    """Write a Moth 3DOF LQG simulation to a .rrd file.

    Args:
        params: Optional fallback MothParams for boat geometry. If
            ``result.params`` is present, that value is used instead to ensure
            visualization matches the simulation configuration.
        result: ClosedLoopResult from simulate_closed_loop.
        output_path: Path for the output .rrd file.
        forces: Optional MothForceLog with per-component force arrays.
            If not provided, falls back to ``result.force_log``. If neither
            is available, force arrows will not be rendered.
        force_scale: Arrow length per Newton (m/N). Default 0.001 = 1m per 1000N.
        blueprint: Whether to send a default blueprint layout. Default True.
        heel_angle: Heel (roll) angle in radians. If None, uses
            ``result.heel_angle`` or defaults to 30 degrees.
        water_elevation_fn: Optional callable(n, e, t) -> eta for wave fields.

    Returns:
        Path to the written .rrd file.
    """
    # Prefer simulation-embedded params/force/heel to avoid caller-side drift.
    result_params = getattr(result, "params", None)
    if result_params is not None:
        params = result_params
    if params is None:
        raise ValueError(
            "Moth parameters are required: pass params explicitly or provide "
            "result.params from simulate_closed_loop(..., params=...)."
        )

    # Fall back to result-embedded force_log and heel_angle
    if forces is None and hasattr(result, "force_log"):
        forces = result.force_log
    if heel_angle is None:
        result_heel = getattr(result, "heel_angle", None)
        heel_angle = np.deg2rad(30.0) if result_heel is None else float(result_heel)

    require_rerun()
    import rerun as rr
    import rerun.blueprint as rrb

    output_path = Path(output_path)

    times = result.times
    true_states = result.true_states
    est_states = result.est_states
    controls = result.controls
    cov_diags = result.covariance_diagonals
    est_errors = result.estimation_errors
    cov_traces = result.covariance_traces
    num_steps = len(times)

    # LQG state-level arrays include initial condition (n_steps + 1).
    expected_states = num_steps + 1
    if len(true_states) != expected_states:
        raise ValueError(
            f"result.true_states must have length {expected_states}, "
            f"got {len(true_states)}"
        )
    if len(est_states) != expected_states:
        raise ValueError(
            f"result.est_states must have length {expected_states}, "
            f"got {len(est_states)}"
        )
    if len(cov_diags) != expected_states:
        raise ValueError(
            f"result.covariance_diagonals must have length {expected_states}, "
            f"got {len(cov_diags)}"
        )
    if len(est_errors) != expected_states:
        raise ValueError(
            f"result.estimation_errors must have length {expected_states}, "
            f"got {len(est_errors)}"
        )
    if len(cov_traces) != expected_states:
        raise ValueError(
            f"result.covariance_traces must have length {expected_states}, "
            f"got {len(cov_traces)}"
        )
    if len(controls) != num_steps:
        raise ValueError(
            f"result.controls must have length {num_steps}, got {len(controls)}"
        )
    if forces is not None:
        _validate_force_lengths(forces, num_steps)

    # Initialize recording
    rr.init("moth_lqg_viz", spawn=False)
    rr.save(str(output_path))

    # Set global coordinate system to Z-up
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- Static geometry ---

    # Water mesh (translucent surface at z=0), child of world/water
    water_size = 5.0
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
    # Use true state for 3D visualization
    u = true_states[:num_steps, _U]
    theta = true_states[:num_steps, _THETA]
    pos_n_dot = u * np.cos(theta)
    pos_n = np.zeros(num_steps)
    if num_steps > 1:
        pos_n[1:] = np.cumsum(pos_n_dot[:-1] * np.diff(times))

    # --- Height computation ---
    # pos_d is NED depth (negative = above water). Rerun Z-up: negate pos_d.
    z_rerun = -true_states[:num_steps, _POS_D]
    positions_rerun = np.column_stack([
        np.zeros(num_steps),  # X_rerun = East = 0
        pos_n,                 # Y_rerun = North
        z_rerun,               # Z_rerun = up
    ])

    # Trail: keep only a short tail behind the boat to avoid expanding
    # scene bounds (which causes the tracking camera to zoom out).
    _TRAIL_LENGTH = 100  # number of trailing positions to show

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

        rerun_pos = positions_rerun[i]
        rerun_quat = moth_3dof_to_rerun_quat(true_states[i, _THETA], heel_angle)

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

        # Short trailing wake behind the boat (fixed length to keep
        # scene bounds tight for camera tracking)
        if i > 0:
            trail_start = max(0, i - _TRAIL_LENGTH)
            rr.log(
                "world/trail",
                rr.LineStrips3D(
                    [positions_rerun[trail_start:i + 1]],
                    colors=[[255, 255, 255, 128]],
                    radii=[0.02],
                ),
            )

        # Waterline indicators on all 4 surfaces (foils + struts)
        for surface_name, verts in surface_rects.items():
            wl = compute_surface_waterline(
                vertices_body=verts,
                pos_d=true_states[i, _POS_D],
                theta=true_states[i, _THETA],
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

        # Force arrows
        if forces is not None:
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

        # Flap angle indicator
        flap_rad = controls[i, _MAIN_FLAP]
        flap_len = 0.15
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
    if num_steps > 0:
        time_column = rr.TimeColumn("sim_time", duration=times)

    # Use n_steps+1 time base for state-level data (includes initial condition)
    if num_steps == 0:
        # Keep a valid one-sample timebase for initial-condition channels.
        times_full = np.array([0.0])
    else:
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        times_full = np.concatenate([times, [times[-1] + dt]])
    time_column_full = rr.TimeColumn("sim_time", duration=times_full)

    # State names and display conversion flags
    state_names = ["pos_d", "theta", "w", "q", "u"]
    state_is_angle = [False, True, False, True, False]

    def _to_display(values, is_angle):
        """Convert to display units (radians -> degrees where applicable)."""
        return np.degrees(values) if is_angle else values

    # --- Estimation time series: true vs estimated per state ---
    for j, (name, is_angle) in enumerate(
        zip(state_names, state_is_angle)
    ):
        true_vals = _to_display(true_states[:, j], is_angle)
        est_vals = _to_display(est_states[:, j], is_angle)

        rr.send_columns(
            f"estimation/{name}/true",
            indexes=[time_column_full],
            columns=rr.Scalars.columns(scalars=true_vals),
        )
        rr.send_columns(
            f"estimation/{name}/estimated",
            indexes=[time_column_full],
            columns=rr.Scalars.columns(scalars=est_vals),
        )

    # --- Error time series with 2-sigma bounds ---
    for j, (name, is_angle) in enumerate(
        zip(state_names, state_is_angle)
    ):
        error_vals = _to_display(est_errors[:, j], is_angle)
        sigma = np.sqrt(cov_diags[:, j])
        sigma_display = _to_display(sigma, is_angle)
        plus_2sigma = 2.0 * sigma_display
        minus_2sigma = -2.0 * sigma_display

        rr.send_columns(
            f"error/{name}/error",
            indexes=[time_column_full],
            columns=rr.Scalars.columns(scalars=error_vals),
        )
        rr.send_columns(
            f"error/{name}/plus_2sigma",
            indexes=[time_column_full],
            columns=rr.Scalars.columns(scalars=plus_2sigma),
        )
        rr.send_columns(
            f"error/{name}/minus_2sigma",
            indexes=[time_column_full],
            columns=rr.Scalars.columns(scalars=minus_2sigma),
        )

    # --- Controls ---
    if num_steps > 0:
        rr.log("controls/main_flap_deg", rr.SeriesLines(colors=_COLOR_MAIN_FOIL), static=True)
        rr.send_columns(
            "controls/main_flap_deg",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=np.degrees(controls[:, _MAIN_FLAP])),
        )
        rr.log("controls/rudder_elev_deg", rr.SeriesLines(colors=_COLOR_RUDDER), static=True)
        rr.send_columns(
            "controls/rudder_elev_deg",
            indexes=[time_column],
            columns=rr.Scalars.columns(
                scalars=np.degrees(controls[:, _RUDDER_ELEVATOR])
            ),
        )

    # --- Covariance trace ---
    rr.send_columns(
        "estimation/cov_trace",
        indexes=[time_column_full],
        columns=rr.Scalars.columns(scalars=cov_traces),
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
                    rrb.TimeSeriesView(
                        name="Height (true vs est)",
                        origin="estimation/pos_d",
                    ),
                    rrb.TimeSeriesView(
                        name="Pitch (true vs est)",
                        origin="estimation/theta",
                    ),
                    rrb.TimeSeriesView(
                        name="Height error +/- 2sigma",
                        origin="error/pos_d",
                    ),
                    rrb.TimeSeriesView(
                        name="Controls",
                        origin="controls",
                    ),
                ),
                column_shares=[0.5, 0.5],
            ),
        )
        rr.send_blueprint(bp)

    return output_path
