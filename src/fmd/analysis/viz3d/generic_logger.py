"""Generic Rerun .rrd writer for any RigidBody6DOF simulation.

Writes a 6DOF simulation to a .rrd file for viewing in the Rerun Viewer.
Works with any system that follows the 13-state RigidBody6DOF layout:
[pos_n, pos_e, pos_d, vel_u, vel_v, vel_w, qw, qx, qy, qz, omega_p, omega_q, omega_r]

Requires: rerun-sdk (included in default `uv sync`)

Example:
    from fmd.simulator import RigidBody6DOF, simulate
    from fmd.analysis.viz3d import write_rrd

    body = RigidBody6DOF(params)
    result = simulate(body, body.default_state(), dt=0.01, duration=5.0)
    write_rrd(result, "sim.rrd")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fmd.analysis.viz3d._rerun_check import require_rerun
from fmd.analysis.viz3d.coordinates import ned_to_rerun, frd_to_rerun, fmd_quat_to_rerun
from fmd.simulator.integrator import SimulationResult

# RigidBody6DOF state indices
_POS_N = 0
_POS_E = 1
_POS_D = 2
_VEL_U = 3
_VEL_V = 4
_VEL_W = 5
_QW = 6
_QX = 7
_QY = 8
_QZ = 9
_OMEGA_P = 10
_OMEGA_Q = 11
_OMEGA_R = 12


@dataclass
class ForceVizData:
    """Force visualization data for write_rrd().

    Attributes:
        name: Entity name (e.g., "thrust", "drag").
        origins: Application points in body frame. Shape (num_steps, 3).
        vectors: Force vectors in body frame. Shape (num_steps, 3).
        color: RGBA color tuple. Default blue.
    """

    name: str
    origins: NDArray
    vectors: NDArray
    color: tuple[int, int, int, int] = (100, 100, 255, 255)


def write_rrd(
    result: SimulationResult,
    output_path: str | Path,
    *,
    forces: list[ForceVizData] | None = None,
    wireframe: dict[str, NDArray] | None = None,
    wireframe_colors: dict[str, tuple[int, int, int, int]] | None = None,
    force_scale: float = 0.001,
    state_names: list[str] | None = None,
    control_names: list[str] | None = None,
    application_id: str = "rigidbody6dof_viz",
) -> Path:
    """Write a 6DOF simulation to a .rrd file.

    Expects 13-state RigidBody6DOF layout:
    [pos_n, pos_e, pos_d, vel_u, vel_v, vel_w, qw, qx, qy, qz, omega_p, omega_q, omega_r]

    Args:
        result: SimulationResult with times, states, controls.
        output_path: Path for the output .rrd file.
        forces: Optional list of ForceVizData for force arrows.
        wireframe: Optional dict mapping names to vertex arrays (body frame FRD).
        wireframe_colors: Optional dict mapping wireframe names to RGBA colors.
        force_scale: Arrow length per Newton (m/N). Default 0.001 = 1m per 1000N.
        state_names: Optional list of names for state channels.
        control_names: Optional list of names for control channels.
        application_id: Rerun application identifier.

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

    # Validate state dimension
    if states.shape[1] < 13:
        raise ValueError(
            f"Expected at least 13 states for RigidBody6DOF, got {states.shape[1]}"
        )

    # Initialize recording
    rr.init(application_id, spawn=False)
    rr.save(str(output_path))

    # Set global coordinate system to Z-up
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- Static geometry ---

    # Ground/water reference plane
    plane_size = 20.0
    plane_vertices = np.array([
        [-plane_size, -plane_size, 0.0],
        [plane_size, -plane_size, 0.0],
        [plane_size, plane_size, 0.0],
        [-plane_size, plane_size, 0.0],
    ])
    plane_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    rr.log(
        "world/ground",
        rr.Mesh3D(
            vertex_positions=plane_vertices,
            triangle_indices=plane_indices,
            vertex_colors=[[100, 100, 100, 40]] * 4,
        ),
        static=True,
    )

    # Wireframe geometry (optional)
    default_wireframe_color = (180, 180, 180, 255)
    if wireframe is not None:
        wireframe_colors = wireframe_colors or {}
        for name, verts in wireframe.items():
            verts_rerun = frd_to_rerun(verts)
            color = wireframe_colors.get(name, default_wireframe_color)
            rr.log(
                f"world/body/{name}",
                rr.LineStrips3D([verts_rerun], colors=[color], radii=[0.01]),
                static=True,
            )

    # --- Extract position and orientation ---
    positions_ned = states[:, _POS_N:_POS_D + 1]  # (N, 3)
    quaternions_blur = states[:, _QW:_QZ + 1]  # (N, 4)

    positions_rerun = ned_to_rerun(positions_ned)  # (N, 3)
    quaternions_rerun = fmd_quat_to_rerun(quaternions_blur)  # (N, 4)

    # --- Position trail ---
    rr.log(
        "world/trail",
        rr.LineStrips3D(
            [positions_rerun],
            colors=[[255, 255, 255, 128]],
            radii=[0.02],
        ),
        static=True,
    )

    # --- Batch logging with send_columns ---
    time_column = rr.TimeColumn("sim_time", duration=times)

    # Log body transform per-timestep (cannot batch Transform3D easily)
    for i in range(num_steps):
        rr.set_time("sim_time", duration=float(times[i]))
        rr.log(
            "world/body",
            rr.Transform3D(
                translation=positions_rerun[i].tolist(),
                quaternion=rr.Quaternion(xyzw=quaternions_rerun[i].tolist()),
            ),
        )

    # Log forces if provided
    if forces is not None:
        for force_data in forces:
            for i in range(num_steps):
                rr.set_time("sim_time", duration=float(times[i]))
                origin_rr = frd_to_rerun(force_data.origins[i])
                vector_rr = frd_to_rerun(force_data.vectors[i]) * force_scale
                mag = float(np.linalg.norm(force_data.vectors[i]))
                rr.log(
                    f"world/body/forces/{force_data.name}",
                    rr.Arrows3D(
                        origins=[origin_rr],
                        vectors=[vector_rr],
                        colors=[force_data.color],
                        labels=[f"{mag:.0f}N"],
                    ),
                )

    # --- Scalar channels for time-series plots ---
    default_state_names = [
        "pos_n", "pos_e", "pos_d",
        "vel_u", "vel_v", "vel_w",
        "qw", "qx", "qy", "qz",
        "omega_p", "omega_q", "omega_r",
    ]
    state_names = state_names or default_state_names

    for i, name in enumerate(state_names[:states.shape[1]]):
        rr.send_columns(
            f"body/state/{name}",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=states[:, i]),
        )

    # Log controls
    num_controls = controls.shape[1] if controls.ndim > 1 else 1
    control_names = control_names or [f"control_{i}" for i in range(num_controls)]
    for i, name in enumerate(control_names[:num_controls]):
        rr.send_columns(
            f"body/controls/{name}",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=controls[:, i]),
        )

    # --- Default blueprint ---
    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="State", origin="body/state"),
                rrb.TimeSeriesView(name="Controls", origin="body/controls"),
            ),
            column_shares=[0.6, 0.4],
        ),
    )
    rr.send_blueprint(bp)

    return output_path
