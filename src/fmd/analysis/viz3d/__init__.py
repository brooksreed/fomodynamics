"""3D visualization for fomodynamics simulations using Rerun.

Requires: rerun-sdk (included in default `uv sync`)

Example (Moth 3DOF):
    from fmd.simulator import Moth3D, simulate
    from fmd.simulator.params import MOTH_BIEKER_V3
    from fmd.simulator.moth_forces_extract import extract_forces
    from fmd.analysis.viz3d import write_moth_rrd

    moth = Moth3D(MOTH_BIEKER_V3)
    result = simulate(moth, moth.default_state(), dt=0.01, duration=5.0)
    forces = extract_forces(moth, result)
    write_moth_rrd(MOTH_BIEKER_V3, result, "moth_test.rrd", forces=forces)

Example (Generic 6DOF):
    from fmd.simulator import RigidBody6DOF, simulate
    from fmd.analysis.viz3d import write_rrd

    body = RigidBody6DOF(params)
    result = simulate(body, body.default_state(), dt=0.01, duration=5.0)
    write_rrd(result, "sim.rrd")
"""

# These imports don't require rerun
from fmd.analysis.viz3d.geometry import build_moth_wireframe, compute_surface_waterline
from fmd.analysis.viz3d.generic_logger import ForceVizData


# Lazy import for rerun-dependent code
def write_moth_rrd(*args, **kwargs):
    """Write a Moth 3DOF simulation to a .rrd file.

    See moth_logger.write_moth_rrd for full documentation.
    """
    from fmd.analysis.viz3d.moth_logger import write_moth_rrd as _write
    return _write(*args, **kwargs)


def write_moth_lqg_rrd(*args, **kwargs):
    """Write a Moth 3DOF LQG simulation to a .rrd file.

    See moth_lqg_logger.write_moth_lqg_rrd for full documentation.
    """
    from fmd.analysis.viz3d.moth_lqg_logger import write_moth_lqg_rrd as _write
    return _write(*args, **kwargs)


def write_rrd(*args, **kwargs):
    """Write any RigidBody6DOF simulation to a .rrd file.

    See generic_logger.write_rrd for full documentation.
    """
    from fmd.analysis.viz3d.generic_logger import write_rrd as _write
    return _write(*args, **kwargs)


__all__ = [
    "write_moth_rrd",
    "write_moth_lqg_rrd",
    "write_rrd",
    "ForceVizData",
    "build_moth_wireframe",
    "compute_surface_waterline",
]
