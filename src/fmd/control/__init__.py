"""fmd.control - Public control surface (LQR).

This package re-exports the discrete LQR solver (which lives at
`fmd.simulator.lqr`) as a convenience so users can write
`from fmd.control import LQR`.

Example:
    from fmd.control import LQR, lqr
    from fmd.simulator import RigidBody6DOF

    # design an LQR gain via the helper:
    K = lqr(A, B, Q, R)
"""

# Ensure float64 is enabled before any JAX imports
from fmd.control import _config  # noqa: F401

from fmd.simulator.lqr import (
    LQRController as LQR,
    TrajectoryLQRController,
    TVLQRController,
    lqr,
    compute_lqr_gain,
)

__all__ = [
    "LQR",
    "TrajectoryLQRController",
    "TVLQRController",
    "lqr",
    "compute_lqr_gain",
]
