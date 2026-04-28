"""Time-grid helpers shared across analysis modules.

Kept separate to avoid circular imports between modules like:
- blur.analysis.core
- blur.analysis.processing
"""

import numpy as np


def make_time_grid_inclusive(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """Create a time grid that includes the endpoint exactly.

    Uses a constant step `dt` until the last step, and shortens the final step
    so that the final time equals `t_end`.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    t = float(t_start)
    times: list[float] = [t]
    while t < t_end:
        step = min(dt, t_end - t)
        t = t + step
        times.append(t)
    return np.asarray(times, dtype=float)
