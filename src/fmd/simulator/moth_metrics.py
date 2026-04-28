"""Moth closed-loop performance metrics.

Computes post-simulation metrics from ClosedLoopResult for comparing
different sensor/estimator/controller configurations.
"""

from __future__ import annotations

import numpy as np

from fmd.simulator.closed_loop_pipeline import ClosedLoopResult


def compute_metrics(result: ClosedLoopResult, trim_state) -> dict[str, float]:
    """Compute performance metrics from a ClosedLoopResult.

    Args:
        result: ClosedLoopResult from a simulation run.
        trim_state: Trim state vector (numpy array).

    Returns:
        Dict with RMS theta, RMS pos_d, control effort, settling time,
        breach fraction, and NaN flag.
    """
    true_states = result.true_states[1:]  # skip initial
    controls = result.controls
    times = result.times

    has_nan = bool(np.any(np.isnan(true_states)) or np.any(np.isnan(controls)))

    # RMS tracking errors
    pos_d_err = true_states[:, 0] - trim_state[0]
    theta_err = true_states[:, 1] - trim_state[1]
    rms_pos_d = float(np.sqrt(np.mean(pos_d_err ** 2)))
    rms_theta = float(np.sqrt(np.mean(theta_err ** 2)))

    # Control effort: RMS of control rates
    if len(controls) > 1:
        du = np.diff(controls, axis=0)
        control_effort = float(np.sqrt(np.mean(du ** 2)))
    else:
        control_effort = 0.0

    # Settling time: first time |pos_d_err| < 0.01 m and stays within
    settling_time = float(times[-1])  # default: never settled
    threshold = 0.01
    if len(pos_d_err) > 0:
        within = np.abs(pos_d_err) < threshold
        for i in range(len(within)):
            if np.all(within[i:]):
                settling_time = float(times[i])
                break

    # Breach fraction: fraction of time main foil tip is above water
    # Approximate: pos_d more negative than trim means boat is higher
    breach_fraction = float(np.mean(true_states[:, 0] < trim_state[0] - 0.05))

    return {
        "rms_pos_d": rms_pos_d,
        "rms_theta": rms_theta,
        "control_effort": control_effort,
        "settling_time": settling_time,
        "breach_fraction": breach_fraction,
        "has_nan": has_nan,
    }
