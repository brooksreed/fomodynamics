"""Analysis utilities for OCP results.

This module provides functions for analyzing optimal control trajectories,
including chattering detection and control smoothness metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fmd.ocp.result import OCPResult


def compute_control_smoothness(result: OCPResult) -> dict:
    """Analyze control smoothness and detect chattering.

    Computes metrics to assess control signal quality, particularly
    detecting high-frequency chattering which indicates the optimizer
    found a trajectory with rapid control oscillations (often undesirable
    for real-world implementation).

    Args:
        result: OCPResult from an OCP solver.

    Returns:
        Dictionary with smoothness metrics:
        - sign_change_ratio: Fraction of timesteps with control sign changes (0-1).
            High values (>0.5) indicate chattering.
        - max_du_dt: Maximum control rate magnitude across all controls.
        - mean_du_dt: Mean control rate magnitude.
        - total_variation: Sum of absolute control changes (L1 variation).
        - is_chattering: True if sign_change_ratio > 0.5.
        - chattering_severity: Qualitative assessment:
            "none" (ratio < 0.1), "mild" (0.1-0.3),
            "moderate" (0.3-0.5), "severe" (>= 0.5).
        - per_control: List of per-control metrics (for multi-control systems).
    """
    controls = result.controls  # (N, num_controls)
    N, nu = controls.shape
    dt = result.T / N

    # Compute control differences
    du = np.diff(controls, axis=0)  # (N-1, nu)
    du_dt = du / dt  # Control rate

    # Per-control metrics
    per_control = []
    total_sign_changes = 0

    for i in range(nu):
        u_i = controls[:, i]
        du_i = du[:, i]

        # Sign changes (ignoring zero crossings where sign is 0)
        signs = np.sign(u_i)
        nonzero_mask = signs != 0
        if np.sum(nonzero_mask) > 1:
            nonzero_signs = signs[nonzero_mask]
            sign_changes = np.sum(nonzero_signs[1:] != nonzero_signs[:-1])
        else:
            sign_changes = 0

        total_sign_changes += sign_changes

        per_control.append({
            "index": i,
            "sign_changes": int(sign_changes),
            "sign_change_ratio": float(sign_changes / (N - 1)) if N > 1 else 0.0,
            "max_du_dt": float(np.max(np.abs(du_i / dt))) if len(du_i) > 0 else 0.0,
            "mean_du_dt": float(np.mean(np.abs(du_i / dt))) if len(du_i) > 0 else 0.0,
            "total_variation": float(np.sum(np.abs(du_i))),
        })

    # Aggregate metrics
    sign_change_ratio = total_sign_changes / (nu * (N - 1)) if N > 1 else 0.0
    max_du_dt = float(np.max(np.abs(du_dt))) if du_dt.size > 0 else 0.0
    mean_du_dt = float(np.mean(np.abs(du_dt))) if du_dt.size > 0 else 0.0
    total_variation = float(np.sum(np.abs(du)))

    # Chattering classification
    if sign_change_ratio < 0.1:
        severity = "none"
    elif sign_change_ratio < 0.3:
        severity = "mild"
    elif sign_change_ratio < 0.5:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "sign_change_ratio": sign_change_ratio,
        "max_du_dt": max_du_dt,
        "mean_du_dt": mean_du_dt,
        "total_variation": total_variation,
        "is_chattering": sign_change_ratio >= 0.5,
        "chattering_severity": severity,
        "per_control": per_control,
    }
