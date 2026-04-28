"""Generic validation toolkit for open-loop simulation diagnostics.

Provides generic case-runner data shapes and diagnostic computation that
do not depend on any specific vehicle model. Vehicle-specific case
factories, sweep machinery, and runners (e.g. for the moth model) live
in `fmd.simulator.moth_validation`.

Example:
    from fmd.simulator.validation import SimCase, CaseDiagnostics, compute_diagnostics

    diag = compute_diagnostics(simulation_result)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from fmd.simulator.integrator import SimulationResult
from fmd.simulator.control import ControlSchedule


@dataclass
class SimCase:
    """Definition of a simulation case for open-loop characterization.

    Generic shape; vehicle-specific case factories live in
    `fmd.simulator.moth_validation` (or analogous module per vehicle).

    Attributes:
        name: Case identifier string.
        initial_state: Initial state (None = use trim state).
        control_schedule: Control schedule (None = use trim control as constant).
        u_forward: Forward speed (m/s).
        heel_angle: Static heel angle (rad). Default 30 deg (nominal foiling heel).
        ventilation_mode: "smooth" or "binary".
        ventilation_threshold: Exposed span fraction for ventilation onset.
        surge_enabled: If True, forward speed u is a dynamic state.
        duration: Simulation duration (s).
        dt: Integration timestep (s).
        params: Vehicle params (None = vehicle default at run time).
        description: Human-readable description.
        check_bounded: Whether to check state boundedness.
        check_nan: Whether to check for NaN/Inf.
        expected_signs: Optional dict mapping state names to expected drift
            directions, e.g., {"pos_d": "decrease"} means pos_d should
            decrease over the simulation.
    """

    name: str
    initial_state: Optional[np.ndarray] = None
    control_schedule: Optional[ControlSchedule] = None
    u_forward: float = 10.0
    heel_angle: float = np.deg2rad(30.0)
    ventilation_mode: str = "smooth"
    ventilation_threshold: float = 0.30
    surge_enabled: bool = False
    duration: float = 10.0
    dt: float = 0.005
    params: Optional[Any] = None
    description: str = ""
    check_bounded: bool = True
    check_nan: bool = True
    expected_signs: Optional[dict] = None


@dataclass
class CaseDiagnostics:
    """Diagnostics computed from a simulation result.

    Attributes:
        state_min: Per-state minimum values.
        state_max: Per-state maximum values.
        state_drift: Final minus initial state values.
        has_nan: Whether NaN was found.
        has_inf: Whether Inf was found.
        state_names: Names of state variables.
    """

    state_min: np.ndarray
    state_max: np.ndarray
    state_drift: np.ndarray
    has_nan: bool
    has_inf: bool
    state_names: tuple


def compute_diagnostics(
    result: SimulationResult,
    state_names: tuple = ("pos_d", "theta", "w", "q", "u"),
) -> CaseDiagnostics:
    """Compute diagnostics from a simulation result.

    Args:
        result: SimulationResult with states array.
        state_names: Tuple of state variable names.

    Returns:
        CaseDiagnostics with min/max/drift/NaN/Inf checks.
    """
    states = np.array(result.states)

    return CaseDiagnostics(
        state_min=np.min(states, axis=0),
        state_max=np.max(states, axis=0),
        state_drift=states[-1] - states[0],
        has_nan=bool(np.any(np.isnan(states))),
        has_inf=bool(np.any(np.isinf(states))),
        state_names=state_names,
    )
