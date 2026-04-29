"""Moth-specific open-loop validation, sweep, and case-runner toolkit.

Built on top of the generic primitives in `fmd.simulator.validation`. Use this
module for moth-specific runs: `run_case`, perturbation/configuration sweeps,
transient/speed/damping comparisons, etc.

Example:
    from fmd.simulator.moth_validation import SimCase, run_case

    case = SimCase(name="trim_equilibrium", duration=10.0)
    result, diagnostics, trim, moth = run_case(case)

Sweep Example:
    from fmd.simulator.moth_validation import (
        PerturbationConfig, ConfigurationVariation,
        run_perturbation_sweep, run_configuration_comparison,
    )

    perturbations = [
        PerturbationConfig("theta", np.radians(1.0), "pitch +1°"),
        PerturbationConfig("w", 0.1, "heave vel +0.1 m/s"),
    ]
    results = run_perturbation_sweep(perturbations)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import jax.numpy as jnp
from jax import Array

import attrs

from fmd.simulator.params import MothParams, MOTH_BIEKER_V3
from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
from fmd.simulator.trim_casadi import find_moth_trim, CasadiTrimResult
from fmd.simulator.integrator import SimulationResult, simulate
from fmd.simulator.control import ControlSchedule, ConstantControl
from fmd.simulator.linearize import linearize
from fmd.simulator.components.moth_forces import (
    compute_depth_factor,
    compute_leeward_tip_depth,
)

# Re-export generic shapes so existing callers can transition to this module
# without changing their imports of SimCase / CaseDiagnostics / compute_diagnostics.
from fmd.simulator.validation import (
    SimCase,
    CaseDiagnostics,
    compute_diagnostics,
)


def run_case(case: SimCase) -> tuple[SimulationResult, CaseDiagnostics, CasadiTrimResult, Moth3D]:
    """Run a simulation case and compute diagnostics.

    Creates Moth3D, finds trim (if needed), runs simulation, computes
    diagnostics.

    Args:
        case: SimCase definition.

    Returns:
        Tuple of (SimulationResult, CaseDiagnostics, CasadiTrimResult, Moth3D).
    """
    params = case.params if case.params is not None else MOTH_BIEKER_V3

    u_forward = float(case.u_forward)
    moth = Moth3D(
        params,
        u_forward=ConstantSchedule(u_forward),
        heel_angle=float(case.heel_angle),
        ventilation_mode=case.ventilation_mode,
        ventilation_threshold=float(case.ventilation_threshold),
        surge_enabled=case.surge_enabled,
    )

    # Find trim for initial state / control defaults
    trim = find_moth_trim(
        params, u_forward=case.u_forward,
        heel_angle=float(case.heel_angle),
        ventilation_mode=case.ventilation_mode,
        ventilation_threshold=float(case.ventilation_threshold),
    )

    # Initial state
    if case.initial_state is not None:
        x0 = jnp.array(case.initial_state)
    else:
        x0 = jnp.array(trim.state)

    # Control schedule
    if case.control_schedule is not None:
        ctrl = case.control_schedule
    else:
        ctrl = ConstantControl(jnp.array(trim.control))

    # Run simulation
    result = simulate(
        moth, x0, dt=case.dt, duration=case.duration, control=ctrl,
    )

    # Compute diagnostics
    diag = compute_diagnostics(result, moth.state_names)

    return result, diag, trim, moth


# ===================================================================
# Predefined Case Factories
# ===================================================================


def case_trim_equilibrium(u_forward: float = 10.0, duration: float = 10.0) -> SimCase:
    """Trim equilibrium: start at trim, hold trim controls.

    Args:
        u_forward: Forward speed in m/s.
        duration: Simulation duration in seconds.

    Returns:
        SimCase configured for trim equilibrium validation.
    """
    return SimCase(
        name="trim_equilibrium",
        u_forward=u_forward,
        duration=duration,
        description=f"Trim equilibrium at {u_forward} m/s for {duration}s",
    )


def case_flap_impulse(
    sign: float = 1.0,
    magnitude_deg: float = 5.0,
    impulse_duration: float = 0.1,
    u_forward: float = 10.0,
    duration: float = 10.0,
) -> SimCase:
    """Flap impulse: step flap deflection for a short duration, then return to trim.

    Note: The control schedule is built when run_case() is called,
    since we need trim control values. This factory stores configuration
    and the schedule is built in a wrapper.

    Args:
        sign: Direction of impulse (+1.0 or -1.0).
        magnitude_deg: Impulse magnitude in degrees.
        impulse_duration: Duration of the impulse in seconds.
        u_forward: Forward speed in m/s.
        duration: Total simulation duration in seconds.

    Returns:
        SimCase configured for flap impulse (control_schedule=None,
        must be set up by the caller before calling run_case).
    """
    return SimCase(
        name=f"flap_impulse_{'pos' if sign > 0 else 'neg'}",
        u_forward=u_forward,
        duration=duration,
        description=(
            f"{'Positive' if sign > 0 else 'Negative'} flap impulse "
            f"({sign * magnitude_deg:.1f} deg for {impulse_duration}s)"
        ),
        # control_schedule will be set up by the script/runner
        # Store metadata for the script to interpret
    )


def case_elevator_impulse(
    sign: float = 1.0,
    magnitude_deg: float = 3.0,
    impulse_duration: float = 0.1,
    u_forward: float = 10.0,
    duration: float = 10.0,
) -> SimCase:
    """Elevator impulse: step elevator deflection for a short duration.

    Note: The control schedule is built when run_case() is called,
    since we need trim control values. This factory stores configuration
    and the schedule is built in a wrapper.

    Args:
        sign: Direction of impulse (+1.0 or -1.0).
        magnitude_deg: Impulse magnitude in degrees.
        impulse_duration: Duration of the impulse in seconds.
        u_forward: Forward speed in m/s.
        duration: Total simulation duration in seconds.

    Returns:
        SimCase configured for elevator impulse (control_schedule=None,
        must be set up by the caller before calling run_case).
    """
    return SimCase(
        name=f"elevator_impulse_{'pos' if sign > 0 else 'neg'}",
        u_forward=u_forward,
        duration=duration,
        description=(
            f"{'Positive' if sign > 0 else 'Negative'} elevator impulse "
            f"({sign * magnitude_deg:.1f} deg for {impulse_duration}s)"
        ),
    )


# ===================================================================
# Sweep Analysis Data Structures
# ===================================================================

# State indices for Moth3D
STATE_NAMES = ("pos_d", "theta", "w", "q", "u")
STATE_INDICES = {"pos_d": 0, "theta": 1, "w": 2, "q": 3, "u": 4}


@dataclass
class PerturbationConfig:
    """Configuration for a single state perturbation.

    Attributes:
        state_name: Name of state to perturb ("pos_d", "theta", "w", "q").
        magnitude: Perturbation magnitude in SI units (m, rad, m/s, rad/s).
        description: Human-readable description.
    """

    state_name: str
    magnitude: float
    description: str = ""

    def __post_init__(self):
        if self.state_name not in STATE_INDICES:
            raise ValueError(
                f"Unknown state '{self.state_name}'. "
                f"Valid states: {list(STATE_INDICES.keys())}"
            )
        if not self.description:
            sign = "+" if self.magnitude >= 0 else ""
            self.description = f"{self.state_name} {sign}{self.magnitude}"

    @property
    def state_index(self) -> int:
        """Get the index of this state in the state vector."""
        return STATE_INDICES[self.state_name]


@dataclass
class ConfigurationVariation:
    """Configuration for a model parameter/control variation.

    All offsets are relative to baseline (MOTH_BIEKER_V3 defaults).
    Set to 0.0 for no change in that parameter.

    Attributes:
        sailor_x_offset: Sailor x-position offset (m), positive = forward.
        flap_offset_deg: Main flap angle offset (degrees), positive = more lift.
        elevator_offset_deg: Rudder elevator offset (degrees), positive = more lift.
        name: Human-readable name for this configuration.
    """

    sailor_x_offset: float = 0.0
    flap_offset_deg: float = 0.0
    elevator_offset_deg: float = 0.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            parts = []
            if self.sailor_x_offset != 0:
                sign = "+" if self.sailor_x_offset >= 0 else ""
                parts.append(f"sailor {sign}{self.sailor_x_offset*100:.0f}cm")
            if self.flap_offset_deg != 0:
                sign = "+" if self.flap_offset_deg >= 0 else ""
                parts.append(f"flap {sign}{self.flap_offset_deg:.0f}°")
            if self.elevator_offset_deg != 0:
                sign = "+" if self.elevator_offset_deg >= 0 else ""
                parts.append(f"elev {sign}{self.elevator_offset_deg:.0f}°")
            self.name = ", ".join(parts) if parts else "baseline"

    @property
    def is_baseline(self) -> bool:
        """Check if this is the baseline configuration (no changes)."""
        return (
            self.sailor_x_offset == 0.0
            and self.flap_offset_deg == 0.0
            and self.elevator_offset_deg == 0.0
        )


@dataclass
class PerturbationResult:
    """Result from a single perturbation simulation.

    Attributes:
        config: The perturbation configuration.
        trim_state: Trim state before perturbation.
        trim_control: Trim control values.
        initial_state: State after perturbation applied.
        final_state: State at end of simulation.
        max_deviation: Maximum deviation from trim during simulation.
        divergence_time: Time to reach threshold (None if stable).
        stable: Whether system remained bounded.
        result: Full SimulationResult (optional, for detailed analysis).
    """

    config: PerturbationConfig
    trim_state: np.ndarray
    trim_control: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray
    max_deviation: np.ndarray
    divergence_time: Optional[float]
    stable: bool
    min_depth_factor: Optional[float] = None
    min_tip_depth: Optional[float] = None
    ventilation_time: Optional[float] = None
    result: Optional[SimulationResult] = None


@dataclass
class ConfigurationResult:
    """Result from a single configuration variation.

    Attributes:
        config: The configuration variation.
        params: Modified MothParams used.
        trim_state: Equilibrium state for this configuration.
        trim_control: Equilibrium control for this configuration.
        trim_residual: Trim solver residual.
        trim_success: Whether trim solver converged.
        delta_state: Change from baseline trim state.
        delta_control: Change from baseline trim control.
        simulation_result: Optional open-loop simulation result.
        final_state: Final state after open-loop simulation (if run).
    """

    config: ConfigurationVariation
    params: MothParams
    trim_state: np.ndarray
    trim_control: np.ndarray
    trim_residual: float
    trim_success: bool
    delta_state: Optional[np.ndarray] = None
    delta_control: Optional[np.ndarray] = None
    simulation_result: Optional[SimulationResult] = None
    final_state: Optional[np.ndarray] = None


@dataclass
class SweepResult:
    """Aggregated results from a sweep analysis.

    Attributes:
        perturbation_results: List of perturbation results (if run).
        configuration_results: List of configuration results (if run).
        baseline_trim_state: Baseline trim state.
        baseline_trim_control: Baseline trim control.
        u_forward: Forward speed used.
        duration: Simulation duration.
        dt: Integration timestep.
    """

    perturbation_results: List[PerturbationResult] = field(default_factory=list)
    configuration_results: List[ConfigurationResult] = field(default_factory=list)
    transient_results: List["TransientResult"] = field(default_factory=list)
    speed_variation_results: List["SpeedVariationResult"] = field(default_factory=list)
    damping_comparison_results: List["DampingComparisonResult"] = field(default_factory=list)
    baseline_trim_state: Optional[np.ndarray] = None
    baseline_trim_control: Optional[np.ndarray] = None
    u_forward: float = 10.0
    duration: float = 5.0
    dt: float = 0.005


# ===================================================================
# Sweep Analysis Functions
# ===================================================================


def run_perturbation_sweep(
    perturbations: List[PerturbationConfig],
    u_forward: float = 10.0,
    duration: float = 5.0,
    dt: float = 0.005,
    divergence_threshold_deg: float = 30.0,
    params: Optional[MothParams] = None,
    store_results: bool = False,
    target_pos_d: Optional[float] = None,
    ventilation_threshold: float = 0.30,
) -> SweepResult:
    """Run a sweep of state perturbations from trim.

    For each perturbation:
    1. Find trim equilibrium
    2. Apply perturbation to trim state
    3. Simulate with trim controls held constant
    4. Record response metrics

    Args:
        perturbations: List of PerturbationConfig to test.
        u_forward: Forward speed (m/s).
        duration: Simulation duration (s).
        dt: Integration timestep (s).
        divergence_threshold_deg: Pitch angle threshold for divergence detection.
        params: MothParams to use (default: MOTH_BIEKER_V3).
        store_results: Whether to store full SimulationResult in output.
        target_pos_d: If set, fix pos_d to this value during trim.
        ventilation_threshold: Depth factor threshold. Set very high to disable.

    Returns:
        SweepResult with perturbation_results populated.
    """
    params = params if params is not None else MOTH_BIEKER_V3
    divergence_threshold = np.radians(divergence_threshold_deg)

    # Create model and find baseline trim
    moth = Moth3D(
        params,
        u_forward=ConstantSchedule(float(u_forward)),
        ventilation_threshold=ventilation_threshold,
    )
    baseline_trim = find_moth_trim(
        params, u_forward=u_forward, target_pos_d=target_pos_d,
        ventilation_threshold=ventilation_threshold,
    )

    sweep_result = SweepResult(
        baseline_trim_state=np.array(baseline_trim.state),
        baseline_trim_control=np.array(baseline_trim.control),
        u_forward=u_forward,
        duration=duration,
        dt=dt,
    )

    for pert in perturbations:
        # Apply perturbation to trim state
        perturbed_state = baseline_trim.state.copy()
        perturbed_state[pert.state_index] += pert.magnitude

        # Run simulation
        result = simulate(
            moth,
            jnp.array(perturbed_state),
            dt=dt,
            duration=duration,
            control=ConstantControl(jnp.array(baseline_trim.control)),
        )

        states = np.array(result.states)
        times = np.array(result.times)

        # Compute metrics
        deviation = states - baseline_trim.state
        max_deviation = np.max(np.abs(deviation), axis=0)

        # Check for divergence (pitch exceeds threshold)
        theta_deviation = np.abs(states[:, STATE_INDICES["theta"]] - baseline_trim.state[STATE_INDICES["theta"]])
        diverged_mask = theta_deviation > divergence_threshold
        if np.any(diverged_mask):
            divergence_time = float(times[np.argmax(diverged_mask)])
            stable = False
        else:
            divergence_time = None
            stable = True

        # Compute ventilation metric (min depth factor across both foils)
        pos_d_traj = states[:, STATE_INDICES["pos_d"]]
        theta_traj = states[:, STATE_INDICES["theta"]]

        # CG offset from sailor (same as forward_dynamics applies)
        cg_offset = params.combined_cg_offset
        main_pos = params.main_foil_position - cg_offset
        rudder_pos = params.rudder_position - cg_offset

        # Leeward tip depth (primary geometric diagnostic)
        main_tip_depth = np.array(compute_leeward_tip_depth(
            jnp.array(pos_d_traj), main_pos[0], main_pos[2],
            jnp.array(theta_traj), moth.main_foil.heel_angle,
            params.main_foil_span,
        ))
        min_tip_depth = float(np.min(main_tip_depth))

        # Depth factor (hydrodynamic metric, retained alongside tip depth)
        main_depth = pos_d_traj + main_pos[2] - main_pos[0] * np.sin(theta_traj)
        rudder_depth = pos_d_traj + rudder_pos[2] - rudder_pos[0] * np.sin(theta_traj)

        main_df = np.array(compute_depth_factor(
            jnp.array(main_depth), params.main_foil_span, moth.main_foil.heel_angle,
        ))
        rudder_df = np.array(compute_depth_factor(
            jnp.array(rudder_depth), params.rudder_span, moth.rudder.heel_angle,
        ))
        combined_df = np.minimum(main_df, rudder_df)
        min_depth_factor = float(np.min(combined_df))

        # Ventilation detection: both depth_factor and tip depth checks
        ventilation_threshold_factor = 0.01
        vent_mask_df = combined_df < ventilation_threshold_factor
        vent_mask_tip = main_tip_depth < 0.0
        vent_mask = vent_mask_df | vent_mask_tip
        ventilation_time = float(times[np.argmax(vent_mask)]) if np.any(vent_mask) else None

        pert_result = PerturbationResult(
            config=pert,
            trim_state=baseline_trim.state.copy(),
            trim_control=baseline_trim.control.copy(),
            initial_state=perturbed_state,
            final_state=states[-1].copy(),
            max_deviation=max_deviation,
            divergence_time=divergence_time,
            stable=stable,
            min_depth_factor=min_depth_factor,
            min_tip_depth=min_tip_depth,
            ventilation_time=ventilation_time,
            result=result if store_results else None,
        )
        sweep_result.perturbation_results.append(pert_result)

    return sweep_result


def run_configuration_comparison(
    configurations: List[ConfigurationVariation],
    u_forward: float = 10.0,
    duration: float = 5.0,
    dt: float = 0.005,
    run_simulation: bool = True,
    base_params: Optional[MothParams] = None,
) -> SweepResult:
    """Compare equilibria and open-loop behavior across configurations.

    For each configuration:
    1. Modify params (sailor position) and/or apply control constraints
    2. Find new trim equilibrium with constrained controls
    3. Optionally simulate open-loop response from that equilibrium
    4. Record equilibrium differences and response

    Note: Control offsets are applied as CONSTRAINTS during trim finding.
    E.g., flap_offset_deg=2 means the flap is FIXED at (baseline_flap + 2°)
    and the trim finder optimizes the OTHER control (elevator) to find equilibrium.

    Args:
        configurations: List of ConfigurationVariation to test.
        u_forward: Forward speed (m/s).
        duration: Simulation duration for open-loop test (s).
        dt: Integration timestep (s).
        run_simulation: Whether to run open-loop simulation from each equilibrium.
        base_params: Base MothParams (default: MOTH_BIEKER_V3).

    Returns:
        SweepResult with configuration_results populated.
    """
    from fmd.simulator.moth_3d import POS_D, THETA, W, Q, U

    base_params = base_params if base_params is not None else MOTH_BIEKER_V3

    # Find baseline trim first
    baseline_moth = Moth3D(
        base_params,
        u_forward=ConstantSchedule(float(u_forward)),
    )
    baseline_trim = find_moth_trim(base_params, u_forward=u_forward)

    sweep_result = SweepResult(
        baseline_trim_state=np.array(baseline_trim.state),
        baseline_trim_control=np.array(baseline_trim.control),
        u_forward=u_forward,
        duration=duration,
        dt=dt,
    )

    for config in configurations:
        # Modify params if sailor position changed
        if config.sailor_x_offset != 0:
            base_sailor_pos = base_params.sailor_position
            new_sailor_pos = base_sailor_pos.copy()
            new_sailor_pos[0] += config.sailor_x_offset  # x is forward
            modified_params = base_params.with_sailor_position(new_sailor_pos)
        else:
            modified_params = base_params

        # Create model with modified params
        moth = Moth3D(
            modified_params,
            u_forward=ConstantSchedule(float(u_forward)),
        )

        # Control offsets
        flap_offset = np.radians(config.flap_offset_deg)
        elev_offset = np.radians(config.elevator_offset_deg)

        # Determine which controls are free vs constrained
        has_flap_offset = config.flap_offset_deg != 0
        has_elev_offset = config.elevator_offset_deg != 0

        if not has_flap_offset and not has_elev_offset:
            if config.sailor_x_offset == 0:
                # Pure baseline — reuse the trim already computed above
                trim = baseline_trim
            else:
                # Sailor offset only — CasADi is robust, no multistart needed
                trim = find_moth_trim(
                    modified_params,
                    u_forward=u_forward,
                )
        else:
            # Use CasADi trim with fixed_controls for constrained controls.
            # Offsets are relative to BASELINE control.
            fixed_controls = {}
            if has_flap_offset:
                fixed_controls["main_flap"] = float(
                    baseline_trim.control[0] + flap_offset
                )
            if has_elev_offset:
                fixed_controls["rudder_elevator"] = float(
                    baseline_trim.control[1] + elev_offset
                )

            # Seed from baseline trim for better convergence when
            # controls are pinned (geometry guess may be far from feasible).
            baseline_z0 = np.concatenate([
                baseline_trim.state,
                baseline_trim.control,
                [baseline_trim.thrust],
            ])

            trim = find_moth_trim(
                modified_params,
                u_forward=u_forward,
                fixed_controls=fixed_controls,
                z0=baseline_z0,
            )

        config_result = ConfigurationResult(
            config=config,
            params=modified_params,
            trim_state=np.array(trim.state),
            trim_control=np.array(trim.control),
            trim_residual=trim.residual,
            trim_success=trim.success,
            delta_state=np.array(trim.state) - baseline_trim.state,
            delta_control=np.array(trim.control) - baseline_trim.control,
        )

        # Optionally run open-loop simulation
        if run_simulation:
            result = simulate(
                moth,
                jnp.array(trim.state),
                dt=dt,
                duration=duration,
                control=ConstantControl(jnp.array(trim.control)),
            )
            config_result.simulation_result = result
            config_result.final_state = np.array(result.states[-1])

        sweep_result.configuration_results.append(config_result)

    return sweep_result


# ===================================================================
# Sweep Configuration Factories
# ===================================================================


def default_perturbation_configs() -> List[PerturbationConfig]:
    """Generate default perturbation configurations.

    Pitch: ±0.1°, ±0.5°, ±1°, ±2°, ±5°
    Heave velocity: ±0.01, ±0.05, ±0.1, ±0.2 m/s

    Returns:
        List of 18 PerturbationConfig instances.
    """
    configs = []

    # Pitch perturbations (degrees -> radians)
    for deg in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            rad = sign * np.radians(deg)
            sign_str = "+" if sign > 0 else "-"
            configs.append(
                PerturbationConfig("theta", rad, f"pitch {sign_str}{deg}°")
            )

    # Heave velocity perturbations
    for vel in [0.01, 0.05, 0.1, 0.2]:
        for sign in [1, -1]:
            v = sign * vel
            sign_str = "+" if sign > 0 else "-"
            configs.append(
                PerturbationConfig("w", v, f"heave vel {sign_str}{vel} m/s")
            )

    return configs


def default_configuration_variations() -> List[ConfigurationVariation]:
    """Generate default configuration variations (single-axis + selected combos).

    Single-axis variations:
    - Sailor: ±10cm, ±20cm fore/aft (NOTE: not yet implemented in Moth3D v1)
    - Flap: ±2°, ±5°
    - Elevator: ±1°, ±2°

    Selected combinations (most informative):
    - Flap up + elevator up (reinforcing nose-down)
    - Flap down + elevator down (reinforcing nose-up)
    - Flap up + elevator down (opposing effects)
    - Flap down + elevator up (opposing effects)

    Note: Sailor position variations are included for completeness but the
    Moth3D v1 model does not compute gravity moment from CG offset, so these
    will not affect trim. See moth_3dof_equations.md for details.

    Returns:
        List of ConfigurationVariation instances.
    """
    configs = [
        # Baseline
        ConfigurationVariation(name="baseline"),
        # Sailor position variations
        ConfigurationVariation(sailor_x_offset=0.1, name="sailor +10cm fwd"),
        ConfigurationVariation(sailor_x_offset=-0.1, name="sailor -10cm aft"),
        # Flap variations
        ConfigurationVariation(flap_offset_deg=2.0, name="flap +2°"),
        ConfigurationVariation(flap_offset_deg=-2.0, name="flap -2°"),
        ConfigurationVariation(flap_offset_deg=5.0, name="flap +5°"),
        ConfigurationVariation(flap_offset_deg=-5.0, name="flap -5°"),
        # Elevator variations
        ConfigurationVariation(elevator_offset_deg=1.0, name="elev +1°"),
        ConfigurationVariation(elevator_offset_deg=-1.0, name="elev -1°"),
        ConfigurationVariation(elevator_offset_deg=2.0, name="elev +2°"),
        ConfigurationVariation(elevator_offset_deg=-2.0, name="elev -2°"),
        # Reinforcing combinations (same direction effects)
        ConfigurationVariation(
            flap_offset_deg=2.0, elevator_offset_deg=1.0,
            name="flap +2°, elev +1° (reinforce nose-down)"
        ),
        ConfigurationVariation(
            flap_offset_deg=-2.0, elevator_offset_deg=-1.0,
            name="flap -2°, elev -1° (reinforce nose-up)"
        ),
        # Opposing combinations
        ConfigurationVariation(
            flap_offset_deg=2.0, elevator_offset_deg=-1.0,
            name="flap +2°, elev -1° (height up, nose-up)"
        ),
        ConfigurationVariation(
            flap_offset_deg=-2.0, elevator_offset_deg=1.0,
            name="flap -2°, elev +1° (height down, nose-down)"
        ),
        # Extreme single-axis
        ConfigurationVariation(flap_offset_deg=8.0, name="flap +8° (extreme)"),
        ConfigurationVariation(flap_offset_deg=-8.0, name="flap -8° (extreme)"),
    ]
    return configs


def print_perturbation_summary(sweep_result: SweepResult) -> None:
    """Print a summary table of perturbation results.

    Args:
        sweep_result: SweepResult with perturbation_results.
    """
    print("\n" + "=" * 80)
    print("PERTURBATION SWEEP RESULTS")
    print("=" * 80)
    print(f"Baseline trim: pos_d={sweep_result.baseline_trim_state[0]:.4f} m, "
          f"theta={np.degrees(sweep_result.baseline_trim_state[1]):.3f}°")
    print(f"Duration: {sweep_result.duration}s, dt: {sweep_result.dt}s")
    print("-" * 80)
    print(f"{'Perturbation':<30} {'Stable':<8} {'Div Time':<10} "
          f"{'Final θ (°)':<12} {'Max |Δθ| (°)':<12}")
    print("-" * 80)

    for r in sweep_result.perturbation_results:
        div_str = f"{r.divergence_time:.2f}s" if r.divergence_time else "N/A"
        final_theta = np.degrees(r.final_state[1])
        max_theta_dev = np.degrees(r.max_deviation[1])
        print(f"{r.config.description:<30} {'Yes' if r.stable else 'No':<8} "
              f"{div_str:<10} {final_theta:<12.3f} {max_theta_dev:<12.3f}")


def print_configuration_summary(sweep_result: SweepResult) -> None:
    """Print a summary table of configuration comparison results.

    Args:
        sweep_result: SweepResult with configuration_results.
    """
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON RESULTS")
    print("=" * 80)
    print(f"Baseline trim: pos_d={sweep_result.baseline_trim_state[0]:.4f} m, "
          f"theta={np.degrees(sweep_result.baseline_trim_state[1]):.3f}°")
    print("-" * 80)
    print(f"{'Configuration':<45} {'Δpos_d (cm)':<12} {'Δθ (°)':<10} "
          f"{'Residual':<10}")
    print("-" * 80)

    for r in sweep_result.configuration_results:
        delta_pos_d = r.delta_state[0] * 100 if r.delta_state is not None else 0
        delta_theta = np.degrees(r.delta_state[1]) if r.delta_state is not None else 0
        print(f"{r.config.name:<45} {delta_pos_d:<12.2f} {delta_theta:<10.3f} "
              f"{r.trim_residual:<10.2e}")


# ===================================================================
# Category 3/4/5 Data Structures
# ===================================================================


@dataclass
class TransientConfig:
    """Off-equilibrium transient test config (Category 3).

    Specifies a control offset from trim to apply as a constant control,
    simulating the system response when held at a non-equilibrium control.

    Attributes:
        flap_offset_deg: Main flap offset from trim (degrees).
        elevator_offset_deg: Rudder elevator offset from trim (degrees).
        description: Human-readable description.
        duration: Override default simulation duration (s).
    """

    flap_offset_deg: float = 0.0
    elevator_offset_deg: float = 0.0
    description: str = ""
    duration: Optional[float] = None


@dataclass
class TransientResult:
    """Result from a single off-equilibrium transient simulation.

    Attributes:
        config: The transient configuration.
        trim_state: Baseline trim state.
        trim_control: Baseline trim control.
        applied_control: Actual control applied (trim + offset).
        initial_acceleration: State derivative at t=0.
        divergence_time: Time theta deviation exceeds threshold (None if stable).
        max_deviation: Per-state maximum absolute deviation from trim.
        final_state: State at end of simulation.
        result: Full SimulationResult (optional).
    """

    config: TransientConfig
    trim_state: np.ndarray
    trim_control: np.ndarray
    applied_control: np.ndarray
    initial_acceleration: np.ndarray
    divergence_time: Optional[float]
    max_deviation: np.ndarray
    final_state: np.ndarray
    result: Optional[SimulationResult] = None


@dataclass
class SpeedVariationConfig:
    """Speed variation test config (Category 5).

    Specifies a trim speed (used to find equilibrium) and a run speed
    (the actual forward speed during simulation).

    Attributes:
        trim_speed: Speed at which to find trim (m/s).
        run_speed: Speed at which to run simulation (m/s).
        description: Human-readable description.
        duration: Override default simulation duration (s).
    """

    trim_speed: float
    run_speed: float
    description: str = ""
    duration: Optional[float] = None


@dataclass
class SpeedVariationResult:
    """Result from a single speed variation simulation.

    Attributes:
        config: The speed variation configuration.
        trim_state: Trim state found at trim_speed.
        trim_control: Trim control found at trim_speed.
        run_speed: Actual forward speed used in simulation.
        divergence_time: Time theta deviation exceeds threshold (None if stable).
        max_deviation: Per-state maximum absolute deviation from trim.
        final_state: State at end of simulation.
        result: Full SimulationResult (optional).
    """

    config: SpeedVariationConfig
    trim_state: np.ndarray
    trim_control: np.ndarray
    run_speed: float
    divergence_time: Optional[float]
    max_deviation: np.ndarray
    final_state: np.ndarray
    result: Optional[SimulationResult] = None


@dataclass
class DampingComparisonConfig:
    """Damping comparison config (Category 4).

    Specifies a parameter variant and a perturbation to apply,
    enabling comparison of damping behavior across parameter sets.

    Attributes:
        params_name: Human-readable name for this parameter variant.
        params: MothParams instance to use.
        perturbation: Perturbation to apply to trim state.
        description: Human-readable description.
        duration: Override default simulation duration (s).
    """

    params_name: str
    params: MothParams
    perturbation: PerturbationConfig
    description: str = ""
    duration: Optional[float] = None


@dataclass
class DampingComparisonResult:
    """Result from a single damping comparison simulation.

    Attributes:
        config: The damping comparison configuration.
        trim_state: Trim state for this parameter set.
        trim_control: Trim control for this parameter set.
        perturbation_result: Full perturbation result from simulation.
        eigenvalues: Eigenvalues of linearized A matrix (complex).
    """

    config: DampingComparisonConfig
    trim_state: np.ndarray
    trim_control: np.ndarray
    perturbation_result: PerturbationResult
    eigenvalues: Optional[np.ndarray] = None


@dataclass
class ResponseMetrics:
    """Standardized metrics from any simulation response.

    Attributes:
        initial_acceleration: State derivative at t=0.
        initial_response_direction: Per-state direction string (e.g. "increase").
        divergence_time: Time theta exceeds threshold (None if stable).
        max_deviation: Per-state maximum absolute deviation from trim.
        final_state: State at end of simulation.
        stable: Whether theta remained within divergence threshold.
    """

    initial_acceleration: np.ndarray
    initial_response_direction: Dict[str, str]
    divergence_time: Optional[float]
    max_deviation: np.ndarray
    final_state: np.ndarray
    stable: bool


# ===================================================================
# Metrics Extraction
# ===================================================================


def extract_metrics(
    result: SimulationResult,
    trim_state: np.ndarray,
    divergence_threshold_deg: float = 30.0,
) -> ResponseMetrics:
    """Extract standardized response metrics from a simulation result.

    Computes initial acceleration, response direction, divergence time,
    max deviation, and stability from a SimulationResult relative to
    a trim state.

    Args:
        result: SimulationResult with times, states, controls.
        trim_state: The trim/reference state to measure deviations from.
        divergence_threshold_deg: Pitch deviation threshold for divergence (degrees).

    Returns:
        ResponseMetrics with all fields populated.
    """
    states = np.array(result.states)
    times = np.array(result.times)
    divergence_threshold = np.radians(divergence_threshold_deg)

    # Initial acceleration: (state[1] - state[0]) / dt
    dt = float(times[1] - times[0])
    initial_acceleration = (states[1] - states[0]) / dt

    # Initial response direction for each state
    initial_response_direction: Dict[str, str] = {}
    for name, idx in STATE_INDICES.items():
        diff = states[1][idx] - states[0][idx]
        if diff > 0:
            initial_response_direction[name] = "increase"
        elif diff < 0:
            initial_response_direction[name] = "decrease"
        else:
            initial_response_direction[name] = "none"

    # Deviation from trim
    deviation = states - trim_state
    max_deviation = np.max(np.abs(deviation), axis=0)

    # Divergence detection (theta exceeds threshold)
    theta_idx = STATE_INDICES["theta"]
    theta_deviation = np.abs(states[:, theta_idx] - trim_state[theta_idx])
    diverged_mask = theta_deviation > divergence_threshold
    if np.any(diverged_mask):
        divergence_time = float(times[np.argmax(diverged_mask)])
        stable = False
    else:
        divergence_time = None
        stable = True

    return ResponseMetrics(
        initial_acceleration=initial_acceleration,
        initial_response_direction=initial_response_direction,
        divergence_time=divergence_time,
        max_deviation=max_deviation,
        final_state=states[-1].copy(),
        stable=stable,
    )


# ===================================================================
# Category 3: Off-Equilibrium Transient Runner
# ===================================================================


def run_transient_sweep(
    configs: List[TransientConfig],
    u_forward: float = 10.0,
    duration: float = 5.0,
    dt: float = 0.005,
    divergence_threshold_deg: float = 30.0,
    params: Optional[MothParams] = None,
    store_results: bool = False,
) -> SweepResult:
    """Run a sweep of off-equilibrium transient tests.

    Finds baseline trim once, then for each config applies a control
    offset and simulates from the trim state with the offset control
    held constant.

    Args:
        configs: List of TransientConfig to test.
        u_forward: Forward speed (m/s).
        duration: Default simulation duration (s).
        dt: Integration timestep (s).
        divergence_threshold_deg: Pitch threshold for divergence detection (degrees).
        params: MothParams to use (default: MOTH_BIEKER_V3).
        store_results: Whether to store full SimulationResult in output.

    Returns:
        SweepResult with transient_results populated.
    """
    params = params if params is not None else MOTH_BIEKER_V3
    divergence_threshold = np.radians(divergence_threshold_deg)

    # Create model and find baseline trim
    moth = Moth3D(
        params,
        u_forward=ConstantSchedule(float(u_forward)),
    )
    baseline_trim = find_moth_trim(params, u_forward=u_forward)

    sweep_result = SweepResult(
        baseline_trim_state=np.array(baseline_trim.state),
        baseline_trim_control=np.array(baseline_trim.control),
        u_forward=u_forward,
        duration=duration,
        dt=dt,
    )

    for cfg in configs:
        sim_duration = cfg.duration if cfg.duration is not None else duration

        # Apply control offset (convert degrees to radians)
        flap_offset_rad = np.radians(cfg.flap_offset_deg)
        elev_offset_rad = np.radians(cfg.elevator_offset_deg)
        applied_control = baseline_trim.control.copy()
        applied_control[0] += flap_offset_rad
        applied_control[1] += elev_offset_rad

        # Compute initial acceleration from forward_dynamics
        trim_state_jax = jnp.array(baseline_trim.state)
        applied_ctrl_jax = jnp.array(applied_control)
        initial_accel = np.array(
            moth.forward_dynamics(trim_state_jax, applied_ctrl_jax, 0.0)
        )

        # Simulate from trim state with offset control
        result = simulate(
            moth,
            trim_state_jax,
            dt=dt,
            duration=sim_duration,
            control=ConstantControl(applied_ctrl_jax),
        )

        states = np.array(result.states)
        times = np.array(result.times)

        # Deviation from trim
        deviation = states - baseline_trim.state
        max_deviation = np.max(np.abs(deviation), axis=0)

        # Divergence detection
        theta_idx = STATE_INDICES["theta"]
        theta_deviation = np.abs(
            states[:, theta_idx] - baseline_trim.state[theta_idx]
        )
        diverged_mask = theta_deviation > divergence_threshold
        if np.any(diverged_mask):
            divergence_time = float(times[np.argmax(diverged_mask)])
        else:
            divergence_time = None

        transient_result = TransientResult(
            config=cfg,
            trim_state=baseline_trim.state.copy(),
            trim_control=baseline_trim.control.copy(),
            applied_control=applied_control,
            initial_acceleration=initial_accel,
            divergence_time=divergence_time,
            max_deviation=max_deviation,
            final_state=states[-1].copy(),
            result=result if store_results else None,
        )
        sweep_result.transient_results.append(transient_result)

    return sweep_result


# ===================================================================
# Category 5: Speed Variation Runner
# ===================================================================


def run_speed_variation(
    configs: List[SpeedVariationConfig],
    duration: float = 5.0,
    dt: float = 0.005,
    divergence_threshold_deg: float = 30.0,
    params: Optional[MothParams] = None,
    store_results: bool = False,
) -> SweepResult:
    """Run a sweep of speed variation tests.

    For each config, finds trim at config.trim_speed, then simulates
    at config.run_speed. If trim_speed == run_speed (Category 5A),
    runs a short equilibrium verification. If they differ (Category 5B),
    simulates the speed-mismatch transient.

    Args:
        configs: List of SpeedVariationConfig to test.
        duration: Default simulation duration (s).
        dt: Integration timestep (s).
        divergence_threshold_deg: Pitch threshold for divergence detection (degrees).
        params: MothParams to use (default: MOTH_BIEKER_V3).
        store_results: Whether to store full SimulationResult in output.

    Returns:
        SweepResult with speed_variation_results populated.
    """
    params = params if params is not None else MOTH_BIEKER_V3
    divergence_threshold = np.radians(divergence_threshold_deg)

    sweep_result = SweepResult(
        u_forward=0.0,  # varies per config
        duration=duration,
        dt=dt,
    )

    for cfg in configs:
        sim_duration = cfg.duration if cfg.duration is not None else duration

        # Create model at trim speed and find trim
        # Speed variation tests use surge_enabled=False so speed comes from
        # the u_forward callback (external schedule), not from the state.
        trim_moth = Moth3D(
            params,
            u_forward=ConstantSchedule(float(cfg.trim_speed)),
            surge_enabled=False,
        )
        trim_result = find_moth_trim(params, u_forward=cfg.trim_speed)

        trim_state = trim_result.state.copy()
        trim_control = trim_result.control.copy()

        if cfg.trim_speed == cfg.run_speed:
            # Category 5A: equilibrium verification at this speed
            result = simulate(
                trim_moth,
                jnp.array(trim_state),
                dt=dt,
                duration=sim_duration,
                control=ConstantControl(jnp.array(trim_control)),
            )
        else:
            # Category 5B: create model at run speed, simulate from trim state
            run_moth = Moth3D(
                params,
                u_forward=ConstantSchedule(float(cfg.run_speed)),
                surge_enabled=False,
            )
            result = simulate(
                run_moth,
                jnp.array(trim_state),
                dt=dt,
                duration=sim_duration,
                control=ConstantControl(jnp.array(trim_control)),
            )

        states = np.array(result.states)
        times = np.array(result.times)

        # Deviation from trim
        deviation = states - trim_state
        max_deviation = np.max(np.abs(deviation), axis=0)

        # Divergence detection
        theta_idx = STATE_INDICES["theta"]
        theta_deviation = np.abs(states[:, theta_idx] - trim_state[theta_idx])
        diverged_mask = theta_deviation > divergence_threshold
        if np.any(diverged_mask):
            divergence_time = float(times[np.argmax(diverged_mask)])
        else:
            divergence_time = None

        speed_result = SpeedVariationResult(
            config=cfg,
            trim_state=trim_state,
            trim_control=trim_control,
            run_speed=cfg.run_speed,
            divergence_time=divergence_time,
            max_deviation=max_deviation,
            final_state=states[-1].copy(),
            result=result if store_results else None,
        )
        sweep_result.speed_variation_results.append(speed_result)

    return sweep_result


# ===================================================================
# Category 4: Damping Comparison Runner
# ===================================================================


def run_damping_comparison(
    configs: List[DampingComparisonConfig],
    u_forward: float = 10.0,
    duration: float = 5.0,
    dt: float = 0.005,
    divergence_threshold_deg: float = 30.0,
    store_results: bool = False,
) -> SweepResult:
    """Run a sweep of damping comparison tests.

    For each config, creates a Moth3D with the specified params, finds
    trim, applies a perturbation, simulates, and computes eigenvalues
    of the linearized system.

    Args:
        configs: List of DampingComparisonConfig to test.
        u_forward: Forward speed (m/s).
        duration: Default simulation duration (s).
        dt: Integration timestep (s).
        divergence_threshold_deg: Pitch threshold for divergence detection (degrees).
        store_results: Whether to store full SimulationResult in output.

    Returns:
        SweepResult with damping_comparison_results populated.
    """
    divergence_threshold = np.radians(divergence_threshold_deg)

    sweep_result = SweepResult(
        u_forward=u_forward,
        duration=duration,
        dt=dt,
    )

    for cfg in configs:
        sim_duration = cfg.duration if cfg.duration is not None else duration

        # Create model with this config's params
        moth = Moth3D(
            cfg.params,
            u_forward=ConstantSchedule(float(u_forward)),
        )

        # Find trim
        trim = find_moth_trim(cfg.params, u_forward=u_forward)
        trim_state = trim.state.copy()
        trim_control = trim.control.copy()

        # Apply perturbation to trim state
        perturbed_state = trim_state.copy()
        perturbed_state[cfg.perturbation.state_index] += cfg.perturbation.magnitude

        # Simulate
        result = simulate(
            moth,
            jnp.array(perturbed_state),
            dt=dt,
            duration=sim_duration,
            control=ConstantControl(jnp.array(trim_control)),
        )

        states = np.array(result.states)
        times = np.array(result.times)

        # Compute metrics
        deviation = states - trim_state
        max_deviation = np.max(np.abs(deviation), axis=0)

        # Divergence detection
        theta_idx = STATE_INDICES["theta"]
        theta_deviation = np.abs(states[:, theta_idx] - trim_state[theta_idx])
        diverged_mask = theta_deviation > divergence_threshold
        if np.any(diverged_mask):
            divergence_time = float(times[np.argmax(diverged_mask)])
            stable = False
        else:
            divergence_time = None
            stable = True

        # Build PerturbationResult for this run
        pert_result = PerturbationResult(
            config=cfg.perturbation,
            trim_state=trim_state,
            trim_control=trim_control,
            initial_state=perturbed_state,
            final_state=states[-1].copy(),
            max_deviation=max_deviation,
            divergence_time=divergence_time,
            stable=stable,
            result=result if store_results else None,
        )

        # Compute eigenvalues from linearization
        try:
            A, _B = linearize(
                moth,
                jnp.array(trim_state),
                jnp.array(trim_control),
            )
            eigenvalues = np.linalg.eigvals(np.array(A))
        except Exception:
            eigenvalues = None

        damping_result = DampingComparisonResult(
            config=cfg,
            trim_state=trim_state,
            trim_control=trim_control,
            perturbation_result=pert_result,
            eigenvalues=eigenvalues,
        )
        sweep_result.damping_comparison_results.append(damping_result)

    return sweep_result


# ===================================================================
# Category 3/4/5 Default Config Factories
# ===================================================================


def default_transient_configs() -> List[TransientConfig]:
    """Generate default off-equilibrium transient configurations.

    Returns 9 configs:
    - Flap offsets: +2deg, -2deg, +5deg, -5deg
    - Elevator offsets: +1deg, -1deg, +2deg, -2deg
    - Combined: flap +2deg, elev +1deg (reinforcing nose-down)

    Returns:
        List of 9 TransientConfig instances.
    """
    return [
        TransientConfig(
            flap_offset_deg=2.0,
            description="flap +2°",
        ),
        TransientConfig(
            flap_offset_deg=-2.0,
            description="flap -2°",
        ),
        TransientConfig(
            flap_offset_deg=5.0,
            description="flap +5°",
        ),
        TransientConfig(
            flap_offset_deg=-5.0,
            description="flap -5°",
        ),
        TransientConfig(
            elevator_offset_deg=1.0,
            description="elev +1°",
        ),
        TransientConfig(
            elevator_offset_deg=-1.0,
            description="elev -1°",
        ),
        TransientConfig(
            elevator_offset_deg=2.0,
            description="elev +2°",
        ),
        TransientConfig(
            elevator_offset_deg=-2.0,
            description="elev -2°",
        ),
        TransientConfig(
            flap_offset_deg=2.0,
            elevator_offset_deg=1.0,
            description="flap +2°, elev +1° (reinforcing)",
        ),
    ]


def default_speed_variation_configs() -> List[SpeedVariationConfig]:
    """Generate default speed variation configurations.

    Returns 7 configs:
    - 5A (equilibrium at each speed): 8, 10, 12, 14 m/s
    - 5B (speed step from 10 m/s trim): run at 8, 12, 14 m/s

    Returns:
        List of 7 SpeedVariationConfig instances.
    """
    return [
        # Category 5A: trim and run at same speed
        SpeedVariationConfig(
            trim_speed=8.0, run_speed=8.0,
            description="5A: trim=8, run=8 m/s",
        ),
        SpeedVariationConfig(
            trim_speed=10.0, run_speed=10.0,
            description="5A: trim=10, run=10 m/s",
        ),
        SpeedVariationConfig(
            trim_speed=12.0, run_speed=12.0,
            description="5A: trim=12, run=12 m/s",
        ),
        SpeedVariationConfig(
            trim_speed=14.0, run_speed=14.0,
            description="5A: trim=14, run=14 m/s",
        ),
        # Category 5B: trim at 10 m/s, run at different speed
        SpeedVariationConfig(
            trim_speed=10.0, run_speed=8.0,
            description="5B: trim=10, run=8 m/s (speed decrease)",
        ),
        SpeedVariationConfig(
            trim_speed=10.0, run_speed=12.0,
            description="5B: trim=10, run=12 m/s (speed increase)",
        ),
        SpeedVariationConfig(
            trim_speed=10.0, run_speed=14.0,
            description="5B: trim=10, run=14 m/s (large speed increase)",
        ),
    ]


def default_damping_configs() -> List[DampingComparisonConfig]:
    """Generate default damping comparison configurations.

    Returns 6 configs: 3 parameter variants x 2 perturbation sizes.
    - Parameter variants: full_damping, no_added_mass, high_added_mass
    - Perturbation sizes: theta +0.5deg, theta +2.0deg

    Returns:
        List of 6 DampingComparisonConfig instances.
    """
    # Parameter variants
    param_variants = [
        ("full_damping", MOTH_BIEKER_V3),
        (
            "no_added_mass",
            attrs.evolve(
                MOTH_BIEKER_V3,
                added_mass_heave=0.0,
                added_inertia_pitch=0.0,
            ),
        ),
        (
            "high_added_mass",
            attrs.evolve(
                MOTH_BIEKER_V3,
                added_mass_heave=20.0,
                added_inertia_pitch=17.5,
            ),
        ),
    ]

    # Perturbation sizes
    perturbation_sizes = [
        PerturbationConfig("theta", np.radians(0.5), "pitch +0.5°"),
        PerturbationConfig("theta", np.radians(2.0), "pitch +2.0°"),
    ]

    configs = []
    for params_name, params_val in param_variants:
        for pert in perturbation_sizes:
            configs.append(
                DampingComparisonConfig(
                    params_name=params_name,
                    params=params_val,
                    perturbation=pert,
                    description=f"{params_name}, {pert.description}",
                )
            )

    return configs


# ===================================================================
# Category 3/4/5 Print Summaries
# ===================================================================


def print_transient_summary(sweep_result: SweepResult) -> None:
    """Print a summary table of off-equilibrium transient results.

    Args:
        sweep_result: SweepResult with transient_results.
    """
    print("\n" + "=" * 90)
    print("OFF-EQUILIBRIUM TRANSIENT RESULTS (Category 3)")
    print("=" * 90)
    if sweep_result.baseline_trim_state is not None:
        print(
            f"Baseline trim: pos_d={sweep_result.baseline_trim_state[0]:.4f} m, "
            f"theta={np.degrees(sweep_result.baseline_trim_state[1]):.3f}°"
        )
    print("-" * 90)
    print(
        f"{'Config':<35} {'Div Time':<10} {'Max |Δθ| (°)':<14} "
        f"{'Max |Δd| (cm)':<14} {'Init ẍ_d':<10} {'Init q̇':<10}"
    )
    print("-" * 90)

    for r in sweep_result.transient_results:
        div_str = f"{r.divergence_time:.2f}s" if r.divergence_time else "N/A"
        max_theta_dev = np.degrees(r.max_deviation[STATE_INDICES["theta"]])
        max_pos_d_dev = r.max_deviation[STATE_INDICES["pos_d"]] * 100  # m -> cm
        init_accel_d = r.initial_acceleration[STATE_INDICES["w"]]
        init_accel_q = r.initial_acceleration[STATE_INDICES["q"]]
        desc = r.config.description or f"flap {r.config.flap_offset_deg}°, elev {r.config.elevator_offset_deg}°"
        print(
            f"{desc:<35} {div_str:<10} {max_theta_dev:<14.3f} "
            f"{max_pos_d_dev:<14.2f} {init_accel_d:<10.3f} {init_accel_q:<10.4f}"
        )


def print_speed_variation_summary(sweep_result: SweepResult) -> None:
    """Print a summary table of speed variation results.

    Args:
        sweep_result: SweepResult with speed_variation_results.
    """
    print("\n" + "=" * 90)
    print("SPEED VARIATION RESULTS (Category 5)")
    print("=" * 90)
    print("-" * 90)
    print(
        f"{'Config':<40} {'Trim (m/s)':<12} {'Run (m/s)':<12} "
        f"{'Div Time':<10} {'Max |Δθ| (°)':<14}"
    )
    print("-" * 90)

    for r in sweep_result.speed_variation_results:
        div_str = f"{r.divergence_time:.2f}s" if r.divergence_time else "N/A"
        max_theta_dev = np.degrees(r.max_deviation[STATE_INDICES["theta"]])
        desc = r.config.description or f"trim={r.config.trim_speed}, run={r.config.run_speed}"
        print(
            f"{desc:<40} {r.config.trim_speed:<12.1f} {r.config.run_speed:<12.1f} "
            f"{div_str:<10} {max_theta_dev:<14.3f}"
        )


def print_damping_comparison_summary(sweep_result: SweepResult) -> None:
    """Print a summary table of damping comparison results.

    Args:
        sweep_result: SweepResult with damping_comparison_results.
    """
    print("\n" + "=" * 100)
    print("DAMPING COMPARISON RESULTS (Category 4)")
    print("=" * 100)
    print("-" * 100)
    print(
        f"{'Config':<35} {'Stable':<8} {'Div Time':<10} "
        f"{'Max |Δθ| (°)':<14} {'Eigenvalues (real parts)':<30}"
    )
    print("-" * 100)

    for r in sweep_result.damping_comparison_results:
        pr = r.perturbation_result
        div_str = f"{pr.divergence_time:.2f}s" if pr.divergence_time else "N/A"
        max_theta_dev = np.degrees(pr.max_deviation[STATE_INDICES["theta"]])
        stable_str = "Yes" if pr.stable else "No"

        if r.eigenvalues is not None:
            real_parts = np.real(r.eigenvalues)
            eig_str = ", ".join(f"{v:.2f}" for v in sorted(real_parts))
        else:
            eig_str = "N/A"

        desc = r.config.description or f"{r.config.params_name}"
        print(
            f"{desc:<35} {stable_str:<8} {div_str:<10} "
            f"{max_theta_dev:<14.3f} {eig_str:<30}"
        )
