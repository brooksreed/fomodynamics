"""Numerical integrators for JAX dynamic systems.

Provides RK4 (4th-order Runge-Kutta) integration using JAX.
Uses jax.lax.scan for efficient JIT-compiled simulation loops.

The fixed-step RK4 implementation matches the numpy version exactly,
enabling golden master testing for numerical equivalence.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator._config import FMD_DTYPE  # noqa: F401 (side-effect import)

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass, field
from typing import NamedTuple, Union, Optional, TYPE_CHECKING
import numpy as np

from fmd.simulator.base import JaxDynamicSystem
from fmd.simulator.control import (
    ControlSchedule,
    ConstantControl,
    ZeroControl,
    PiecewiseConstantControl,
)
from fmd.simulator.constraints.base import ConstraintSet, Capability
from fmd.simulator.noise.base import NoiseModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SimulationResult(NamedTuple):
    """Result from simulation (JIT-safe, arrays only).

    This result type is safe to use inside @jax.jit decorated functions.
    For logging/analysis, use result_with_meta() to add metadata.

    Attributes:
        times: Time points array, shape (num_steps,)
        states: State trajectory array, shape (num_steps, num_states)
        controls: Control history array, shape (num_steps, num_controls)
    """

    times: Array
    states: Array
    controls: Array


# Backwards compatibility alias
JaxSimulationResult = SimulationResult


@dataclass
class RichSimulationResult:
    """Simulation result with metadata for logging/analysis.

    NOT JIT-safe - use outside of jitted functions only.

    Attributes:
        times: Time points array
        states: State trajectory array
        controls: Control history array
        state_names: Names of state variables
        control_names: Names of control variables
        outputs: Additional computed outputs (e.g., derived quantities)
    """

    times: "NDArray"
    states: "NDArray"
    controls: "NDArray"
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    outputs: dict[str, "NDArray"] = field(default_factory=dict)


def compute_aux_trajectory(
    system: JaxDynamicSystem,
    result: SimulationResult,
    env=None,
) -> dict[str, np.ndarray]:
    """Compute auxiliary outputs for all timesteps in a simulation result.

    Evaluates system.compute_aux at each (state, control, time) triple
    in the trajectory using jax.vmap for efficiency.

    Args:
        system: The dynamic system that was simulated
        result: SimulationResult from simulate()
        env: Optional Environment

    Returns:
        Dict mapping aux_names to 1D numpy arrays of shape (num_steps,).
        Returns {} if system.num_aux == 0.
    """
    if system.num_aux == 0:
        return {}

    times = result.times
    states = result.states
    controls = result.controls

    def aux_at_step(state, control, t):
        return system.compute_aux(state, control, t, env=env)

    aux_all = jax.vmap(aux_at_step)(states, controls, times)
    aux_np = np.asarray(aux_all)

    return {name: aux_np[:, i] for i, name in enumerate(system.aux_names)}


def result_with_meta(
    system: JaxDynamicSystem,
    result: SimulationResult,
    outputs: Optional[dict[str, Array]] = None,
    env=None,
) -> RichSimulationResult:
    """Attach metadata to a JIT-safe result for logging.

    Call this OUTSIDE of jitted functions. Automatically computes
    auxiliary outputs from system.compute_aux if the system has any.

    Merge precedence (later overwrites earlier on key collision):
    1. Auto-computed aux outputs (from system.compute_aux)
    2. Environment-derived outputs (e.g., wave channels)
    3. User-provided outputs dict

    Aux-name collisions from env outputs raise an error.
    User-provided collisions override with a warning.

    Args:
        system: The dynamic system that was simulated
        result: JIT-safe SimulationResult from simulate()
        outputs: Optional dict of additional computed outputs
        env: Optional Environment with wave_field for wave output computation

    Returns:
        RichSimulationResult with state/control names and outputs

    Example:
        result = simulate(pendulum, initial, dt=0.01, duration=10.0)
        rich_result = result_with_meta(pendulum, result)
        writer.write(rich_result, "simulation.csv")
    """
    import warnings

    # Convert JAX arrays to numpy
    times_np = np.asarray(result.times)
    states_np = np.asarray(result.states)
    controls_np = np.asarray(result.controls)

    # Step 1: Auto-compute aux outputs
    outputs_np = compute_aux_trajectory(system, result, env=env)

    # Step 2: Merge env-derived outputs (wave channels)
    if env is not None:
        from fmd.simulator.output import compute_wave_outputs

        rich_for_waves = RichSimulationResult(
            times=times_np,
            states=states_np,
            controls=controls_np,
            state_names=tuple(system.state_names),
            control_names=tuple(system.control_names),
            outputs={},
        )
        wave_outputs = compute_wave_outputs(rich_for_waves, env)
        # Check for collisions with aux names
        aux_collisions = set(wave_outputs.keys()) & set(outputs_np.keys())
        if aux_collisions:
            raise ValueError(
                f"Environment output keys collide with aux_names: {aux_collisions}"
            )
        outputs_np.update(wave_outputs)

    # Step 3: Merge user-provided outputs (override with warning)
    if outputs:
        for key, val in outputs.items():
            if key in outputs_np:
                warnings.warn(
                    f"User output '{key}' overrides auto-computed output",
                    stacklevel=2,
                )
            outputs_np[key] = np.asarray(val)

    return RichSimulationResult(
        times=times_np,
        states=states_np,
        controls=controls_np,
        state_names=tuple(system.state_names),
        control_names=tuple(system.control_names),
        outputs=outputs_np,
    )


def _build_time_grid(dt: float, duration: float) -> tuple[list[float], int]:
    """Build integration-driven time grid.

    Steps by dt until reaching or exceeding duration, with a shortened
    final step if needed.

    Args:
        dt: Target time step (seconds, Python float)
        duration: Total simulation duration (seconds, Python float)

    Returns:
        Tuple of (times_list, num_steps) where times_list contains
        all time points starting from 0.0 and num_steps is len(times_list).

    Note:
        dt and duration must be Python floats (not JAX arrays) because
        they determine the loop structure which must be static for JIT.
    """
    n_steps = int(np.ceil(duration / dt))
    times_list = list(np.minimum(np.arange(n_steps + 1) * dt, duration))
    num_steps = len(times_list)
    return times_list, num_steps


def _resolve_control_schedule(
    system: JaxDynamicSystem,
    control: Optional[Union[ControlSchedule, Array]],
    times: Array,
) -> ControlSchedule:
    """Resolve control input to a ControlSchedule.

    Handles three cases:
    1. None -> Use system's default control (wrapped in ConstantControl or ZeroControl)
    2. ControlSchedule -> Return as-is
    3. Array -> Wrap in PiecewiseConstantControl

    Args:
        system: JaxDynamicSystem instance
        control: Control input (None, ControlSchedule, or pre-sampled array)
        times: Time array for PiecewiseConstantControl if needed

    Returns:
        ControlSchedule instance
    """
    if control is None:
        default_ctrl = system.default_control()
        if default_ctrl.shape[0] == 0:
            return ZeroControl(dim=0)
        else:
            return ConstantControl(default_ctrl)
    elif isinstance(control, ControlSchedule):
        return control
    else:
        # Assume it's a pre-sampled array
        return PiecewiseConstantControl(times, control)


def _collect_results(
    initial_state: Array,
    init_ctrl: Array,
    states_rest: Array,
    controls_rest: Array,
    times: Array,
    num_steps: int,
) -> JaxSimulationResult:
    """Collect simulation results by prepending initial state/control.

    Handles the common pattern of prepending initial conditions to
    the arrays returned from jax.lax.scan, with special handling
    for empty control vectors.

    Args:
        initial_state: Initial state vector
        init_ctrl: Initial control vector
        states_rest: States from scan (excludes initial)
        controls_rest: Controls from scan (excludes initial)
        times: Time array
        num_steps: Total number of time steps

    Returns:
        JaxSimulationResult with times, states, and controls
    """
    states = jnp.vstack([initial_state[None, :], states_rest])

    if init_ctrl.shape[0] == 0:
        controls = jnp.zeros((num_steps, 0))
    else:
        controls = jnp.vstack([init_ctrl[None, :], controls_rest])

    return JaxSimulationResult(times, states, controls)


def _enforce_constraints_jit(
    constraints: Optional[ConstraintSet],
    enforcement: Capability,
    t: float,
    state: Array,
    control: Array,
    u_prev: Optional[Array] = None,
    dt: Optional[float] = None,
) -> tuple[Array, Array, Array]:
    """JIT-compatible constraint enforcement.

    Args:
        constraints: ConstraintSet to enforce (may be None)
        enforcement: Capability specifying enforcement method
        t: Current simulation time
        state: State vector
        control: Control vector
        u_prev: Previous control (required for HAS_RATE_LIMIT enforcement)
        dt: Timestep (required for HAS_RATE_LIMIT enforcement)

    Returns: (state, control, clipped_flag)
    where clipped_flag is 1.0 if any constraint was active, 0.0 otherwise.
    """
    if constraints is None or len(constraints) == 0:
        return state, control, jnp.array(0.0)

    applicable = constraints.by_capability(enforcement)
    if not applicable:
        return state, control, jnp.array(0.0)

    state_orig, control_orig = state, control

    for c in applicable:
        if enforcement == Capability.HARD_CLIP:
            state, control = c.clip(t, state, control)
        elif enforcement == Capability.PROJECTION:
            state, control = c.project(t, state, control)
        elif enforcement == Capability.HAS_RATE_LIMIT:
            # Rate limit constraints require u_prev and dt
            if u_prev is None:
                raise ValueError(
                    f"Rate limit constraint '{c.name}' enforcement requires u_prev, "
                    f"but None was provided. Provide an initial u_prev or skip rate "
                    f"limit enforcement for the first step."
                )
            state, control = c.clip_with_prev(t, state, control, u_prev, dt)

    clipped = jnp.logical_or(
        jnp.any(state != state_orig),
        jnp.any(control != control_orig)
    ).astype(FMD_DTYPE)

    return state, control, clipped


def euler_step(
    system: JaxDynamicSystem,
    state: Array,
    control: Array,
    dt: float,
    t: float = 0.0,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> Array:
    """Perform a single forward Euler integration step.

    x_{k+1} = x_k + dt * f(x_k, u_k, t)

    Args:
        system: JaxDynamicSystem instance
        state: Current state vector
        control: Control input (held constant during step)
        dt: Time step size
        t: Current simulation time (default 0.0)
        constraints: Optional ConstraintSet for constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        New state vector after integration step

    Note:
        Forward Euler is first-order accurate and conditionally stable.
        For RK4 stability, require dt < 2.785/|lambda_max|.
        For Euler stability, require dt < 2/|lambda_max| (stricter).
        For production use, prefer simulate() which uses RK4.
    """
    deriv = system.forward_dynamics(state, control, t, env=env)
    new_state = state + dt * deriv
    new_state = system.post_step(new_state)
    if constraints is not None:
        new_state, _, _ = _enforce_constraints_jit(constraints, enforcement, t + dt, new_state, control)
    return new_state


def rk4_step(
    system: JaxDynamicSystem,
    state: Array,
    control: Array,
    dt: float,
    t: float = 0.0,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> Array:
    """Perform a single RK4 integration step.

    Matches numpy implementation exactly: control held constant,
    post_step applied after final combination.

    Args:
        system: JaxDynamicSystem instance
        state: Current state vector
        control: Control input (held constant during step)
        dt: Time step size
        t: Current simulation time (default 0.0)
        constraints: Optional ConstraintSet for constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        New state vector after integration step
    """
    k1 = system.forward_dynamics(state, control, t, env=env)
    k2 = system.forward_dynamics(state + 0.5 * dt * k1, control, t + 0.5 * dt, env=env)
    k3 = system.forward_dynamics(state + 0.5 * dt * k2, control, t + 0.5 * dt, env=env)
    k4 = system.forward_dynamics(state + dt * k3, control, t + dt, env=env)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Apply post-step (e.g., quaternion normalization) - matches numpy exactly
    new_state = system.post_step(new_state)
    if constraints is not None:
        new_state, _, _ = _enforce_constraints_jit(constraints, enforcement, t + dt, new_state, control)
    return new_state


@eqx.filter_jit
def simulate(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate a JAX dynamic system using fixed-step RK4.

    Uses jax.lax.scan for efficient JIT-compiled integration.
    Automatically JIT-compiled via eqx.filter_jit for performance.

    Args:
        system: JaxDynamicSystem instance to simulate
        initial_state: Initial state vector
        dt: Fixed time step (seconds) - must be a Python float, not JAX array
        duration: Total simulation duration (seconds) - must be a Python float
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_steps, num_controls) for pre-sampled
            - None for system.default_control()
        constraints: Optional ConstraintSet for post-step constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times, states, and controls

    Note:
        This uses fixed-step RK4 to match numpy implementation exactly.
        dt and duration must be Python floats (not JAX arrays) because
        they determine the loop structure which must be static for JIT.
        The first call with a given (dt, duration) pair incurs JIT
        compilation overhead (~70-200ms); subsequent calls are cached.
        For production use with adaptive stepping, consider using diffrax.

    Example:
        from fmd.simulator import SimplePendulumJax, simulate
        from fmd.simulator.params import PENDULUM_1M

        pendulum = SimplePendulumJax(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # 0.5 rad initial angle
        result = simulate(pendulum, initial, dt=0.01, duration=10.0)
    """
    # Build time grid using helper function
    times_list, num_steps = _build_time_grid(dt, duration)
    times = jnp.array(times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def step_fn(carry, idx):
        """Single integration step for lax.scan.

        Constraint Enforcement Order:
            1. Rate limits (HAS_RATE_LIMIT) are enforced BEFORE integration.
               This clips the control input to respect slew rate limits.
            2. State constraints (HARD_CLIP, PROJECTION) are enforced AFTER
               integration inside rk4_step. This ensures the resulting state
               satisfies bounds like position limits or ground planes.

        This ordering ensures actuator rate limits are respected when computing
        the control to apply, while state constraints correct any violations
        that arise from the dynamics integration.
        """
        state, t_prev, u_prev = carry
        t_curr = times[idx]
        dt_actual = t_curr - t_prev
        ctrl = schedule(t_prev, state)

        # Enforce rate limits on control BEFORE integration (see docstring)
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_actual
            )

        # rk4_step enforces state constraints AFTER integration
        new_state = rk4_step(system, state, ctrl, dt_actual, t_prev, constraints, enforcement, env=env)
        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Run integration loop with lax.scan
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        step_fn,
        (initial_state, 0.0, init_ctrl),
        jnp.arange(1, num_steps),
    )

    return _collect_results(initial_state, init_ctrl, states_rest, controls_rest, times, num_steps)


def simulate_euler(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate a JAX dynamic system using fixed-step forward Euler.

    Uses jax.lax.scan for efficient JIT-compiled integration.

    Args:
        system: JaxDynamicSystem instance to simulate
        initial_state: Initial state vector
        dt: Fixed time step (seconds) - must be a Python float, not JAX array
        duration: Total simulation duration (seconds) - must be a Python float
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_steps, num_controls) for pre-sampled
            - None for system.default_control()
        constraints: Optional ConstraintSet for post-step constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times, states, and controls

    Note:
        Forward Euler is first-order accurate and less stable than RK4.
        For production use, prefer simulate() which uses RK4.

    Example:
        from fmd.simulator import Cartpole, simulate_euler
        from fmd.simulator.params import CARTPOLE_CLASSIC

        cartpole = Cartpole(CARTPOLE_CLASSIC)
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])
        result = simulate_euler(cartpole, initial, dt=0.02, duration=5.0)
    """
    # Build time grid using helper function
    times_list, num_steps = _build_time_grid(dt, duration)
    times = jnp.array(times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def step_fn(carry, idx):
        """Single Euler integration step for lax.scan."""
        state, t_prev, u_prev = carry
        t_curr = times[idx]
        dt_actual = t_curr - t_prev
        ctrl = schedule(t_prev, state)

        # First enforce rate limits on control before integration
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_actual
            )

        new_state = euler_step(system, state, ctrl, dt_actual, t_prev, constraints, enforcement, env=env)
        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Run integration loop with lax.scan
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        step_fn,
        (initial_state, 0.0, init_ctrl),
        jnp.arange(1, num_steps),
    )

    return _collect_results(initial_state, init_ctrl, states_rest, controls_rest, times, num_steps)


def simulate_euler_substepped(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt_sim: float,
    dt_control: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate with substepped integration (split physics / control rates).

    Physics integration runs at ``dt_sim`` while the controller updates at
    the slower ``dt_control``. The control is held constant during substeps
    (zero-order hold), and results are sampled at the control rate.

    Args:
        system: JaxDynamicSystem instance to simulate
        initial_state: Initial state vector
        dt_sim: Simulation timestep (e.g., 0.001 = 1ms)
        dt_control: Control update rate (e.g., 0.02 = 20ms)
        duration: Total simulation duration (seconds)
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_control_steps, num_controls)
            - None for system.default_control()
        constraints: Optional ConstraintSet for constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times at control rate (not simulation rate)

    Note:
        This is the standard approach for embedded systems where:
        1. Control computation is expensive (LQR, MPC, neural nets)
        2. Real hardware can't run controllers at physics rates
        3. Sensors don't update at physics rates
        4. Controller gains are tuned for specific control rates

        Rate limit constraints are applied at the control rate (dt_control),
        NOT the physics rate (dt_sim).

    Example:
        # 1000 Hz physics, 50 Hz control
        result = simulate_euler_substepped(
            system, x0,
            dt_sim=0.001,     # 1ms physics
            dt_control=0.02,  # 20ms control
            duration=5.0
        )
    """
    # Validate timestep relationship
    substeps_per_control = int(round(dt_control / dt_sim))
    if abs(substeps_per_control * dt_sim - dt_control) > 1e-10:
        raise ValueError(
            f"dt_control ({dt_control}) must be an integer multiple of "
            f"dt_sim ({dt_sim}). Got ratio: {dt_control / dt_sim}"
        )

    # Build control time grid
    t = 0.0
    control_times_list = [t]
    while t < duration:
        step = min(dt_control, duration - t)
        t = t + step
        control_times_list.append(t)
    num_control_steps = len(control_times_list)
    control_times = jnp.array(control_times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, control_times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def run_substeps(state: Array, t_start: float, ctrl: Array) -> Array:
        """Run multiple Euler substeps with fixed control."""

        def substep_fn(state_and_t, _):
            """Single Euler substep within a control period."""
            state, t = state_and_t
            new_state = euler_step(system, state, ctrl, dt_sim, t, constraints, enforcement, env=env)
            return (new_state, t + dt_sim), None

        (final_state, _), _ = jax.lax.scan(
            substep_fn,
            (state, t_start),
            None,
            length=substeps_per_control,
        )
        return final_state

    def control_step_fn(carry, idx):
        """One control period with multiple physics substeps."""
        state, t_prev, u_prev = carry
        t_curr = control_times[idx]

        # Get control at start of this control period
        ctrl = schedule(t_prev, state)

        # Enforce rate limits on control before integration (at control rate)
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_control
            )

        # Run substeps within this control period
        new_state = run_substeps(state, t_prev, ctrl)

        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Run integration loop with lax.scan over control steps
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        control_step_fn,
        (initial_state, 0.0, init_ctrl),
        jnp.arange(1, num_control_steps),
    )

    return _collect_results(initial_state, init_ctrl, states_rest, controls_rest, control_times, num_control_steps)


def _validate_times(times) -> None:
    """Validate a user-supplied time array.

    Checks that the array has at least 2 points and is strictly increasing.
    Called only from ``simulate_trajectory`` where times are user-supplied.
    Other simulate functions use ``_build_time_grid`` which is monotonic
    by construction.

    Raises:
        ValueError: If times has fewer than 2 points or is not strictly increasing.
    """
    if len(times) < 2:
        raise ValueError(
            f"times must have at least 2 points, got {len(times)}"
        )
    diffs = np.diff(np.asarray(times))
    if np.any(diffs <= 0):
        raise ValueError(
            "times must be strictly increasing (no duplicate or decreasing values)"
        )


def simulate_trajectory(
    system: JaxDynamicSystem,
    initial_state: Array,
    times: Array,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate with explicit time array.

    Useful when you need specific time points or variable time steps.

    Args:
        system: JaxDynamicSystem instance to simulate
        initial_state: Initial state vector
        times: Array of time points (must be strictly increasing, at least 2 points)
        control: Control schedule or pre-sampled array
        constraints: Optional ConstraintSet for post-step constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times, states, and controls

    Raises:
        ValueError: If times has fewer than 2 points or is not strictly increasing.
    """
    _validate_times(times)
    num_steps = len(times)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def step_fn(carry, idx):
        state, t_prev, u_prev = carry
        t_curr = times[idx]
        dt_actual = t_curr - t_prev
        ctrl = schedule(t_prev, state)

        # First enforce rate limits on control before integration
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_actual
            )

        new_state = rk4_step(system, state, ctrl, dt_actual, t_prev, constraints, enforcement, env=env)
        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        step_fn,
        (initial_state, times[0], init_ctrl),
        jnp.arange(1, num_steps),
    )

    return _collect_results(initial_state, init_ctrl, states_rest, controls_rest, times, num_steps)


def semi_implicit_euler_step(
    system: JaxDynamicSystem,
    state: Array,
    control: Array,
    dt: float,
    t: float = 0.0,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> Array:
    """Perform a single semi-implicit (symplectic) Euler integration step.

    Semi-implicit Euler updates velocity first using old position,
    then updates position using NEW velocity. This preserves the
    symplectic structure of Hamiltonian systems.

    Algorithm:
        v_new = v_old + dt * a(x_old, v_old)
        x_new = x_old + dt * v_new  # Uses NEW velocity!

    Args:
        system: JaxDynamicSystem instance (must have position_indices and velocity_indices)
        state: Current state vector
        control: Control input (held constant during step)
        dt: Time step size
        t: Current simulation time (default 0.0)
        constraints: Optional ConstraintSet for constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        New state vector after integration step

    Note:
        This method requires the system to define position_indices and
        velocity_indices properties. It will raise an error if
        supports_symplectic is False.
    """
    # Get indices (these are static, known at trace time)
    pos_idx = jnp.array(system.position_indices)
    vel_idx = jnp.array(system.velocity_indices)

    # Compute derivatives with OLD state
    deriv = system.forward_dynamics(state, control, t, env=env)

    # Step 1: Update velocities using OLD state derivatives
    new_state = state.at[vel_idx].add(dt * deriv[vel_idx])

    # Step 2: Recompute derivatives with UPDATED velocities
    deriv_new = system.forward_dynamics(new_state, control, t, env=env)

    # Step 3: Update positions using NEW velocity derivatives
    new_state = new_state.at[pos_idx].add(dt * deriv_new[pos_idx])

    # Apply post-step normalization (e.g., quaternions)
    new_state = system.post_step(new_state)
    if constraints is not None:
        new_state, _, _ = _enforce_constraints_jit(constraints, enforcement, t + dt, new_state, control)
    return new_state


def simulate_symplectic(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate a JAX dynamic system using semi-implicit (symplectic) Euler.

    Uses jax.lax.scan for efficient JIT-compiled integration.
    This integrator preserves the symplectic structure of Hamiltonian
    systems, providing better energy conservation than explicit Euler.

    Args:
        system: JaxDynamicSystem instance (must support symplectic integration)
        initial_state: Initial state vector
        dt: Fixed time step (seconds) - must be a Python float, not JAX array
        duration: Total simulation duration (seconds) - must be a Python float
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_steps, num_controls) for pre-sampled
            - None for system.default_control()
        constraints: Optional ConstraintSet for post-step constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times, states, and controls

    Raises:
        ValueError: If system does not support symplectic integration

    Note:
        Semi-implicit Euler is first-order accurate but has much better
        energy behavior than explicit Euler for conservative systems.
        Energy oscillates around the true value rather than drifting.

    Example:
        from fmd.simulator import SimplePendulum, simulate_symplectic
        from fmd.simulator.params import PENDULUM_1M
        import jax.numpy as jnp

        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # 0.5 rad initial angle
        result = simulate_symplectic(pendulum, initial, dt=0.01, duration=10.0)
    """
    # Verify system supports symplectic integration
    if not system.supports_symplectic:
        raise ValueError(
            f"System {type(system).__name__} does not support symplectic integration. "
            "Override position_indices and velocity_indices properties."
        )

    # Build time grid using helper function
    times_list, num_steps = _build_time_grid(dt, duration)
    times = jnp.array(times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def step_fn(carry, idx):
        """Single semi-implicit Euler integration step for lax.scan."""
        state, t_prev, u_prev = carry
        t_curr = times[idx]
        dt_actual = t_curr - t_prev
        ctrl = schedule(t_prev, state)

        # First enforce rate limits on control before integration
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_actual
            )

        new_state = semi_implicit_euler_step(system, state, ctrl, dt_actual, t_prev, constraints, enforcement, env=env)
        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Run integration loop with lax.scan
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        step_fn,
        (initial_state, 0.0, init_ctrl),
        jnp.arange(1, num_steps),
    )

    return _collect_results(initial_state, init_ctrl, states_rest, controls_rest, times, num_steps)


def simulate_symplectic_substepped(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt_sim: float,
    dt_control: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    env=None,
) -> JaxSimulationResult:
    """Simulate with substepped symplectic Euler (split physics / control rates).

    Combines semi-implicit (symplectic) Euler integration with substepping:
    physics runs at the fast ``dt_sim`` rate while the controller updates at
    the slower ``dt_control``.

    The symplectic integrator preserves energy better than explicit Euler.
    Combined with substepping, this provides accurate physics simulation with
    realistic control timing.

    The control is held constant during substeps (zero-order hold), and
    results are sampled at the control rate.

    Args:
        system: JaxDynamicSystem instance (must support symplectic integration)
        initial_state: Initial state vector
        dt_sim: Simulation timestep (e.g., 0.001 = 1ms)
        dt_control: Control update rate (e.g., 0.02 = 20ms)
        duration: Total simulation duration (seconds)
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_control_steps, num_controls)
            - None for system.default_control()
        constraints: Optional ConstraintSet for constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)

    Returns:
        JaxSimulationResult with times at control rate (not simulation rate)

    Raises:
        ValueError: If system does not support symplectic integration
        ValueError: If dt_control is not an integer multiple of dt_sim

    Note:
        This is the standard approach for embedded systems where:
        1. Control computation is expensive (LQR, MPC, neural nets)
        2. Real hardware can't run controllers at physics rates
        3. Sensors don't update at physics rates
        4. Controller gains are tuned for specific control rates

        Rate limit constraints are applied at the control rate (dt_control),
        NOT the physics rate (dt_sim).

    Example:
        # 1000 Hz physics, 50 Hz control with symplectic integration
        result = simulate_symplectic_substepped(
            system, x0,
            dt_sim=0.001,     # 1ms physics
            dt_control=0.02,  # 20ms control
            duration=5.0
        )
    """
    # Verify system supports symplectic integration
    if not system.supports_symplectic:
        raise ValueError(
            f"System {type(system).__name__} does not support symplectic integration. "
            "Override position_indices and velocity_indices properties."
        )

    # Validate timestep relationship
    substeps_per_control = int(round(dt_control / dt_sim))
    if abs(substeps_per_control * dt_sim - dt_control) > 1e-10:
        raise ValueError(
            f"dt_control ({dt_control}) must be an integer multiple of "
            f"dt_sim ({dt_sim}). Got ratio: {dt_control / dt_sim}"
        )

    # Build control time grid
    t = 0.0
    control_times_list = [t]
    while t < duration:
        step = min(dt_control, duration - t)
        t = t + step
        control_times_list.append(t)
    num_control_steps = len(control_times_list)
    control_times = jnp.array(control_times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, control_times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    def run_substeps(state: Array, t_start: float, ctrl: Array) -> Array:
        """Run multiple semi-implicit Euler substeps with fixed control."""

        def substep_fn(state_and_t, _):
            """Single symplectic substep within a control period."""
            state, t = state_and_t
            new_state = semi_implicit_euler_step(system, state, ctrl, dt_sim, t, constraints, enforcement, env=env)
            return (new_state, t + dt_sim), None

        (final_state, _), _ = jax.lax.scan(
            substep_fn,
            (state, t_start),
            None,
            length=substeps_per_control,
        )
        return final_state

    def control_step_fn(carry, idx):
        """One control period with multiple physics substeps."""
        state, t_prev, u_prev = carry
        t_curr = control_times[idx]

        # Get control at start of this control period
        ctrl = schedule(t_prev, state)

        # Enforce rate limits on control before integration (at control rate)
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_control
            )

        # Run substeps within this control period
        new_state = run_substeps(state, t_prev, ctrl)

        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Run integration loop with lax.scan over control steps
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        control_step_fn,
        (initial_state, 0.0, init_ctrl),
        jnp.arange(1, num_control_steps),
    )

    return _collect_results(
        initial_state, init_ctrl, states_rest, controls_rest,
        control_times, num_control_steps
    )


def simulate_noisy(
    system: JaxDynamicSystem,
    initial_state: Array,
    dt: float,
    duration: float,
    control: Optional[Union[ControlSchedule, Array]] = None,
    constraints: Optional[ConstraintSet] = None,
    enforcement: Capability = Capability.HARD_CLIP,
    process_noise: Optional[NoiseModel] = None,
    prng_key: Optional[jax.random.PRNGKey] = None,
    env=None,
) -> JaxSimulationResult:
    """Simulate a JAX dynamic system with process noise using fixed-step RK4.

    This is the stochastic version of simulate(), adding process noise after
    each integration step. The noise is applied AFTER the RK4 step but BEFORE
    post_step processing (e.g., quaternion normalization).

    The discrete dynamics with noise are:
        x[k+1] = f_discrete(x[k], u[k]) + w[k]

    where w[k] ~ NoiseModel is independent process noise at each step.

    Args:
        system: JaxDynamicSystem instance to simulate
        initial_state: Initial state vector
        dt: Fixed time step (seconds) - must be a Python float, not JAX array
        duration: Total simulation duration (seconds) - must be a Python float
        control: One of:
            - ControlSchedule (eqx.Module with __call__(t, state))
            - Array of shape (num_steps, num_controls) for pre-sampled
            - None for system.default_control()
        constraints: Optional ConstraintSet for post-step constraint enforcement
        enforcement: Capability specifying enforcement method (default HARD_CLIP)
        process_noise: Optional NoiseModel for process noise. If None,
            simulation is deterministic (equivalent to simulate()).
        prng_key: JAX PRNG key for noise generation. Required if process_noise
            is not None.

    Returns:
        JaxSimulationResult with times, states, and controls

    Raises:
        ValueError: If process_noise is provided but prng_key is None

    Note:
        - For reproducible simulations, use the same prng_key
        - Different keys will produce different trajectories
        - ZeroNoise produces deterministic results (same as simulate())
        - Uses fixed-step RK4 to match numpy implementation exactly

    Example:
        from fmd.simulator import SimplePendulum, simulate_noisy
        from fmd.simulator.noise import GaussianNoise
        from fmd.simulator.params import PENDULUM_1M
        import jax

        pendulum = SimplePendulum(PENDULUM_1M)
        initial = jnp.array([0.5, 0.0])  # 0.5 rad initial angle

        # Create process noise
        noise = GaussianNoise.isotropic(dim=2, variance=0.001)
        key = jax.random.key(42)

        result = simulate_noisy(
            pendulum, initial, dt=0.01, duration=10.0,
            process_noise=noise, prng_key=key
        )
    """
    # Validate PRNG key requirement
    if process_noise is not None and prng_key is None:
        raise ValueError(
            "prng_key is required when process_noise is provided. "
            "Use jax.random.key(seed) to create one."
        )

    # Validate noise dimension matches system state dimension (EDGE-5)
    if process_noise is not None:
        n_states = initial_state.shape[0]
        if process_noise.dim != n_states:
            raise ValueError(
                f"process_noise.dim ({process_noise.dim}) must match "
                f"initial_state dimension ({n_states})"
            )

    # Build time grid using helper function
    times_list, num_steps = _build_time_grid(dt, duration)
    times = jnp.array(times_list)

    # Resolve control to schedule
    schedule = _resolve_control_schedule(system, control, times)

    # Get initial control
    init_ctrl = schedule(0.0, initial_state)

    # Pre-split keys for all steps (num_steps - 1 integration steps)
    # We need keys for steps 1 to num_steps-1 (indices 1, 2, ..., num_steps-1)
    if process_noise is not None:
        # Split into (num_steps - 1) subkeys for the integration steps
        noise_keys = jax.random.split(prng_key, num_steps - 1)
    else:
        # Placeholder - won't be used but needed for scan type consistency
        noise_keys = jnp.zeros((num_steps - 1, 2), dtype=jnp.uint32)

    def step_fn(carry, scan_input):
        """Single integration step with optional noise for lax.scan.

        Noise is added AFTER RK4 integration but BEFORE post_step.
        This allows post_step to re-normalize (e.g., quaternions).

        Constraint Enforcement Order:
            1. Rate limits (HAS_RATE_LIMIT) are enforced BEFORE integration.
               This clips the control input to respect slew rate limits.
            2. State constraints (HARD_CLIP, PROJECTION) are enforced AFTER
               integration inside rk4_step. This ensures the resulting state
               satisfies bounds like position limits or ground planes.
        """
        state, t_prev, u_prev = carry
        idx, noise_key = scan_input
        t_curr = times[idx]
        dt_actual = t_curr - t_prev
        ctrl = schedule(t_prev, state)

        # Enforce rate limits on control BEFORE integration (see docstring)
        if constraints is not None:
            _, ctrl, _ = _enforce_constraints_jit(
                constraints, Capability.HAS_RATE_LIMIT, t_prev, state, ctrl, u_prev, dt_actual
            )

        # RK4 step WITHOUT post_step (we'll apply it after noise)
        k1 = system.forward_dynamics(state, ctrl, t_prev, env=env)
        k2 = system.forward_dynamics(state + 0.5 * dt_actual * k1, ctrl, t_prev + 0.5 * dt_actual, env=env)
        k3 = system.forward_dynamics(state + 0.5 * dt_actual * k2, ctrl, t_prev + 0.5 * dt_actual, env=env)
        k4 = system.forward_dynamics(state + dt_actual * k3, ctrl, t_prev + dt_actual, env=env)
        new_state = state + (dt_actual / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Add process noise BEFORE post_step
        if process_noise is not None:
            noise_sample = process_noise.sample(noise_key, (process_noise.dim,))
            new_state = new_state + noise_sample

        # Apply post-step (e.g., quaternion normalization) AFTER noise
        new_state = system.post_step(new_state)

        # Apply state constraints AFTER post_step
        if constraints is not None:
            new_state, _, _ = _enforce_constraints_jit(
                constraints, enforcement, t_curr, new_state, ctrl
            )

        return (new_state, t_curr, ctrl), (new_state, ctrl)

    # Create scan input: (indices, noise_keys)
    indices = jnp.arange(1, num_steps)

    # Run integration loop with lax.scan
    # Initial u_prev = init_ctrl (first step always feasible)
    _, (states_rest, controls_rest) = jax.lax.scan(
        step_fn,
        (initial_state, 0.0, init_ctrl),
        (indices, noise_keys),
    )

    return _collect_results(
        initial_state, init_ctrl, states_rest, controls_rest, times, num_steps
    )
