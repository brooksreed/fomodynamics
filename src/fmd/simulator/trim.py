"""Generic trim finder for equilibrium analysis of dynamic systems.

Finds steady-state operating points where all state derivatives are zero.
Uses SciPy optimization (L-BFGS-B + polish) to minimize residual.

For Moth-specific trim solving, use the CasADi solver in trim_casadi.py.
The Moth-specific code that was previously in this file has been archived
to src/blur-archive/scipy-trim-solver/.

Example:
    from fmd.simulator import Moth3D
    from fmd.simulator.trim import find_trim

    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(10.0))
    state_guess = moth.default_state()
    control_guess = moth.default_control()
    result = find_trim(moth, state_guess, control_guess)
"""

from __future__ import annotations

from fmd.simulator import _config  # noqa: F401

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize, OptimizeResult

from fmd.simulator.base import JaxDynamicSystem


@dataclass
class TrimResult:
    """Result from trim finder optimization.

    Attributes:
        state: Trim state vector (numpy array)
        control: Trim control vector (numpy array)
        residual: L2 norm of state derivative at trim point
        success: Whether optimizer converged
        optimize_result: Full scipy OptimizeResult for diagnostics
        warnings: List of plausibility warning strings
        calibrated_thrust: Calibrated thrust (N) when applicable, None otherwise.
    """
    state: np.ndarray
    control: np.ndarray
    residual: float
    success: bool
    optimize_result: OptimizeResult
    warnings: list[str] = field(default_factory=list)
    calibrated_thrust: Optional[float] = None


def trim_residual(
    system: JaxDynamicSystem,
    state: Array,
    control: Array,
    t: float = 0.0,
) -> Array:
    """Compute state derivative residual at given state/control.

    At trim, all elements should be zero (or near-zero).

    Args:
        system: Dynamic system to evaluate.
        state: State vector.
        control: Control vector.
        t: Simulation time.

    Returns:
        State derivative vector (residual). Zero at perfect trim.
    """
    return system.forward_dynamics(state, control, t)


def find_trim(
    system: JaxDynamicSystem,
    x_guess: np.ndarray,
    u_guess: np.ndarray,
    free_state_indices: list[int] | None = None,
    free_control_indices: list[int] | None = None,
    state_bounds: list[tuple[float, float]] | None = None,
    control_bounds: list[tuple[float, float]] | None = None,
    state_constraints: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    t: float = 0.0,
    method: str = "SLSQP",
    tol: float = 1e-10,
    regularization: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> TrimResult:
    """Find trim (equilibrium) point for a generic dynamic system.

    Minimizes ||x_dot||^2 by varying the free state and control variables.
    Uses L-BFGS-B for the initial search (robust to ill-conditioned gradients),
    then polishes with ``method`` (default SLSQP) for tight convergence.

    Args:
        system: Dynamic system to trim.
        x_guess: Initial state guess.
        u_guess: Initial control guess.
        free_state_indices: Which state indices are free to vary (None = all).
        free_control_indices: Which control indices are free to vary (None = all).
        state_bounds: Bounds for free states [(lo, hi), ...].
        control_bounds: Bounds for free controls [(lo, hi), ...].
        state_constraints: Optional function(state, control) -> constrained_state.
        t: Simulation time for evaluation.
        method: SciPy optimization method for the polish phase.
        tol: Convergence tolerance.
        regularization: Optional function(state, control) -> float penalty.

    Returns:
        TrimResult with optimized state, control, residual norm, and success flag.
    """
    x = np.array(x_guess, dtype=np.float64)
    u = np.array(u_guess, dtype=np.float64)

    if free_state_indices is None:
        free_state_indices = list(range(len(x)))
    if free_control_indices is None:
        free_control_indices = list(range(len(u)))

    n_free_state = len(free_state_indices)

    def pack(state, control):
        parts = []
        for i in free_state_indices:
            parts.append(state[i])
        for i in free_control_indices:
            parts.append(control[i])
        return np.array(parts)

    def unpack(z):
        state = x.copy()
        control = u.copy()
        for k, i in enumerate(free_state_indices):
            state[i] = z[k]
        for k, i in enumerate(free_control_indices):
            control[i] = z[n_free_state + k]
        return state, control

    def objective(z):
        state, control = unpack(z)
        if state_constraints is not None:
            state = state_constraints(state, control)
        jax_state = jnp.array(state)
        jax_control = jnp.array(control)
        residual = trim_residual(system, jax_state, jax_control, t)
        obj = float(jnp.sum(residual ** 2))
        if regularization is not None:
            obj += regularization(state, control)
        return obj

    z0 = pack(x, u)

    # Build bounds
    bounds_list = []
    if state_bounds is not None:
        bounds_list.extend(state_bounds)
    else:
        bounds_list.extend([(None, None)] * n_free_state)
    if control_bounds is not None:
        bounds_list.extend(control_bounds)
    else:
        n_free_control = len(free_control_indices)
        bounds_list.extend([(None, None)] * n_free_control)

    # Two-phase optimization: L-BFGS-B + polish
    result = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        bounds=bounds_list,
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": tol},
    )

    for z_start in [result.x, z0]:
        r2 = minimize(
            objective,
            z_start,
            method=method,
            bounds=bounds_list,
            tol=tol,
            options={"maxiter": 2000, "ftol": tol},
        )
        if r2.fun <= result.fun * (1 + 1e-8) + 1e-12:
            result = r2

    opt_state, opt_control = unpack(result.x)
    if state_constraints is not None:
        opt_state = state_constraints(opt_state, opt_control)

    # Compute final residual norm
    jax_state = jnp.array(opt_state)
    jax_control = jnp.array(opt_control)
    final_residual = trim_residual(system, jax_state, jax_control, t)
    residual_norm = float(jnp.sqrt(jnp.sum(final_residual ** 2)))

    return TrimResult(
        state=opt_state,
        control=opt_control,
        residual=residual_norm,
        success=result.success,
        optimize_result=result,
    )
