"""CasADi RK4 integrator for MPC applications.

Provides RK4 (4th-order Runge-Kutta) integration using CasADi symbolic math.
These integrators create symbolic expressions that can be embedded in
optimization problems (MPC, trajectory optimization) or compiled to
efficient numerical functions.

The RK4 formula matches the JAX implementation exactly, enabling rigorous
equivalence testing between backends. The key distinction is:

- `rk4_step_casadi`: Pure RK4, no post_step - use for MPC transcription
  where constraints handle normalization explicitly
- `rk4_step_with_post_step_casadi`: RK4 + post_step - use for equivalence
  testing against JAX simulation trajectories
- `rk4_step_function`: Compiled CasADi Function for efficient evaluation

Example:
    >>> from fmd.simulator.casadi import Box1DCasadiExact, rk4_step_function
    >>> from fmd.simulator.params import BOX1D_DEFAULT
    >>> import casadi as cs
    >>> import numpy as np
    >>>
    >>> model = Box1DCasadiExact(BOX1D_DEFAULT)
    >>> f = rk4_step_function(model, dt=0.01)
    >>> x0 = np.array([0.0, 0.0])
    >>> u0 = np.array([1.0])
    >>> x1 = f(x0, u0)  # Numerical evaluation
"""

from __future__ import annotations

import casadi as cs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmd.simulator.casadi.base import CasadiDynamicSystem


def rk4_step_casadi(
    model: "CasadiDynamicSystem",
    x: cs.SX,
    u: cs.SX,
    dt: float,
    t: float = 0.0,
) -> cs.SX:
    """Single RK4 step - matches JAX rk4_step formula exactly.

    Computes one step of 4th-order Runge-Kutta integration with zero-order
    hold on the control input. The formula is:

        k1 = f(x, u, t)
        k2 = f(x + 0.5*dt*k1, u, t + 0.5*dt)
        k3 = f(x + 0.5*dt*k2, u, t + 0.5*dt)
        k4 = f(x + dt*k3, u, t + dt)
        x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    NOTE: This function does NOT apply post_step (e.g., quaternion
    normalization). Use rk4_step_with_post_step_casadi for simulation-
    equivalent behavior that includes post-processing.

    For MPC transcription, omitting post_step is typically correct because:
    - Quaternion normalization is handled via explicit constraints
    - Smooth optimization requires differentiable dynamics without discontinuities

    Args:
        model: CasADi dynamics model implementing forward_dynamics
        x: State vector (symbolic SX expression)
        u: Control vector (symbolic SX expression)
        dt: Time step (Python float)
        t: Current time (Python float, default 0.0)

    Returns:
        x_next: Next state (symbolic SX expression), without post-processing
    """
    k1 = model.forward_dynamics(x, u, t)
    k2 = model.forward_dynamics(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = model.forward_dynamics(x + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = model.forward_dynamics(x + dt * k3, u, t + dt)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def rk4_step_with_post_step_casadi(
    model: "CasadiDynamicSystem",
    x: cs.SX,
    u: cs.SX,
    dt: float,
    t: float = 0.0,
) -> cs.SX:
    """RK4 step with post_step - matches JAX simulation semantics.

    This function applies model.post_step after the RK4 integration,
    matching the JAX simulator behavior exactly. Use this for equivalence
    testing against JAX trajectories.

    The post_step typically handles:
    - Quaternion normalization (for 3D rigid body systems)
    - Angle wrapping (for systems with angular states)
    - Other state constraint enforcement

    Args:
        model: CasADi dynamics model implementing forward_dynamics and post_step
        x: State vector (symbolic SX expression)
        u: Control vector (symbolic SX expression)
        dt: Time step (Python float)
        t: Current time (Python float, default 0.0)

    Returns:
        x_next: Next state (symbolic SX expression), with post-processing applied
    """
    x_next = rk4_step_casadi(model, x, u, dt, t)
    return model.post_step(x_next)


def rk4_step_function(
    model: "CasadiDynamicSystem",
    dt: float,
    name: str = "rk4_step",
    include_post_step: bool = False,
) -> cs.Function:
    """Return compiled CasADi Function for RK4 step.

    Creates a compiled CasADi Function that can be called efficiently
    with numerical inputs. The function signature is (x, u) -> x_next.

    This is useful for:
    - Building MPC problems with multiple shooting
    - Numerical simulation outside of optimization
    - Testing dynamics with concrete values

    Args:
        model: CasADi dynamics model implementing forward_dynamics
        dt: Time step (Python float)
        name: Function name for debugging/code generation (default "rk4_step")
        include_post_step: If True, include model.post_step after RK4.
                          Use True for simulation equivalence tests,
                          False for MPC transcription (default False).

    Returns:
        CasADi Function with signature: (x, u) -> x_next
            - Inputs named "x" and "u"
            - Output named "x_next"

    Example:
        >>> model = Box1DCasadiExact(BOX1D_DEFAULT)
        >>> f = rk4_step_function(model, dt=0.01)
        >>> x_next = f([0.0, 0.0], [1.0])  # Returns DM (dense matrix)
        >>> x_next_np = np.array(x_next).flatten()  # Convert to numpy
    """
    # Create symbolic inputs
    x = cs.SX.sym("x", model.num_states)
    u = cs.SX.sym("u", model.num_controls)

    # Compute symbolic RK4 step
    if include_post_step:
        x_next = rk4_step_with_post_step_casadi(model, x, u, dt)
    else:
        x_next = rk4_step_casadi(model, x, u, dt)

    # Compile to CasADi Function
    return cs.Function(name, [x, u], [x_next], ["x", "u"], ["x_next"])
