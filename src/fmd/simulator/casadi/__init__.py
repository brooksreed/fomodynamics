"""fmd.simulator.casadi - CasADi-based dynamics models for MPC applications.

This package provides CasADi-compatible implementations of fomodynamics
dynamics models. Each model has a JAX counterpart in `fmd.simulator` with
identical physics, verified through rigorous equivalence testing.

Includes: generic + simple/teaching models (box_1d, cartpole,
planar_quadrotor, boat_2d) + the moth (moth_3d).

Key design principles:
- Models follow the DynamicsProtocol interface (shared with JAX)
- *Exact variants match JAX implementations exactly (for equivalence tests)
- *Smooth variants use solver-friendly approximations (for MPC)
- Shared parameter classes with JAX models (backend-agnostic)
- Cached compiled functions for efficient MPC re-solves

Usage:
    >>> from fmd.simulator.casadi import Box1DCasadiExact, rk4_step_function
    >>> from fmd.simulator.params import BOX1D_DEFAULT
    >>> import casadi as cs
    >>>
    >>> model = Box1DCasadiExact(BOX1D_DEFAULT)
    >>> f = model.dynamics_function()
    >>> x = cs.SX.sym("x", model.num_states)
    >>> u = cs.SX.sym("u", model.num_controls)
    >>> xdot = f(x, u)

Note:
    CasADi is a required core dependency.
"""

from fmd.simulator.casadi.base import CasadiDynamicSystem
from fmd.simulator.casadi.integrator import (
    rk4_step_casadi,
    rk4_step_with_post_step_casadi,
    rk4_step_function,
)
from fmd.simulator.casadi.box_1d import (
    Box1DCasadiExact,
    Box1DFrictionCasadiExact,
    Box1DFrictionCasadiSmooth,
)
from fmd.simulator.casadi.cartpole import CartpoleCasadiExact
from fmd.simulator.casadi.planar_quadrotor import PlanarQuadrotorCasadiExact
from fmd.simulator.casadi.boat_2d import Boat2DCasadiExact
from fmd.simulator.casadi.moth_3d import Moth3DCasadiExact

__all__ = [
    "CasadiDynamicSystem",
    "rk4_step_casadi",
    "rk4_step_with_post_step_casadi",
    "rk4_step_function",
    "Box1DCasadiExact",
    "Box1DFrictionCasadiExact",
    "Box1DFrictionCasadiSmooth",
    "CartpoleCasadiExact",
    "PlanarQuadrotorCasadiExact",
    "Boat2DCasadiExact",
    "Moth3DCasadiExact",
]
