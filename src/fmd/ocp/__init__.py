"""fmd.ocp - Optimal Control Problem solvers (generic infrastructure).

This package provides generic OCP infrastructure for trajectory
optimization using CasADi and IPOPT.

Classes:
    MultipleShootingOCP: Direct multiple shooting with RK4 integration.
    OCPResult: Result container for OCP solutions.

Functions:
    compute_control_smoothness: Analyze control smoothness and detect chattering.

Example:
    >>> from fmd.ocp import MultipleShootingOCP, OCPResult
    >>> from fmd.simulator.casadi import CartpoleCasadiExact
    >>> from fmd.simulator.params import CARTPOLE_CLASSIC
    >>>
    >>> model = CartpoleCasadiExact(CARTPOLE_CLASSIC)
    >>> ocp = MultipleShootingOCP(model, N=100, T_fixed=2.0)
    >>> ocp.set_initial_state([0, 0, 0, 0])
    >>> ocp.set_terminal_state([1, 0, 0, 0])
    >>> ocp.set_control_bounds(-10.0, 10.0)
    >>> result = ocp.solve()
    >>> print(f"Converged: {result.converged}, Cost: {result.cost:.3f}")

Note:
    CasADi is a required core dependency.
"""

from fmd.ocp.analysis import compute_control_smoothness
from fmd.ocp.multiple_shooting import MultipleShootingOCP
from fmd.ocp.result import OCPResult

__all__ = [
    "MultipleShootingOCP",
    "OCPResult",
    "compute_control_smoothness",
]
