"""OCP result container.

This module defines the result type returned by OCP solvers.
"""

from typing import NamedTuple

import numpy as np


class OCPResult(NamedTuple):
    """Result of an optimal control problem solve.

    Attributes:
        states: State trajectory of shape (N+1, num_states).
            states[0] is the initial state, states[-1] is the terminal state.
        controls: Control trajectory of shape (N, num_controls).
            controls[k] is the control applied at time step k.
        times: Time points of shape (N+1,).
            times[0] = 0, times[-1] = T.
        T: Total time horizon in seconds.
        cost: Optimal objective value.
        converged: Whether the solver converged successfully.
        solver_stats: Dictionary of solver statistics (iterations, solve time, etc.).
    """

    states: np.ndarray
    controls: np.ndarray
    times: np.ndarray
    T: float
    cost: float
    converged: bool
    solver_stats: dict
