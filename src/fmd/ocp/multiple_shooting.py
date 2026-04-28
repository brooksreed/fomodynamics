"""Direct multiple shooting optimal control problem solver.

This module provides a generic OCP solver using direct multiple shooting
with RK4 integration and IPOPT as the NLP solver.

Multiple shooting discretizes the trajectory into N intervals with N+1
state variables (shooting nodes). RK4 integration is used to propagate
dynamics within each interval. Continuity constraints ensure the
integrated state matches the next shooting node.

References:
    - Bock, H. G., & Plitt, K. J. (1984). A multiple shooting algorithm for
      direct solution of optimal control problems.
    - CasADi documentation: https://web.casadi.org/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np

from fmd.ocp.result import OCPResult

if TYPE_CHECKING:
    from fmd.simulator.casadi.base import CasadiDynamicSystem


class MultipleShootingOCP:
    """Direct multiple shooting OCP solver with RK4 integration.

    This class formulates and solves optimal control problems using:
    - Direct multiple shooting transcription
    - RK4 integration for dynamics
    - IPOPT for the resulting NLP

    The OCP formulation is:
        min   integral(L(x,u)) + Phi(x_N) + time_weight * T
        s.t.  x_{k+1} = RK4(x_k, u_k, dt)     (dynamics)
              x_0 = x_initial                   (initial condition)
              x_N = x_target (optionally masked) (terminal condition)
              x_lb <= x <= x_ub                 (state bounds)
              u_lb <= u <= u_ub                 (control bounds)
              T_min <= T <= T_max               (time bounds, if free-time)

    Usage:
        >>> model = CartpoleCasadiExact(params)
        >>> ocp = MultipleShootingOCP(model, N=100, T_fixed=2.0)
        >>> ocp.set_initial_state([0, 0, 0, 0])
        >>> ocp.set_terminal_state([1, 0, 0, 0])
        >>> ocp.set_control_bounds(-10.0, 10.0)
        >>> result = ocp.solve()

    Attributes:
        model: CasADi dynamics model.
        N: Number of control intervals (N+1 state nodes).
        T_fixed: Fixed time horizon (None if free-time problem).
        T_bounds: (T_min, T_max) for free-time problems.
        T_init: Initial guess for T in free-time problems.
    """

    def __init__(
        self,
        model: CasadiDynamicSystem,
        N: int,
        T_fixed: float | None = None,
        T_bounds: tuple[float, float] | None = None,
        T_init: float | None = None,
        M: int = 1,
    ):
        """Initialize the OCP.

        Args:
            model: CasADi-compatible dynamics model.
            N: Number of control intervals (shooting nodes).
            T_fixed: Fixed time horizon in seconds. Mutually exclusive with T_bounds.
            T_bounds: (T_min, T_max) bounds for free-time problems.
                      Mutually exclusive with T_fixed.
            T_init: Initial guess for T in free-time problems. Defaults to midpoint
                    of T_bounds.
            M: Number of RK4 integration sub-steps per control interval.
                Default 1 (single RK4 step). Use M>1 when the interval dt
                exceeds the RK4 stability limit for stiff dynamics.

        Raises:
            ValueError: If neither or both of T_fixed and T_bounds are specified,
                or if M is not a positive integer.
        """
        if T_fixed is None and T_bounds is None:
            raise ValueError("Must specify either T_fixed or T_bounds")
        if T_fixed is not None and T_bounds is not None:
            raise ValueError("Cannot specify both T_fixed and T_bounds")
        if not isinstance(M, int) or M < 1:
            raise ValueError(f"M must be a positive integer, got {M!r}")

        self.model = model
        self.N = N
        self.T_fixed = T_fixed
        self.T_bounds = T_bounds
        self.T_init = T_init
        self.M = M

        # State/control dimensions
        self.nx = model.num_states
        self.nu = model.num_controls

        # Configuration (set by user)
        self._x0: np.ndarray | None = None
        self._x_target: np.ndarray | None = None
        self._x_target_mask: np.ndarray | None = None
        self._u_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._state_bounds: dict[int, tuple[float, float]] = {}
        self._Q: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._x_ref: np.ndarray | None = None
        self._x_refs: np.ndarray | None = None  # Time-varying reference (N+1, nx)
        self._Qf: np.ndarray | None = None
        self._xf_ref: np.ndarray | None = None
        self._time_weight: float = 0.0
        self._initial_guess_states: np.ndarray | None = None
        self._initial_guess_controls: np.ndarray | None = None
        self._u_ref: np.ndarray | None = None
        self._Rd: np.ndarray | None = None
        self._du_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def set_initial_state(self, x0: np.ndarray | list) -> MultipleShootingOCP:
        """Set the initial state constraint.

        Args:
            x0: Initial state vector of length num_states.

        Returns:
            self for method chaining.
        """
        self._x0 = np.asarray(x0, dtype=np.float64)
        if self._x0.shape != (self.nx,):
            raise ValueError(f"x0 must have shape ({self.nx},), got {self._x0.shape}")
        return self

    def set_terminal_state(
        self,
        x_target: np.ndarray | list,
        mask: np.ndarray | list | None = None,
    ) -> MultipleShootingOCP:
        """Set the terminal state constraint.

        Args:
            x_target: Target terminal state of length num_states.
            mask: Boolean array of length num_states. If provided, only
                  states where mask[i] is True are constrained.
                  If None, all states are constrained.

        Returns:
            self for method chaining.
        """
        self._x_target = np.asarray(x_target, dtype=np.float64)
        if self._x_target.shape != (self.nx,):
            raise ValueError(f"x_target must have shape ({self.nx},)")

        if mask is not None:
            self._x_target_mask = np.asarray(mask, dtype=bool)
            if self._x_target_mask.shape != (self.nx,):
                raise ValueError(f"mask must have shape ({self.nx},)")
        else:
            self._x_target_mask = None

        return self

    def set_control_bounds(
        self,
        lower: float | np.ndarray | list,
        upper: float | np.ndarray | list,
    ) -> MultipleShootingOCP:
        """Set control input bounds.

        Args:
            lower: Lower bound (scalar or array of length num_controls).
            upper: Upper bound (scalar or array of length num_controls).

        Returns:
            self for method chaining.
        """
        lower = np.atleast_1d(np.asarray(lower, dtype=np.float64))
        upper = np.atleast_1d(np.asarray(upper, dtype=np.float64))

        # Broadcast scalars to full dimension
        if lower.shape == (1,) and self.nu > 1:
            lower = np.full(self.nu, lower[0])
        if upper.shape == (1,) and self.nu > 1:
            upper = np.full(self.nu, upper[0])

        if lower.shape != (self.nu,) or upper.shape != (self.nu,):
            raise ValueError(f"Control bounds must have shape ({self.nu},)")

        self._u_bounds = (lower, upper)
        return self

    def set_state_bounds(
        self,
        index: int,
        lower: float,
        upper: float,
    ) -> MultipleShootingOCP:
        """Set bounds on a specific state variable.

        Args:
            index: State index (0-based).
            lower: Lower bound for this state.
            upper: Upper bound for this state.

        Returns:
            self for method chaining.
        """
        if not 0 <= index < self.nx:
            raise ValueError(f"State index {index} out of range [0, {self.nx})")
        self._state_bounds[index] = (lower, upper)
        return self

    def set_running_cost(
        self,
        Q: np.ndarray | list,
        R: np.ndarray | list,
        x_ref: np.ndarray | list | None = None,
        u_ref: np.ndarray | list | None = None,
    ) -> MultipleShootingOCP:
        """Set running (stage) cost.

        Cost: L(x,u) = (x-x_ref)'Q(x-x_ref) + (u-u_ref)'R(u-u_ref)

        Args:
            Q: State cost matrix (num_states x num_states).
            R: Control cost matrix (num_controls x num_controls).
            x_ref: Reference state. Can be:
                - None: Uses zeros
                - 1D array (num_states,): Fixed reference for all timesteps
                - 2D array (N+1, num_states): Time-varying reference trajectory
            u_ref: Reference control. Can be:
                - None: Uses zeros (penalises absolute control)
                - 1D array (num_controls,): Fixed reference for all timesteps

        Returns:
            self for method chaining.
        """
        self._Q = np.asarray(Q, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)

        if x_ref is not None:
            x_ref = np.asarray(x_ref, dtype=np.float64)
            if x_ref.ndim == 1:
                # Fixed reference - replicate for all timesteps
                self._x_ref = x_ref
                self._x_refs = None
            elif x_ref.ndim == 2:
                # Time-varying reference trajectory (N+1, nx)
                if x_ref.shape[0] != self.N + 1:
                    raise ValueError(
                        f"Time-varying reference must have shape (N+1, nx)=({self.N + 1}, {self.nx}), "
                        f"got {x_ref.shape}"
                    )
                self._x_refs = x_ref  # Store trajectory
                self._x_ref = None  # Clear fixed reference
            else:
                raise ValueError(f"x_ref must be 1D or 2D, got {x_ref.ndim}D")
        else:
            self._x_ref = np.zeros(self.nx)
            self._x_refs = None

        if u_ref is not None:
            self._u_ref = np.asarray(u_ref, dtype=np.float64)
            if self._u_ref.shape != (self.nu,):
                raise ValueError(
                    f"u_ref must have shape ({self.nu},), got {self._u_ref.shape}"
                )
        else:
            self._u_ref = None

        return self

    def set_terminal_cost(
        self,
        Qf: np.ndarray | list,
        x_ref: np.ndarray | list | None = None,
    ) -> MultipleShootingOCP:
        """Set terminal cost: Phi(x_N) = (x_N - x_ref)'Qf(x_N - x_ref).

        Args:
            Qf: Terminal cost matrix (num_states x num_states).
            x_ref: Reference terminal state. Default is zeros or x_target if set.

        Returns:
            self for method chaining.
        """
        self._Qf = np.asarray(Qf, dtype=np.float64)

        if x_ref is not None:
            self._xf_ref = np.asarray(x_ref, dtype=np.float64)
        else:
            self._xf_ref = None  # Will use x_target if available

        return self

    def set_time_cost(self, weight: float = 1.0) -> MultipleShootingOCP:
        """Set cost on total time (for minimum-time problems).

        The objective includes: time_weight * T

        Args:
            weight: Weight on total time. Default 1.0.

        Returns:
            self for method chaining.
        """
        self._time_weight = weight
        return self

    def set_control_rate_cost(
        self,
        Rd: float | np.ndarray | list,
    ) -> MultipleShootingOCP:
        """Set cost on control rate (du/dt) to reduce chattering.

        Penalizes control changes between timesteps. The cost term is:
        sum_k (u_{k+1} - u_k)' Rd (u_{k+1} - u_k) / dt^2

        Args:
            Rd: Control rate cost weight. Can be:
                - Scalar: Applied as Rd * I (identity scaled)
                - 1D array of length num_controls: Diagonal matrix
                - 2D array (num_controls x num_controls): Full matrix

        Returns:
            self for method chaining.
        """
        Rd_arr = np.atleast_1d(np.asarray(Rd, dtype=np.float64))

        # Handle scalar -> diagonal matrix
        if Rd_arr.shape == (1,):
            self._Rd = Rd_arr[0] * np.eye(self.nu)
        elif Rd_arr.ndim == 1:
            if Rd_arr.shape != (self.nu,):
                raise ValueError(f"Rd array must have length {self.nu}, got {len(Rd_arr)}")
            self._Rd = np.diag(Rd_arr)
        elif Rd_arr.ndim == 2:
            if Rd_arr.shape != (self.nu, self.nu):
                raise ValueError(f"Rd matrix must have shape ({self.nu}, {self.nu})")
            self._Rd = Rd_arr
        else:
            raise ValueError("Rd must be scalar, 1D array, or 2D matrix")

        return self

    def set_control_rate_bounds(
        self,
        lower: float | np.ndarray | list,
        upper: float | np.ndarray | list,
    ) -> MultipleShootingOCP:
        """Set bounds on control rate (du/dt).

        Hard constraints on how fast controls can change between timesteps.
        Constraints: lower <= (u_{k+1} - u_k) / dt <= upper

        Args:
            lower: Lower bound on du/dt (typically negative for deceleration).
                Scalar or array of length num_controls.
            upper: Upper bound on du/dt (typically positive for acceleration).
                Scalar or array of length num_controls.

        Returns:
            self for method chaining.
        """
        lower = np.atleast_1d(np.asarray(lower, dtype=np.float64))
        upper = np.atleast_1d(np.asarray(upper, dtype=np.float64))

        # Broadcast scalars to full dimension
        if lower.shape == (1,) and self.nu > 1:
            lower = np.full(self.nu, lower[0])
        if upper.shape == (1,) and self.nu > 1:
            upper = np.full(self.nu, upper[0])

        if lower.shape != (self.nu,) or upper.shape != (self.nu,):
            raise ValueError(f"Control rate bounds must have shape ({self.nu},)")

        self._du_bounds = (lower, upper)
        return self

    def set_initial_guess(
        self,
        states: np.ndarray | list | None = None,
        controls: np.ndarray | list | None = None,
    ) -> MultipleShootingOCP:
        """Set initial guess for the solver.

        Args:
            states: State trajectory guess of shape (N+1, num_states).
            controls: Control trajectory guess of shape (N, num_controls).

        Returns:
            self for method chaining.
        """
        if states is not None:
            self._initial_guess_states = np.asarray(states, dtype=np.float64)
        if controls is not None:
            self._initial_guess_controls = np.asarray(controls, dtype=np.float64)
        return self

    def solve(self, solver_options: dict | None = None) -> OCPResult:
        """Solve the OCP using IPOPT.

        Args:
            solver_options: Dictionary of IPOPT options. Default uses quiet mode.

        Returns:
            OCPResult with optimal trajectory, cost, and solver statistics.

        Raises:
            RuntimeError: If solver fails to converge.
        """
        # Validate required settings
        if self._x0 is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")

        # Create CasADi Opti stack
        opti = cs.Opti()

        # Decision variables
        X = opti.variable(self.nx, self.N + 1)  # States
        U = opti.variable(self.nu, self.N)  # Controls

        # Time variable
        if self.T_fixed is not None:
            T = self.T_fixed
            dt = T / self.N
        else:
            T = opti.variable()
            dt = T / self.N

            # Time bounds
            T_min, T_max = self.T_bounds
            opti.subject_to(opti.bounded(T_min, T, T_max))

            # Initial guess for T
            if self.T_init is not None:
                opti.set_initial(T, self.T_init)
            else:
                opti.set_initial(T, (T_min + T_max) / 2)

        # Dynamics constraints (RK4 shooting with M sub-steps per interval)
        dt_sub = dt / self.M
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # M RK4 sub-steps (dt_sub may be symbolic for free-time)
            x_end = x_k
            for _m in range(self.M):
                k1 = self.model.forward_dynamics(x_end, u_k)
                k2 = self.model.forward_dynamics(x_end + 0.5 * dt_sub * k1, u_k)
                k3 = self.model.forward_dynamics(x_end + 0.5 * dt_sub * k2, u_k)
                k4 = self.model.forward_dynamics(x_end + dt_sub * k3, u_k)
                x_end = x_end + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            opti.subject_to(x_next == x_end)

        # Initial state constraint
        opti.subject_to(X[:, 0] == self._x0)

        # Terminal state constraint
        if self._x_target is not None:
            if self._x_target_mask is not None:
                # Only constrain masked states
                for i in range(self.nx):
                    if self._x_target_mask[i]:
                        opti.subject_to(X[i, -1] == self._x_target[i])
            else:
                opti.subject_to(X[:, -1] == self._x_target)

        # Control bounds
        if self._u_bounds is not None:
            u_lb, u_ub = self._u_bounds
            for i in range(self.nu):
                opti.subject_to(opti.bounded(u_lb[i], U[i, :], u_ub[i]))

        # Control rate bounds
        if self._du_bounds is not None:
            du_lb, du_ub = self._du_bounds
            for k in range(self.N - 1):
                du = U[:, k + 1] - U[:, k]
                for i in range(self.nu):
                    # Constraints: du_lb * dt <= u_{k+1} - u_k <= du_ub * dt
                    opti.subject_to(opti.bounded(du_lb[i] * dt, du[i], du_ub[i] * dt))

        # State bounds (skip node 0 to avoid infeasibility when initial state
        # violates bounds due to model mismatch or disturbances)
        for idx, (lb, ub) in self._state_bounds.items():
            opti.subject_to(opti.bounded(lb, X[idx, 1:], ub))

        # Build objective
        obj = 0.0

        # Running cost
        if self._Q is not None and self._R is not None:
            Q = self._Q
            R = self._R

            for k in range(self.N):
                # Get reference for this timestep (time-varying or fixed)
                if self._x_refs is not None:
                    x_ref_k = self._x_refs[k]
                elif self._x_ref is not None:
                    x_ref_k = self._x_ref
                else:
                    x_ref_k = np.zeros(self.nx)

                x_err = X[:, k] - x_ref_k
                u_k = U[:, k]
                if self._u_ref is not None:
                    u_err = u_k - self._u_ref
                else:
                    u_err = u_k
                obj += cs.mtimes([x_err.T, Q, x_err]) + cs.mtimes([u_err.T, R, u_err])

        # Control rate cost
        if self._Rd is not None:
            for k in range(self.N - 1):
                du = U[:, k + 1] - U[:, k]
                # Normalize by dt^2 to get rate in physical units
                obj += cs.mtimes([du.T, self._Rd, du]) / (dt * dt)

        # Terminal cost
        if self._Qf is not None:
            xf_ref = self._xf_ref
            if xf_ref is None:
                if self._x_refs is not None:
                    xf_ref = self._x_refs[-1]
                elif self._x_ref is not None:
                    xf_ref = self._x_ref
                elif self._x_target is not None:
                    xf_ref = self._x_target
                else:
                    xf_ref = np.zeros(self.nx)

            x_err_f = X[:, -1] - xf_ref
            obj += cs.mtimes([x_err_f.T, self._Qf, x_err_f])

        # Time cost
        if self._time_weight > 0 and self.T_fixed is None:
            obj += self._time_weight * T

        opti.minimize(obj)

        # Initial guess for states
        if self._initial_guess_states is not None:
            for k in range(self.N + 1):
                opti.set_initial(X[:, k], self._initial_guess_states[k])
        elif self._x0 is not None and self._x_target is not None:
            # Linear interpolation from x0 to x_target
            for k in range(self.N + 1):
                alpha = k / self.N
                x_guess = (1 - alpha) * self._x0 + alpha * self._x_target
                opti.set_initial(X[:, k], x_guess)
        elif self._x0 is not None:
            for k in range(self.N + 1):
                opti.set_initial(X[:, k], self._x0)

        # Initial guess for controls
        # Note: default is zero, not trim control. For systems with nonzero
        # trim control (e.g. Moth: ~-1.6deg flap, +0.8deg elevator), using
        # trim as the default initial guess would improve cold-start convergence.
        if self._initial_guess_controls is not None:
            for k in range(self.N):
                opti.set_initial(U[:, k], self._initial_guess_controls[k])
        else:
            opti.set_initial(U, np.zeros((self.nu, self.N)))

        # Solver options
        default_opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",  # Suppress IPOPT banner
        }
        if solver_options is not None:
            default_opts.update(solver_options)

        opti.solver("ipopt", default_opts)

        # Solve
        converged = True
        solver_stats = {}

        try:
            sol = opti.solve()

            # Extract solution
            X_opt = sol.value(X)
            U_opt = sol.value(U)
            if self.T_fixed is not None:
                T_opt = self.T_fixed
            else:
                T_opt = sol.value(T)
            cost_opt = sol.value(obj)

            # Solver stats
            stats = sol.stats()
            solver_stats = {
                "iterations": stats.get("iter_count", -1),
                "success": stats.get("success", True),
                "return_status": stats.get("return_status", "Solve_Succeeded"),
            }

        except RuntimeError as e:
            # Solver failed - try to get debug values
            converged = False
            X_opt = opti.debug.value(X)
            U_opt = opti.debug.value(U)
            if self.T_fixed is not None:
                T_opt = self.T_fixed
            else:
                T_opt = opti.debug.value(T)
            cost_opt = float("inf")
            solver_stats = {"error": str(e)}

        # Build time vector
        times = np.linspace(0, T_opt, self.N + 1)

        # Transpose to (N+1, nx) and (N, nu) format
        states = np.array(X_opt).T
        controls = np.array(U_opt).T
        # Ensure controls is 2D even for single control
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

        return OCPResult(
            states=states,
            controls=controls,
            times=times,
            T=T_opt,
            cost=cost_opt,
            converged=converged,
            solver_stats=solver_stats,
        )
