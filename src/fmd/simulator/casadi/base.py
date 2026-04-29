"""Base class for CasADi-compatible dynamical systems.

This module provides the abstract base class for CasADi dynamics models.
CasADi models support symbolic differentiation and efficient compiled
functions for use in optimization-based control (MPC, trajectory optimization).

Design principles:
- Symbolic dynamics with cached compiled CasADi Functions
- Jacobian computation via CasADi's symbolic differentiation (exact, not numerical)
- post_step for simulation equivalence testing (identity by default for MPC)
- Compatible interface with JAX models (same state_names, control_names, etc.)

Note:
    CasADi is an optional dependency (included in default `uv sync`).

Example:
    >>> import casadi as cs
    >>> from fmd.simulator.casadi import CasadiDynamicSystem
    >>>
    >>> class MySystem(CasadiDynamicSystem):
    ...     state_names = ("x", "v")
    ...     control_names = ("f",)
    ...
    ...     def forward_dynamics(self, x, u, t=0.0):
    ...         return cs.vertcat(x[1], u[0])
    ...
    >>> model = MySystem()
    >>> f = model.dynamics_function()  # Compiled CasADi Function
    >>> x = cs.SX([1.0, 0.0])
    >>> u = cs.SX([0.5])
    >>> print(f(x, u))  # [0.0; 0.5]
"""

from __future__ import annotations

import casadi as cs
from abc import ABC, abstractmethod
from functools import lru_cache


class CasadiDynamicSystem(ABC):
    """Abstract base class for CasADi-compatible dynamical systems.

    Provides:
    - Symbolic dynamics with cached compiled functions
    - Jacobian computation via CasADi symbolic differentiation
    - post_step for simulation equivalence (identity by default)

    The system is defined by:
        dx/dt = f(x, u, t)

    where:
        x: state vector (cs.SX)
        u: control input vector (cs.SX)
        t: time (scalar, optional - default 0.0 for time-invariant systems)

    Conventions:
        - Coordinate frame: NED (North-East-Down)
        - Quaternion: Hamilton convention, scalar-first [qw, qx, qy, qz]
        - Units: SI (m, m/s, rad, rad/s)
        - Signature: (state, control, t=0.0) with time optional

    Subclasses must define:
        - state_names: Tuple of state variable names
        - control_names: Tuple of control input names
        - forward_dynamics: Method returning dx/dt as cs.SX

    Example:
        class SimplePendulumCasadi(CasadiDynamicSystem):
            state_names = ("theta", "theta_dot")
            control_names = ()

            def __init__(self, length: float = 1.0, g: float = 9.80665):
                self.length = length
                self.g = g

            def forward_dynamics(self, x, u, t=0.0):
                theta = x[0]
                theta_dot = x[1]
                theta_ddot = -(self.g / self.length) * cs.sin(theta)
                return cs.vertcat(theta_dot, theta_ddot)
    """

    # Subclasses must define these as class attributes or instance attributes
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]

    @property
    def num_states(self) -> int:
        """Number of state variables.

        Returns:
            Length of the state vector.
        """
        return len(self.state_names)

    @property
    def num_controls(self) -> int:
        """Number of control inputs.

        Returns:
            Length of the control vector.
        """
        return len(self.control_names)

    @abstractmethod
    def forward_dynamics(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute state derivative: dx/dt = f(x, u, t).

        This method defines the continuous-time dynamics of the system.
        Must be implemented by subclasses.

        Args:
            x: Current state vector as CasADi symbolic (cs.SX)
            u: Control input vector as CasADi symbolic (cs.SX)
            t: Current simulation time (default 0.0 for time-invariant systems)

        Returns:
            State derivative vector as cs.SX (same size as x)

        Note:
            Use cs.vertcat() to combine scalar expressions into a vector.
            Avoid Python control flow (if/else) - use cs.if_else() for
            conditional expressions.
        """
        pass

    def post_step(self, x: cs.SX) -> cs.SX:
        """Post-process state after integration step.

        Default is identity (no modification). Override in subclasses
        for operations like angle wrapping or quaternion normalization.

        For MPC, prefer keeping identity and using explicit constraints
        instead of post-processing, as this preserves smoothness for
        the optimizer.

        Args:
            x: State vector after integration step

        Returns:
            Post-processed state vector (same size as input)

        Note:
            This is primarily used for simulation equivalence testing
            with JAX models. MPC formulations should typically use
            constraints instead of post_step modifications.
        """
        return x

    def get_state_jacobian(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute df/dx (A matrix for linearization) using CasADi symbolic diff.

        Uses CasADi's cs.jacobian() for exact symbolic differentiation,
        which is more accurate than numerical finite differences.

        Args:
            x: State vector (symbolic) at which to compute Jacobian
            u: Control vector (symbolic) at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            State Jacobian matrix df/dx with shape (num_states, num_states)
        """
        xdot = self.forward_dynamics(x, u, t)
        return cs.jacobian(xdot, x)

    def get_control_jacobian(self, x: cs.SX, u: cs.SX, t: float = 0.0) -> cs.SX:
        """Compute df/du (B matrix for linearization) using CasADi symbolic diff.

        Uses CasADi's cs.jacobian() for exact symbolic differentiation,
        which is more accurate than numerical finite differences.

        Args:
            x: State vector (symbolic) at which to compute Jacobian
            u: Control vector (symbolic) at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            Control Jacobian matrix df/du with shape (num_states, num_controls)
        """
        xdot = self.forward_dynamics(x, u, t)
        return cs.jacobian(xdot, u)

    # --- Cached compiled functions ---

    @lru_cache(maxsize=1)
    def dynamics_function(self, name: str = "f") -> cs.Function:
        """Return cached CasADi Function for forward_dynamics.

        Creates symbolic variables and compiles the dynamics into a
        reusable CasADi Function. The function is cached for efficiency.

        Args:
            name: Name for the compiled function (default "f")

        Returns:
            CasADi Function with signature (x, u) -> xdot

        Note:
            The cache key includes the `name` argument. Prefer leaving `name`
            at its default so repeated calls return the same cached Function.

        Example:
            >>> model = MyModel()
            >>> f = model.dynamics_function()
            >>> xdot = f([1.0, 0.0], [0.5])  # Evaluate at specific point
        """
        x = cs.SX.sym("x", self.num_states)
        u = cs.SX.sym("u", self.num_controls)
        xdot = self.forward_dynamics(x, u, 0.0)
        return cs.Function(name, [x, u], [xdot], ["x", "u"], ["xdot"])

    @lru_cache(maxsize=1)
    def linearization_function(self, name: str = "AB") -> cs.Function:
        """Return cached CasADi Function for (A, B) Jacobians.

        Creates a compiled function that returns both the state Jacobian
        (A = df/dx) and control Jacobian (B = df/du) at any operating point.

        Args:
            name: Name for the compiled function (default "AB")

        Returns:
            CasADi Function with signature (x, u) -> (A, B)

        Note:
            The cache key includes the `name` argument. Prefer leaving `name`
            at its default so repeated calls return the same cached Function.

        Example:
            >>> model = MyModel()
            >>> AB = model.linearization_function()
            >>> A, B = AB([1.0, 0.0], [0.5])  # Get Jacobians at point
        """
        x = cs.SX.sym("x", self.num_states)
        u = cs.SX.sym("u", self.num_controls)
        A = self.get_state_jacobian(x, u, 0.0)
        B = self.get_control_jacobian(x, u, 0.0)
        return cs.Function(name, [x, u], [A, B], ["x", "u"], ["A", "B"])

    def rk4_step_function(
        self,
        dt: float,
        name: str = "rk4",
        include_post_step: bool = False,
    ) -> cs.Function:
        """Return CasADi Function for RK4 integration step.

        Creates a compiled function that performs one RK4 integration step.
        The function is cached by (dt, name, include_post_step) for efficiency.

        Args:
            dt: Time step size in seconds
            name: Name for the compiled function (default "rk4")
            include_post_step: Whether to apply post_step after RK4
                - False (default): Pure RK4, suitable for MPC optimization
                - True: RK4 + post_step, for simulation equivalence tests

        Returns:
            CasADi Function with signature (x, u) -> x_next

        Example:
            >>> model = MyModel()
            >>> step = model.rk4_step_function(dt=0.01)
            >>> x_next = step([1.0, 0.0], [0.5])  # Single RK4 step

        Note:
            For MPC, use include_post_step=False to avoid non-smooth
            operations in the optimization problem. Use constraints
            to enforce state bounds instead.

            For simulation equivalence testing with JAX models, use
            include_post_step=True to match the JAX post_step behavior.
        """
        return self._get_rk4_step_function(dt, name, include_post_step)

    @lru_cache(maxsize=8)
    def _get_rk4_step_function(
        self,
        dt: float,
        name: str,
        include_post_step: bool,
    ) -> cs.Function:
        """Internal cached implementation of RK4 step function.

        This method is cached by (dt, name, include_post_step) tuple,
        allowing efficient reuse of compiled functions for the same
        timestep across multiple calls.

        Args:
            dt: Time step size in seconds
            name: Name for the compiled function
            include_post_step: Whether to apply post_step after RK4

        Returns:
            CasADi Function with signature (x, u) -> x_next
        """
        # Import here to avoid circular dependency
        from fmd.simulator.casadi.integrator import rk4_step_function

        return rk4_step_function(self, dt, name, include_post_step)
