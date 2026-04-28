"""Abstract base classes for BLUR packages.

This module defines the core abstractions used across BLUR packages.
Implementations should avoid if/else on array values for compatibility
with autodiff frameworks (JAX, CasADi).

Note: The blur.simulator package uses JAX with Equinox modules.
A separate CasADi implementation would be needed for MPC/NMPC applications.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Sequence, Any, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class DynamicsProtocol(Protocol):
    """Contract for dynamics implementations across backends.

    Both JAX and CasADi implementations must satisfy this interface.
    Enables polymorphic testing and solver-agnostic high-level code.

    This protocol defines the minimal interface for a dynamical system
    that can be integrated and linearized. Implementations may use
    JAX arrays (jax.Array) or CasADi symbolic types (cs.SX).

    Example:
        def verify_physics(system: DynamicsProtocol, x0, u0):
            '''Test that works with any backend.'''
            xdot = system.forward_dynamics(x0, u0)
            A = system.get_state_jacobian(x0, u0)
            B = system.get_control_jacobian(x0, u0)
            return xdot, A, B
    """

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        ...

    @property
    def num_controls(self) -> int:
        """Number of control inputs."""
        ...

    @property
    def state_names(self) -> tuple[str, ...]:
        """Names of state variables, in order matching state vector."""
        ...

    @property
    def control_names(self) -> tuple[str, ...]:
        """Names of control inputs, in order matching control vector."""
        ...

    def forward_dynamics(self, x: Any, u: Any, t: float = 0.0) -> Any:
        """Return dx/dt (continuous-time derivative).

        Args:
            x: State vector (jax.Array for JAX, cs.SX for CasADi)
            u: Control vector
            t: Time (optional, default 0.0 for time-invariant systems)

        Returns:
            State derivative dx/dt
        """
        ...

    def get_state_jacobian(self, x: Any, u: Any, t: float = 0.0) -> Any:
        """Return df/dx (A matrix for linearization).

        Args:
            x: State vector at which to compute Jacobian
            u: Control vector at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            State Jacobian matrix df/dx with shape (num_states, num_states)
        """
        ...

    def get_control_jacobian(self, x: Any, u: Any, t: float = 0.0) -> Any:
        """Return df/du (B matrix for linearization).

        Args:
            x: State vector at which to compute Jacobian
            u: Control vector at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            Control Jacobian matrix df/du with shape (num_states, num_controls)
        """
        ...

    def post_step(self, x: Any) -> Any:
        """Post-process state after integration step.

        Default is identity. Override for angle wrapping, quaternion
        normalization, etc.

        Args:
            x: State vector after integration step

        Returns:
            Post-processed state vector
        """
        ...


class DynamicSystem(ABC):
    """Abstract base class for dynamical systems.

    Defines the interface for systems that can be simulated using
    numerical integration. All implementations should be CasADi-compatible
    (vectorized operations, no if/else branches in state_derivative).

    The system is defined by:
        ẋ = f(x, u, t)

    where:
        x: state vector
        u: control input vector
        t: time

    Example:
        class SimplePendulum(DynamicSystem):
            def __init__(self, length: float = 1.0, gravity: float = 9.81):
                self.length = length
                self.gravity = gravity

            @property
            def state_names(self) -> list[str]:
                return ["theta", "theta_dot"]

            def state_derivative(self, state, control, time):
                theta, theta_dot = state[0], state[1]
                theta_ddot = -self.gravity / self.length * np.sin(theta)
                return np.array([theta_dot, theta_ddot])
    """

    @abstractmethod
    def state_derivative(
        self,
        state: NDArray,
        control: NDArray,
        time: float,
    ) -> NDArray:
        """Compute state derivative: ẋ = f(x, u, t).

        Args:
            state: Current state vector
            control: Control input vector
            time: Current simulation time

        Returns:
            State derivative vector (same shape as state)
        """
        pass

    @property
    @abstractmethod
    def state_names(self) -> Sequence[str]:
        """Names of state variables.

        Returns:
            Sequence of state variable names, in order matching state vector.
        """
        pass

    @property
    def control_names(self) -> Sequence[str]:
        """Names of control inputs.

        Returns:
            Sequence of control input names. Default is empty (no controls).
        """
        return []

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return len(self.state_names)

    @property
    def num_controls(self) -> int:
        """Number of control inputs."""
        return len(self.control_names)

    def default_state(self) -> NDArray:
        """Default initial state (zeros).

        Returns:
            Zero-initialized state vector.
        """
        return np.zeros(len(self.state_names))

    def default_control(self) -> NDArray:
        """Default control input (zeros).

        Returns:
            Zero-initialized control vector.
        """
        return np.zeros(len(self.control_names))

    def post_step(self, state: NDArray) -> NDArray:
        """Post-process state after integration step.

        Called after each integration step to enforce constraints
        (e.g., quaternion normalization).

        Args:
            state: State vector after integration step

        Returns:
            Post-processed state vector
        """
        return state

    def get_outputs(
        self,
        state: NDArray,
        control: NDArray,
        time: float,
    ) -> dict[str, float]:
        """Compute derived outputs for logging.

        Override to compute additional quantities (e.g., Euler angles
        from quaternions) that should be logged but aren't part of state.

        Args:
            state: Current state vector
            control: Control input vector
            time: Current simulation time

        Returns:
            Dictionary of output name -> value
        """
        return {}
