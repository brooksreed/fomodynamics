"""Base classes for JAX dynamic systems.

This module defines the core abstraction for JAX-compatible dynamic systems.
All systems are Equinox modules (PyTrees) that can be JIT-compiled and
differentiated.

The signature convention is (state, control, t=0.0) with time as an optional
parameter, since most robotic systems are time-invariant.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple
from abc import abstractmethod


class JaxDynamicSystem(eqx.Module):
    """Abstract base class for JAX-compatible dynamical systems.

    Defines the interface for systems that can be simulated using
    numerical integration with JAX. All implementations should be
    JIT-compatible (no Python control flow in forward_dynamics).

    # TODO: Subclasses share ~50-80 lines of boilerplate each (state_names,
    # default_state, post_step). Consider a mixin or code-gen approach.

    The system is defined by:
        dx/dt = f(x, u, t)

    where:
        x: state vector (Array)
        u: control input vector (Array)
        t: time (scalar, optional - default 0.0 for time-invariant systems)

    Conventions:
        - Coordinate frame: NED (North-East-Down)
        - Quaternion: Hamilton convention, scalar-first [qw, qx, qy, qz]
        - Units: SI (m, m/s, rad, rad/s)
        - Signature: (state, control, t=0.0) with time optional

    Example:
        class SimplePendulumJax(JaxDynamicSystem):
            length: float
            g: float = 9.80665

            state_names = ("theta", "theta_dot")
            control_names = ()

            def forward_dynamics(self, state, control, t=0.0):
                theta, theta_dot = state[0], state[1]
                theta_ddot = -(self.g / self.length) * jnp.sin(theta)
                return jnp.array([theta_dot, theta_ddot])
    """

    # Static metadata (not part of PyTree leaves)
    state_names: Tuple[str, ...] = eqx.field(static=True)
    control_names: Tuple[str, ...] = eqx.field(static=True, default=())
    aux_names: Tuple[str, ...] = eqx.field(static=True, default=())

    @abstractmethod
    def forward_dynamics(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute state derivative: dx/dt = f(x, u, t).

        This method must be JIT-compatible: no Python control flow,
        no side effects, pure JAX operations only.

        Args:
            state: Current state vector
            control: Control input vector
            t: Current simulation time (default 0.0 for time-invariant systems)
            env: Optional Environment with wave/wind/current fields.

        Returns:
            State derivative vector (same shape as state)
        """
        pass

    def get_state_jacobian(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute df/dx (A matrix for linearization) using JAX autodiff.

        Args:
            x: State vector at which to compute Jacobian
            u: Control vector at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            State Jacobian matrix df/dx with shape (num_states, num_states)
        """
        return jax.jacobian(lambda x_: self.forward_dynamics(x_, u, t))(x)

    def get_control_jacobian(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute df/du (B matrix for linearization) using JAX autodiff.

        Args:
            x: State vector at which to compute Jacobian
            u: Control vector at which to compute Jacobian
            t: Time at which to compute Jacobian (default 0.0)

        Returns:
            Control Jacobian matrix df/du with shape (num_states, num_controls)
        """
        return jax.jacobian(lambda u_: self.forward_dynamics(x, u_, t))(u)

    @property
    def num_aux(self) -> int:
        """Number of auxiliary output variables."""
        return len(self.aux_names)

    def compute_aux(
        self,
        state: Array,
        control: Array,
        t: float = 0.0,
        env=None,
    ) -> Array:
        """Compute auxiliary outputs for logging/analysis.

        Returns a vector of auxiliary quantities computed from the same
        state/control/time inputs used by forward_dynamics. Must be
        JAX-traceable (no Python control flow on array values).

        Override in subclasses to provide model-specific auxiliary outputs.
        Default returns an empty array for systems with no aux outputs.

        Args:
            state: Current state vector
            control: Control input vector
            t: Current simulation time
            env: Optional Environment

        Returns:
            Auxiliary output vector of shape (num_aux,)
        """
        return jnp.zeros((0,))

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return len(self.state_names)

    @property
    def num_controls(self) -> int:
        """Number of control inputs."""
        return len(self.control_names)

    def default_state(self) -> Array:
        """Default initial state (zeros).

        Override in subclasses for non-zero defaults (e.g., identity quaternion).

        Returns:
            Zero-initialized state vector.
        """
        return jnp.zeros(self.num_states)

    def default_control(self) -> Array:
        """Default control input (zeros).

        Override in subclasses for non-zero defaults (e.g., hover thrust).

        Returns:
            Zero-initialized control vector.
        """
        return jnp.zeros(self.num_controls)

    def post_step(self, state: Array) -> Array:
        """Post-process state after integration step.

        Called after each integration step to enforce constraints
        (e.g., quaternion normalization). Override in subclasses
        that require constraint enforcement.

        Note:
            This approach matches the numpy implementation exactly.
            TODO: Investigate Diffrax manifold projection for cleaner
            constraint handling that normalizes during solver internal steps.

        Args:
            state: State vector after integration step

        Returns:
            Post-processed state vector
        """
        return state


    @property
    def position_indices(self) -> tuple[int, ...]:
        """Indices of position-like states in the state vector.

        For symplectic (semi-implicit Euler) integration, position states
        are updated using the NEW velocity after velocity is updated.

        Override in subclasses to enable symplectic integration.
        Returns empty tuple by default (symplectic not supported).

        Returns:
            Tuple of integer indices for position states.
        """
        return ()

    @property
    def velocity_indices(self) -> tuple[int, ...]:
        """Indices of velocity-like states in the state vector.

        For symplectic (semi-implicit Euler) integration, velocity states
        are updated first using the OLD position.

        Override in subclasses to enable symplectic integration.
        Returns empty tuple by default (symplectic not supported).

        Returns:
            Tuple of integer indices for velocity states.
        """
        return ()

    @property
    def supports_symplectic(self) -> bool:
        """Whether this system supports symplectic integration.

        Returns True if both position_indices and velocity_indices are
        non-empty, indicating the system has the necessary structure
        for semi-implicit Euler integration.

        Returns:
            True if symplectic integration is supported.
        """
        return len(self.position_indices) > 0 and len(self.velocity_indices) > 0
