"""Measurement model abstractions for state estimation.

This module provides base classes and implementations for measurement
models used in Kalman Filters and other estimators.

Conventions:
    - All models are Equinox modules (JAX PyTrees)
    - Measurements are returned as JAX arrays
    - Jacobians computed via JAX autodiff when not provided analytically
    - R matrix (measurement noise covariance) is part of the model

Example:
    # Create a linear measurement model
    H = jnp.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
    R = jnp.eye(2) * 0.01
    model = LinearMeasurementModel(
        output_names=("position", "angle"),
        H=H,
        R=R,
    )

    # Get measurement and Jacobian
    y = model.measure(state, control)
    H_jac = model.get_measurement_jacobian(state, control)
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class MeasurementModel(eqx.Module):
    """Abstract base class for measurement models.

    A measurement model describes how system states map to observed
    measurements:
        y = h(x, u, t) + v

    where:
        x: state vector
        u: control input
        t: time
        y: measurement vector
        v: measurement noise ~ N(0, R)

    Attributes:
        output_names: Names of measurement outputs (static, not part of PyTree)
        R: Measurement noise covariance matrix (num_outputs, num_outputs)

    Conventions:
        - Must be JIT-compatible (no Python control flow in measure)
        - R matrix is positive semi-definite
        - Jacobian H = dh/dx computed via autodiff by default

    Subclasses must implement:
        - measure(x, u, t) -> y
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    R: Array  # Measurement noise covariance

    @abstractmethod
    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute measurement from state.

        This method must be JIT-compatible: no Python control flow,
        no side effects, pure JAX operations only.

        Args:
            x: State vector of shape (num_states,)
            u: Control input vector of shape (num_controls,)
            t: Current time (default 0.0 for time-invariant measurements)

        Returns:
            Measurement vector y = h(x, u, t) of shape (num_outputs,)
        """
        pass

    def get_measurement_jacobian(
        self,
        x: Array,
        u: Array,
        t: float = 0.0,
    ) -> Array:
        """Compute measurement Jacobian H = dh/dx using JAX autodiff.

        Override this method for analytical Jacobians if desired for
        performance or numerical reasons.

        Args:
            x: State vector at which to compute Jacobian
            u: Control input vector
            t: Time

        Returns:
            Measurement Jacobian matrix H with shape (num_outputs, num_states)
        """
        return jax.jacobian(lambda x_: self.measure(x_, u, t))(x)

    @property
    def num_outputs(self) -> int:
        """Number of measurement outputs."""
        return len(self.output_names)

    @property
    def state_index_map(self) -> dict[str, int | None]:
        """Map output names to state vector indices.

        Returns a dict mapping each output name to the state index it
        directly observes, or None for derived/nonlinear outputs.
        Override in subclasses to provide model-specific mappings.

        Returns:
            Dict of {output_name: state_index_or_None}
        """
        return {}

    def noisy_measure(
        self,
        x: Array,
        u: Array,
        key: jax.random.PRNGKey,
        t: float = 0.0,
    ) -> Array:
        """Compute measurement with additive Gaussian noise.

        Convenience method for simulation with realistic sensor noise.

        Args:
            x: State vector
            u: Control input vector
            key: JAX PRNG key for noise generation
            t: Current time

        Returns:
            Noisy measurement y = h(x, u, t) + v where v ~ N(0, R)
        """
        y = self.measure(x, u, t)
        noise = jax.random.multivariate_normal(
            key, jnp.zeros(self.num_outputs), self.R
        )
        return y + noise


class LinearMeasurementModel(MeasurementModel):
    """Linear measurement model: y = H @ x + D @ u.

    For linear systems where measurements are a linear combination
    of states and optionally inputs (feedthrough).

    Attributes:
        output_names: Names of measurement outputs
        H: Measurement matrix of shape (num_outputs, num_states)
        R: Measurement noise covariance of shape (num_outputs, num_outputs)
        D: Optional feedthrough matrix of shape (num_outputs, num_controls)

    Example:
        # Measure first two states directly
        H = jnp.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
        R = jnp.eye(2) * 0.01  # Small measurement noise
        model = LinearMeasurementModel(
            output_names=("x", "y"),
            H=H,
            R=R,
        )
    """

    output_names: tuple[str, ...] = eqx.field(static=True)
    H: Array  # Measurement matrix
    R: Array  # Noise covariance
    D: Array | None = None  # Feedthrough (optional)

    def __check_init__(self):
        """Validate dimensions after initialization.

        Equinox calls this automatically after __init__.
        """
        if self.H.shape[0] != len(self.output_names):
            raise ValueError(
                f"H rows ({self.H.shape[0]}) must match "
                f"number of outputs ({len(self.output_names)})"
            )
        if self.R.shape != (len(self.output_names), len(self.output_names)):
            raise ValueError(
                f"R shape {self.R.shape} must be "
                f"({len(self.output_names)}, {len(self.output_names)})"
            )
        if self.D is not None and self.D.shape[0] != len(self.output_names):
            raise ValueError(
                f"D rows ({self.D.shape[0]}) must match "
                f"number of outputs ({len(self.output_names)})"
            )

    def measure(self, x: Array, u: Array, t: float = 0.0) -> Array:
        """Compute linear measurement: y = H @ x + D @ u.

        Args:
            x: State vector
            u: Control input vector (unused if D is None)
            t: Current time (unused - time-invariant)

        Returns:
            Measurement vector y
        """
        y = self.H @ x
        if self.D is not None:
            y = y + self.D @ u
        return y

    def get_measurement_jacobian(
        self,
        x: Array,
        u: Array,
        t: float = 0.0,
    ) -> Array:
        """Return the H matrix directly (analytical Jacobian).

        For linear models, H is constant regardless of state.
        This avoids autodiff overhead.
        """
        return self.H

    @property
    def state_index_map(self) -> dict[str, int | None]:
        """Derive output-name-to-state-index mapping from H matrix.

        For each row of H, if the row is a unit selection vector (single
        1.0 entry), maps that output name to the corresponding state index.
        Otherwise maps to None.
        """
        result: dict[str, int | None] = {}
        H_np = np.asarray(self.H)
        for i, name in enumerate(self.output_names):
            row = H_np[i]
            nonzero = np.nonzero(row)[0]
            if len(nonzero) == 1 and np.isclose(row[nonzero[0]], 1.0):
                result[name] = int(nonzero[0])
            else:
                result[name] = None
        return result

    @classmethod
    def from_indices(
        cls,
        output_names: tuple[str, ...],
        state_indices: tuple[int, ...],
        num_states: int,
        R: Array,
    ) -> LinearMeasurementModel:
        """Create measurement model that selects specific states.

        Convenience factory for the common case of measuring a subset
        of states directly (selection matrix).

        Args:
            output_names: Names for measured outputs
            state_indices: Indices of states to measure (0-based)
            num_states: Total number of states in the system
            R: Measurement noise covariance

        Returns:
            LinearMeasurementModel with selection matrix H

        Example:
            # Measure states 0 and 2 from a 4-state system
            model = LinearMeasurementModel.from_indices(
                output_names=("pos", "angle"),
                state_indices=(0, 2),
                num_states=4,
                R=jnp.eye(2) * 0.01,
            )

        Raises:
            ValueError: If any state_index >= num_states
        """
        num_outputs = len(state_indices)
        if len(output_names) != num_outputs:
            raise ValueError(
                f"output_names length ({len(output_names)}) must match "
                f"state_indices length ({num_outputs})"
            )

        # Validate indices
        for idx in state_indices:
            if idx >= num_states or idx < 0:
                raise ValueError(
                    f"state_index {idx} out of range for {num_states} states"
                )

        # Build selection matrix
        H = jnp.zeros((num_outputs, num_states))
        for i, idx in enumerate(state_indices):
            H = H.at[i, idx].set(1.0)

        return cls(output_names=output_names, H=H, R=R)


class FullStateMeasurement(LinearMeasurementModel):
    """Measurement model that observes all states directly.

    Useful for testing and as a baseline (perfect sensing).
    y = I @ x (identity measurement)

    Example:
        from fmd.simulator import Cartpole
        from fmd.simulator.params import CARTPOLE_CLASSIC

        cartpole = Cartpole(CARTPOLE_CLASSIC)
        model = FullStateMeasurement.for_system(
            cartpole,
            R=0.01 * jnp.eye(cartpole.num_states)
        )
    """

    @classmethod
    def for_system(
        cls,
        system,  # JaxDynamicSystem - can't type hint due to circular import
        R: Array,
    ) -> FullStateMeasurement:
        """Create full-state measurement for a dynamic system.

        Args:
            system: JaxDynamicSystem instance (must have state_names, num_states)
            R: Measurement noise covariance of shape (num_states, num_states)

        Returns:
            FullStateMeasurement observing all states

        Raises:
            ValueError: If R shape doesn't match system.num_states
        """
        num_states = system.num_states
        if R.shape != (num_states, num_states):
            raise ValueError(
                f"R shape {R.shape} must be ({num_states}, {num_states})"
            )

        H = jnp.eye(num_states)
        return cls(
            output_names=system.state_names,
            H=H,
            R=R,
        )
