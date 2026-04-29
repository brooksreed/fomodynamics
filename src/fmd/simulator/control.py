"""JIT-safe control interfaces for JAX simulation.

All control schedules are Equinox modules to ensure JIT compilation
works without retracing. Python callables cannot be used directly
as control functions in JIT-compiled simulation loops.

The signature convention is (t, state) -> control to match common usage.
"""

from __future__ import annotations

# Ensure float64 is enabled before any JAX imports
from fmd.simulator import _config  # noqa: F401

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from abc import abstractmethod


class ControlSchedule(eqx.Module):
    """Abstract base for JIT-safe control policies.

    All control schedules must be Equinox modules to ensure
    JIT compilation works without retracing.

    Subclasses should implement __call__ to return the control
    vector at a given time and state.

    Example:
        class PDController(ControlSchedule):
            kp: float
            kd: float
            target: Array

            def __call__(self, t: float, state: Array) -> Array:
                error = self.target - state[:3]
                derror = -state[3:6]
                return self.kp * error + self.kd * derror
    """

    @abstractmethod
    def __call__(self, t: float, state: Array) -> Array:
        """Return control vector at time t given state.

        Args:
            t: Current simulation time (scalar)
            state: Current state vector

        Returns:
            Control input vector
        """
        pass


class ConstantControl(ControlSchedule):
    """Constant control for all time.

    Useful for open-loop simulations with fixed inputs.

    Attributes:
        value: Control vector to return at all times

    Example:
        control = ConstantControl(jnp.array([10.0, 0.0]))  # Constant thrust
    """

    value: Array

    def __call__(self, t: float, state: Array) -> Array:
        """Return the constant control value."""
        return self.value


class ZeroControl(ControlSchedule):
    """Zero control of specified dimension.

    Useful as default for systems with no control inputs
    or for free-response simulations.

    Attributes:
        dim: Dimension of the control vector
    """

    dim: int = eqx.field(static=True)

    def __call__(self, t: float, state: Array) -> Array:
        """Return zero control vector."""
        return jnp.zeros(self.dim)


class PiecewiseConstantControl(ControlSchedule):
    """Pre-sampled control array with zero-order hold.

    Looks up control by time index. The control at time t is the
    control value at the largest time point <= t.

    Useful for golden master tests where controls are pre-computed
    at each time step.

    Attributes:
        times: Time points array, shape (num_steps,)
        controls: Control values array, shape (num_steps, num_controls)

    Example:
        times = jnp.linspace(0, 10, 101)
        controls = jnp.zeros((101, 2))
        schedule = PiecewiseConstantControl(times, controls)
    """

    times: Array
    controls: Array

    def __call__(self, t: float, state: Array) -> Array:
        """Return control at time t using zero-order hold.

        Finds the largest time index where times[idx] <= t
        and returns the corresponding control.
        """
        # Find index: largest i where times[i] <= t
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.controls) - 1)
        return self.controls[idx]


class LinearInterpolatedControl(ControlSchedule):
    """Pre-sampled control with linear interpolation.

    Interpolates control between time points.
    Useful when smoother control transitions are desired.

    Attributes:
        times: Time points array, shape (num_steps,)
        controls: Control values array, shape (num_steps, num_controls)
    """

    times: Array
    controls: Array

    def __call__(self, t: float, state: Array) -> Array:
        """Return linearly interpolated control at time t."""
        # Find indices for interpolation
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)

        t0 = self.times[idx]
        t1 = self.times[idx + 1]
        c0 = self.controls[idx]
        c1 = self.controls[idx + 1]

        # Linear interpolation factor
        alpha = (t - t0) / (t1 - t0 + 1e-10)  # Epsilon avoids div-by-zero when t0==t1; bias is negligible (<1e-10)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        return c0 + alpha * (c1 - c0)
