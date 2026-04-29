"""Estimator implementations for the closed-loop pipeline.

Provides EKFEstimator, which wraps the ExtendedKalmanFilter from
fmd.estimation.ekf to produce state estimates with proper state
management (x_est, P) as a pytree carry.

Also provides PassthroughEstimator, an identity estimator for
mechanical wand control that maps wand angle into a pseudo-state vector.

Example:
    from fmd.estimation import ExtendedKalmanFilter
    from fmd.simulator.estimators import EKFEstimator, PassthroughEstimator

    ekf = ExtendedKalmanFilter(dt=0.005)
    estimator = EKFEstimator(
        ekf=ekf,
        measurement_model=meas_model,
        Q_ekf=Q_ekf,
    )

    passthrough = PassthroughEstimator(n_states=5)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from fmd.estimation.measurement import MeasurementModel
from fmd.simulator.base import JaxDynamicSystem

if TYPE_CHECKING:
    from fmd.estimation.ekf import ExtendedKalmanFilter


class EKFEstimator(eqx.Module):
    """EKF-based estimator for the closed-loop pipeline.

    Wraps ExtendedKalmanFilter predict + update steps. The estimator
    state is a (x_est, P) tuple carried through the scan loop.

    Attributes:
        ekf: ExtendedKalmanFilter instance.
        measurement_model: MeasurementModel for updates.
        Q_ekf: Process noise covariance for EKF predict.
        num_controls: Number of control inputs (for dummy u vector).
    """

    ekf: ExtendedKalmanFilter
    measurement_model: MeasurementModel
    Q_ekf: Array
    num_controls: int = eqx.field(static=True, default=2)

    def estimate(
        self,
        est_state: tuple[Array, Array],
        y: Array,
        u_prev: Array,
        system: JaxDynamicSystem,
        t: float,
    ) -> tuple[Array, tuple[Array, Array], Array]:
        """Update state estimate from measurement and previous control.

        Runs EKF predict (using u_prev) then update (using measurement y).
        Computes innovation manually to capture it for diagnostics.

        Args:
            est_state: Tuple of (x_est, P) from previous step.
            y: Noisy measurement vector.
            u_prev: Control applied at previous timestep.
            system: Dynamic system for EKF prediction.
            t: Current simulation time.

        Returns:
            Tuple of (x_est_new, (x_est_new, P_new), innovation).
        """
        x_est, P = est_state

        # Predict using previous control
        x_pred, P_pred = self.ekf.predict(x_est, P, system, u_prev, self.Q_ekf, t)

        # Compute innovation manually for diagnostics
        u_dummy = jnp.zeros(self.num_controls)
        y_pred = self.measurement_model.measure(x_pred, u_dummy, t)
        innovation = y - y_pred

        # Update with measurement
        x_est_new, P_new = self.ekf.update(
            x_pred, P_pred, y, self.measurement_model, u_dummy, t, system=system
        )

        return x_est_new, (x_est_new, P_new), innovation

    def init_state(self, x0_est: Array, P0: Array) -> tuple[Array, Array]:
        """Return initial estimator state.

        Args:
            x0_est: Initial state estimate.
            P0: Initial covariance.

        Returns:
            Tuple of (x0_est, P0).
        """
        return (x0_est, P0)


class PassthroughEstimator(eqx.Module):
    """Identity estimator for mechanical wand control.

    Maps the wand angle measurement into slot 0 of an n_states-element
    pseudo-state vector. Conforms to the EKF ``(x_est, P)`` state
    convention so the pipeline scan loop's covariance extraction works
    without changes.

    The covariance trace/diagonals in ClosedLoopResult will be meaningless
    for this estimator — downstream code should check the estimator type
    before interpreting covariance data.

    Attributes:
        n_states: Number of states in pseudo-state vector (default 5).
    """

    n_states: int = eqx.field(static=True, default=5)

    def estimate(
        self,
        est_state: tuple[Array, Array],
        y: Array,
        u_prev: Array,
        system: Any,
        t: float,
    ) -> tuple[Array, tuple[Array, Array], Array]:
        """Map measurement into pseudo-state vector.

        Places y[0] (wand angle) at slot 0 of the state vector.
        Remaining slots are zeros. Returns zero innovation.

        Args:
            est_state: Tuple of (x_est, P) from previous step.
            y: Measurement vector (wand angle at index 0).
            u_prev: Control applied at previous timestep (unused).
            system: Dynamic system (unused).
            t: Current simulation time (unused).

        Returns:
            Tuple of (x_est_new, (x_est_new, P), zero_innovation).
        """
        _, P = est_state
        x_est_new = jnp.zeros(self.n_states).at[0].set(y[0])
        return x_est_new, (x_est_new, P), jnp.zeros_like(y)

    def init_state(self, x0_est: Array, P0: Array) -> tuple[Array, Array]:
        """Return initial estimator state.

        Args:
            x0_est: Initial state estimate.
            P0: Initial covariance.

        Returns:
            Tuple of (x0_est, P0) — dummy P carried for scan loop compatibility.
        """
        return (x0_est, P0)
