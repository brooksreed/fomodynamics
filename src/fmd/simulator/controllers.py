"""Controller implementations for the closed-loop pipeline.

Provides LQRController and MechanicalWandController.

Example:
    from fmd.simulator.controllers import LQRController, MechanicalWandController

    controller = LQRController(
        K=K, x_trim=x_trim, u_trim=u_trim,
        u_min=u_min, u_max=u_max,
    )
    u = controller.control(x_est, t)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from fmd.simulator.components.moth_wand import WandLinkage


class LQRController(eqx.Module):
    """LQR state-feedback controller with saturation.

    Computes: u = u_trim - K @ (x_est[:n_ctrl] - x_trim[:n_ctrl])
    then clips to [u_min, u_max].

    Attributes:
        K: LQR gain matrix, shape (m, n_ctrl).
        x_trim: Trim state, shape (n,).
        u_trim: Trim control, shape (m,).
        u_min: Lower control bounds, shape (m,).
        u_max: Upper control bounds, shape (m,).
    """

    K: Array
    x_trim: Array
    u_trim: Array
    u_min: Array
    u_max: Array

    def control(self, x_est: Array, t: float) -> Array:
        """Compute saturated LQR control.

        Args:
            x_est: Estimated state vector.
            t: Current simulation time (unused for LQR).

        Returns:
            Control vector, clipped to bounds.
        """
        n_ctrl = self.K.shape[1]
        u = self.u_trim - self.K @ (x_est[:n_ctrl] - self.x_trim[:n_ctrl])
        return jnp.clip(u, self.u_min, self.u_max)


class MechanicalWandController(eqx.Module):
    """Mechanical wand-to-flap controller with saturation.

    Uses WandLinkage to convert wand angle to flap deflection.
    Elevator is held at a fixed trim value.

    The wand angle is read from ``x_est[0]`` (slot 0), as placed there
    by PassthroughEstimator.

    Attributes:
        linkage: WandLinkage instance for angle-to-flap conversion.
        elevator_trim: Fixed elevator angle from trim solver (rad).
        u_min: Lower control bounds [flap_min, elevator_min].
        u_max: Upper control bounds [flap_max, elevator_max].
    """

    linkage: WandLinkage
    elevator_trim: float
    u_min: Array
    u_max: Array

    def control(self, x_est: Array, t: float) -> Array:
        """Compute saturated mechanical wand control.

        Args:
            x_est: Estimated state vector (wand angle at index 0).
            t: Current simulation time (unused).

        Returns:
            Control vector [flap, elevator], clipped to bounds.
        """
        wand_angle = x_est[0]
        flap = self.linkage.compute(wand_angle)
        u = jnp.array([flap, self.elevator_trim])
        return jnp.clip(u, self.u_min, self.u_max)
