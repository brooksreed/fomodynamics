"""Boat2D parameter class.

Immutable, validated parameters for the Boat2D planar dynamics model.
"""

from __future__ import annotations

import attrs

from fmd.simulator.params.base import is_finite, positive


@attrs.define(frozen=True, slots=True)
class Boat2DParams:
    """Immutable parameters for Boat2D planar dynamics model.

    All values must be finite (no NaN/Inf) and positive.

    The Boat2D model uses a Fossen-style 3-DOF maneuvering formulation with
    linear drag in each DOF and Coriolis coupling between surge/sway/yaw.

    Attributes:
        mass: Vehicle mass (kg). Must be positive.
        izz: Yaw moment of inertia about the vertical axis (kg*m^2). Must be positive.
        drag_surge: Surge (forward) damping coefficient (kg/s). Must be positive.
        drag_sway: Sway (lateral) damping coefficient (kg/s). Must be positive.
        drag_yaw: Yaw (rotational) damping coefficient (kg*m^2/s). Must be positive.

    Example:
        >>> params = Boat2DParams(
        ...     mass=100.0,
        ...     izz=50.0,
        ...     drag_surge=10.0,
        ...     drag_sway=20.0,
        ...     drag_yaw=5.0,
        ... )
        >>> from fmd.simulator import Boat2D
        >>> boat = Boat2D(params)
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Vehicle mass"},
    )
    izz: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg*m^2", "description": "Yaw moment of inertia"},
    )
    drag_surge: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg/s", "description": "Surge damping coefficient"},
    )
    drag_sway: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg/s", "description": "Sway damping coefficient"},
    )
    drag_yaw: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg*m^2/s", "description": "Yaw damping coefficient"},
    )

    def with_mass(self, mass: float) -> Boat2DParams:
        """Return new params with updated mass.

        Args:
            mass: New mass value (kg).

        Returns:
            New Boat2DParams instance with updated mass.
        """
        return attrs.evolve(self, mass=mass)

    def with_inertia(self, izz: float) -> Boat2DParams:
        """Return new params with updated yaw inertia.

        Args:
            izz: New yaw moment of inertia (kg*m^2).

        Returns:
            New Boat2DParams instance with updated inertia.
        """
        return attrs.evolve(self, izz=izz)

    def with_drag(
        self,
        surge: float | None = None,
        sway: float | None = None,
        yaw: float | None = None,
    ) -> Boat2DParams:
        """Return new params with updated drag coefficients.

        Only specified values are updated; others remain unchanged.

        Args:
            surge: New surge drag coefficient (kg/s), or None to keep current.
            sway: New sway drag coefficient (kg/s), or None to keep current.
            yaw: New yaw drag coefficient (kg*m^2/s), or None to keep current.

        Returns:
            New Boat2DParams instance with updated drag values.
        """
        return attrs.evolve(
            self,
            drag_surge=surge if surge is not None else self.drag_surge,
            drag_sway=sway if sway is not None else self.drag_sway,
            drag_yaw=yaw if yaw is not None else self.drag_yaw,
        )

    # Analytical properties

    def surge_time_constant(self) -> float:
        """Time constant for surge dynamics: tau = m / D_u (s)."""
        return self.mass / self.drag_surge

    def yaw_time_constant(self) -> float:
        """Time constant for yaw dynamics: tau_r = Izz / D_r (s)."""
        return self.izz / self.drag_yaw

    def steady_state_surge(self, thrust: float) -> float:
        """Steady-state surge velocity for given thrust: u_ss = thrust / D_u (m/s)."""
        return thrust / self.drag_surge

    def steady_state_yaw_rate(self, yaw_moment: float) -> float:
        """Steady-state yaw rate for given moment: r_ss = moment / D_r (rad/s)."""
        return yaw_moment / self.drag_yaw
