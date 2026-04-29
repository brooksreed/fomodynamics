"""Physical and environmental constraints.

This module provides constraints for physical/environmental boundaries
suitable for optimization and control.

Note: Surface contact and collision FORCES are better implemented as
JaxForceElement components (see fmd.simulator.components). This module
contains constraints for optimization that don't involve continuous
force generation (e.g., keep-out zones for path planning).

Coordinate Convention:
    fomodynamics uses NED (North-East-Down) coordinates:
    - pos_n, pos_e, pos_d (state indices 0, 1, 2 for RigidBody6DOF)
    - +D is DOWN, so altitude increase means pos_d decreases
    - Water surface at pos_d=0 means "above water" is pos_d <= 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from fmd.simulator.constraints.base import (
    AbstractConstraint,
    Capability,
    ConstraintCategory,
)

if TYPE_CHECKING:
    from jax import Array


class KeepOutZone(AbstractConstraint):
    """Spherical exclusion zone constraint.

    Constraint: stay outside a sphere centered at `center` with given `radius`.

    Returns scalar: radius - distance
    - <= 0 when outside the sphere (satisfied)
    - > 0 when inside the sphere (violated)

    This is useful for obstacle avoidance in optimization/MPC.

    Attributes:
        center: NED position of zone center (3-element array)
        radius: Radius of exclusion zone (must be positive)
        pos_indices: Tuple of state indices for position (default: (0, 1, 2))

    Example:
        # Keep out of a 5m sphere centered at (10, 20, -5) NED
        zone = KeepOutZone(
            name="obstacle_1",
            center=jnp.array([10.0, 20.0, -5.0]),
            radius=5.0
        )

        # For a 2D system where position is at indices (0, 1)
        # and you want to ignore the vertical component:
        zone_2d = KeepOutZone(
            name="obstacle_2d",
            center=jnp.array([10.0, 20.0]),
            radius=5.0,
            pos_indices=(0, 1)
        )
    """

    center: Array
    radius: float
    pos_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        center: Array,
        radius: float,
        pos_indices: tuple[int, ...] = (0, 1, 2),
    ):
        """Create a keep-out zone constraint.

        Args:
            name: Unique constraint name
            center: Position of zone center (array matching pos_indices length)
            radius: Radius of exclusion zone
            pos_indices: Tuple of state indices for position

        Raises:
            ValueError: If radius is not positive
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        center_arr = jnp.asarray(center)
        if center_arr.shape[0] != len(pos_indices):
            raise ValueError(
                f"center has {center_arr.shape[0]} elements but "
                f"pos_indices has {len(pos_indices)} elements"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", ConstraintCategory.SAFETY)
        object.__setattr__(
            self, "capabilities", frozenset({Capability.HAS_SYMBOLIC_FORM})
        )
        object.__setattr__(self, "center", center_arr)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "pos_indices", pos_indices)

    def value(self, t: float, x: Array, u: Array) -> Array:
        """Evaluate constraint: radius - distance.

        Args:
            t: Time (unused, for interface consistency)
            x: State vector
            u: Control vector (unused)

        Returns:
            Scalar: <= 0 when outside zone (satisfied).
        """
        pos = x[jnp.array(self.pos_indices)]
        dist = jnp.linalg.norm(pos - self.center)
        return self.radius - dist  # <= 0 when outside zone


class HalfSpaceConstraint(AbstractConstraint):
    """Half-space constraint defined by a plane.

    Constraint: stay on one side of a plane defined by normal and offset.
    The constraint is: normal.dot(pos) <= offset

    Returns scalar: normal.dot(pos) - offset
    - <= 0 when on the correct side (satisfied)
    - > 0 when on the wrong side (violated)

    Attributes:
        normal: Unit normal vector pointing toward forbidden region
        offset: Signed distance from origin to plane
        pos_indices: Tuple of state indices for position

    Example:
        # Stay below altitude 100m (in NED, +D is down, so pos_d >= -100)
        # Normal points up (negative D), constraint is pos_d >= -100
        # Equivalently: -pos_d <= 100, or normal=[-1] at index 2
        ceiling = HalfSpaceConstraint(
            name="altitude_ceiling",
            normal=jnp.array([0.0, 0.0, -1.0]),  # Points up (forbidden above)
            offset=-100.0,  # Plane at pos_d = -100 (100m altitude)
            pos_indices=(0, 1, 2)
        )

        # Ground plane constraint: stay above ground (pos_d <= 0 in NED)
        ground = HalfSpaceConstraint(
            name="ground",
            normal=jnp.array([0.0, 0.0, 1.0]),  # Points down (forbidden below)
            offset=0.0,  # Plane at pos_d = 0
            pos_indices=(0, 1, 2)
        )
    """

    normal: Array
    offset: float
    pos_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        normal: Array,
        offset: float,
        pos_indices: tuple[int, ...] = (0, 1, 2),
    ):
        """Create a half-space constraint.

        Args:
            name: Unique constraint name
            normal: Normal vector (will be normalized)
            offset: Signed distance from origin to plane
            pos_indices: Tuple of state indices for position
        """
        normal_arr = jnp.asarray(normal)
        if normal_arr.shape[0] != len(pos_indices):
            raise ValueError(
                f"normal has {normal_arr.shape[0]} elements but "
                f"pos_indices has {len(pos_indices)} elements"
            )

        # Normalize the normal vector
        norm = jnp.linalg.norm(normal_arr)
        # Use dtype-aware threshold for near-zero check
        eps = jnp.finfo(normal_arr.dtype).eps * 100
        if norm < eps:
            raise ValueError("normal vector cannot be zero")
        normal_arr = normal_arr / norm

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", ConstraintCategory.PHYSICAL)
        object.__setattr__(
            self, "capabilities", frozenset({Capability.HAS_SYMBOLIC_FORM, Capability.HARD_CLIP})
        )
        object.__setattr__(self, "normal", normal_arr)
        object.__setattr__(self, "offset", float(offset))
        object.__setattr__(self, "pos_indices", pos_indices)

    def value(self, t: float, x: Array, u: Array) -> Array:
        """Evaluate constraint: normal.dot(pos) - offset.

        Args:
            t: Time (unused, for interface consistency)
            x: State vector
            u: Control vector (unused)

        Returns:
            Scalar: <= 0 when on correct side of plane.
        """
        pos = x[jnp.array(self.pos_indices)]
        return jnp.dot(self.normal, pos) - self.offset

    def clip(self, t: float, x: Array, u: Array) -> tuple[Array, Array]:
        """Project state onto feasible half-space.

        If the position is on the wrong side of the plane (violated),
        project it back onto the plane boundary.

        Args:
            t: Time (unused)
            x: State vector
            u: Control vector (unchanged)

        Returns:
            Tuple (x_clipped, u) with position on or inside the half-space.
        """
        # Extract position components using JAX-idiomatic indexing
        pos = x[jnp.array(self.pos_indices)]

        # Compute violation: normal.dot(pos) - offset
        # Positive means violated (on wrong side of plane)
        violation = jnp.dot(self.normal, pos) - self.offset

        # Only correct if violated (violation > 0)
        correction = jnp.maximum(violation, 0.0)

        # Project back along normal direction
        # new_pos = pos - correction * normal
        new_x = x
        for i, idx in enumerate(self.pos_indices):
            new_x = new_x.at[idx].add(-correction * self.normal[i])

        return new_x, u
