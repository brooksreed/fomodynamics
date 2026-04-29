"""RigidBody6DOF parameter class.

Immutable, validated parameters for the RigidBody6DOF dynamics model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import attrs

from fmd.simulator.params.base import (
    is_finite,
    is_finite_array,
    is_valid_inertia,
    positive,
    to_float_array,
)


@attrs.define(frozen=True, slots=True, eq=False)
class RigidBody6DOFParams:
    """Immutable parameters for RigidBody6DOF dynamics.

    RigidBody6DOF is a base class for 6-DOF rigid body dynamics using
    quaternion attitude representation in NED frame. It uses a
    force-accumulator pattern where external components provide forces
    and moments.

    Inertia accepts either a 3-element diagonal [Ixx, Iyy, Izz] or a full 3x3
    inertia tensor. Full tensors must be symmetric and positive semi-definite.

    Note:
        This params class contains only the core physical properties.
        Force/moment components (like Gravity) are configured separately
        when constructing the RigidBody6DOF model.

    Attributes:
        mass: Body mass (kg). Must be positive.
        inertia: Moments of inertia. Either [Ixx, Iyy, Izz] (kg*m^2) diagonal
            or full 3x3 tensor. All diagonal elements must be positive.

    Example:
        >>> params = RigidBody6DOFParams(mass=10.0, inertia=[1.0, 2.0, 3.0])
    """

    mass: float = attrs.field(
        validator=[is_finite, positive],
        metadata={"unit": "kg", "description": "Body mass"},
    )
    inertia: NDArray = attrs.field(
        converter=to_float_array,
        validator=[is_finite_array, is_valid_inertia],
        metadata={
            "unit": "kg*m^2",
            "description": "Moments of inertia [Ixx,Iyy,Izz] or 3x3 tensor",
        },
    )

    @property
    def inertia_matrix(self) -> NDArray:
        """Full 3x3 inertia matrix.

        Converts diagonal [Ixx, Iyy, Izz] to diag(I) if needed.
        """
        if self.inertia.shape == (3,):
            return np.diag(self.inertia)
        return self.inertia

    @property
    def inertia_inverse(self) -> NDArray:
        """Inverse of the inertia matrix."""
        return np.linalg.inv(self.inertia_matrix)

    def with_mass(self, mass: float) -> RigidBody6DOFParams:
        """Return new params with updated mass.

        Args:
            mass: New mass value (kg).

        Returns:
            New RigidBody6DOFParams instance with updated mass.
        """
        return attrs.evolve(self, mass=mass)

    def with_inertia(self, inertia: NDArray | list) -> RigidBody6DOFParams:
        """Return new params with updated inertia.

        Args:
            inertia: New inertia values [Ixx, Iyy, Izz] or 3x3 tensor.

        Returns:
            New RigidBody6DOFParams instance with updated inertia.
        """
        return attrs.evolve(self, inertia=np.asarray(inertia))

    def __eq__(self, other: object) -> bool:
        """Compare equality with proper numpy array handling."""
        if not isinstance(other, RigidBody6DOFParams):
            return NotImplemented
        return (
            self.mass == other.mass
            and np.array_equal(self.inertia, other.inertia)
        )

    def __hash__(self) -> int:
        """Hash based on all fields."""
        return hash((self.mass, self.inertia.tobytes()))
