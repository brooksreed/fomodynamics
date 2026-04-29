"""Inertia estimation for composite bodies using parallel axis theorem.

Provides a reusable function for estimating moments of inertia from
component masses and positions, primarily for Moth sailboat geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ComponentSpec:
    """Specification for a single mass component in inertia estimation.

    Attributes:
        name: Human-readable component name.
        mass: Component mass in kg.
        position: CG position [x, y, z] in hull-datum coordinates (m).
        local_inertia: Optional [Ixx, Iyy, Izz] about component's own CG (kg*m^2).
            Defaults to zero (point mass assumption).
    """
    name: str
    mass: float
    position: NDArray
    local_inertia: NDArray | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.local_inertia is not None:
            self.local_inertia = np.asarray(self.local_inertia, dtype=float)


def estimate_composite_inertia(
    components: list[ComponentSpec],
    reference_cg: NDArray | None = None,
) -> tuple[NDArray, NDArray, float]:
    """Estimate composite inertia from components using parallel axis theorem.

    Computes the composite CG and moments of inertia about either the
    computed composite CG or a specified reference point.

    Args:
        components: List of ComponentSpec defining each mass element.
        reference_cg: If provided, compute inertia about this point instead
            of the mass-weighted CG. Useful when the CG is prescribed by
            engineering judgment rather than purely mass-weighted.

    Returns:
        Tuple of (inertia, cg_position, total_mass) where:
            inertia: [Ixx, Iyy, Izz] about reference point (kg*m^2)
            cg_position: Mass-weighted composite CG in hull-datum (m)
            total_mass: Sum of all component masses (kg)
    """
    total_mass = sum(c.mass for c in components)
    if total_mass <= 0:
        raise ValueError("Total mass must be positive")

    # Composite CG (mass-weighted average position)
    cg = sum(c.mass * c.position for c in components) / total_mass

    # Use reference CG for inertia calculation if specified
    ref = reference_cg if reference_cg is not None else cg

    # Parallel axis theorem: I_total = sum(I_local_i + m_i * d_i^2)
    # where d_i is distance from component CG to reference point
    ixx = 0.0
    iyy = 0.0
    izz = 0.0

    for c in components:
        d = c.position - ref
        # Parallel axis: I_axis = I_local + m * (sum of squared distances
        # in the other two axes)
        # Ixx: rotation about x, distances in y and z
        # Iyy: rotation about y, distances in x and z
        # Izz: rotation about z, distances in x and y
        ixx += c.mass * (d[1]**2 + d[2]**2)
        iyy += c.mass * (d[0]**2 + d[2]**2)
        izz += c.mass * (d[0]**2 + d[1]**2)

        if c.local_inertia is not None:
            ixx += c.local_inertia[0]
            iyy += c.local_inertia[1]
            izz += c.local_inertia[2]

    return np.array([ixx, iyy, izz]), cg, total_mass


def estimate_moth_inertia() -> tuple[NDArray, NDArray, float, list[ComponentSpec]]:
    """Estimate Bieker Moth V3 non-sailor inertia from component breakdown.

    Components are positioned in hull-datum coordinates (x aft from bow,
    z up from hull bottom). The returned CG should be used to set
    hull_cg_above_bottom and hull_cg_from_bow in the preset. A sync test
    in test_inertia.py verifies consistency. If the CG changes frequently,
    consider making hull_cg_* computed properties instead.

    Returns:
        Tuple of (inertia, cg_position, total_mass, components) where:
            inertia: [Ixx, Iyy, Izz] about estimated CG (kg*m^2)
            cg_position: Composite CG in hull-datum (m)
            total_mass: Sum of component masses (kg)
            components: The component list used
    """
    components = [
        # Hull shell: ~15 kg, CG roughly at mid-length, at mid-hull-depth
        ComponentSpec(
            name="Hull shell",
            mass=15.0,
            position=[1.7, 0.0, 0.22],
            # Thin shell: Iyy significant (long), Izz similar, Ixx small (narrow)
            local_inertia=[0.1, 3.0, 3.0],
        ),
        # Mast + sail: ~12 kg, at mast position, CG halfway up
        ComponentSpec(
            name="Mast + sail",
            mass=12.0,
            position=[1.2, 0.0, 3.0],
            # Long vertical rod: Ixx and Izz from length, Iyy similar
            local_inertia=[4.0, 4.0, 0.05],
        ),
        # Main foil + strut: ~8 kg, at main foil position
        ComponentSpec(
            name="Main foil + strut",
            mass=8.0,
            position=[1.6, 0.0, -0.5],
            # Strut is vertical rod, foil is horizontal span
            local_inertia=[0.6, 0.3, 0.6],
        ),
        # Rudder + strut: ~5 kg, at rudder position
        ComponentSpec(
            name="Rudder + strut",
            mass=5.0,
            position=[3.855, 0.0, -0.47],
            local_inertia=[0.2, 0.1, 0.2],
        ),
        # Rigging + misc (lines, fittings, trampoline, wing racks): ~10 kg
        # Distributed along hull, CG near boat CG
        ComponentSpec(
            name="Rigging + misc",
            mass=10.0,
            position=[1.9, 0.0, 0.8],
            # Wing racks give significant Ixx (wide span)
            local_inertia=[1.5, 0.5, 1.5],
        ),
    ]

    inertia, cg, total_mass = estimate_composite_inertia(components)
    return inertia, cg, total_mass, components
