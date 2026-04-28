"""JAX force and moment components for rigid body simulation.

Components implement the JaxForceElement interface as Equinox modules,
computing forces and moments acting on a rigid body in the body frame.

All components are JIT-compatible and work with JAX autodiff.

Includes: generic base + gravity, plus the moth-specific
components (foil, sail, hull-drag, strut-drag, wand).

Example:
    from fmd.simulator.components import JaxGravity, JaxForceElement

    class MyForce(JaxForceElement):
        my_param: float

        def compute(self, t, state, control):
            force = jnp.array([0.0, 0.0, 0.0])
            moment = jnp.array([0.0, 0.0, 0.0])
            return force, moment
"""

# Generic / base components
from fmd.simulator.components.base import JaxForceElement
from fmd.simulator.components.gravity import JaxGravity

# Moth components
from fmd.simulator.components.moth_forces import (
    MothMainFoil,
    MothRudderElevator,
    MothSailForce,
    MothHullDrag,
    MothStrutDrag,
    create_moth_components,
    compute_foil_ned_depth,
    compute_leeward_tip_depth,
    compute_tip_at_surface_pos_d,
)
from fmd.simulator.components.moth_wand import (
    WandLinkage,
    WandLinkageState,
    wand_angle_from_state,
    wand_angle_from_state_waves,
    create_wand_linkage,
    gearing_ratio_from_rod,
)

__all__ = [
    "JaxForceElement",
    "JaxGravity",
    "MothMainFoil",
    "MothRudderElevator",
    "MothSailForce",
    "MothHullDrag",
    "MothStrutDrag",
    "create_moth_components",
    "compute_foil_ned_depth",
    "compute_leeward_tip_depth",
    "compute_tip_at_surface_pos_d",
    "WandLinkage",
    "WandLinkageState",
    "wand_angle_from_state",
    "wand_angle_from_state_waves",
    "create_wand_linkage",
    "gearing_ratio_from_rod",
]
