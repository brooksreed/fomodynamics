"""JAX PyTree registration for parameter classes.

This module registers all parameter classes as JAX PyTrees, enabling:
- vmap over parameter variations (domain randomization)
- Gradient computation through parameters
- JIT compilation of functions using parameters

Usage:
    # Import to register PyTrees (side effect import)
    import fmd.simulator.params._pytree

    # Now parameters work with JAX transformations
    import jax
    import jax.numpy as jnp
    from fmd.simulator.params import Boat2DParams, BOAT2D_TEST_DEFAULT

    def simulate_with_params(params):
        boat = Boat2D(params)
        return simulate(boat, ...)

    # vmap over parameter variations
    params_batch = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        [params1, params2, params3]
    )
    results = jax.vmap(simulate_with_params)(params_batch)

Note:
    This module has no effect if JAX is not installed. It safely handles
    ImportError when JAX is unavailable.
"""

from __future__ import annotations

try:
    import jax
    from jax import tree_util

    HAS_JAX = True
except Exception:
    HAS_JAX = False


def _register_attrs_pytree(cls: type) -> None:
    """Register an attrs class as a JAX PyTree.

    For frozen attrs classes, the PyTree structure is:
    - children: tuple of all field values (the "leaves")
    - aux_data: tuple of field names (static metadata)

    Args:
        cls: The attrs class to register.
    """
    import attrs

    field_names = tuple(f.name for f in attrs.fields(cls))

    def flatten(obj):
        """Flatten object to (children, aux_data)."""
        children = tuple(getattr(obj, name) for name in field_names)
        return children, field_names

    def unflatten(aux_data, children):
        """Reconstruct object from (aux_data, children)."""
        return cls(**dict(zip(aux_data, children)))

    # JAX raises on duplicate registration (e.g., if this module is reloaded).
    # Make registration idempotent so importing this module is safe.
    try:
        tree_util.register_pytree_node(cls, flatten, unflatten)
    except ValueError as e:
        msg = str(e)
        if "Duplicate" in msg and "PyTree" in msg:
            return
        raise


# Register all parameter classes if JAX is available.
if HAS_JAX:
    from fmd.simulator.params.boat_2d import Boat2DParams
    from fmd.simulator.params.rigid_body import RigidBody6DOFParams
    from fmd.simulator.params.pendulum import SimplePendulumParams
    from fmd.simulator.params.cartpole import CartpoleParams
    from fmd.simulator.params.planar_quadrotor import PlanarQuadrotorParams
    from fmd.simulator.params.moth import MothParams
    from fmd.simulator.params.wave import WaveParams
    from fmd.simulator.params.box_1d import Box1DParams

    _register_attrs_pytree(Boat2DParams)
    _register_attrs_pytree(RigidBody6DOFParams)
    _register_attrs_pytree(SimplePendulumParams)
    _register_attrs_pytree(CartpoleParams)
    _register_attrs_pytree(PlanarQuadrotorParams)
    _register_attrs_pytree(MothParams)
    _register_attrs_pytree(WaveParams)
    _register_attrs_pytree(Box1DParams)