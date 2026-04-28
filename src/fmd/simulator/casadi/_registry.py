"""Registry of JAX <-> CasADi model pairs.

This documents which models have CasADi implementations and their status.
Update when adding new CasADi models or modifying dynamics.

This is a documentation/maintenance tool, not enforced in CI.
"""

MODEL_REGISTRY = {
    "Box1D": {
        "jax": "fmd.simulator.box_1d.Box1DJax",
        "casadi_exact": "fmd.simulator.casadi.box_1d.Box1DCasadiExact",
        "casadi_smooth": None,  # No smooth variant needed (no discontinuities)
        "equivalence_test": "tests/simulator/casadi/test_equivalence.py::TestBox1DEquivalence",
        "notes": "Hello world model for architecture validation",
    },
    "Box1DFriction": {
        "jax": "fmd.simulator.box_1d.Box1DFrictionJax",
        "casadi_exact": "fmd.simulator.casadi.box_1d.Box1DFrictionCasadiExact",
        "casadi_smooth": "fmd.simulator.casadi.box_1d.Box1DFrictionCasadiSmooth",
        "equivalence_test": "tests/simulator/casadi/test_equivalence.py::TestBox1DFrictionEquivalence",
        "notes": "Smooth variant uses tanh(v/eps) for Coulomb friction",
    },
    "Cartpole": {
        "jax": "fmd.simulator.cartpole.CartpoleJax",
        "casadi_exact": "fmd.simulator.casadi.cartpole.CartpoleCasadiExact",
        "casadi_smooth": None,  # Cartpole has no discontinuities
        "equivalence_test": "tests/simulator/casadi/test_equivalence.py::TestCartpoleEquivalence",
        "notes": "Classic control benchmark - Barto/Sutton/Anderson equations",
    },
    "PlanarQuadrotor": {
        "jax": "fmd.simulator.planar_quadrotor.PlanarQuadrotorJax",
        "casadi_exact": "fmd.simulator.casadi.planar_quadrotor.PlanarQuadrotorCasadiExact",
        "casadi_smooth": None,  # No discontinuities
        "equivalence_test": "tests/simulator/casadi/test_equivalence.py::TestPlanarQuadrotorEquivalence",
        "notes": "2D quadrotor with thrust control",
    },
    "Boat2D": {
        "jax": "fmd.simulator.boat_2d.Boat2DJax",
        "casadi_exact": "fmd.simulator.casadi.boat_2d.Boat2DCasadiExact",
        "casadi_smooth": None,  # No discontinuities
        "equivalence_test": "tests/simulator/casadi/test_equivalence.py::TestBoat2DEquivalence",
        "notes": "3-DOF boat with Coriolis coupling, heading wrapped to [-pi, pi]",
    },
    # Additional vehicle mirrors are available in optional companion packages.
}


def get_model_info(model_name: str) -> dict | None:
    """Get registry info for a model by name.

    Args:
        model_name: Model name (e.g., "Box1D", "Cartpole")

    Returns:
        Registry dict or None if not found
    """
    return MODEL_REGISTRY.get(model_name)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())


def list_models_with_smooth() -> list[str]:
    """List models that have smooth CasADi variants."""
    return [
        name
        for name, info in MODEL_REGISTRY.items()
        if info.get("casadi_smooth") is not None
    ]
