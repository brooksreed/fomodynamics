"""Tests for CasADi model registry."""

# Registry doesn't require casadi, but we test it in the casadi test directory
# since it documents casadi model mappings
from fmd.simulator.casadi._registry import (
    MODEL_REGISTRY,
    get_model_info,
    list_models,
    list_models_with_smooth,
)


class TestModelRegistry:
    """Tests for model registry functions."""

    def test_registry_has_box1d_models(self):
        """Registry contains Box1D models from Phase 1."""
        assert "Box1D" in MODEL_REGISTRY
        assert "Box1DFriction" in MODEL_REGISTRY

    def test_registry_structure(self):
        """Registry entries have required keys."""
        required_keys = {"jax", "casadi_exact", "casadi_smooth", "equivalence_test", "notes"}
        for model_name, info in MODEL_REGISTRY.items():
            assert required_keys <= set(info.keys()), f"{model_name} missing keys"

    def test_get_model_info_existing(self):
        """get_model_info returns info for existing models."""
        info = get_model_info("Box1D")
        assert info is not None
        assert "jax" in info
        assert info["jax"] == "fmd.simulator.box_1d.Box1DJax"

    def test_get_model_info_nonexistent(self):
        """get_model_info returns None for unknown models."""
        info = get_model_info("NonexistentModel")
        assert info is None

    def test_list_models(self):
        """list_models returns all registered model names."""
        models = list_models()
        assert isinstance(models, list)
        assert "Box1D" in models
        assert "Box1DFriction" in models

    def test_list_models_with_smooth(self):
        """list_models_with_smooth returns only models with smooth variants."""
        models_with_smooth = list_models_with_smooth()
        assert isinstance(models_with_smooth, list)
        # Box1DFriction has a smooth variant
        assert "Box1DFriction" in models_with_smooth
        # Box1D does not have a smooth variant (no discontinuities)
        assert "Box1D" not in models_with_smooth

    def test_box1d_no_smooth_variant(self):
        """Box1D correctly has no smooth variant (no discontinuities)."""
        info = get_model_info("Box1D")
        assert info["casadi_smooth"] is None

    def test_box1d_friction_has_smooth_variant(self):
        """Box1DFriction has smooth variant for Coulomb friction."""
        info = get_model_info("Box1DFriction")
        assert info["casadi_smooth"] is not None
        assert "Smooth" in info["casadi_smooth"]
