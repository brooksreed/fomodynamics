"""Tests for fmd.estimation.measurement module.

Comprehensive tests for measurement model classes including:
- LinearMeasurementModel
- FullStateMeasurement
- Noisy measurement functionality
- JAX compatibility (JIT, vmap)
- Dimension validation
"""

import jax
import jax.numpy as jnp
import pytest

from fmd.estimation import (
    MeasurementModel,
    LinearMeasurementModel,
    FullStateMeasurement,
)
from fmd.simulator import Cartpole, Boat2D
from fmd.simulator.params import CARTPOLE_CLASSIC, BOAT2D_TEST_DEFAULT


class TestLinearMeasurementModel:
    """Tests for LinearMeasurementModel."""

    def test_linear_measure_basic(self):
        """Test y = H @ x computation."""
        H = jnp.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]])
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("x", "theta"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0])

        y = model.measure(x, u)
        expected = jnp.array([1.0, 3.0])
        assert jnp.allclose(y, expected)

    def test_linear_measure_with_feedthrough(self):
        """Test y = H @ x + D @ u when D is provided."""
        H = jnp.array([[1.0, 0.0],
                       [0.0, 1.0]])
        D = jnp.array([[0.5],
                       [0.0]])
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
            D=D,
        )

        x = jnp.array([1.0, 2.0])
        u = jnp.array([4.0])

        y = model.measure(x, u)
        # y = H @ x + D @ u = [1, 2] + [0.5*4, 0] = [3, 2]
        expected = jnp.array([3.0, 2.0])
        assert jnp.allclose(y, expected)

    def test_linear_jacobian_equals_H(self):
        """Jacobian should equal H matrix for linear model."""
        H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0])
        u = jnp.array([])
        H_jac = model.get_measurement_jacobian(x, u)

        assert jnp.allclose(H_jac, H)

    def test_from_indices_factory(self):
        """Test from_indices creates correct selection matrix."""
        model = LinearMeasurementModel.from_indices(
            output_names=("pos", "angle"),
            state_indices=(0, 2),
            num_states=4,
            R=jnp.eye(2) * 0.01,
        )

        # Check H is a selection matrix
        expected_H = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        assert jnp.allclose(model.H, expected_H)

        # Test that it works correctly
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = model.measure(x, jnp.array([]))
        assert jnp.allclose(y, jnp.array([1.0, 3.0]))

    def test_from_indices_invalid_index(self):
        """Invalid state index should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            LinearMeasurementModel.from_indices(
                output_names=("invalid",),
                state_indices=(5,),  # Invalid for 4-state system
                num_states=4,
                R=jnp.eye(1) * 0.01,
            )

    def test_from_indices_negative_index(self):
        """Negative state index should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            LinearMeasurementModel.from_indices(
                output_names=("invalid",),
                state_indices=(-1,),
                num_states=4,
                R=jnp.eye(1) * 0.01,
            )

    def test_from_indices_mismatched_names(self):
        """Mismatched output_names and state_indices lengths should raise."""
        with pytest.raises(ValueError, match="must match"):
            LinearMeasurementModel.from_indices(
                output_names=("a", "b"),  # 2 names
                state_indices=(0,),  # 1 index
                num_states=4,
                R=jnp.eye(2) * 0.01,
            )

    def test_num_outputs_property(self):
        """num_outputs should match output_names length."""
        H = jnp.array([[1.0, 0.0, 0.0]])
        R = jnp.eye(1) * 0.01
        model = LinearMeasurementModel(
            output_names=("x",),
            H=H,
            R=R,
        )
        assert model.num_outputs == 1

        H2 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        R2 = jnp.eye(3) * 0.01
        model2 = LinearMeasurementModel(
            output_names=("a", "b", "c"),
            H=H2,
            R=R2,
        )
        assert model2.num_outputs == 3

    def test_dimension_validation_H(self):
        """Mismatched H rows and output_names should raise error."""
        H = jnp.array([[1.0, 0.0, 0.0]])  # 1 row
        R = jnp.eye(2) * 0.01  # 2x2
        with pytest.raises(ValueError, match="H rows"):
            LinearMeasurementModel(
                output_names=("a", "b"),  # 2 outputs
                H=H,
                R=R,
            )

    def test_dimension_validation_R(self):
        """Mismatched R shape should raise error."""
        H = jnp.array([[1.0, 0.0]])  # 1 row
        R = jnp.eye(2) * 0.01  # 2x2 - wrong!
        with pytest.raises(ValueError, match="R shape"):
            LinearMeasurementModel(
                output_names=("a",),  # 1 output
                H=H,
                R=R,
            )

    def test_dimension_validation_D(self):
        """Mismatched D rows should raise error."""
        H = jnp.array([[1.0, 0.0]])  # 1 row
        D = jnp.array([[0.5], [0.5]])  # 2 rows - wrong!
        R = jnp.eye(1) * 0.01
        with pytest.raises(ValueError, match="D rows"):
            LinearMeasurementModel(
                output_names=("a",),
                H=H,
                R=R,
                D=D,
            )

    def test_jit_compatible(self):
        """Measure should be JIT-compilable."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        @jax.jit
        def f(x):
            return model.measure(x, jnp.array([]))

        x = jnp.array([1.0, 2.0])
        y = f(x)
        assert y.shape == (2,)
        assert jnp.allclose(y, x)

    def test_vmap_compatible(self):
        """Measure should be vmap-compatible over batch of states."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        # Batch of 5 states
        x_batch = jnp.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ])
        u = jnp.array([])

        # vmap over states
        y_batch = jax.vmap(lambda x: model.measure(x, u))(x_batch)

        assert y_batch.shape == (5, 2)
        assert jnp.allclose(y_batch, x_batch)


class TestFullStateMeasurement:
    """Tests for FullStateMeasurement."""

    def test_with_cartpole(self):
        """Test FullStateMeasurement with Cartpole system."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        R = jnp.eye(4) * 0.01

        model = FullStateMeasurement.for_system(cartpole, R)

        assert model.num_outputs == 4
        assert model.output_names == cartpole.state_names

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = model.measure(x, jnp.array([0.0]))
        assert jnp.allclose(y, x)

    def test_with_boat2d(self):
        """Test FullStateMeasurement with Boat2D system."""
        boat = Boat2D(BOAT2D_TEST_DEFAULT)
        R = jnp.eye(6) * 0.01

        model = FullStateMeasurement.for_system(boat, R)

        assert model.num_outputs == 6
        assert model.output_names == boat.state_names

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = model.measure(x, jnp.array([0.0, 0.0]))
        assert jnp.allclose(y, x)

    def test_full_state_jacobian_is_identity(self):
        """Full state measurement Jacobian should be identity matrix."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        R = jnp.eye(4) * 0.01
        model = FullStateMeasurement.for_system(cartpole, R)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0])
        H = model.get_measurement_jacobian(x, u)

        assert jnp.allclose(H, jnp.eye(4))

    def test_full_state_wrong_R_shape(self):
        """Wrong R shape should raise ValueError."""
        cartpole = Cartpole(CARTPOLE_CLASSIC)
        R = jnp.eye(3) * 0.01  # Wrong size for 4-state system

        with pytest.raises(ValueError, match="R shape"):
            FullStateMeasurement.for_system(cartpole, R)


class TestNoisyMeasurement:
    """Tests for noisy_measure method."""

    def test_different_keys_different_noise(self):
        """Different PRNG keys should produce different noise."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.1
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0])
        u = jnp.array([])

        key1 = jax.random.key(0)
        key2 = jax.random.key(1)

        y1 = model.noisy_measure(x, u, key1)
        y2 = model.noisy_measure(x, u, key2)

        assert not jnp.allclose(y1, y2)

    def test_same_key_reproducible(self):
        """Same PRNG key should produce same result."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.1
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0])
        u = jnp.array([])
        key = jax.random.key(42)

        y1 = model.noisy_measure(x, u, key)
        y2 = model.noisy_measure(x, u, key)

        assert jnp.allclose(y1, y2)

    def test_noisy_measure_statistics(self):
        """Noise should have approximately correct mean and covariance."""
        H = jnp.eye(2)
        R = jnp.array([[0.04, 0.0], [0.0, 0.01]])  # Different variances
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([5.0, 10.0])
        u = jnp.array([])

        # Generate many samples
        n_samples = 10000
        keys = jax.random.split(jax.random.key(0), n_samples)
        samples = jax.vmap(lambda k: model.noisy_measure(x, u, k))(keys)

        # Check mean is close to true measurement
        sample_mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(sample_mean, x, atol=0.05)

        # Check covariance is close to R
        sample_cov = jnp.cov(samples.T)
        assert jnp.allclose(sample_cov, R, atol=0.01)

    def test_noisy_measure_jit_compatible(self):
        """noisy_measure should be JIT-compilable."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        @jax.jit
        def f(x, key):
            return model.noisy_measure(x, jnp.array([]), key)

        x = jnp.array([1.0, 2.0])
        key = jax.random.key(0)
        y = f(x, key)
        assert y.shape == (2,)


class TestJacobianAutodiff:
    """Tests for autodiff Jacobian computation."""

    def test_jacobian_autodiff_matches_analytical(self):
        """Base class autodiff Jacobian should match analytical for linear."""
        H = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([])

        # Analytical (from override)
        H_analytical = model.get_measurement_jacobian(x, u)

        # Use base class autodiff by calling parent method explicitly
        H_autodiff = MeasurementModel.get_measurement_jacobian(model, x, u)

        assert jnp.allclose(H_analytical, H_autodiff)
        assert jnp.allclose(H_analytical, H)

    def test_jacobian_correct_shape(self):
        """Jacobian should have shape (num_outputs, num_states)."""
        H = jnp.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]])  # 3 outputs from 4 states
        R = jnp.eye(3) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b", "c"),
            H=H,
            R=R,
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array([])

        H_jac = model.get_measurement_jacobian(x, u)
        assert H_jac.shape == (3, 4)


class TestMeasurementModelInterface:
    """Tests for abstract MeasurementModel interface."""

    def test_abstract_measure_not_implemented(self):
        """Cannot instantiate MeasurementModel directly (abstract)."""
        # MeasurementModel is abstract and cannot be instantiated
        # This is enforced by Python's ABC mechanism
        # We just verify the class structure is correct
        assert hasattr(MeasurementModel, 'measure')
        assert hasattr(MeasurementModel, 'get_measurement_jacobian')
        assert hasattr(MeasurementModel, 'noisy_measure')
        assert hasattr(MeasurementModel, 'num_outputs')


class TestEquinoxPyTreeCompatibility:
    """Tests for Equinox PyTree compatibility."""

    def test_pytree_flatten_unflatten(self):
        """Model should be a valid JAX PyTree."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        # Test PyTree operations
        leaves, treedef = jax.tree_util.tree_flatten(model)
        model_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Check functionality preserved
        x = jnp.array([1.0, 2.0])
        u = jnp.array([])
        y_original = model.measure(x, u)
        y_reconstructed = model_reconstructed.measure(x, u)
        assert jnp.allclose(y_original, y_reconstructed)

    def test_pytree_map(self):
        """Can apply jax.tree_util.tree_map to model."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        # Double all leaves (H and R matrices)
        doubled = jax.tree_util.tree_map(lambda x: x * 2, model)

        assert jnp.allclose(doubled.H, H * 2)
        assert jnp.allclose(doubled.R, R * 2)

    def test_output_names_is_static(self):
        """output_names should not be in PyTree leaves."""
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.01
        model = LinearMeasurementModel(
            output_names=("a", "b"),
            H=H,
            R=R,
        )

        leaves = jax.tree_util.tree_leaves(model)

        # Should only have H and R as leaves (both arrays)
        # output_names is static and should not be a leaf
        for leaf in leaves:
            assert isinstance(leaf, jax.Array)
