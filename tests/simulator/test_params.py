"""Tests for parameter classes (validation, immutability, properties)."""

import pytest
import numpy as np
import attrs

from fmd.simulator.params import (
    Boat2DParams,
    RigidBody6DOFParams,
    SimplePendulumParams,
    CartpoleParams,
    PlanarQuadrotorParams,
    BOAT2D_TEST_DEFAULT,
    SIMPLE_MOTORBOAT,
    RIGIDBODY_TEST_SYMMETRIC,
    RIGIDBODY_TEST_ASYMMETRIC,
    PENDULUM_1M,
    SECONDS_PENDULUM,
    CARTPOLE_CLASSIC,
    CARTPOLE_HEAVY_POLE,
    PLANAR_QUAD_TEST_DEFAULT,
    PLANAR_QUAD_CRAZYFLIE,
)
from fmd.simulator.params.base import STANDARD_GRAVITY


# ============================================================================
# Boat2DParams Tests
# ============================================================================


class TestBoat2DParams:
    """Tests for Boat2DParams validation and behavior."""

    def test_valid_construction(self):
        """Valid params should construct without error."""
        params = Boat2DParams(
            mass=100.0,
            izz=50.0,
            drag_surge=10.0,
            drag_sway=20.0,
            drag_yaw=5.0,
        )
        assert params.mass == 100.0
        assert params.izz == 50.0
        assert params.drag_surge == 10.0
        assert params.drag_sway == 20.0
        assert params.drag_yaw == 5.0

    def test_negative_mass_raises(self):
        """Negative mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            Boat2DParams(
                mass=-1.0, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
            )

    def test_zero_mass_raises(self):
        """Zero mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            Boat2DParams(
                mass=0.0, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
            )

    def test_zero_inertia_raises(self):
        """Zero inertia should raise ValueError."""
        with pytest.raises(ValueError, match="izz must be positive"):
            Boat2DParams(
                mass=100.0, izz=0.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
            )

    def test_nan_mass_raises(self):
        """NaN mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be finite"):
            Boat2DParams(
                mass=np.nan, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
            )

    def test_inf_drag_raises(self):
        """Infinite drag should raise ValueError."""
        with pytest.raises(ValueError, match="drag_surge must be finite"):
            Boat2DParams(
                mass=100.0, izz=50.0, drag_surge=np.inf, drag_sway=20.0, drag_yaw=5.0
            )

    def test_immutability(self):
        """Params should be frozen (immutable)."""
        params = BOAT2D_TEST_DEFAULT
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            params.mass = 200.0

    def test_evolve_creates_new_instance(self):
        """attrs.evolve should create new instance."""
        params = BOAT2D_TEST_DEFAULT
        new_params = attrs.evolve(params, mass=200.0)

        assert new_params is not params
        assert new_params.mass == 200.0
        assert params.mass == 100.0  # Original unchanged

    def test_with_mass_helper(self):
        """with_mass should return new params with updated mass."""
        params = BOAT2D_TEST_DEFAULT
        new_params = params.with_mass(200.0)

        assert new_params.mass == 200.0
        assert params.mass == 100.0  # Original unchanged

    def test_with_inertia_helper(self):
        """with_inertia should return new params with updated inertia."""
        params = BOAT2D_TEST_DEFAULT
        new_params = params.with_inertia(100.0)

        assert new_params.izz == 100.0
        assert params.izz == 50.0

    def test_with_drag_helper(self):
        """with_drag should update specified drag values."""
        params = BOAT2D_TEST_DEFAULT

        # Update single value
        new_params = params.with_drag(surge=50.0)
        assert new_params.drag_surge == 50.0
        assert new_params.drag_sway == params.drag_sway  # Unchanged
        assert new_params.drag_yaw == params.drag_yaw  # Unchanged

        # Update multiple values
        new_params2 = params.with_drag(surge=50.0, sway=100.0, yaw=25.0)
        assert new_params2.drag_surge == 50.0
        assert new_params2.drag_sway == 100.0
        assert new_params2.drag_yaw == 25.0

    def test_equality(self):
        """Equal params should compare equal."""
        p1 = Boat2DParams(
            mass=100.0, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
        )
        p2 = Boat2DParams(
            mass=100.0, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
        )
        assert p1 == p2

    def test_inequality(self):
        """Different params should not be equal."""
        p1 = BOAT2D_TEST_DEFAULT
        p2 = p1.with_mass(200.0)
        assert p1 != p2

    def test_hash_consistency(self):
        """Equal params should have equal hashes."""
        p1 = BOAT2D_TEST_DEFAULT
        p2 = Boat2DParams(
            mass=100.0, izz=50.0, drag_surge=10.0, drag_sway=20.0, drag_yaw=5.0
        )
        assert hash(p1) == hash(p2)

    def test_hash_usable_in_set(self):
        """Params should be usable in sets."""
        params_set = {BOAT2D_TEST_DEFAULT, SIMPLE_MOTORBOAT}
        assert len(params_set) == 2
        assert BOAT2D_TEST_DEFAULT in params_set

    def test_analytical_properties(self):
        """Analytical properties should compute correctly."""
        params = BOAT2D_TEST_DEFAULT  # mass=100, drag_surge=10, drag_yaw=5, izz=50

        assert params.surge_time_constant() == 10.0  # 100 / 10
        assert params.yaw_time_constant() == 10.0  # 50 / 5
        assert params.steady_state_surge(100.0) == 10.0  # 100 / 10
        assert params.steady_state_yaw_rate(10.0) == 2.0  # 10 / 5


# ============================================================================
# RigidBody6DOFParams Tests
# ============================================================================


class TestRigidBody6DOFParams:
    """Tests for RigidBody6DOFParams validation and behavior."""

    def test_valid_construction(self):
        """Valid params should construct."""
        params = RigidBody6DOFParams(mass=10.0, inertia=[1.0, 2.0, 3.0])
        assert params.mass == 10.0
        np.testing.assert_array_equal(params.inertia, [1.0, 2.0, 3.0])

    def test_3x3_inertia_accepted(self):
        """Full 3x3 inertia tensor should be accepted."""
        inertia = np.diag([1.0, 2.0, 3.0])
        params = RigidBody6DOFParams(mass=10.0, inertia=inertia)
        np.testing.assert_array_equal(params.inertia, inertia)

    def test_inertia_matrix_property(self):
        """inertia_matrix should convert diagonal to 3x3."""
        params = RigidBody6DOFParams(mass=10.0, inertia=[1.0, 2.0, 3.0])
        expected = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(params.inertia_matrix, expected)


# ============================================================================
# SimplePendulumParams Tests
# ============================================================================


class TestSimplePendulumParams:
    """Tests for SimplePendulumParams validation and behavior."""

    def test_valid_construction(self):
        """Valid params should construct."""
        params = SimplePendulumParams(length=1.0)
        assert params.length == 1.0
        assert params.g == STANDARD_GRAVITY

    def test_custom_gravity(self):
        """Custom gravity should be accepted."""
        params = SimplePendulumParams(length=1.0, g=10.0)
        assert params.g == 10.0

    def test_negative_length_raises(self):
        """Negative length should raise ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            SimplePendulumParams(length=-1.0)

    def test_zero_length_raises(self):
        """Zero length should raise ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            SimplePendulumParams(length=0.0)

    def test_period_small_angle_property(self):
        """period_small_angle should compute correctly."""
        params = SimplePendulumParams(length=1.0, g=STANDARD_GRAVITY)
        expected = 2 * np.pi * np.sqrt(1.0 / STANDARD_GRAVITY)
        assert abs(params.period_small_angle - expected) < 1e-10

    def test_natural_frequency_property(self):
        """natural_frequency should compute correctly."""
        params = SimplePendulumParams(length=1.0, g=STANDARD_GRAVITY)
        expected = np.sqrt(STANDARD_GRAVITY / 1.0)
        assert abs(params.natural_frequency - expected) < 1e-10

    def test_with_length_helper(self):
        """with_length should return new params."""
        params = PENDULUM_1M
        new_params = params.with_length(2.0)
        assert new_params.length == 2.0
        assert params.length == 1.0

    def test_with_gravity_helper(self):
        """with_gravity should return new params."""
        params = PENDULUM_1M
        new_params = params.with_gravity(10.0)
        assert new_params.g == 10.0
        assert params.g == STANDARD_GRAVITY


# ============================================================================
# Preset Tests
# ============================================================================


class TestPresets:
    """Tests for preset configurations."""

    def test_boat2d_test_default_values(self):
        """BOAT2D_TEST_DEFAULT should have documented values."""
        assert BOAT2D_TEST_DEFAULT.mass == 100.0
        assert BOAT2D_TEST_DEFAULT.izz == 50.0
        assert BOAT2D_TEST_DEFAULT.drag_surge == 10.0
        assert BOAT2D_TEST_DEFAULT.drag_sway == 20.0
        assert BOAT2D_TEST_DEFAULT.drag_yaw == 5.0

    def test_simple_motorboat_values(self):
        """SIMPLE_MOTORBOAT should have expected values."""
        assert SIMPLE_MOTORBOAT.mass == 125.0
        assert SIMPLE_MOTORBOAT.izz == 100.0

    def test_rigidbody_symmetric_values(self):
        """RIGIDBODY_TEST_SYMMETRIC should have equal inertia."""
        np.testing.assert_array_equal(
            RIGIDBODY_TEST_SYMMETRIC.inertia, [1.0, 1.0, 1.0]
        )

    def test_rigidbody_asymmetric_values(self):
        """RIGIDBODY_TEST_ASYMMETRIC should have [1, 2, 3] inertia."""
        np.testing.assert_array_equal(
            RIGIDBODY_TEST_ASYMMETRIC.inertia, [1.0, 2.0, 3.0]
        )

    def test_pendulum_1m_values(self):
        """PENDULUM_1M should have length 1.0."""
        assert PENDULUM_1M.length == 1.0

    def test_seconds_pendulum_period(self):
        """SECONDS_PENDULUM should have period close to 2.0 s."""
        period = SECONDS_PENDULUM.period_small_angle
        assert abs(period - 2.0) < 0.01  # Within 1%


# ============================================================================
# Metadata Tests
# ============================================================================


class TestFieldMetadata:
    """Tests for field metadata accessibility."""

    def test_boat2d_mass_metadata(self):
        """Boat2DParams.mass should have unit and description metadata."""
        fields = attrs.fields(Boat2DParams)
        mass_field = fields.mass
        assert mass_field.metadata["unit"] == "kg"
        assert "mass" in mass_field.metadata["description"].lower()

    def test_boat2d_drag_metadata(self):
        """Boat2DParams drag fields should have metadata."""
        fields = attrs.fields(Boat2DParams)
        assert fields.drag_surge.metadata["unit"] == "kg/s"
        assert fields.drag_yaw.metadata["unit"] == "kg*m^2/s"

    def test_pendulum_length_metadata(self):
        """SimplePendulumParams.length should have metadata."""
        fields = attrs.fields(SimplePendulumParams)
        length_field = fields.length
        assert length_field.metadata["unit"] == "m"
        assert "length" in length_field.metadata["description"].lower()


# ============================================================================
# CartpoleParams Tests
# ============================================================================


class TestCartpoleParams:
    """Tests for CartpoleParams validation and behavior."""

    def test_valid_construction(self):
        """Valid params should construct without error."""
        params = CartpoleParams(
            mass_cart=1.0,
            mass_pole=0.1,
            pole_length=0.5,
        )
        assert params.mass_cart == 1.0
        assert params.mass_pole == 0.1
        assert params.pole_length == 0.5
        assert params.g == STANDARD_GRAVITY

    def test_custom_gravity(self):
        """Custom gravity should be accepted."""
        params = CartpoleParams(
            mass_cart=1.0, mass_pole=0.1, pole_length=0.5, g=10.0
        )
        assert params.g == 10.0

    def test_negative_mass_cart_raises(self):
        """Negative cart mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_cart must be positive"):
            CartpoleParams(mass_cart=-1.0, mass_pole=0.1, pole_length=0.5)

    def test_negative_mass_pole_raises(self):
        """Negative pole mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_pole must be positive"):
            CartpoleParams(mass_cart=1.0, mass_pole=-0.1, pole_length=0.5)

    def test_zero_mass_cart_raises(self):
        """Zero cart mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_cart must be positive"):
            CartpoleParams(mass_cart=0.0, mass_pole=0.1, pole_length=0.5)

    def test_zero_mass_pole_raises(self):
        """Zero pole mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_pole must be positive"):
            CartpoleParams(mass_cart=1.0, mass_pole=0.0, pole_length=0.5)

    def test_negative_pole_length_raises(self):
        """Negative pole length should raise ValueError."""
        with pytest.raises(ValueError, match="pole_length must be positive"):
            CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=-0.5)

    def test_zero_pole_length_raises(self):
        """Zero pole length should raise ValueError."""
        with pytest.raises(ValueError, match="pole_length must be positive"):
            CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=0.0)

    def test_nan_mass_cart_raises(self):
        """NaN cart mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_cart must be finite"):
            CartpoleParams(mass_cart=np.nan, mass_pole=0.1, pole_length=0.5)

    def test_nan_mass_pole_raises(self):
        """NaN pole mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_pole must be finite"):
            CartpoleParams(mass_cart=1.0, mass_pole=np.nan, pole_length=0.5)

    def test_nan_pole_length_raises(self):
        """NaN pole length should raise ValueError."""
        with pytest.raises(ValueError, match="pole_length must be finite"):
            CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=np.nan)

    def test_inf_mass_cart_raises(self):
        """Infinite cart mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_cart must be finite"):
            CartpoleParams(mass_cart=np.inf, mass_pole=0.1, pole_length=0.5)

    def test_inf_mass_pole_raises(self):
        """Infinite pole mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass_pole must be finite"):
            CartpoleParams(mass_cart=1.0, mass_pole=np.inf, pole_length=0.5)

    def test_inf_pole_length_raises(self):
        """Infinite pole length should raise ValueError."""
        with pytest.raises(ValueError, match="pole_length must be finite"):
            CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=np.inf)

    def test_immutability(self):
        """Params should be frozen (immutable)."""
        params = CARTPOLE_CLASSIC
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            params.mass_cart = 2.0

    def test_evolve_creates_new_instance(self):
        """attrs.evolve should create new instance."""
        params = CARTPOLE_CLASSIC
        new_params = attrs.evolve(params, mass_cart=2.0)

        assert new_params is not params
        assert new_params.mass_cart == 2.0
        assert params.mass_cart == 1.0  # Original unchanged

    def test_with_mass_cart_helper(self):
        """with_mass_cart should return new params with updated cart mass."""
        params = CARTPOLE_CLASSIC
        new_params = params.with_mass_cart(2.0)

        assert new_params.mass_cart == 2.0
        assert params.mass_cart == 1.0  # Original unchanged

    def test_with_mass_pole_helper(self):
        """with_mass_pole should return new params with updated pole mass."""
        params = CARTPOLE_CLASSIC
        new_params = params.with_mass_pole(0.2)

        assert new_params.mass_pole == 0.2
        assert params.mass_pole == 0.1  # Original unchanged

    def test_with_pole_length_helper(self):
        """with_pole_length should return new params with updated pole length."""
        params = CARTPOLE_CLASSIC
        new_params = params.with_pole_length(1.0)

        assert new_params.pole_length == 1.0
        assert params.pole_length == 0.5  # Original unchanged

    def test_with_masses_helper(self):
        """with_masses should update specified mass values."""
        params = CARTPOLE_CLASSIC

        # Update single value
        new_params = params.with_masses(cart=2.0)
        assert new_params.mass_cart == 2.0
        assert new_params.mass_pole == params.mass_pole  # Unchanged

        # Update multiple values
        new_params2 = params.with_masses(cart=2.0, pole=0.2)
        assert new_params2.mass_cart == 2.0
        assert new_params2.mass_pole == 0.2

    def test_equality(self):
        """Equal params should compare equal."""
        p1 = CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=0.5)
        p2 = CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=0.5)
        assert p1 == p2

    def test_inequality(self):
        """Different params should not be equal."""
        p1 = CARTPOLE_CLASSIC
        p2 = p1.with_mass_cart(2.0)
        assert p1 != p2

    def test_hash_consistency(self):
        """Equal params should have equal hashes."""
        p1 = CARTPOLE_CLASSIC
        p2 = CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=0.5)
        assert hash(p1) == hash(p2)

    def test_hash_usable_in_set(self):
        """Params should be usable in sets."""
        params_set = {CARTPOLE_CLASSIC, CARTPOLE_HEAVY_POLE}
        assert len(params_set) == 2
        assert CARTPOLE_CLASSIC in params_set

    def test_total_mass_property(self):
        """total_mass should compute correctly."""
        params = CARTPOLE_CLASSIC  # mass_cart=1.0, mass_pole=0.1
        assert params.total_mass == 1.1

    def test_natural_frequency_property(self):
        """natural_frequency should compute correctly."""
        params = CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=1.0)
        expected = np.sqrt(STANDARD_GRAVITY / 1.0)
        assert abs(params.natural_frequency - expected) < 1e-10

    def test_linearized_period_property(self):
        """linearized_period should compute correctly."""
        params = CartpoleParams(mass_cart=1.0, mass_pole=0.1, pole_length=1.0)
        expected = 2 * np.pi * np.sqrt(1.0 / STANDARD_GRAVITY)
        assert abs(params.linearized_period - expected) < 1e-10

    def test_mass_ratio_property(self):
        """mass_ratio should compute correctly."""
        params = CARTPOLE_CLASSIC  # mass_cart=1.0, mass_pole=0.1
        expected = 0.1 / 1.1
        assert abs(params.mass_ratio - expected) < 1e-10


# ============================================================================
# PlanarQuadrotorParams Tests
# ============================================================================


class TestPlanarQuadrotorParams:
    """Tests for PlanarQuadrotorParams validation and behavior."""

    def test_valid_construction(self):
        """Valid params should construct without error."""
        params = PlanarQuadrotorParams(
            mass=1.0,
            arm_length=0.25,
            inertia_pitch=0.01,
        )
        assert params.mass == 1.0
        assert params.arm_length == 0.25
        assert params.inertia_pitch == 0.01
        assert params.g == STANDARD_GRAVITY

    def test_custom_gravity(self):
        """Custom gravity should be accepted."""
        params = PlanarQuadrotorParams(
            mass=1.0, arm_length=0.25, inertia_pitch=0.01, g=10.0
        )
        assert params.g == 10.0

    def test_negative_mass_raises(self):
        """Negative mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            PlanarQuadrotorParams(mass=-1.0, arm_length=0.25, inertia_pitch=0.01)

    def test_zero_mass_raises(self):
        """Zero mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            PlanarQuadrotorParams(mass=0.0, arm_length=0.25, inertia_pitch=0.01)

    def test_negative_arm_length_raises(self):
        """Negative arm length should raise ValueError."""
        with pytest.raises(ValueError, match="arm_length must be positive"):
            PlanarQuadrotorParams(mass=1.0, arm_length=-0.25, inertia_pitch=0.01)

    def test_zero_arm_length_raises(self):
        """Zero arm length should raise ValueError."""
        with pytest.raises(ValueError, match="arm_length must be positive"):
            PlanarQuadrotorParams(mass=1.0, arm_length=0.0, inertia_pitch=0.01)

    def test_negative_inertia_raises(self):
        """Negative inertia should raise ValueError."""
        with pytest.raises(ValueError, match="inertia_pitch must be positive"):
            PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=-0.01)

    def test_zero_inertia_raises(self):
        """Zero inertia should raise ValueError."""
        with pytest.raises(ValueError, match="inertia_pitch must be positive"):
            PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=0.0)

    def test_nan_mass_raises(self):
        """NaN mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be finite"):
            PlanarQuadrotorParams(mass=np.nan, arm_length=0.25, inertia_pitch=0.01)

    def test_nan_arm_length_raises(self):
        """NaN arm length should raise ValueError."""
        with pytest.raises(ValueError, match="arm_length must be finite"):
            PlanarQuadrotorParams(mass=1.0, arm_length=np.nan, inertia_pitch=0.01)

    def test_nan_inertia_raises(self):
        """NaN inertia should raise ValueError."""
        with pytest.raises(ValueError, match="inertia_pitch must be finite"):
            PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=np.nan)

    def test_inf_mass_raises(self):
        """Infinite mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be finite"):
            PlanarQuadrotorParams(mass=np.inf, arm_length=0.25, inertia_pitch=0.01)

    def test_inf_arm_length_raises(self):
        """Infinite arm length should raise ValueError."""
        with pytest.raises(ValueError, match="arm_length must be finite"):
            PlanarQuadrotorParams(mass=1.0, arm_length=np.inf, inertia_pitch=0.01)

    def test_inf_inertia_raises(self):
        """Infinite inertia should raise ValueError."""
        with pytest.raises(ValueError, match="inertia_pitch must be finite"):
            PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=np.inf)

    def test_immutability(self):
        """Params should be frozen (immutable)."""
        params = PLANAR_QUAD_TEST_DEFAULT
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            params.mass = 2.0

    def test_evolve_creates_new_instance(self):
        """attrs.evolve should create new instance."""
        params = PLANAR_QUAD_TEST_DEFAULT
        new_params = attrs.evolve(params, mass=2.0)

        assert new_params is not params
        assert new_params.mass == 2.0
        assert params.mass == 1.0  # Original unchanged

    def test_with_mass_helper(self):
        """with_mass should return new params with updated mass."""
        params = PLANAR_QUAD_TEST_DEFAULT
        new_params = params.with_mass(2.0)

        assert new_params.mass == 2.0
        assert params.mass == 1.0  # Original unchanged

    def test_with_arm_length_helper(self):
        """with_arm_length should return new params with updated arm length."""
        params = PLANAR_QUAD_TEST_DEFAULT
        new_params = params.with_arm_length(0.5)

        assert new_params.arm_length == 0.5
        assert params.arm_length == 0.25  # Original unchanged

    def test_with_inertia_helper(self):
        """with_inertia should return new params with updated inertia."""
        params = PLANAR_QUAD_TEST_DEFAULT
        new_params = params.with_inertia(0.02)

        assert new_params.inertia_pitch == 0.02
        assert params.inertia_pitch == 0.01  # Original unchanged

    def test_equality(self):
        """Equal params should compare equal."""
        p1 = PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=0.01)
        p2 = PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=0.01)
        assert p1 == p2

    def test_inequality(self):
        """Different params should not be equal."""
        p1 = PLANAR_QUAD_TEST_DEFAULT
        p2 = p1.with_mass(2.0)
        assert p1 != p2

    def test_hash_consistency(self):
        """Equal params should have equal hashes."""
        p1 = PLANAR_QUAD_TEST_DEFAULT
        p2 = PlanarQuadrotorParams(mass=1.0, arm_length=0.25, inertia_pitch=0.01)
        assert hash(p1) == hash(p2)

    def test_hash_usable_in_set(self):
        """Params should be usable in sets."""
        params_set = {PLANAR_QUAD_TEST_DEFAULT, PLANAR_QUAD_CRAZYFLIE}
        assert len(params_set) == 2
        assert PLANAR_QUAD_TEST_DEFAULT in params_set

    def test_hover_thrust_total_property(self):
        """hover_thrust_total should compute correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT  # mass=1.0
        expected = 1.0 * STANDARD_GRAVITY
        assert abs(params.hover_thrust_total - expected) < 1e-10

    def test_hover_thrust_per_rotor_property(self):
        """hover_thrust_per_rotor should compute correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT  # mass=1.0
        expected = (1.0 * STANDARD_GRAVITY) / 2.0
        assert abs(params.hover_thrust_per_rotor - expected) < 1e-10

    def test_moment_arm_property(self):
        """moment_arm should equal arm_length."""
        params = PLANAR_QUAD_TEST_DEFAULT
        assert params.moment_arm == params.arm_length

    def test_moment_from_thrust_difference(self):
        """moment_from_thrust_difference should compute correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT  # arm_length=0.25
        moment = params.moment_from_thrust_difference(4.0)  # 4 N difference
        expected = 4.0 * 0.25
        assert abs(moment - expected) < 1e-10

    def test_thrust_difference_for_moment(self):
        """thrust_difference_for_moment should compute correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT  # arm_length=0.25
        delta_T = params.thrust_difference_for_moment(1.0)  # 1 N*m moment
        expected = 1.0 / 0.25
        assert abs(delta_T - expected) < 1e-10

    def test_angular_acceleration_for_moment(self):
        """angular_acceleration_for_moment should compute correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT  # inertia_pitch=0.01
        alpha = params.angular_acceleration_for_moment(0.1)  # 0.1 N*m moment
        expected = 0.1 / 0.01
        assert abs(alpha - expected) < 1e-10

    def test_scaled_method(self):
        """scaled should scale parameters correctly."""
        params = PLANAR_QUAD_TEST_DEFAULT
        scaled = params.scaled(2.0)  # 2x scale

        # Length scales linearly
        assert abs(scaled.arm_length - params.arm_length * 2.0) < 1e-10
        # Mass scales cubically
        assert abs(scaled.mass - params.mass * 8.0) < 1e-10
        # Inertia scales to 5th power
        assert abs(scaled.inertia_pitch - params.inertia_pitch * 32.0) < 1e-10


