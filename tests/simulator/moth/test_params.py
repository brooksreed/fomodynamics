"""Tests for MothParams validation and behavior."""

import numpy as np
import pytest
import attrs

from fmd.simulator.params import (
    MothParams,
    MOTH_BIEKER_V3,
    STANDARD_GRAVITY,
    WATER_DENSITY_SALT,
)


# ============================================================================
# Helper function to create valid params
# ============================================================================


def make_valid_params(**overrides) -> MothParams:
    """Create valid MothParams with optional overrides.

    Uses hull-datum geometry fields. Body-frame positions like
    main_foil_position, rudder_position, etc. are computed @properties.
    """
    defaults = dict(
        # Hull
        hull_mass=50.0,
        hull_inertia=np.array([91.1, 118.6, 31.3]),
        hull_length=3.355,
        hull_beam=0.35,
        # Hull geometry (hull-datum reference)
        hull_depth=0.45,
        hull_cg_above_bottom=0.82,
        hull_cg_from_bow=1.82,
        main_foil_strut_depth=1.0,
        rudder_strut_depth=0.95,
        wing_rack_span=2.25,
        wing_dihedral=np.radians(30.0),
        # Hull-datum positions
        main_foil_from_bow=1.6,
        wing_rack_from_bow=2.0,
        rudder_from_bow=3.855,
        sail_ce_hull_datum=np.array([1.9, 0.0, 2.95]),
        bowsprit_hull_datum=np.array([0.0, 0.0, 0.45]),
        wand_pivot_hull_datum=np.array([0.0, 0.0, 0.35]),
        # Sailor
        sailor_mass=75.0,
        sailor_position=np.array([-0.3, 0.0, -0.2]),
        # Main foil
        main_foil_span=0.95,
        main_foil_chord=0.089,
        main_foil_area=0.08455,
        main_foil_cl_alpha=5.7,
        main_foil_cd0=0.01,
        # Rudder
        rudder_span=0.68,
        rudder_chord=0.075,
        rudder_area=0.051,
        rudder_elevator_min=np.radians(-10.0),
        rudder_elevator_max=np.radians(10.0),
        rudder_cl_alpha=5.0,
        # Wand
        wand_length=1.2,
        # Sail
        sail_area=8.0,
    )
    defaults.update(overrides)
    return MothParams(**defaults)


# ============================================================================
# Valid Construction Tests
# ============================================================================


class TestMothParamsConstruction:
    """Tests for valid MothParams construction."""

    def test_valid_construction(self):
        """Valid params should construct without error."""
        params = make_valid_params()
        assert params.hull_mass == 50.0
        assert params.sailor_mass == 75.0
        assert params.main_foil_span == 0.95

    def test_default_environmental_params(self):
        """Default g and rho_water should be set correctly."""
        params = make_valid_params()
        assert params.g == STANDARD_GRAVITY
        assert params.rho_water == WATER_DENSITY_SALT

    def test_custom_environmental_params(self):
        """Custom g and rho_water should be accepted."""
        params = make_valid_params(g=10.0, rho_water=1000.0)
        assert params.g == 10.0
        assert params.rho_water == 1000.0

    def test_default_optional_params(self):
        """Default optional params should be set correctly."""
        params = make_valid_params()
        assert params.main_foil_cl0 == 0.0
        assert params.main_foil_oswald == 0.85
        assert params.main_foil_flap_effectiveness == 0.5
        assert params.wand_gearing_ratio == 1.0

    def test_array_conversion(self):
        """Lists should be converted to numpy arrays."""
        params = make_valid_params(
            hull_inertia=[1.0, 2.0, 3.0],
            sailor_position=[0.1, 0.2, 0.3],
        )
        assert isinstance(params.hull_inertia, np.ndarray)
        assert isinstance(params.sailor_position, np.ndarray)


# ============================================================================
# Validation Error Tests
# ============================================================================


class TestMothParamsValidation:
    """Tests for MothParams validation errors."""

    def test_negative_hull_mass_raises(self):
        """Negative hull mass should raise ValueError."""
        with pytest.raises(ValueError, match="hull_mass must be positive"):
            make_valid_params(hull_mass=-1.0)

    def test_zero_hull_mass_raises(self):
        """Zero hull mass should raise ValueError."""
        with pytest.raises(ValueError, match="hull_mass must be positive"):
            make_valid_params(hull_mass=0.0)

    def test_negative_sailor_mass_raises(self):
        """Negative sailor mass should raise ValueError."""
        with pytest.raises(ValueError, match="sailor_mass must be positive"):
            make_valid_params(sailor_mass=-1.0)

    def test_zero_sailor_mass_raises(self):
        """Zero sailor mass should raise ValueError."""
        with pytest.raises(ValueError, match="sailor_mass must be positive"):
            make_valid_params(sailor_mass=0.0)

    def test_invalid_inertia_raises(self):
        """Non-positive inertia elements should raise ValueError."""
        with pytest.raises(ValueError, match="must all be positive"):
            make_valid_params(hull_inertia=np.array([1.0, -2.0, 3.0]))

    def test_zero_inertia_element_raises(self):
        """Zero inertia element should raise ValueError."""
        with pytest.raises(ValueError, match="must all be positive"):
            make_valid_params(hull_inertia=np.array([1.0, 0.0, 3.0]))

    def test_nan_hull_mass_raises(self):
        """NaN hull mass should raise ValueError."""
        with pytest.raises(ValueError, match="hull_mass must be finite"):
            make_valid_params(hull_mass=np.nan)

    def test_nan_inertia_raises(self):
        """NaN in inertia should raise ValueError."""
        with pytest.raises(ValueError, match="must have all finite"):
            make_valid_params(hull_inertia=np.array([np.nan, 2.0, 3.0]))

    def test_inf_hull_mass_raises(self):
        """Infinite hull mass should raise ValueError."""
        with pytest.raises(ValueError, match="hull_mass must be finite"):
            make_valid_params(hull_mass=np.inf)

    def test_inf_inertia_raises(self):
        """Infinite value in inertia should raise ValueError."""
        with pytest.raises(ValueError, match="must have all finite"):
            make_valid_params(hull_inertia=np.array([np.inf, 2.0, 3.0]))

    def test_wrong_position_array_shape_raises(self):
        """Wrong shape position array should raise ValueError."""
        with pytest.raises(ValueError, match="must be a 3-element vector"):
            make_valid_params(sailor_position=np.array([0.1, 0.2]))

    def test_wrong_inertia_shape_raises(self):
        """Wrong shape inertia should raise ValueError."""
        with pytest.raises(ValueError, match="must be a 3-element vector or 3x3"):
            make_valid_params(hull_inertia=np.array([1.0, 2.0]))

    def test_negative_main_foil_span_raises(self):
        """Negative main foil span should raise ValueError."""
        with pytest.raises(ValueError, match="main_foil_span must be positive"):
            make_valid_params(main_foil_span=-1.0)

    def test_negative_main_foil_cd0_raises(self):
        """Negative main foil cd0 should raise ValueError."""
        with pytest.raises(ValueError, match="main_foil_cd0 must be non-negative"):
            make_valid_params(main_foil_cd0=-0.01)

    def test_nan_position_array_raises(self):
        """NaN in position array should raise ValueError."""
        with pytest.raises(ValueError, match="must have all finite"):
            make_valid_params(sail_ce_hull_datum=np.array([np.nan, 0.0, 2.95]))

    def test_sail_thrust_speeds_nan_raises(self):
        """NaN in sail thrust speed table should raise ValueError."""
        with pytest.raises(ValueError, match="sail_thrust_speeds must contain only finite values"):
            make_valid_params(
                sail_thrust_speeds=(6.0, np.nan, 10.0),
                sail_thrust_values=(100.0, 120.0, 140.0),
            )

    def test_sail_thrust_values_inf_raises(self):
        """Inf in sail thrust value table should raise ValueError."""
        with pytest.raises(ValueError, match="sail_thrust_values must contain only finite values"):
            make_valid_params(
                sail_thrust_speeds=(6.0, 8.0, 10.0),
                sail_thrust_values=(100.0, np.inf, 140.0),
            )

    def test_sail_thrust_speeds_negative_raises(self):
        """Negative sail thrust speed should raise ValueError."""
        with pytest.raises(ValueError, match="sail_thrust_speeds must be non-negative"):
            make_valid_params(
                sail_thrust_speeds=(-1.0, 6.0, 8.0),
                sail_thrust_values=(90.0, 100.0, 120.0),
            )


# ============================================================================
# Immutability Tests
# ============================================================================


class TestMothParamsImmutability:
    """Tests for MothParams immutability."""

    def test_frozen_instance(self):
        """Frozen instance should not allow attribute modification."""
        params = make_valid_params()
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            params.hull_mass = 20.0

    def test_evolve_creates_new_instance(self):
        """attrs.evolve should create new instance."""
        params = make_valid_params()
        new_params = attrs.evolve(params, hull_mass=20.0)

        assert new_params is not params
        assert new_params.hull_mass == 20.0
        assert params.hull_mass == 50.0  # Original unchanged


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestMothParamsHelpers:
    """Tests for MothParams helper methods."""

    def test_with_sailor_mass_helper(self):
        """with_sailor_mass should return new params with updated sailor mass."""
        params = make_valid_params()
        new_params = params.with_sailor_mass(80.0)

        assert new_params.sailor_mass == 80.0
        assert params.sailor_mass == 75.0  # Original unchanged

    def test_with_sailor_position_helper(self):
        """with_sailor_position should return new params with updated position."""
        params = make_valid_params()
        new_position = [0.5, 0.1, 0.3]
        new_params = params.with_sailor_position(new_position)

        np.testing.assert_array_almost_equal(new_params.sailor_position, new_position)
        np.testing.assert_array_almost_equal(params.sailor_position, [-0.3, 0.0, -0.2])

    def test_with_wand_gearing_helper(self):
        """with_wand_gearing should return new params with updated gearing."""
        params = make_valid_params()
        new_params = params.with_wand_gearing(1.5)

        assert new_params.wand_gearing_ratio == 1.5
        assert params.wand_gearing_ratio == 1.0  # Original unchanged


# ============================================================================
# Equality and Hash Tests
# ============================================================================


class TestMothParamsEqualityHash:
    """Tests for MothParams equality and hashing."""

    def test_equality_same_params(self):
        """Equal params should compare equal."""
        p1 = make_valid_params()
        p2 = make_valid_params()
        assert p1 == p2

    def test_inequality_different_scalar(self):
        """Different scalar values should not be equal."""
        p1 = make_valid_params()
        p2 = make_valid_params(hull_mass=20.0)
        assert p1 != p2

    def test_inequality_different_array(self):
        """Different array values should not be equal."""
        p1 = make_valid_params()
        p2 = make_valid_params(sailor_position=np.array([0.4, 0.0, 0.2]))
        assert p1 != p2

    def test_inequality_with_non_moth_params(self):
        """Comparison with non-MothParams should return NotImplemented."""
        p1 = make_valid_params()
        result = p1.__eq__("not a MothParams")
        assert result is NotImplemented

    def test_hash_consistency(self):
        """Equal params should have equal hashes."""
        p1 = make_valid_params()
        p2 = make_valid_params()
        assert hash(p1) == hash(p2)

    def test_hash_different_params(self):
        """Different params should (likely) have different hashes."""
        p1 = make_valid_params()
        p2 = make_valid_params(hull_mass=20.0)
        # While hash collisions are possible, they should be rare
        assert hash(p1) != hash(p2)

    def test_hash_usable_in_set(self):
        """Params should be usable in sets."""
        p1 = make_valid_params()
        p2 = make_valid_params(hull_mass=20.0)
        params_set = {p1, p2}
        assert len(params_set) == 2
        assert p1 in params_set
        assert p2 in params_set

    def test_hash_usable_in_dict(self):
        """Params should be usable as dict keys."""
        p1 = make_valid_params()
        p2 = make_valid_params(hull_mass=20.0)
        params_dict = {p1: "original", p2: "modified"}
        assert params_dict[p1] == "original"
        assert params_dict[p2] == "modified"


# ============================================================================
# Computed Property Tests
# ============================================================================


class TestMothParamsProperties:
    """Tests for MothParams computed properties."""

    def test_total_mass_property(self):
        """total_mass should return hull + sailor mass."""
        params = make_valid_params(hull_mass=50.0, sailor_mass=75.0)
        assert params.total_mass == 125.0

    def test_hull_inertia_matrix_from_diagonal(self):
        """hull_inertia_matrix should convert diagonal to 3x3."""
        params = make_valid_params(hull_inertia=np.array([2.0, 8.0, 8.5]))
        expected = np.diag([2.0, 8.0, 8.5])
        np.testing.assert_array_equal(params.hull_inertia_matrix, expected)

    def test_hull_inertia_matrix_from_3x3(self):
        """hull_inertia_matrix should return original for 3x3 input."""
        inertia_3x3 = np.array([
            [2.0, 0.1, 0.0],
            [0.1, 8.0, 0.0],
            [0.0, 0.0, 8.5],
        ])
        params = make_valid_params(hull_inertia=inertia_3x3)
        np.testing.assert_array_equal(params.hull_inertia_matrix, inertia_3x3)

    def test_main_foil_aspect_ratio(self):
        """main_foil_aspect_ratio should be span^2 / area."""
        params = make_valid_params(main_foil_span=1.0, main_foil_area=0.12)
        expected = 1.0**2 / 0.12  # ~8.33
        assert abs(params.main_foil_aspect_ratio - expected) < 1e-10

    def test_rudder_aspect_ratio(self):
        """rudder_aspect_ratio should be span^2 / area."""
        params = make_valid_params(rudder_span=0.5, rudder_area=0.04)
        expected = 0.5**2 / 0.04  # 6.25
        assert abs(params.rudder_aspect_ratio - expected) < 1e-10

    def test_combined_cg_offset(self):
        """combined_cg_offset should compute weighted position."""
        params = make_valid_params(
            hull_mass=50.0,
            sailor_mass=75.0,
            sailor_position=np.array([-0.3, 0.0, -0.2]),
        )
        # CG offset = sailor_mass * sailor_position / total_mass
        # = 75.0 * [-0.3, 0.0, -0.2] / 125.0
        expected = 75.0 * np.array([-0.3, 0.0, -0.2]) / 125.0
        np.testing.assert_array_almost_equal(params.combined_cg_offset, expected)

    def test_composite_pitch_inertia_basic(self):
        """composite_pitch_inertia should match reduced-mass parallel axis formula."""
        params = make_valid_params(
            hull_mass=50.0,
            sailor_mass=75.0,
            hull_inertia=np.array([91.1, 118.6, 31.3]),
            sailor_position=np.array([-0.3, 0.0, -0.2]),
        )
        x_s, z_s = -0.3, -0.2
        reduced = 50.0 * 75.0 / 125.0
        expected = 118.6 + reduced * (x_s**2 + z_s**2)
        assert params.composite_pitch_inertia == pytest.approx(expected, rel=1e-10)

    def test_composite_pitch_inertia_sailor_at_hull_cg(self):
        """When sailor is at hull CG, composite inertia equals hull-only."""
        params = make_valid_params(
            hull_inertia=np.array([91.1, 118.6, 31.3]),
            sailor_position=np.array([0.0, 0.0, 0.0]),
        )
        assert params.composite_pitch_inertia == pytest.approx(118.6, rel=1e-10)

    def test_composite_pitch_inertia_always_ge_hull(self):
        """Composite inertia should always be >= hull-only inertia."""
        for x in [-1.0, -0.3, 0.0, 0.3, 1.0]:
            for z in [-1.0, -0.6, 0.0, 0.2, 1.0]:
                params = make_valid_params(sailor_position=np.array([x, 0.0, z]))
                hull_iyy = params.hull_inertia_matrix[1, 1]
                assert params.composite_pitch_inertia >= hull_iyy


# ============================================================================
# Preset Tests
# ============================================================================


class TestMothBiekerV3Preset:
    """Tests for MOTH_BIEKER_V3 preset."""

    def test_preset_loads(self):
        """MOTH_BIEKER_V3 should load without error."""
        assert MOTH_BIEKER_V3 is not None
        assert isinstance(MOTH_BIEKER_V3, MothParams)

    def test_preset_total_mass(self):
        """MOTH_BIEKER_V3 should have expected total mass."""
        assert MOTH_BIEKER_V3.total_mass == 125.0

    def test_preset_hull_length(self):
        """MOTH_BIEKER_V3 should have class-legal hull length."""
        assert MOTH_BIEKER_V3.hull_length == 3.355

    def test_preset_sail_area(self):
        """MOTH_BIEKER_V3 should have class-legal sail area."""
        assert MOTH_BIEKER_V3.sail_area == 8.0

    def test_preset_main_foil_aspect_ratio(self):
        """MOTH_BIEKER_V3 main foil AR should be reasonable."""
        ar = MOTH_BIEKER_V3.main_foil_aspect_ratio
        # Typical T-foil AR is 6-10
        assert 6.0 < ar < 12.0

    def test_preset_sailor_position_z_above_hull(self):
        """MOTH_BIEKER_V3 sailor z should be negative (above hull CG)."""
        assert MOTH_BIEKER_V3.sailor_position[2] < 0
        np.testing.assert_allclose(MOTH_BIEKER_V3.sailor_position, [-0.30, 0.0, -0.2], atol=1e-12)

    def test_preset_environmental_defaults(self):
        """MOTH_BIEKER_V3 should use default environmental params."""
        assert MOTH_BIEKER_V3.g == STANDARD_GRAVITY
        assert MOTH_BIEKER_V3.rho_water == WATER_DENSITY_SALT


# ============================================================================
# Metadata Tests
# ============================================================================


class TestMothParamsMetadata:
    """Tests for MothParams field metadata."""

    def test_hull_mass_metadata(self):
        """hull_mass should have unit and description metadata."""
        fields = attrs.fields(MothParams)
        field = fields.hull_mass
        assert field.metadata["unit"] == "kg"
        assert "mass" in field.metadata["description"].lower()

    def test_hull_inertia_metadata(self):
        """hull_inertia should have unit and description metadata."""
        fields = attrs.fields(MothParams)
        field = fields.hull_inertia
        assert field.metadata["unit"] == "kg*m^2"
        assert "inertia" in field.metadata["description"].lower()

    def test_position_fields_have_units(self):
        """Position fields should have meter units."""
        fields = attrs.fields(MothParams)
        # Stored position/hull-datum fields (body-frame positions are @properties)
        position_fields = [
            fields.sailor_position,
            fields.sail_ce_hull_datum,
            fields.bowsprit_hull_datum,
            fields.wand_pivot_hull_datum,
        ]
        for field in position_fields:
            assert field.metadata["unit"] == "m", f"{field.name} missing unit 'm'"

    def test_foil_cl_alpha_metadata(self):
        """Lift curve slope should have 1/rad units."""
        fields = attrs.fields(MothParams)
        assert fields.main_foil_cl_alpha.metadata["unit"] == "1/rad"
        assert fields.rudder_cl_alpha.metadata["unit"] == "1/rad"

    def test_elevator_max_metadata(self):
        """Elevator max should have rad units."""
        fields = attrs.fields(MothParams)
        assert fields.rudder_elevator_max.metadata["unit"] == "rad"


# ============================================================================
# Added Mass Parameter Tests
# ============================================================================


class TestMothParamsAddedMass:
    """Tests for added mass parameters."""

    def test_added_mass_defaults(self):
        """Added mass parameters have expected defaults."""
        params = make_valid_params()
        assert params.added_mass_heave == 10.0
        assert params.added_inertia_pitch == 8.75

    def test_added_mass_custom_values(self):
        """Added mass parameters can be customized."""
        params = make_valid_params(added_mass_heave=15.0, added_inertia_pitch=12.0)
        assert params.added_mass_heave == 15.0
        assert params.added_inertia_pitch == 12.0

    def test_added_mass_zero_allowed(self):
        """Zero added mass is allowed (no added mass)."""
        params = make_valid_params(added_mass_heave=0.0, added_inertia_pitch=0.0)
        assert params.added_mass_heave == 0.0
        assert params.added_inertia_pitch == 0.0

    def test_added_mass_negative_raises(self):
        """Negative added mass should raise ValueError."""
        with pytest.raises(ValueError, match="added_mass_heave must be non-negative"):
            make_valid_params(added_mass_heave=-1.0)

    def test_added_inertia_negative_raises(self):
        """Negative added inertia should raise ValueError."""
        with pytest.raises(ValueError, match="added_inertia_pitch must be non-negative"):
            make_valid_params(added_inertia_pitch=-1.0)

    def test_added_mass_nan_raises(self):
        """NaN added mass should raise ValueError."""
        with pytest.raises(ValueError, match="added_mass_heave must be finite"):
            make_valid_params(added_mass_heave=np.nan)

    def test_added_inertia_inf_raises(self):
        """Infinite added inertia should raise ValueError."""
        with pytest.raises(ValueError, match="added_inertia_pitch must be finite"):
            make_valid_params(added_inertia_pitch=np.inf)

    def test_added_mass_in_equality(self):
        """Added mass parameters affect equality."""
        p1 = make_valid_params()
        p2 = make_valid_params(added_mass_heave=15.0)
        p3 = make_valid_params(added_inertia_pitch=10.0)
        assert p1 != p2
        assert p1 != p3
        assert p2 != p3

    def test_added_mass_in_hash(self):
        """Added mass parameters affect hash."""
        p1 = make_valid_params()
        p2 = make_valid_params(added_mass_heave=15.0)
        assert hash(p1) != hash(p2)

    def test_added_mass_metadata(self):
        """Added mass parameters have correct units."""
        fields = attrs.fields(MothParams)
        assert fields.added_mass_heave.metadata["unit"] == "kg"
        assert fields.added_inertia_pitch.metadata["unit"] == "kg*m^2"


# ============================================================================
# Frame Conversion Tests
# ============================================================================


class TestFrameConversion:
    """Tests for hull_datum_to_body() and body_to_hull_datum()."""

    def test_round_trip_origin(self):
        """Round-trip through both conversions preserves the original point."""
        p = MOTH_BIEKER_V3
        point = np.array([0.0, 0.0, 0.0])
        result = p.body_to_hull_datum(p.hull_datum_to_body(point))
        np.testing.assert_allclose(result, point, atol=1e-15)

    def test_round_trip_arbitrary_point(self):
        """Round-trip works for an arbitrary point."""
        p = MOTH_BIEKER_V3
        point = np.array([1.5, 0.3, -0.7])
        result = p.body_to_hull_datum(p.hull_datum_to_body(point))
        np.testing.assert_allclose(result, point, atol=1e-15)

    def test_round_trip_several_points(self):
        """Round-trip works for several diverse points."""
        p = MOTH_BIEKER_V3
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.9, 0.0, 1.0]),
            np.array([3.355, 0.5, -1.0]),
            np.array([0.0, -1.0, 2.5]),
        ]
        for pt in points:
            result = p.body_to_hull_datum(p.hull_datum_to_body(pt))
            np.testing.assert_allclose(result, pt, atol=1e-15)

    def test_cg_maps_to_body_origin(self):
        """CG in hull-datum should map to [0, 0, 0] in body FRD."""
        p = MOTH_BIEKER_V3
        cg_datum = np.array([p.hull_cg_from_bow, 0.0, p.hull_cg_above_bottom])
        result = p.hull_datum_to_body(cg_datum)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-15)

    def test_hull_bottom_at_cg_x(self):
        """Hull bottom at CG x-position maps to [0, 0, cg_above_bottom] in body."""
        p = MOTH_BIEKER_V3
        hull_bottom = np.array([p.hull_cg_from_bow, 0.0, 0.0])
        result = p.hull_datum_to_body(hull_bottom)
        np.testing.assert_allclose(result, [0.0, 0.0, p.hull_cg_above_bottom], atol=1e-15)

    def test_y_axis_passthrough(self):
        """Y-coordinate is unchanged through hull_datum_to_body."""
        p = MOTH_BIEKER_V3
        for y_val in [-1.0, 0.0, 0.5, 2.0]:
            point = np.array([1.0, y_val, 0.5])
            result = p.hull_datum_to_body(point)
            assert result[1] == y_val

    def test_y_axis_passthrough_body_to_hull_datum(self):
        """Y-coordinate is unchanged through body_to_hull_datum."""
        p = MOTH_BIEKER_V3
        for y_val in [-1.0, 0.0, 0.5, 2.0]:
            point = np.array([0.0, y_val, 0.0])
            result = p.body_to_hull_datum(point)
            assert result[1] == y_val

    def test_known_main_foil_position(self):
        """Main foil hull-datum -> body matches expected value."""
        p = MOTH_BIEKER_V3
        datum_pos = np.array([p.main_foil_from_bow, 0.0, -p.main_foil_strut_depth])
        result = p.hull_datum_to_body(datum_pos)
        # Updated: measured geometry (hull_cg_from_bow=1.99, main_foil_from_bow=1.57, strut_depth=1.03)
        np.testing.assert_allclose(result, [0.42, 0.0, 1.85], atol=1e-12)

    def test_known_rudder_position(self):
        """Rudder hull-datum -> body matches expected value."""
        p = MOTH_BIEKER_V3
        datum_pos = np.array([p.rudder_from_bow, 0.0, -p.rudder_strut_depth])
        result = p.hull_datum_to_body(datum_pos)
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(result, [-1.865, 0.0, 1.77], atol=1e-12)

    def test_known_sail_ce_position(self):
        """Sail CE hull-datum -> body matches expected value."""
        p = MOTH_BIEKER_V3
        result = p.hull_datum_to_body(p.sail_ce_hull_datum)
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(result, [-0.51, 0.0, -1.18], atol=1e-12)


# ============================================================================
# Derived Body-Frame Position Tests
# ============================================================================


class TestDerivedPositions:
    """Tests for @property body-frame positions derived from hull-datum."""

    def test_main_foil_position_value(self):
        """main_foil_position matches expected body-frame value."""
        # Updated: measured geometry (hull_cg_from_bow=1.99, main_foil_from_bow=1.57, strut_depth=1.03)
        np.testing.assert_allclose(
            MOTH_BIEKER_V3.main_foil_position, [0.42, 0.0, 1.85], atol=1e-12
        )

    def test_rudder_position_value(self):
        """rudder_position matches expected body-frame value."""
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(
            MOTH_BIEKER_V3.rudder_position, [-1.865, 0.0, 1.77], atol=1e-12
        )

    def test_sail_ce_position_value(self):
        """sail_ce_position matches expected body-frame value."""
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(
            MOTH_BIEKER_V3.sail_ce_position, [-0.51, 0.0, -1.18], atol=1e-12
        )

    def test_bowsprit_position_value(self):
        """bowsprit_position matches expected body-frame value."""
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(
            MOTH_BIEKER_V3.bowsprit_position, [1.99, 0.0, 0.37], atol=1e-12
        )

    def test_wand_pivot_position_value(self):
        """wand_pivot_position matches expected body-frame value."""
        # Updated: measured geometry (hull_cg_from_bow=1.99)
        np.testing.assert_allclose(
            MOTH_BIEKER_V3.wand_pivot_position, [1.99, 0.0, 0.47], atol=1e-12
        )

    def test_hull_contact_depth_value(self):
        """hull_contact_depth matches expected value for preset."""
        assert MOTH_BIEKER_V3.hull_contact_depth == pytest.approx(0.94, abs=1e-10)


# ============================================================================
# Geometry Consistency Tests
# ============================================================================


class TestGeometryConsistency:
    """Verify internal consistency of geometry-derived values."""

    def test_foil_position_consistent_with_strut_geometry(self):
        """Main foil z-position equals hull_cg_above_bottom + strut_depth."""
        p = MOTH_BIEKER_V3
        expected_z = p.hull_cg_above_bottom + p.main_foil_strut_depth
        assert abs(p.main_foil_position[2] - expected_z) < 0.01

    def test_rudder_position_consistent_with_strut_geometry(self):
        """Rudder z-position equals hull_cg_above_bottom + rudder_strut_depth."""
        p = MOTH_BIEKER_V3
        expected_z = p.hull_cg_above_bottom + p.rudder_strut_depth
        assert abs(p.rudder_position[2] - expected_z) < 0.01

    def test_hull_contact_depth_tracks_system_cg(self):
        """hull_contact_depth adjusts when sailor_position changes."""
        p = attrs.evolve(MOTH_BIEKER_V3, sailor_position=np.array([-0.3, 0.0, -0.4]))
        expected = p.hull_cg_above_bottom - p.combined_cg_offset[2]
        assert abs(p.hull_contact_depth - expected) < 1e-12

    def test_frame_helpers_round_trip(self):
        """body_to_hull_datum(hull_datum_to_body(x)) == x for foil position."""
        p = MOTH_BIEKER_V3
        body = np.array([0.55, 0.0, 1.82])
        np.testing.assert_allclose(
            p.hull_datum_to_body(p.body_to_hull_datum(body)), body, atol=1e-12
        )
