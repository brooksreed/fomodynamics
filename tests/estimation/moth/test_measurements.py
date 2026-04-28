"""Tests for Moth 3DOF measurement models.

Tests cover:
- MothParams bowsprit_position field
- Full state identity measurement
- Vakaros measurement at known states
- Ride height nonlinearity
- Autodiff Jacobian correctness
- ArduPilot base and accel output shapes
- Noisy measurement statistics
- Factory function for all variants
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.estimation import (
    SpeedPitchHeightMeasurement,
    SpeedPitchRateHeightMeasurement,
    SpeedPitchRateHeightAccelMeasurement,
    create_moth_measurement,
    bow_ride_height,
)
from fmd.simulator.params import MOTH_BIEKER_V3


# Moth3D state indices
POS_D = 0
THETA = 1
W = 2
Q = 3
U = 4

# Default bowsprit for tests
_BP = jnp.array([1.6, 0.0, 0.0])

# Output name tuples for direct construction
_VAKAROS_NAMES = ("forward_speed", "pitch", "ride_height")
_ARDUPILOT_BASE_NAMES = ("forward_speed", "pitch", "pitch_rate", "ride_height")
_ARDUPILOT_ACCEL_NAMES = ("forward_speed", "pitch", "pitch_rate", "ride_height", "vertical_accel")


class TestBowspritInMothParams:
    """Verify bowsprit_position is properly defined in MothParams."""

    def test_bowsprit_in_preset(self):
        """MOTH_BIEKER_V3 has bowsprit_position with shape (3,)."""
        assert hasattr(MOTH_BIEKER_V3, "bowsprit_position")
        assert MOTH_BIEKER_V3.bowsprit_position.shape == (3,)

    def test_bowsprit_values(self):
        """Bowsprit is forward of CG at deck level."""
        bp = MOTH_BIEKER_V3.bowsprit_position
        assert bp[0] == pytest.approx(1.99, abs=1e-12)  # forward
        assert bp[1] == 0.0  # centerline
        assert bp[2] == pytest.approx(0.37, abs=1e-12)  # above CG


class TestFullStateMeasurement:
    """Test full_state variant returns identity measurement."""

    def test_full_state_identity(self):
        """H=I, measure returns x unchanged."""
        model = create_moth_measurement("full_state", R=jnp.eye(5) * 0.01)
        x = jnp.array([0.1, 0.05, -0.2, 0.01, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)
        assert jnp.allclose(y, x)

    def test_full_state_jacobian_is_identity(self):
        """Jacobian of full state is I."""
        model = create_moth_measurement("full_state", R=jnp.eye(5) * 0.01)
        x = jnp.array([0.1, 0.05, -0.2, 0.01, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert jnp.allclose(H, jnp.eye(5))


class TestVakarosMeasurement:
    """Tests for SpeedPitchHeightMeasurement (speed, pitch, ride_height)."""

    def test_vakaros_at_trim(self):
        """Verify each component analytically at a known state."""
        bp = jnp.array([1.6, 0.0, 0.0])
        R = jnp.eye(3) * 0.01
        model = SpeedPitchHeightMeasurement(
            output_names=("forward_speed", "pitch", "ride_height"),
            bowsprit_position=bp, R=R,
        )

        # State: pos_d=-0.3 (above datum), theta=0.1 rad, w=0, q=0, u=6.0
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)

        assert y.shape == (3,)

        # forward_speed = x[U] = 6.0
        assert jnp.isclose(y[0], 6.0)
        # pitch = x[THETA] = 0.1
        assert jnp.isclose(y[1], 0.1)
        # ride_height = -pos_d + bx*sin(theta) - bz*cos(theta)
        #             = 0.3 + 1.6*sin(0.1) - 0*cos(0.1)
        expected_rh = 0.3 + 1.6 * np.sin(0.1)
        assert jnp.isclose(y[2], expected_rh, atol=1e-6)

    def test_output_names(self):
        """Vakaros model has correct output names."""
        model = SpeedPitchHeightMeasurement(
            output_names=_VAKAROS_NAMES, bowsprit_position=_BP, R=jnp.eye(3) * 0.01
        )
        assert model.output_names == ("forward_speed", "pitch", "ride_height")
        assert model.num_outputs == 3


class TestRideHeightNonlinear:
    """Test ride height computation at various pitch angles."""

    def test_ride_height_at_zero_pitch(self):
        """At theta=0, h = -pos_d - bz."""
        bp = jnp.array([1.6, 0.0, 0.0])
        # pos_d = -0.2 (above water), theta = 0
        h = bow_ride_height(jnp.array(-0.2), jnp.array(0.0), bp)
        # h = -(-0.2) + 1.6*sin(0) - 0*cos(0) = 0.2
        assert jnp.isclose(h, 0.2)

    def test_ride_height_positive_pitch(self):
        """At theta=0.1, bowsprit rises."""
        bp = jnp.array([1.6, 0.0, 0.0])
        h = bow_ride_height(jnp.array(-0.2), jnp.array(0.1), bp)
        expected = 0.2 + 1.6 * np.sin(0.1)
        assert jnp.isclose(h, expected, atol=1e-7)

    def test_ride_height_negative_pitch(self):
        """At theta=-0.1, bowsprit dips."""
        bp = jnp.array([1.6, 0.0, 0.0])
        h = bow_ride_height(jnp.array(-0.2), jnp.array(-0.1), bp)
        expected = 0.2 + 1.6 * np.sin(-0.1)
        assert jnp.isclose(h, expected, atol=1e-7)

    def test_ride_height_with_bz(self):
        """Bowsprit below CG contributes via cos(theta)."""
        bp = jnp.array([1.6, 0.0, 0.3])  # bz = 0.3 (below CG)
        h = bow_ride_height(jnp.array(0.0), jnp.array(0.1), bp)
        expected = 0.0 + 1.6 * np.sin(0.1) - 0.3 * np.cos(0.1)
        assert jnp.isclose(h, expected, atol=1e-7)

    def test_ride_height_with_heel_and_bz(self):
        """Heel angle reduces the bz contribution via cos(heel)."""
        bp = jnp.array([1.6, 0.0, 0.3])  # bz = 0.3 (below CG)
        heel = np.deg2rad(30.0)
        theta = 0.1
        h = bow_ride_height(jnp.array(0.0), jnp.array(theta), bp, heel_angle=heel)
        expected = 0.0 + 1.6 * np.sin(theta) - 0.3 * np.cos(heel) * np.cos(theta)
        assert jnp.isclose(h, expected, atol=1e-7)
        # With heel, bz contribution is smaller than without heel
        h_no_heel = bow_ride_height(jnp.array(0.0), jnp.array(theta), bp)
        assert h > h_no_heel  # less depth subtracted -> higher ride height


class TestVakarosJacobian:
    """Test autodiff Jacobian for Vakaros model."""

    def test_vakaros_jacobian_shape(self):
        """Jacobian is (3, 5)."""
        model = SpeedPitchHeightMeasurement(
            output_names=_VAKAROS_NAMES, bowsprit_position=_BP, R=jnp.eye(3) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert H.shape == (3, 5)

    def test_vakaros_jacobian_finite(self):
        """Jacobian entries are all finite."""
        model = SpeedPitchHeightMeasurement(
            output_names=_VAKAROS_NAMES, bowsprit_position=_BP, R=jnp.eye(3) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert jnp.all(jnp.isfinite(H))

    def test_vakaros_jacobian_matches_finite_diff(self):
        """Autodiff Jacobian matches numerical finite differences."""
        model = SpeedPitchHeightMeasurement(
            output_names=_VAKAROS_NAMES, bowsprit_position=_BP, R=jnp.eye(3) * 0.01
        )
        x = jnp.array([-0.3, 0.1, -0.05, 0.02, 6.0])
        u = jnp.zeros(2)

        H_auto = model.get_measurement_jacobian(x, u)

        # Numerical finite difference
        eps = 1e-6
        H_num = jnp.zeros((3, 5))
        y0 = model.measure(x, u)
        for i in range(5):
            x_pert = x.at[i].add(eps)
            y_pert = model.measure(x_pert, u)
            H_num = H_num.at[:, i].set((y_pert - y0) / eps)

        assert jnp.allclose(H_auto, H_num, atol=1e-5)


class TestArduPilotBaseJacobian:
    """Test autodiff Jacobian for ArduPilot base model."""

    def test_ardupilot_base_jacobian_shape(self):
        """Jacobian is (4, 5)."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert H.shape == (4, 5)

    def test_ardupilot_base_jacobian_finite(self):
        """Jacobian entries are all finite."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert jnp.all(jnp.isfinite(H))

    def test_ardupilot_base_jacobian_matches_finite_diff(self):
        """Autodiff Jacobian matches numerical finite differences."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        x = jnp.array([-0.3, 0.1, -0.05, 0.02, 6.0])
        u = jnp.zeros(2)

        H_auto = model.get_measurement_jacobian(x, u)

        eps = 1e-6
        H_num = jnp.zeros((4, 5))
        y0 = model.measure(x, u)
        for i in range(5):
            x_pert = x.at[i].add(eps)
            y_pert = model.measure(x_pert, u)
            H_num = H_num.at[:, i].set((y_pert - y0) / eps)

        assert jnp.allclose(H_auto, H_num, atol=1e-5)


class TestArduPilotAccelJacobian:
    """Test autodiff Jacobian for ArduPilot accel model.

    The ardupilot_accel model has 5 outputs including the nonlinear
    vertical_accel = -g*cos(theta) + q*u term. Verify that the autodiff
    Jacobian matches finite differences at several random states near trim.
    """

    def test_ardupilot_accel_jacobian_shape(self):
        """Jacobian is (5, 5)."""
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert H.shape == (5, 5)

    def test_ardupilot_accel_jacobian_finite(self):
        """Jacobian entries are all finite."""
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)
        assert jnp.all(jnp.isfinite(H))

    def test_ardupilot_accel_jacobian_matches_finite_diff(self):
        """Autodiff Jacobian matches numerical finite differences at a single point."""
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        x = jnp.array([-0.3, 0.1, -0.05, 0.02, 6.0])
        u = jnp.zeros(2)

        H_auto = model.get_measurement_jacobian(x, u)

        eps = 1e-6
        H_num = jnp.zeros((5, 5))
        y0 = model.measure(x, u)
        for i in range(5):
            x_pert = x.at[i].add(eps)
            y_pert = model.measure(x_pert, u)
            H_num = H_num.at[:, i].set((y_pert - y0) / eps)

        assert jnp.allclose(H_auto, H_num, atol=1e-5)

    def test_ardupilot_accel_jacobian_random_states(self):
        """Autodiff Jacobian matches FD at several random states near trim."""
        rng = np.random.default_rng(123)
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        u = jnp.zeros(2)
        eps = 1e-6

        for _ in range(10):
            x = jnp.array([
                rng.uniform(-0.5, 0.1),   # pos_d near foiling
                rng.uniform(-0.15, 0.15),  # theta near trim
                rng.uniform(-0.5, 0.5),    # w
                rng.uniform(-0.3, 0.3),    # q
                rng.uniform(4.0, 8.0),     # u forward speed
            ])

            H_auto = model.get_measurement_jacobian(x, u)

            H_num = jnp.zeros((5, 5))
            y0 = model.measure(x, u)
            for i in range(5):
                x_pert = x.at[i].add(eps)
                y_pert = model.measure(x_pert, u)
                H_num = H_num.at[:, i].set((y_pert - y0) / eps)

            assert jnp.allclose(H_auto, H_num, atol=1e-5), (
                f"Jacobian mismatch at state {np.asarray(x)}"
            )

    def test_ardupilot_accel_jacobian_vertical_accel_row(self):
        """Verify the vertical_accel row has expected partial derivatives.

        vertical_accel = -g*cos(theta) + q*u
        d/d(pos_d) = 0
        d/d(theta) = g*sin(theta)
        d/d(w) = 0
        d/d(q) = u
        d/d(u) = q
        """
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        theta = 0.1
        q_val = 0.05
        u_val = 6.0
        x = jnp.array([-0.3, theta, 0.0, q_val, u_val])
        u = jnp.zeros(2)
        H = model.get_measurement_jacobian(x, u)

        g = 9.80665
        # Row 4 is vertical_accel
        assert jnp.isclose(H[4, POS_D], 0.0, atol=1e-10)          # d/d(pos_d) = 0
        assert jnp.isclose(H[4, THETA], g * np.sin(theta), atol=1e-6)  # d/d(theta) = g*sin(theta)
        assert jnp.isclose(H[4, W], 0.0, atol=1e-10)              # d/d(w) = 0
        assert jnp.isclose(H[4, Q], u_val, atol=1e-6)             # d/d(q) = u
        assert jnp.isclose(H[4, U], q_val, atol=1e-6)             # d/d(u) = q


class TestArduPilotBase:
    """Tests for SpeedPitchRateHeightMeasurement."""

    def test_has_4_outputs(self):
        """SpeedPitchRateHeight model has 4 outputs."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        assert model.num_outputs == 4
        assert model.output_names == (
            "forward_speed", "pitch", "pitch_rate", "ride_height"
        )

    def test_pitch_rate_is_q(self):
        """pitch_rate measurement is directly x[Q]."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.05, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)
        assert jnp.isclose(y[2], 0.05)  # pitch_rate = q

    def test_measure_shape(self):
        """Output vector has shape (4,)."""
        model = SpeedPitchRateHeightMeasurement(
            output_names=_ARDUPILOT_BASE_NAMES, bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.05, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)
        assert y.shape == (4,)


class TestArduPilotAccel:
    """Tests for SpeedPitchRateHeightAccelMeasurement."""

    def test_has_5_outputs(self):
        """SpeedPitchRateHeightAccel model has 5 outputs."""
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        assert model.num_outputs == 5
        assert model.output_names == (
            "forward_speed", "pitch", "pitch_rate", "ride_height", "vertical_accel"
        )

    def test_accel_formula(self):
        """Vertical accel = -g*cos(theta) + q*u."""
        g = 9.80665
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01, g=g
        )
        theta = 0.1
        q_val = 0.05
        u_val = 6.0
        x = jnp.array([-0.3, theta, 0.0, q_val, u_val])
        u = jnp.zeros(2)
        y = model.measure(x, u)

        expected_accel = -g * np.cos(theta) + q_val * u_val
        assert jnp.isclose(y[4], expected_accel, atol=1e-6)

    def test_measure_shape(self):
        """Output vector has shape (5,)."""
        model = SpeedPitchRateHeightAccelMeasurement(
            output_names=_ARDUPILOT_ACCEL_NAMES, bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        x = jnp.array([-0.3, 0.1, 0.0, 0.05, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)
        assert y.shape == (5,)


class TestNoisyMeasureStatistics:
    """Test noisy_measure produces correct statistics."""

    def test_noisy_measure_statistics(self):
        """Mean of N=1000 samples ~ clean measure, std ~ sqrt(R_diag)."""
        R_diag = jnp.array([0.04, 0.001, 0.01])
        R = jnp.diag(R_diag)
        model = SpeedPitchHeightMeasurement(output_names=_VAKAROS_NAMES, bowsprit_position=_BP, R=R)

        x = jnp.array([-0.3, 0.1, 0.0, 0.0, 6.0])
        u = jnp.zeros(2)
        y_clean = model.measure(x, u)

        n_samples = 1000
        keys = jax.random.split(jax.random.key(42), n_samples)
        samples = jax.vmap(lambda k: model.noisy_measure(x, u, k))(keys)

        sample_mean = jnp.mean(samples, axis=0)
        sample_std = jnp.std(samples, axis=0)

        # Mean should be close to clean measurement
        assert jnp.allclose(sample_mean, y_clean, atol=0.05)
        # Std should be close to sqrt of R diagonal
        expected_std = jnp.sqrt(R_diag)
        assert jnp.allclose(sample_std, expected_std, atol=0.05)


class TestFactory:
    """Test create_moth_measurement factory for all variants."""

    def test_factory_full_state(self):
        """Factory creates full_state variant."""
        model = create_moth_measurement("full_state", R=jnp.eye(5) * 0.01)
        x = jnp.array([0.1, 0.05, -0.2, 0.01, 6.0])
        u = jnp.zeros(2)
        y = model.measure(x, u)
        assert jnp.allclose(y, x)

    def test_factory_vakaros(self):
        """Factory creates vakaros variant."""
        model = create_moth_measurement(
            "speed_pitch_height", bowsprit_position=_BP, R=jnp.eye(3) * 0.01
        )
        assert model.num_outputs == 3
        assert isinstance(model, SpeedPitchHeightMeasurement)

    def test_factory_ardupilot_base(self):
        """Factory creates ardupilot_base variant."""
        model = create_moth_measurement(
            "speed_pitch_rate_height", bowsprit_position=_BP, R=jnp.eye(4) * 0.01
        )
        assert model.num_outputs == 4
        assert isinstance(model, SpeedPitchRateHeightMeasurement)

    def test_factory_ardupilot_accel(self):
        """Factory creates ardupilot_accel variant."""
        model = create_moth_measurement(
            "speed_pitch_rate_height_accel", bowsprit_position=_BP, R=jnp.eye(5) * 0.01
        )
        assert model.num_outputs == 5
        assert isinstance(model, SpeedPitchRateHeightAccelMeasurement)

    def test_factory_default_R(self):
        """Factory uses default R when not specified."""
        model = create_moth_measurement("speed_pitch_height", bowsprit_position=_BP)
        assert model.R.shape == (3, 3)

    def test_factory_unknown_variant(self):
        """Unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_moth_measurement("invalid_variant")

    def test_factory_missing_bowsprit(self):
        """Missing bowsprit for non-full_state variant raises ValueError."""
        with pytest.raises(ValueError, match="bowsprit_position is required"):
            create_moth_measurement("speed_pitch_height", R=jnp.eye(3) * 0.01)

    def test_factory_wrong_R_shape(self):
        """Wrong R shape raises ValueError."""
        with pytest.raises(ValueError, match="R shape"):
            create_moth_measurement("speed_pitch_height", bowsprit_position=_BP, R=jnp.eye(5))

    def test_factory_all_variants(self):
        """All 4 variants create valid models with correct output count."""
        variants = {
            "full_state": 5,
            "speed_pitch_height": 3,
            "speed_pitch_rate_height": 4,
            "speed_pitch_rate_height_accel": 5,
        }
        for name, expected_outputs in variants.items():
            kwargs = {}
            if name != "full_state":
                kwargs["bowsprit_position"] = _BP
            model = create_moth_measurement(name, **kwargs)
            assert model.num_outputs == expected_outputs, f"Failed for {name}"

            # Verify measure works
            x = jnp.array([-0.3, 0.1, 0.0, 0.05, 6.0])
            u = jnp.zeros(2)
            y = model.measure(x, u)
            assert y.shape == (expected_outputs,), f"Shape failed for {name}"
            assert jnp.all(jnp.isfinite(y)), f"Non-finite output for {name}"
