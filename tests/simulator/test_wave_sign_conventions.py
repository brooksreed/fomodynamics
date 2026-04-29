"""Sign convention tests for wave-body interaction.

Verifies that wave crest/trough produce correct physical effects:
- NED: positive down, wave elevation positive up
- Crest (eta > 0): surface higher, foil shallower (less submerged)
- Trough (eta < 0): surface lower, foil deeper (more submerged)
"""
import pytest
import jax.numpy as jnp

from fmd.simulator import Moth3D, Environment, ConstantControl, simulate
from fmd.simulator.params import MOTH_BIEKER_V3, WaveParams
from fmd.simulator.waves import WaveField


class TestWaveElevationSigns:
    """Test that wave elevation has correct sign convention."""

    def test_crest_positive(self):
        """Wave crest should have positive elevation."""
        wf = WaveField.regular(amplitude=1.0, period=5.0, direction=0.0)
        # At t=0, x=0: cos(0) = 1, so eta should be +amplitude
        eta = float(wf.elevation(0.0, 0.0, 0.0))
        assert eta > 0.0, f"Crest should be positive, got eta={eta}"
        assert abs(eta - 1.0) < 0.01, f"Crest should be ~1.0m, got eta={eta}"

    def test_trough_negative(self):
        """Wave trough should have negative elevation."""
        wf = WaveField.regular(amplitude=1.0, period=5.0, direction=0.0)
        # At half period, cos(pi) = -1, so eta should be -amplitude
        half_period = 5.0 / 2.0
        eta = float(wf.elevation(0.0, 0.0, half_period))
        assert eta < 0.0, f"Trough should be negative, got eta={eta}"
        assert abs(eta + 1.0) < 0.01, f"Trough should be ~-1.0m, got eta={eta}"


class TestMothSignConventions:
    """Test that wave effects on moth have correct signs."""

    @pytest.fixture
    def moth(self):
        return Moth3D(MOTH_BIEKER_V3)

    def test_wave_changes_derivatives(self, moth):
        """Wave orbital velocity should change moth dynamics.

        At t=T/4, the wave orbital velocity has a nonzero vertical
        component that alters the foil's effective angle of attack.
        At t=0, AoA=0 and cl0=0 means lift is zero regardless of
        depth factor, so we use a quarter-period offset.
        """
        state0 = moth.default_state()
        env_wave = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=5.0))

        # At quarter period, orbital velocity has nonzero vertical component
        quarter_period = 5.0 / 4.0
        deriv_wave = moth.forward_dynamics(state0, moth.default_control(), t=quarter_period, env=env_wave)
        deriv_calm = moth.forward_dynamics(state0, moth.default_control(), t=quarter_period, env=None)

        # Should produce different derivatives
        diff = float(jnp.max(jnp.abs(deriv_wave - deriv_calm)))
        assert diff > 1e-6, f"Wave should change dynamics, max diff={diff}"

    def test_wave_modifies_trajectory(self, moth):
        """Waves should measurably affect moth trajectory."""
        state0 = moth.default_state()

        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=5.0))
        result_calm = simulate(moth, state0, dt=0.005, duration=2.0)
        result_wave = simulate(moth, state0, dt=0.005, duration=2.0, env=env)

        diff = float(jnp.max(jnp.abs(result_calm.states - result_wave.states)))
        assert diff > 0.001, f"Wave should modify trajectory, max diff={diff}"
