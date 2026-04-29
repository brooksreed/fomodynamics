"""Tests for Environment module."""

import pytest
import jax.numpy as jnp

from fmd.simulator import (
    Environment, simulate, ConstantControl,
    Moth3D, RigidBody6DOF, SimplePendulum,
)
from fmd.simulator.params import (
    MOTH_BIEKER_V3, PENDULUM_1M,
    WAVE_REGULAR_1M, WaveParams,
)
from fmd.simulator.waves import WaveField


class TestEnvironmentConstruction:
    """Test Environment creation."""

    def test_calm(self):
        """Environment.calm() has no wave field."""
        env = Environment.calm()
        assert env.wave_field is None

    def test_with_wave_params(self):
        """Environment.with_waves() accepts WaveParams."""
        env = Environment.with_waves(WAVE_REGULAR_1M)
        assert env.wave_field is not None

    def test_with_wave_field(self):
        """Environment.with_waves() accepts WaveField directly."""
        wf = WaveField.regular(amplitude=0.5, period=5.0)
        env = Environment.with_waves(wf)
        assert env.wave_field is not None

    def test_default_is_calm(self):
        """Default Environment() should have no wave field."""
        env = Environment()
        assert env.wave_field is None


class TestEnvNoneMatchesCalm:
    """env=None should match Environment.calm() for all models."""

    def test_moth(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        state0 = moth.default_state()

        result_none = simulate(moth, state0, dt=0.005, duration=1.0)
        result_calm = simulate(moth, state0, dt=0.005, duration=1.0,
                               env=Environment.calm())

        assert jnp.allclose(result_none.states, result_calm.states, atol=1e-12)

    def test_pendulum(self):
        pend = SimplePendulum(PENDULUM_1M)
        state0 = pend.default_state()

        result_none = simulate(pend, state0, dt=0.01, duration=0.5)
        result_calm = simulate(pend, state0, dt=0.01, duration=0.5,
                               env=Environment.calm())

        assert jnp.allclose(result_none.states, result_calm.states, atol=1e-12)

    def test_rigid_body(self):
        from fmd.simulator.components import JaxGravity
        body = RigidBody6DOF(mass=10.0, inertia=jnp.array([1.0, 1.0, 1.0]),
                             components=[JaxGravity(10.0)])
        state0 = body.default_state()

        result_none = simulate(body, state0, dt=0.01, duration=0.5)
        result_calm = simulate(body, state0, dt=0.01, duration=0.5,
                               env=Environment.calm())

        assert jnp.allclose(result_none.states, result_calm.states, atol=1e-12)
