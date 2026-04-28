"""Unit tests for ControlRateLimit constraint."""

import pytest
import jax
import jax.numpy as jnp

from fmd.simulator.constraints import ControlRateLimit, Capability, ConstraintCategory


class TestControlRateLimit:
    """Tests for ControlRateLimit constraint."""

    def test_symmetric_construction(self):
        """Test symmetric convenience constructor."""
        c = ControlRateLimit.symmetric(
            name="test_rate", index=0, max_rate=1.0, dt_default=0.01
        )
        assert c.name == "test_rate"
        assert c.index == 0
        assert c.max_rate_up == 1.0
        assert c.max_rate_down == 1.0
        assert c.dt_default == 0.01

    def test_asymmetric_construction(self):
        """Test asymmetric rate limits."""
        c = ControlRateLimit(
            name="asymmetric",
            index=1,
            max_rate_up=2.0,
            max_rate_down=1.0,
            dt_default=0.02,
        )
        assert c.name == "asymmetric"
        assert c.index == 1
        assert c.max_rate_up == 2.0
        assert c.max_rate_down == 1.0
        assert c.dt_default == 0.02

    def test_value_requires_prev(self):
        """Test that value() raises NotImplementedError."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0, 0.0])
        u = jnp.array([0.5])

        with pytest.raises(NotImplementedError, match="requires u_prev"):
            c.value(0.0, x, u)

    def test_value_with_prev_feasible(self):
        """Test value_with_prev returns <= 0 when rate is within limits."""
        # max_rate = 1.0, dt = 0.1, so max delta = 0.1
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.05])  # delta = 0.05, rate = 0.05/0.1 = 0.5 <= 1.0

        val = c.value_with_prev(0.0, x, u, u_prev, dt=0.1)
        assert val.shape == (2,)
        # rate = 0.5
        # [0.5 - 1.0, -0.5 - 1.0] = [-0.5, -1.5]
        assert jnp.all(val <= 0)

    def test_value_with_prev_feasible_at_boundary(self):
        """Test value at exact rate limit boundary."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.1])  # delta = 0.1, rate = 0.1/0.1 = 1.0 exactly

        val = c.value_with_prev(0.0, x, u, u_prev, dt=0.1)
        # rate = 1.0
        # [1.0 - 1.0, -1.0 - 1.0] = [0.0, -2.0]
        assert val[0] == pytest.approx(0.0)  # At boundary
        assert val[1] < 0

    def test_value_with_prev_violated_up(self):
        """Test value_with_prev returns > 0 when rate up exceeds limit."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.2])  # delta = 0.2, rate = 0.2/0.1 = 2.0 > 1.0

        val = c.value_with_prev(0.0, x, u, u_prev, dt=0.1)
        # rate = 2.0
        # [2.0 - 1.0, -2.0 - 1.0] = [1.0, -3.0]
        assert val[0] > 0  # Violated (rate up exceeds limit)
        assert val[1] < 0  # Not violated (rate down is fine)
        # Max of elements > 0 indicates constraint violation
        assert jnp.max(val) > 0

    def test_value_with_prev_violated_down(self):
        """Test value_with_prev returns > 0 when rate down exceeds limit."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([-0.2])  # delta = -0.2, rate = -2.0

        val = c.value_with_prev(0.0, x, u, u_prev, dt=0.1)
        # rate = -2.0
        # [-2.0 - 1.0, -(-2.0) - 1.0] = [-3.0, 1.0]
        assert val[0] < 0  # Not violated (rate up is fine)
        assert val[1] > 0  # Violated (rate down exceeds limit)
        assert jnp.max(val) > 0

    def test_asymmetric_value(self):
        """Test asymmetric rate limits work correctly."""
        # Can increase at 2.0/s but only decrease at 0.5/s
        c = ControlRateLimit(
            name="asymmetric", index=0, max_rate_up=2.0, max_rate_down=0.5
        )
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])

        # Test increase within limit
        u_up = jnp.array([0.15])  # rate = 1.5 <= 2.0, OK
        val_up = c.value_with_prev(0.0, x, u_up, u_prev, dt=0.1)
        assert jnp.all(val_up <= 0)

        # Test increase exceeds limit
        u_up_bad = jnp.array([0.25])  # rate = 2.5 > 2.0, violated
        val_up_bad = c.value_with_prev(0.0, x, u_up_bad, u_prev, dt=0.1)
        assert jnp.max(val_up_bad) > 0

        # Test decrease within limit
        u_down = jnp.array([-0.04])  # rate = -0.4, |rate| = 0.4 <= 0.5, OK
        val_down = c.value_with_prev(0.0, x, u_down, u_prev, dt=0.1)
        assert jnp.all(val_down <= 0)

        # Test decrease exceeds limit
        u_down_bad = jnp.array([-0.1])  # rate = -1.0, |rate| = 1.0 > 0.5, violated
        val_down_bad = c.value_with_prev(0.0, x, u_down_bad, u_prev, dt=0.1)
        assert jnp.max(val_down_bad) > 0

    def test_clip_with_prev(self):
        """Test clipping respects rate limits."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.5])  # rate = 5.0 > 1.0, needs clipping

        x_clip, u_clip = c.clip_with_prev(0.0, x, u, u_prev, dt=0.1)

        # Should be clipped to max delta = 1.0 * 0.1 = 0.1
        assert x_clip[0] == 0.0  # x unchanged
        assert u_clip[0] == pytest.approx(0.1)

    def test_clip_with_prev_down(self):
        """Test clipping for decreasing control."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([-0.5])  # rate = -5.0, magnitude > 1.0

        x_clip, u_clip = c.clip_with_prev(0.0, x, u, u_prev, dt=0.1)

        # Should be clipped to min delta = -1.0 * 0.1 = -0.1
        assert u_clip[0] == pytest.approx(-0.1)

    def test_clip_with_prev_asymmetric(self):
        """Test clipping with asymmetric rate limits."""
        c = ControlRateLimit(
            name="asymmetric", index=0, max_rate_up=2.0, max_rate_down=0.5
        )
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])

        # Large increase should be clipped to 2.0 * 0.1 = 0.2
        u_up = jnp.array([1.0])
        _, u_clip_up = c.clip_with_prev(0.0, x, u_up, u_prev, dt=0.1)
        assert u_clip_up[0] == pytest.approx(0.2)

        # Large decrease should be clipped to -0.5 * 0.1 = -0.05
        u_down = jnp.array([-1.0])
        _, u_clip_down = c.clip_with_prev(0.0, x, u_down, u_prev, dt=0.1)
        assert u_clip_down[0] == pytest.approx(-0.05)

    def test_clip_unchanged_when_feasible(self):
        """Test clipping doesn't change feasible controls."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.05])  # rate = 0.5 <= 1.0, feasible

        x_clip, u_clip = c.clip_with_prev(0.0, x, u, u_prev, dt=0.1)

        assert x_clip[0] == 0.0
        assert u_clip[0] == pytest.approx(0.05)

    def test_dt_missing_raises(self):
        """Test that missing dt raises ValueError."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u = jnp.array([0.1])
        u_prev = jnp.array([0.0])

        with pytest.raises(ValueError, match="dt must be provided"):
            c.value_with_prev(0.0, x, u, u_prev)

        with pytest.raises(ValueError, match="dt must be provided"):
            c.clip_with_prev(0.0, x, u, u_prev)

    def test_dt_default_used(self):
        """Test that dt_default is used when dt not provided."""
        c = ControlRateLimit.symmetric(
            name="test", index=0, max_rate=1.0, dt_default=0.1
        )
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.05])

        # Should work without explicit dt
        val = c.value_with_prev(0.0, x, u, u_prev)
        assert val.shape == (2,)

        x_clip, u_clip = c.clip_with_prev(0.0, x, u, u_prev)
        assert u_clip[0] == pytest.approx(0.05)

    def test_dt_explicit_overrides_default(self):
        """Test that explicit dt overrides dt_default."""
        c = ControlRateLimit.symmetric(
            name="test", index=0, max_rate=1.0, dt_default=0.1
        )
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.1])  # With dt=0.1, rate=1.0; with dt=0.2, rate=0.5

        # With default dt=0.1, rate = 0.1/0.1 = 1.0 (at boundary)
        val_default = c.value_with_prev(0.0, x, u, u_prev)
        assert val_default[0] == pytest.approx(0.0)

        # With explicit dt=0.2, rate = 0.1/0.2 = 0.5 (well within limit)
        val_explicit = c.value_with_prev(0.0, x, u, u_prev, dt=0.2)
        assert val_explicit[0] == pytest.approx(-0.5)  # 0.5 - 1.0 = -0.5

    def test_jit_compatible(self):
        """Test that value_with_prev and clip_with_prev work under JIT."""
        c = ControlRateLimit.symmetric(
            name="test", index=0, max_rate=1.0, dt_default=0.1
        )
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])
        u = jnp.array([0.05])

        # JIT value_with_prev
        jit_value = jax.jit(lambda u: c.value_with_prev(0.0, x, u, u_prev))
        val = jit_value(u)
        assert val.shape == (2,)

        # JIT clip_with_prev
        jit_clip = jax.jit(lambda u: c.clip_with_prev(0.0, x, u, u_prev))
        x_clip, u_clip = jit_clip(u)
        assert u_clip[0] == pytest.approx(0.05)

        # Also test with dt as argument (non-traced)
        jit_value_dt = jax.jit(
            lambda u, dt: c.value_with_prev(0.0, x, u, u_prev, dt),
            static_argnums=(1,),
        )
        val2 = jit_value_dt(u, 0.1)
        assert val2.shape == (2,)

    def test_capabilities(self):
        """Test that constraint has correct capabilities."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)

        assert c.has_capability(Capability.HAS_RATE_LIMIT)
        assert not c.has_capability(Capability.HARD_CLIP)
        assert not c.has_capability(Capability.PROJECTION)
        assert not c.has_capability(Capability.HAS_SYMBOLIC_FORM)

    def test_category(self):
        """Test that constraint has RATE_LIMIT category."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        assert c.category == ConstraintCategory.RATE_LIMIT

    def test_negative_rate_rejected(self):
        """Test that negative rates are rejected."""
        with pytest.raises(ValueError, match="max_rate_up must be positive"):
            ControlRateLimit(
                name="bad", index=0, max_rate_up=-1.0, max_rate_down=1.0
            )

        with pytest.raises(ValueError, match="max_rate_down must be positive"):
            ControlRateLimit(
                name="bad", index=0, max_rate_up=1.0, max_rate_down=-1.0
            )

        with pytest.raises(ValueError, match="max_rate_up must be positive"):
            ControlRateLimit.symmetric(name="bad", index=0, max_rate=-1.0)

    def test_zero_rate_rejected(self):
        """Test that zero rates are rejected."""
        with pytest.raises(ValueError, match="max_rate_up must be positive"):
            ControlRateLimit(
                name="bad", index=0, max_rate_up=0.0, max_rate_down=1.0
            )

        with pytest.raises(ValueError, match="max_rate_down must be positive"):
            ControlRateLimit(
                name="bad", index=0, max_rate_up=1.0, max_rate_down=0.0
            )

    def test_multi_channel_control(self):
        """Test with multi-dimensional control vector."""
        # Constrain only index 1 (second channel)
        c = ControlRateLimit.symmetric(name="ch1_rate", index=1, max_rate=1.0)
        x = jnp.array([0.0, 0.0])
        u_prev = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([10.0, 0.05, 10.0])  # Only channel 1 matters

        val = c.value_with_prev(0.0, x, u, u_prev, dt=0.1)
        # rate = 0.5 for channel 1
        assert jnp.all(val <= 0)

        # Violate on channel 1
        u_bad = jnp.array([0.0, 0.5, 0.0])  # rate = 5.0 on channel 1
        val_bad = c.value_with_prev(0.0, x, u_bad, u_prev, dt=0.1)
        assert jnp.max(val_bad) > 0

    def test_clip_preserves_other_channels(self):
        """Test that clipping only affects the constrained channel."""
        c = ControlRateLimit.symmetric(name="ch1_rate", index=1, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([5.0, 0.5, 5.0])  # Large values on channels 0 and 2

        x_clip, u_clip = c.clip_with_prev(0.0, x, u, u_prev, dt=0.1)

        # Channel 0 and 2 should be unchanged
        assert u_clip[0] == 5.0
        assert u_clip[2] == 5.0
        # Channel 1 should be clipped to 0.1
        assert u_clip[1] == pytest.approx(0.1)

    def test_sign_convention_matches_blur(self):
        """Test sign convention: feasible <= 0, violated > 0."""
        c = ControlRateLimit.symmetric(name="test", index=0, max_rate=1.0)
        x = jnp.array([0.0])
        u_prev = jnp.array([0.0])

        # Feasible: rate = 0.5 <= 1.0
        u_feasible = jnp.array([0.05])
        val_feasible = c.value_with_prev(0.0, x, u_feasible, u_prev, dt=0.1)
        assert jnp.all(val_feasible <= 0), "Feasible point should have all values <= 0"

        # Violated up: rate = 2.0 > 1.0
        u_violated_up = jnp.array([0.2])
        val_violated_up = c.value_with_prev(0.0, x, u_violated_up, u_prev, dt=0.1)
        assert jnp.max(val_violated_up) > 0, "Violated point should have max value > 0"

        # Violated down: rate = -2.0, |rate| > 1.0
        u_violated_down = jnp.array([-0.2])
        val_violated_down = c.value_with_prev(0.0, x, u_violated_down, u_prev, dt=0.1)
        assert (
            jnp.max(val_violated_down) > 0
        ), "Violated point should have max value > 0"
