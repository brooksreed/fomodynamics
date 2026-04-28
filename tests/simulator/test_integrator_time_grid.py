import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fmd.simulator import simulate, SimplePendulum
from fmd.simulator.integrator import simulate_trajectory
from fmd.simulator.params import PENDULUM_1M


class TestSimulateTimeGrid:
    def test_non_divisible_dt_shortens_final_step_and_hits_duration(self):
        """times[] must reflect actual integration times and end exactly at duration.

        If duration is not divisible by dt, the final step is shortened so that
        times[-1] == duration (hard contract).
        """
        system = SimplePendulum(PENDULUM_1M)
        dt = 0.3
        duration = 1.0
        result = simulate(system, system.default_state(), dt=dt, duration=duration)

        assert result.times[0] == pytest.approx(0.0)
        assert result.times[-1] == pytest.approx(duration)

        dts = np.diff(result.times)
        assert np.all(dts > 0)
        assert np.max(dts) == pytest.approx(dt)
        # Final step should be shorter than dt in this case: 0.1
        assert dts[-1] == pytest.approx(0.1)


class TestValidateTimes:
    """Tests for time-grid validation in simulate_trajectory."""

    def test_single_point_rejected(self):
        """A single time point should raise ValueError."""
        system = SimplePendulum(PENDULUM_1M)
        with pytest.raises(ValueError, match="at least 2 points"):
            simulate_trajectory(system, system.default_state(), jnp.array([0.0]))

    def test_non_monotonic_rejected(self):
        """Non-monotonic times should raise ValueError."""
        system = SimplePendulum(PENDULUM_1M)
        with pytest.raises(ValueError, match="strictly increasing"):
            simulate_trajectory(
                system, system.default_state(), jnp.array([0.0, 0.1, 0.05, 0.2])
            )

    def test_duplicate_times_rejected(self):
        """Duplicate time points should raise ValueError."""
        system = SimplePendulum(PENDULUM_1M)
        with pytest.raises(ValueError, match="strictly increasing"):
            simulate_trajectory(
                system, system.default_state(), jnp.array([0.0, 0.1, 0.1, 0.2])
            )

    def test_valid_times_accepted(self):
        """Strictly increasing times with >= 2 points should work."""
        system = SimplePendulum(PENDULUM_1M)
        result = simulate_trajectory(
            system, system.default_state(), jnp.array([0.0, 0.01, 0.02])
        )
        assert result.states.shape[0] == 3


