import numpy as np
import pytest

from fmd.analysis.filters import ExponentialMovingAverage


class TestExponentialMovingAverageNaNs:
    def test_linear_propagates_and_restarts(self):
        """Hard contract: NaNs propagate, reset state, restart on next valid."""
        ema = ExponentialMovingAverage(alpha=0.5, circular=False)
        x = np.array([np.nan, 1.0, 2.0, np.nan, 10.0, 12.0], dtype=float)
        y = ema.apply(x).data

        assert np.isnan(y[0])
        assert y[1] == pytest.approx(1.0)  # restart at first valid
        assert y[2] == pytest.approx(0.5 * 2.0 + 0.5 * 1.0)
        assert np.isnan(y[3])              # propagate + reset
        assert y[4] == pytest.approx(10.0) # restart after NaN
        assert y[5] == pytest.approx(0.5 * 12.0 + 0.5 * 10.0)

    def test_circular_propagates_and_restarts(self):
        """Hard contract: NaNs propagate/reset/restart for circular EMA as well."""
        ema = ExponentialMovingAverage(alpha=0.5, circular=True, period=360.0)
        x = np.array([np.nan, 350.0, 10.0, np.nan, 20.0], dtype=float)
        y = ema.apply(x).data

        assert np.isnan(y[0])
        assert y[1] == pytest.approx(-10.0)  # 350 wrapped to [-180, 180)
        # After 350 -> 10 (short arc through 0), the filtered result should be near 0.
        assert abs(y[2]) < 20.0
        assert np.isnan(y[3])
        assert y[4] == pytest.approx(20.0)


