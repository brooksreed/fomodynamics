import numpy as np
import pandas as pd
import pytest

from fmd.analysis.processing import resample_to_rate


class TestResampleToRateContracts:
    def test_includes_endpoint(self):
        """Hard contract: resampling includes t_end exactly (final step may be shorter)."""
        times = np.arange(0.0, 1.0 + 1e-12, 0.1)  # include 1.0
        df = pd.DataFrame({"time": times, "x": np.arange(len(times), dtype=float)})

        out = resample_to_rate(df, target_rate=5.0)  # dt=0.2
        assert out["time"].iloc[0] == pytest.approx(0.0)
        assert out["time"].iloc[-1] == pytest.approx(1.0)

    def test_downsample_requires_integer_factor(self):
        """Hard contract: downsampling requires current_rate/target_rate to be integer (within eps)."""
        times = np.arange(0.0, 1.0 + 1e-12, 0.1)  # ~10Hz
        df = pd.DataFrame({"time": times, "x": np.sin(times)})

        # 10 / 6 is not integer -> should raise.
        with pytest.raises(ValueError):
            resample_to_rate(df, target_rate=6.0)

    def test_downsample_accepts_close_to_integer_ratio(self):
        """Accept ratios that are close to integer within epsilon."""
        times = np.arange(0.0, 1.0 + 1e-12, 0.1)  # ~10Hz
        df = pd.DataFrame({"time": times, "x": np.sin(times)})

        # Slightly off 5Hz, ratio is close to 2.
        out = resample_to_rate(df, target_rate=5.000001)
        assert out["time"].iloc[-1] == pytest.approx(1.0)
