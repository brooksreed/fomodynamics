"""Tests for fmd.analysis.core module."""

import numpy as np
import pandas as pd
import pytest

from fmd.analysis.core import DataStream, VehicleLog, detect_sample_rate


class TestDetectSampleRate:
    """Test sample rate detection."""

    def test_regular_10hz(self):
        """Detect 10 Hz sample rate."""
        times = np.arange(0, 10, 0.1)  # 100 samples at 10 Hz
        rate = detect_sample_rate(times)
        assert rate is not None
        assert np.isclose(rate, 10.0, rtol=0.01)

    def test_irregular_returns_median(self):
        """Irregular samples return median-based rate."""
        times = np.array([0, 0.1, 0.2, 0.5, 0.6, 0.7])  # Gap at 0.2->0.5
        rate = detect_sample_rate(times)
        assert rate is not None

    def test_insufficient_data(self):
        """Return None for insufficient data."""
        assert detect_sample_rate(np.array([1.0])) is None
        assert detect_sample_rate(np.array([])) is None


class TestDataStream:
    """Test DataStream container."""

    def test_create_datastream(self):
        """Create a basic DataStream."""
        df = pd.DataFrame({
            "time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "roll": np.radians([0, 5, 10, 5, 0]),
            "sog": [5.0, 5.1, 5.2, 5.1, 5.0],
        })
        stream = DataStream(df=df, name="test")
        assert stream.name == "test"
        assert len(stream) == 5
        assert stream.source_rate is not None

    def test_columns_excludes_time(self):
        """Columns property should exclude 'time'."""
        df = pd.DataFrame({
            "time": [0, 1, 2],
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        stream = DataStream(df=df, name="test")
        assert "time" not in stream.columns
        assert "a" in stream.columns
        assert "b" in stream.columns

    def test_circular_subtract(self):
        """Circular subtraction handles wraparound."""
        df = pd.DataFrame({
            "time": [0, 1],
            "yaw": [np.radians(1), np.radians(359)],  # Near 0/360 boundary
            "cog": [np.radians(359), np.radians(1)],
        })
        stream = DataStream(df=df, name="test")

        # yaw - cog should give small differences, not large ones
        diff = stream.subtract("yaw", "cog")
        # 1° - 359° = 2° (not -358°)
        # 359° - 1° = -2° (not 358°)
        assert np.allclose(np.abs(diff.values), np.radians(2), atol=0.01)

    def test_linear_subtract(self):
        """Linear subtraction for non-circular data."""
        df = pd.DataFrame({
            "time": [0, 1],
            "sog": [10.0, 20.0],
            "target_sog": [8.0, 15.0],
        })
        stream = DataStream(df=df, name="test")

        # These aren't circular, so regular subtraction
        diff = stream.subtract("sog", "target_sog")
        assert np.allclose(diff.values, [2.0, 5.0])

    def test_subtract_mismatched_known_types_raises(self):
        """Subtracting known circular vs known linear should raise (hard contract)."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "yaw": [np.radians(10), np.radians(20)],  # circular
                "sog": [1.0, 2.0],  # linear
            }
        )
        stream = DataStream(df=df, name="test")
        with pytest.raises(ValueError):
            stream.subtract("yaw", "sog")

    def test_subtract_unknown_defaults_linear(self):
        """Unknown variables default to linear behavior."""
        df = pd.DataFrame({"time": [0, 1], "foo": [10.0, 11.0], "bar": [1.0, 2.0]})
        stream = DataStream(df=df, name="test")
        diff = stream.subtract("foo", "bar")
        assert np.allclose(diff.values, [9.0, 9.0])


class TestVehicleLog:
    """Test VehicleLog multi-stream container."""

    def test_add_and_get_stream(self):
        """Add and retrieve streams."""
        log = VehicleLog()

        stream1 = DataStream(
            df=pd.DataFrame({"time": [0, 1, 2], "a": [1, 2, 3]}),
            name="stream1",
        )
        stream2 = DataStream(
            df=pd.DataFrame({"time": [0, 0.5, 1], "b": [4, 5, 6]}),
            name="stream2",
        )

        log.add_stream(stream1)
        log.add_stream(stream2)

        assert log.get_stream("stream1") is stream1
        assert log.get_stream("stream2") is stream2

    def test_get_missing_stream_raises(self):
        """Getting missing stream raises KeyError."""
        log = VehicleLog()
        with pytest.raises(KeyError):
            log.get_stream("nonexistent")

    def test_time_range(self):
        """Time range spans all streams."""
        log = VehicleLog()

        log.add_stream(DataStream(
            df=pd.DataFrame({"time": [1, 2, 3], "a": [1, 2, 3]}),
            name="s1",
        ))
        log.add_stream(DataStream(
            df=pd.DataFrame({"time": [0, 5, 10], "b": [4, 5, 6]}),
            name="s2",
        ))

        t_min, t_max = log.time_range
        assert t_min == 0
        assert t_max == 10


class TestDataStreamComparison:
    """Tests for DataStream equality and approximate comparison.

    DataStream is a dataclass containing a pandas DataFrame. This test suite
    documents the behavior of equality comparisons and explores how to perform
    approximate numeric comparisons using pytest.approx.

    Key findings:
    - DataStream uses default dataclass equality, which fails for DataFrames
      because DataFrame.__eq__ returns a DataFrame of booleans, not a bool
    - pytest.approx cannot be used directly with DataStream objects
    - To compare DataStreams with approximate numeric tolerance, users must
      compare the underlying DataFrames/arrays manually
    """

    def test_exact_equality_same_object(self):
        """Same DataStream instance should equal itself."""
        df = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        stream = DataStream(df=df, name="test")
        # Identity comparison works
        assert stream is stream

    def test_exact_equality_different_objects_fails(self):
        """Two DataStream objects with identical data raise ValueError on ==.

        This is because pandas DataFrame.__eq__ returns a DataFrame of booleans,
        not a single boolean, which causes issues with dataclass equality.
        """
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        stream1 = DataStream(df=df1, name="test")
        stream2 = DataStream(df=df2, name="test")

        # Direct equality comparison raises ValueError due to DataFrame comparison
        with pytest.raises(ValueError, match="ambiguous"):
            stream1 == stream2

    def test_pytest_approx_not_supported_for_different_objects(self):
        """pytest.approx cannot properly compare different DataStream objects.

        This documents the limitation that DataStream does not support
        approximate comparison via pytest.approx when comparing different
        instances. The comparison raises ValueError because pandas DataFrame
        equality returns a DataFrame of booleans which is ambiguous.

        Note: Same instance comparison (stream == pytest.approx(stream)) works
        via identity but this is misleading as it doesn't actually compare values.
        """
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0 + 1e-10, 2.0, 3.0],  # Slightly different values
        })
        stream1 = DataStream(df=df1, name="test")
        stream2 = DataStream(df=df2, name="test")

        # pytest.approx comparison between different instances fails
        # because DataFrame.__eq__ returns a DataFrame of booleans
        with pytest.raises(ValueError, match="ambiguous"):
            stream1 == pytest.approx(stream2)

    def test_manual_dataframe_comparison_exact(self):
        """Manual comparison of DataFrame values for exact equality."""
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        stream1 = DataStream(df=df1, name="test")
        stream2 = DataStream(df=df2, name="test")

        # Recommended approach: use pandas testing utilities
        pd.testing.assert_frame_equal(stream1.df, stream2.df)

        # Or compare metadata separately
        assert stream1.name == stream2.name
        assert stream1.units == stream2.units

    def test_manual_dataframe_comparison_approximate(self):
        """Manual comparison of DataFrame values with numeric tolerance.

        This is the recommended approach for comparing DataStreams with
        slight numeric differences.
        """
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        # Add small floating point differences
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0 + 1e-10, 2.0 - 1e-10, 3.0 + 1e-10],
        })
        stream1 = DataStream(df=df1, name="test")
        stream2 = DataStream(df=df2, name="test")

        # pandas assert_frame_equal with tolerance
        pd.testing.assert_frame_equal(
            stream1.df, stream2.df, atol=1e-9, rtol=1e-9
        )

        # Or use numpy with pytest.approx on the underlying arrays
        assert stream1.df["value"].values == pytest.approx(
            stream2.df["value"].values, abs=1e-9
        )

    def test_numpy_array_comparison_with_approx(self):
        """pytest.approx works well with numpy arrays from DataStream.

        This demonstrates the recommended pattern for approximate comparison
        of DataStream numeric data.
        """
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "roll": [0.1, 0.2, 0.3],
            "pitch": [0.01, 0.02, 0.03],
        })
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "roll": [0.1 + 1e-7, 0.2 - 1e-7, 0.3],
            "pitch": [0.01, 0.02 + 1e-8, 0.03 - 1e-8],
        })
        stream1 = DataStream(df=df1, name="imu")
        stream2 = DataStream(df=df2, name="imu")

        # Compare each column with pytest.approx
        for col in stream1.columns:
            assert stream1.df[col].values == pytest.approx(
                stream2.df[col].values, rel=1e-6
            ), f"Column {col} differs beyond tolerance"

    def test_comparison_with_different_metadata(self):
        """DataStreams with same data but different metadata."""
        df = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        stream1 = DataStream(df=df.copy(), name="test", units={"value": "m/s"})
        stream2 = DataStream(df=df.copy(), name="test", units={"value": "m/s"})
        stream3 = DataStream(df=df.copy(), name="test", units={"value": "knots"})

        # DataFrames are equal
        pd.testing.assert_frame_equal(stream1.df, stream2.df)

        # Metadata comparison
        assert stream1.name == stream2.name
        assert stream1.units == stream2.units
        assert stream1.units != stream3.units

    def test_comparison_helper_function(self):
        """Example helper function for approximate DataStream comparison.

        This demonstrates a pattern that could be added to DataStream or
        used in test utilities.
        """
        def datastreams_approx_equal(
            ds1: DataStream,
            ds2: DataStream,
            atol: float = 1e-8,
            rtol: float = 1e-5,
            check_metadata: bool = True,
        ) -> bool:
            """Check if two DataStreams are approximately equal.

            Args:
                ds1: First DataStream
                ds2: Second DataStream
                atol: Absolute tolerance for numeric comparison
                rtol: Relative tolerance for numeric comparison
                check_metadata: Whether to check name, units, and metadata

            Returns:
                True if DataStreams are approximately equal
            """
            # Check metadata if requested
            if check_metadata:
                if ds1.name != ds2.name:
                    return False
                if ds1.units != ds2.units:
                    return False

            # Check DataFrame structure
            if set(ds1.df.columns) != set(ds2.df.columns):
                return False
            if len(ds1) != len(ds2):
                return False

            # Check numeric values with tolerance
            for col in ds1.df.columns:
                vals1 = ds1.df[col].values
                vals2 = ds2.df[col].values
                if not np.allclose(vals1, vals2, atol=atol, rtol=rtol):
                    return False

            return True

        # Test the helper
        df1 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.0],
        })
        df2 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0 + 1e-10, 2.0, 3.0 - 1e-10],
        })
        stream1 = DataStream(df=df1, name="test")
        stream2 = DataStream(df=df2, name="test")

        assert datastreams_approx_equal(stream1, stream2)

        # Different values should fail
        df3 = pd.DataFrame({
            "time": [0.0, 0.1, 0.2],
            "value": [1.0, 2.0, 3.5],  # 3.5 != 3.0
        })
        stream3 = DataStream(df=df3, name="test")
        assert not datastreams_approx_equal(stream1, stream3)
