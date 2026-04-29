"""Tests for fmd.analysis.loaders module."""

import tempfile
import os
from pathlib import Path
import pytest
import pandas as pd

from fmd.analysis.loaders import load_file, load_stream, list_schemas, registry


class TestSchemaRegistry:
    """Test schema registry."""

    def test_list_schemas(self):
        """List available schemas."""
        schemas = list_schemas()
        assert "test_data" in schemas
        assert "vakaros_atlas_telemetry" in schemas
        assert "dynamic_simulator" in schemas


class TestTestDataSchema:
    """Test loading test_data format."""

    def test_load_test_data(self):
        """Load test data CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,data1,data2\n")
            f.write("0.0,1.0,2.0\n")
            f.write("0.1,1.1,2.1\n")
            f.write("0.2,1.2,2.2\n")

        try:
            result = load_file(f.name)
            assert result.schema_name == "test_data"
            assert "time" in result.df.columns
            assert "data1" in result.df.columns
            assert len(result.df) == 3
        finally:
            os.unlink(f.name)

    def test_auto_detect_test_data(self):
        """Auto-detect test_data schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,data1\n")
            f.write("0.0,1.0\n")

        try:
            result = load_file(f.name)
            assert result.schema_name == "test_data"
        finally:
            os.unlink(f.name)


class TestLoadStream:
    """Test load_stream function."""

    def test_load_stream_returns_datastream(self):
        """load_stream returns a DataStream."""
        from fmd.analysis.core import DataStream

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,data1,data2\n")
            f.write("0.0,1.0,2.0\n")
            f.write("0.1,1.1,2.1\n")

        try:
            stream = load_stream(f.name)
            assert isinstance(stream, DataStream)
            assert stream.source_rate is not None
        finally:
            os.unlink(f.name)


class TestSchemaValidation:
    """Test schema validation."""

    def test_missing_columns_raises(self):
        """Missing required columns raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("not_time,not_data\n")
            f.write("0.0,1.0\n")
            f.name

        try:
            with pytest.raises(ValueError):
                load_file(f.name)
        finally:
            os.unlink(f.name)


class TestVakarosAtlasTelemetrySchema:
    """Test loading vakaros_atlas_telemetry schema from a real fixture file."""

    def test_load_fixture_test_vakaros_data(self):
        """Load the repository fixture and verify normalization + units."""
        # Fixture lives alongside this test module under tests/data/.
        # parents[1] is tests/ (parents[0] is tests/analysis/).
        tests_root = Path(__file__).resolve().parents[1]
        fixture_path = tests_root / "data" / "test_vakaros_data.csv"
        assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

        # Explicit schema ensures we're testing this loader specifically.
        result = load_file(str(fixture_path), schema="vakaros_atlas_telemetry")
        assert result.schema_name == "vakaros_atlas_telemetry"

        df = result.df
        expected_cols = [
            "time",
            "timestamp",
            "latitude",
            "longitude",
            "sog",
            "cog",
            "yaw",
            "roll",
            "pitch",
        ]
        for c in expected_cols:
            assert c in df.columns

        # Basic sanity checks on normalization
        assert len(df) > 0
        assert df["time"].iloc[0] == pytest.approx(0.0)
        assert df["time"].is_monotonic_increasing
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_numeric_dtype(df["sog"])
        assert pd.api.types.is_numeric_dtype(df["yaw"])

        # Units should match the schema's advertised output units.
        assert result.units.get("time") == "s"
        assert result.units.get("sog") == "m/s"
        assert result.units.get("yaw") == "rad"
