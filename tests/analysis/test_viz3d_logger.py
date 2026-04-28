"""Integration test for Rerun .rrd writer."""

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if rerun is not installed
rerun = pytest.importorskip("rerun", reason="rerun-sdk not installed")

from fmd.simulator import Moth3D, simulate
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.moth_forces_extract import extract_forces
from fmd.analysis.viz3d.moth_logger import write_moth_rrd


@pytest.fixture
def sim_data():
    """Run a short simulation for testing."""
    moth = Moth3D(MOTH_BIEKER_V3)
    result = simulate(moth, moth.default_state(), dt=0.01, duration=0.5)
    forces = extract_forces(moth, result)
    return MOTH_BIEKER_V3, result, forces


class TestWriteMothRrd:
    def test_creates_rrd_file(self, sim_data, tmp_path):
        params, result, forces = sim_data
        output = tmp_path / "test.rrd"
        returned_path = write_moth_rrd(params, result, output, forces=forces)
        assert returned_path == output
        assert output.exists()

    def test_rrd_file_nonempty(self, sim_data, tmp_path):
        params, result, forces = sim_data
        output = tmp_path / "test.rrd"
        write_moth_rrd(params, result, output, forces=forces)
        assert output.stat().st_size > 0

    def test_custom_force_scale(self, sim_data, tmp_path):
        params, result, forces = sim_data
        output = tmp_path / "test_scaled.rrd"
        write_moth_rrd(params, result, output, forces=forces, force_scale=0.01)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_without_forces(self, sim_data, tmp_path):
        """Test that forces are optional."""
        params, result, _ = sim_data
        output = tmp_path / "no_forces.rrd"
        write_moth_rrd(params, result, output)
        assert output.exists()
        assert output.stat().st_size > 0
