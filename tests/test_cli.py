"""CLI entry point tests.

Tests that CLI commands can be invoked without crashing.
Uses subprocess to test the actual installed entry points.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCLIHelp:
    """Test --help works for all CLI commands."""

    def test_fmd_analyze_help(self):
        """fmd-analyze --help shows usage information."""
        # Test via python -c to call the function with --help
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-analyze', '--help']; from fmd.analysis.cli import analyze_log; analyze_log()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout
        assert "filename" in result.stdout.lower()

    def test_fmd_generate_help(self):
        """fmd-generate --help shows usage information."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-generate', '--help']; from fmd.analysis.cli import generate_data; generate_data()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout
        assert "filename" in result.stdout.lower()

    def test_fmd_plot_vakaros_help(self):
        """fmd-plot-vakaros --help shows usage information."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-plot-vakaros', '--help']; from fmd.analysis.cli import plot_vakaros; plot_vakaros()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout
        assert "vakaros" in result.stdout.lower()

    def test_fmd_sampling_help(self):
        """fmd-sampling --help shows usage information."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-sampling', '--help']; from fmd.analysis.cli import analyze_sampling_cli; analyze_sampling_cli()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout
        assert "sampling" in result.stdout.lower()


class TestCLIBasicInvocation:
    """Test basic CLI invocations."""

    def test_fmd_generate_creates_file(self):
        """fmd-generate creates a CSV file with random data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.argv = ['fmd-generate', '{output_file}', '-n', '2', '-e', '0.5']; from fmd.analysis.cli import generate_data; generate_data()",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert output_file.exists(), "Output file was not created"

            # Verify it's a valid CSV with data
            content = output_file.read_text()
            lines = content.strip().split("\n")
            assert len(lines) > 1, "CSV should have header and data rows"

    def test_fmd_analyze_missing_file_error(self):
        """fmd-analyze shows error for missing file."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-analyze', 'nonexistent_file.csv']; from fmd.analysis.cli import analyze_log; analyze_log()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should exit with error code
        assert result.returncode != 0

    def test_fmd_sampling_missing_file_error(self):
        """fmd-sampling shows error for missing file."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['fmd-sampling', 'nonexistent_file.csv']; from fmd.analysis.cli import analyze_sampling_cli; analyze_sampling_cli()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should exit with error code
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestCLIEntryPoints:
    """Test that installed entry points work (if package is installed)."""

    @pytest.fixture
    def check_entry_point_installed(self):
        """Check if the fomodynamics package is installed with entry points."""
        result = subprocess.run(
            ["which", "fmd-analyze"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def test_fmd_analyze_entry_point_help(self, check_entry_point_installed):
        """Test fmd-analyze entry point if installed."""
        if not check_entry_point_installed:
            pytest.skip("fmd-analyze entry point not installed")

        result = subprocess.run(
            ["fmd-analyze", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_fmd_generate_entry_point_help(self, check_entry_point_installed):
        """Test fmd-generate entry point if installed."""
        if not check_entry_point_installed:
            pytest.skip("fmd-generate entry point not installed")

        result = subprocess.run(
            ["fmd-generate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_fmd_sampling_entry_point_help(self, check_entry_point_installed):
        """Test fmd-sampling entry point if installed."""
        if not check_entry_point_installed:
            pytest.skip("fmd-sampling entry point not installed")

        result = subprocess.run(
            ["fmd-sampling", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_fmd_plot_vakaros_entry_point_help(self, check_entry_point_installed):
        """Test fmd-plot-vakaros entry point if installed."""
        if not check_entry_point_installed:
            pytest.skip("fmd-plot-vakaros entry point not installed")

        result = subprocess.run(
            ["fmd-plot-vakaros", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()
        assert "vakaros" in result.stdout.lower()
