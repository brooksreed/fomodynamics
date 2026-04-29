"""Tests for the generic RigidBody6DOF Rerun logger."""

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if rerun is not installed
rerun = pytest.importorskip("rerun", reason="rerun-sdk not installed")

from fmd.analysis.viz3d.generic_logger import write_rrd, ForceVizData
from fmd.simulator.integrator import SimulationResult


def make_mock_6dof_result(num_steps: int = 50) -> SimulationResult:
    """Create a mock 6DOF simulation result (free fall)."""
    dt = 0.01
    times = np.arange(num_steps) * dt

    # 13-state: [pos_n, pos_e, pos_d, vel_u, vel_v, vel_w, qw, qx, qy, qz, omega_p, omega_q, omega_r]
    states = np.zeros((num_steps, 13))

    # Simple free fall: pos_d increases, vel_w increases (downward)
    g = 9.81
    for i, t in enumerate(times):
        states[i, 0] = 0.0           # pos_n
        states[i, 1] = 0.0           # pos_e
        states[i, 2] = 0.5 * g * t**2  # pos_d (positive down)
        states[i, 3] = 0.0           # vel_u
        states[i, 4] = 0.0           # vel_v
        states[i, 5] = g * t         # vel_w (positive down)
        states[i, 6] = 1.0           # qw (identity quaternion)
        states[i, 7] = 0.0           # qx
        states[i, 8] = 0.0           # qy
        states[i, 9] = 0.0           # qz
        states[i, 10] = 0.0          # omega_p
        states[i, 11] = 0.0          # omega_q
        states[i, 12] = 0.0          # omega_r

    # Simple controls (2 channels)
    controls = np.zeros((num_steps, 2))
    controls[:, 0] = 0.1  # constant control 1
    controls[:, 1] = -0.1  # constant control 2

    return SimulationResult(
        times=times,
        states=states,
        controls=controls,
    )


def make_rotating_result(num_steps: int = 50) -> SimulationResult:
    """Create a 6DOF result with rotation."""
    dt = 0.01
    times = np.arange(num_steps) * dt

    states = np.zeros((num_steps, 13))
    omega = 1.0  # rad/s rotation rate about body z-axis

    for i, t in enumerate(times):
        states[i, 0] = t * 1.0       # pos_n (moving north)
        states[i, 1] = 0.0           # pos_e
        states[i, 2] = -0.5          # pos_d (above water)
        states[i, 3] = 1.0           # vel_u (forward)
        states[i, 4] = 0.0           # vel_v
        states[i, 5] = 0.0           # vel_w
        # Quaternion for rotation about z (scalar-first)
        angle = omega * t
        states[i, 6] = np.cos(angle / 2)  # qw
        states[i, 7] = 0.0                 # qx
        states[i, 8] = 0.0                 # qy
        states[i, 9] = np.sin(angle / 2)  # qz
        states[i, 10] = 0.0          # omega_p
        states[i, 11] = 0.0          # omega_q
        states[i, 12] = omega        # omega_r

    controls = np.zeros((num_steps, 2))

    return SimulationResult(times=times, states=states, controls=controls)


class TestWriteRrd:
    def test_creates_rrd_file(self, tmp_path):
        """Basic test that file gets created."""
        result = make_mock_6dof_result()
        output = tmp_path / "test.rrd"
        returned_path = write_rrd(result, output)
        assert returned_path == output
        assert output.exists()

    def test_rrd_file_nonempty(self, tmp_path):
        """File should have content."""
        result = make_mock_6dof_result()
        output = tmp_path / "test.rrd"
        write_rrd(result, output)
        assert output.stat().st_size > 0

    def test_with_rotating_body(self, tmp_path):
        """Test with a rotating body."""
        result = make_rotating_result()
        output = tmp_path / "rotating.rrd"
        write_rrd(result, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_with_forces(self, tmp_path):
        """Test with force visualization data."""
        result = make_mock_6dof_result()
        num_steps = len(result.times)

        # Create force data
        thrust = ForceVizData(
            name="thrust",
            origins=np.tile([0.0, 0.0, 0.0], (num_steps, 1)),
            vectors=np.tile([100.0, 0.0, 0.0], (num_steps, 1)),
            color=(255, 0, 0, 255),
        )

        output = tmp_path / "with_forces.rrd"
        write_rrd(result, output, forces=[thrust])
        assert output.exists()
        assert output.stat().st_size > 0

    def test_with_multiple_forces(self, tmp_path):
        """Test with multiple force visualizations."""
        result = make_mock_6dof_result()
        num_steps = len(result.times)

        thrust = ForceVizData(
            name="thrust",
            origins=np.tile([0.0, 0.0, 0.0], (num_steps, 1)),
            vectors=np.tile([100.0, 0.0, 0.0], (num_steps, 1)),
            color=(255, 0, 0, 255),
        )
        drag = ForceVizData(
            name="drag",
            origins=np.tile([0.0, 0.0, 0.0], (num_steps, 1)),
            vectors=np.tile([-50.0, 0.0, 0.0], (num_steps, 1)),
            color=(0, 255, 0, 255),
        )

        output = tmp_path / "multi_forces.rrd"
        write_rrd(result, output, forces=[thrust, drag])
        assert output.exists()

    def test_with_wireframe(self, tmp_path):
        """Test with custom wireframe geometry."""
        result = make_mock_6dof_result()

        # Simple box wireframe
        wireframe = {
            "box": np.array([
                [-1, -0.5, -0.2],
                [1, -0.5, -0.2],
                [1, 0.5, -0.2],
                [-1, 0.5, -0.2],
                [-1, -0.5, -0.2],
            ]),
        }

        output = tmp_path / "with_wireframe.rrd"
        write_rrd(result, output, wireframe=wireframe)
        assert output.exists()

    def test_with_wireframe_colors(self, tmp_path):
        """Test with colored wireframe."""
        result = make_mock_6dof_result()

        wireframe = {
            "body": np.array([[-1, 0, 0], [1, 0, 0]]),
            "wing": np.array([[0, -1, 0], [0, 1, 0]]),
        }
        wireframe_colors = {
            "body": (255, 0, 0, 255),
            "wing": (0, 0, 255, 255),
        }

        output = tmp_path / "colored_wireframe.rrd"
        write_rrd(result, output, wireframe=wireframe, wireframe_colors=wireframe_colors)
        assert output.exists()

    def test_custom_state_names(self, tmp_path):
        """Test with custom state channel names."""
        result = make_mock_6dof_result()
        state_names = ["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]

        output = tmp_path / "custom_states.rrd"
        write_rrd(result, output, state_names=state_names)
        assert output.exists()

    def test_custom_control_names(self, tmp_path):
        """Test with custom control channel names."""
        result = make_mock_6dof_result()
        control_names = ["throttle", "steering"]

        output = tmp_path / "custom_controls.rrd"
        write_rrd(result, output, control_names=control_names)
        assert output.exists()

    def test_custom_force_scale(self, tmp_path):
        """Test with custom force scale."""
        result = make_mock_6dof_result()
        num_steps = len(result.times)

        thrust = ForceVizData(
            name="thrust",
            origins=np.tile([0.0, 0.0, 0.0], (num_steps, 1)),
            vectors=np.tile([1000.0, 0.0, 0.0], (num_steps, 1)),
        )

        output = tmp_path / "scaled_forces.rrd"
        write_rrd(result, output, forces=[thrust], force_scale=0.01)
        assert output.exists()

    def test_custom_application_id(self, tmp_path):
        """Test with custom Rerun application ID."""
        result = make_mock_6dof_result()

        output = tmp_path / "custom_app.rrd"
        write_rrd(result, output, application_id="my_custom_sim")
        assert output.exists()

    def test_rejects_insufficient_states(self, tmp_path):
        """Should raise error if states don't have 13 columns."""
        times = np.array([0.0, 0.1, 0.2])
        states = np.zeros((3, 6))  # Only 6 states
        controls = np.zeros((3, 2))

        result = SimulationResult(times=times, states=states, controls=controls)
        output = tmp_path / "bad.rrd"

        with pytest.raises(ValueError, match="at least 13 states"):
            write_rrd(result, output)


class TestForceVizData:
    def test_default_color(self):
        """Default color should be blue."""
        force = ForceVizData(
            name="test",
            origins=np.zeros((10, 3)),
            vectors=np.ones((10, 3)),
        )
        assert force.color == (100, 100, 255, 255)

    def test_custom_color(self):
        """Should accept custom color."""
        force = ForceVizData(
            name="test",
            origins=np.zeros((10, 3)),
            vectors=np.ones((10, 3)),
            color=(255, 0, 0, 255),
        )
        assert force.color == (255, 0, 0, 255)
