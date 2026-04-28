"""Output utilities for saving simulation results.

Provides CSV output compatible with the data-analysis repository's
schema-based loader system.

Note: This module was renamed from 'logging.py' to avoid shadowing
Python's standard library logging module.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from fmd.simulator.integrator import SimulationResult, RichSimulationResult


# Type alias for either result type
AnyResult = Union[SimulationResult, RichSimulationResult]


def _get_state_names(result: AnyResult, num_states: int) -> list[str]:
    """Get state names from result, or generate default names."""
    if hasattr(result, 'state_names') and result.state_names:
        return list(result.state_names)
    return [f"state_{i}" for i in range(num_states)]


def _get_control_names(result: AnyResult, num_controls: int) -> list[str]:
    """Get control names from result, or generate default names."""
    if hasattr(result, 'control_names') and result.control_names:
        return list(result.control_names)
    return [f"control_{i}" for i in range(num_controls)]


def _get_outputs(result: AnyResult) -> dict[str, np.ndarray]:
    """Get outputs dict from result, or empty dict."""
    if hasattr(result, 'outputs') and result.outputs:
        return result.outputs
    return {}


class LogWriter:
    """Write simulation results to CSV files.

    Output format is compatible with the data-analysis repository.
    All values are in SI units.

    Works with both SimulationResult (JIT-safe) and RichSimulationResult.
    For JIT-safe results, state/control names are auto-generated.
    """

    def __init__(
        self,
        output_dir: str | Path = "data",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize the log writer.

        Args:
            output_dir: Directory to write output files
            metadata: Optional metadata to include with logs
        """
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        result: AnyResult,
        filename: str,
        include_outputs: bool = True,
    ) -> Path:
        """Write simulation result to CSV file.

        Args:
            result: SimulationResult or RichSimulationResult from simulate()
            filename: Output filename (without directory)
            include_outputs: Whether to include derived outputs

        Returns:
            Path to the written file
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".csv")

        # Convert to numpy arrays
        # Convert JAX arrays to numpy for CSV serialization (np.asarray is a no-op for numpy inputs)
        times = np.asarray(result.times)
        states = np.asarray(result.states)
        controls = np.asarray(result.controls)
        outputs = _get_outputs(result)

        # Get names
        state_names = _get_state_names(result, states.shape[1] if states.ndim > 1 else 1)
        control_names = _get_control_names(result, controls.shape[1] if controls.ndim > 1 else 1)

        # Validate outputs
        if include_outputs:
            n = len(times)
            for name, values in outputs.items():
                if np.isscalar(values):
                    continue
                arr = np.asarray(values)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Unsupported output shape for '{name}': {arr.shape}. "
                        "Outputs must be scalar or 1D arrays."
                    )
                if len(arr) != n:
                    raise ValueError(
                        f"Output '{name}' has length {len(arr)} but expected {n} (len(times))."
                    )

        # Build column headers
        headers = ["time"] + state_names
        if control_names:
            headers.extend(control_names)
        if include_outputs:
            headers.extend(outputs.keys())

        # Build data array
        num_rows = len(times)
        data = np.zeros((num_rows, len(headers)))

        # Fill in data
        col = 0
        data[:, col] = times
        col += 1

        for i in range(len(state_names)):
            data[:, col] = states[:, i]
            col += 1

        for i in range(len(control_names)):
            data[:, col] = controls[:, i]
            col += 1

        if include_outputs:
            for name in outputs.keys():
                values = outputs[name]
                arr = np.asarray(values)
                if np.isscalar(arr) or arr.ndim == 0:
                    data[:, col] = float(arr)
                elif arr.ndim == 1:
                    data[:, col] = arr
                col += 1

        # Write CSV
        with open(filepath, "w") as f:
            f.write(",".join(headers) + "\n")
            for row in data:
                f.write(",".join(f"{v:.10g}" for v in row) + "\n")

        return filepath

    def write_metadata(self, filename: str, extra: Optional[dict] = None) -> Path:
        """Write metadata JSON file.

        Args:
            filename: Base filename (will add .meta.json suffix)
            extra: Additional metadata to include

        Returns:
            Path to the written file
        """
        filepath = self.output_dir / f"{Path(filename).stem}.meta.json"

        metadata = {
            **self.metadata,
            **(extra or {}),
            "units": {
                "time": "s",
                "pos_n": "m",
                "pos_e": "m",
                "pos_d": "m",
                "vel_u": "m/s",
                "vel_v": "m/s",
                "vel_w": "m/s",
                "omega_p": "rad/s",
                "omega_q": "rad/s",
                "omega_r": "rad/s",
                "roll": "rad",
                "pitch": "rad",
                "yaw": "rad",
            },
            "coordinate_frame": "NED",
            "quaternion_convention": "scalar_first",
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath


def _compute_euler_from_quaternion(states: np.ndarray, state_names: list[str]) -> dict[str, np.ndarray]:
    """Compute Euler angles from quaternion if present in state vector.

    Returns:
        Dict with roll, pitch, yaw arrays if quaternion found, else empty dict.
    """
    # Intentionally uses numpy quat_to_euler from fmd.core (not JAX version from
    # blur.simulator.quaternion) because states are already converted to numpy at this point.
    from fmd.core.quaternion import quat_to_euler

    # Check if quaternion states exist
    quat_indices = []
    for qname in ["qw", "qx", "qy", "qz"]:
        if qname in state_names:
            quat_indices.append(state_names.index(qname))

    if len(quat_indices) != 4:
        return {}

    # Extract quaternions and compute Euler angles
    quaternions = states[:, quat_indices]  # (n_steps, 4)

    roll = np.zeros(len(quaternions))
    pitch = np.zeros(len(quaternions))
    yaw = np.zeros(len(quaternions))

    for i, q in enumerate(quaternions):
        euler = quat_to_euler(q)
        roll[i] = euler[0]
        pitch[i] = euler[1]
        yaw[i] = euler[2]

    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def compute_wave_outputs(
    result: AnyResult,
    env,
) -> dict[str, np.ndarray]:
    """Compute wave-related outputs at body position for each timestep.

    Args:
        result: SimulationResult or RichSimulationResult from simulate()
        env: Environment with wave_field (if None or no wave_field, returns empty dict)

    Returns:
        Dict with wave output arrays:
        - wave_elevation: surface elevation at body position (m)
        - wave_vel_u: orbital velocity x-component at body position (m/s)
        - wave_vel_v: orbital velocity y-component at body position (m/s)
        - wave_vel_w: orbital velocity z-component at body position (m/s)
    """
    if env is None or getattr(env, 'wave_field', None) is None:
        return {}

    import jax
    import jax.numpy as jnp

    times = np.asarray(result.times)
    states = np.asarray(result.states)
    n_states = states.shape[1] if states.ndim > 1 else 1

    wf = env.wave_field

    # Extract positions based on state vector size
    times_jax = jnp.asarray(times)
    states_jax = jnp.asarray(states)

    def _wave_at_step(state, t):
        if n_states == 13:
            x, y, z = state[0], state[1], state[2]
        elif n_states == 5:
            x, y, z = 0.0, 0.0, state[0]
        else:
            x, y, z = 0.0, 0.0, 0.0

        eta = wf.elevation(x, y, t)
        vel = wf.orbital_velocity(x, y, z, t)
        return eta, vel

    etas, vels = jax.vmap(_wave_at_step)(states_jax, times_jax)

    return {
        "wave_elevation": np.asarray(etas),
        "wave_vel_u": np.asarray(vels[:, 0]),
        "wave_vel_v": np.asarray(vels[:, 1]),
        "wave_vel_w": np.asarray(vels[:, 2]),
    }


def result_to_dataframe(result: AnyResult) -> "pd.DataFrame":
    """Convert SimulationResult to pandas DataFrame.

    Requires pandas to be installed.

    Args:
        result: SimulationResult or RichSimulationResult from simulate()

    Returns:
        DataFrame with time, states, controls, and outputs.
        For RigidBody6DOF results, roll/pitch/yaw are automatically derived from quaternion.
    """
    import pandas as pd

    # Convert to numpy
    times = np.asarray(result.times)
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    outputs = _get_outputs(result)

    # Get names
    state_names = _get_state_names(result, states.shape[1] if states.ndim > 1 else 1)
    control_names = _get_control_names(result, controls.shape[1] if controls.ndim > 1 else 1)

    data = {"time": times}

    for i, name in enumerate(state_names):
        data[name] = states[:, i]

    for i, name in enumerate(control_names):
        data[name] = controls[:, i]

    for name, values in outputs.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            data[name] = arr

    # Compute derived Euler angles from quaternion if present
    euler_outputs = _compute_euler_from_quaternion(states, state_names)
    for name, values in euler_outputs.items():
        if name not in data:  # Don't override if already in outputs
            data[name] = values

    return pd.DataFrame(data)


def result_to_datastream(
    result: AnyResult,
    name: str = "simulation",
) -> "DataStream":
    """Convert SimulationResult to DataStream for analysis.

    This provides tight integration between blur.simulator and blur.analysis,
    enabling simulation results to be analyzed with circular-aware operations,
    resampling, and other DataStream features.

    Requires analysis dependencies (included in default `uv sync`).

    Args:
        result: SimulationResult or RichSimulationResult from simulate()
        name: Name for the DataStream (default: "simulation")

    Returns:
        DataStream with time, states, controls, and outputs

    Example:
        >>> from fmd.simulator import simulate, RigidBody6DOF, create_state
        >>> from fmd.simulator.output import result_to_datastream
        >>>
        >>> body = RigidBody6DOF(mass=1.0, inertia=[1, 1, 1])
        >>> result = simulate(body, create_state(), dt=0.01, duration=10.0)
        >>> stream = result_to_datastream(result)
        >>> stream.mean("yaw")  # Circular mean of heading
    """
    from fmd.analysis.core import DataStream

    # Build DataFrame first
    df = result_to_dataframe(result)

    # Get names for metadata
    states = np.asarray(result.states)
    controls = np.asarray(result.controls)
    state_names = _get_state_names(result, states.shape[1] if states.ndim > 1 else 1)
    control_names = _get_control_names(result, controls.shape[1] if controls.ndim > 1 else 1)

    # Determine SI units for known columns
    units = {
        "time": "s",
        # Position
        "pos_n": "m",
        "pos_e": "m",
        "pos_d": "m",
        # Velocity
        "vel_u": "m/s",
        "vel_v": "m/s",
        "vel_w": "m/s",
        # Angular velocity
        "omega_p": "rad/s",
        "omega_q": "rad/s",
        "omega_r": "rad/s",
        # Euler angles (derived)
        "roll": "rad",
        "pitch": "rad",
        "yaw": "rad",
    }

    # Only include units for columns that exist
    existing_units = {k: v for k, v in units.items() if k in df.columns}

    return DataStream(
        df=df,
        name=name,
        units=existing_units,
        metadata={
            "source": "blur.simulator",
            "state_names": state_names,
            "control_names": control_names,
            "num_steps": len(result.times),
        },
    )
