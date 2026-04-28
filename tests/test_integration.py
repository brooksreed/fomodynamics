"""Integration tests for blur monorepo.

Tests cross-package functionality to ensure fmd.core, fmd.simulator,
and fmd.analysis work together correctly.
"""

import numpy as np
import pytest
import jax.numpy as jnp


class TestCoreSimulatorIntegration:
    """Test fmd.core + fmd.simulator integration."""

    def test_quaternion_used_in_rigid_body(self):
        """Rigid body uses quaternion math (JAX implementation)."""
        from fmd.simulator import RigidBody6DOF, create_state, simulate
        from fmd.simulator import quat_to_euler

        # Create a spinning body
        body = RigidBody6DOF(mass=1.0, inertia=jnp.array([1.0, 1.0, 1.0]))
        initial_state = create_state(angular_velocity=jnp.array([0.1, 0.0, 0.0]))

        result = simulate(body, initial_state, dt=0.01, duration=1.0)

        # Check that quaternion remains normalized
        states = np.asarray(result.states)
        for i in range(len(result.times)):
            quat = states[i, 6:10]
            norm = np.linalg.norm(quat)
            assert np.isclose(norm, 1.0, atol=1e-6)

    def test_dynamic_system_interface(self):
        """Simulator systems implement DynamicSystem interface."""
        from fmd.simulator import RigidBody6DOF, SimplePendulum, DynamicSystem

        # JAX systems inherit from JaxDynamicSystem (DynamicSystem alias)
        assert issubclass(RigidBody6DOF, DynamicSystem)
        assert issubclass(SimplePendulum, DynamicSystem)


class TestSimulatorAnalysisIntegration:
    """Test fmd.simulator + fmd.analysis integration."""

    def test_result_to_datastream(self):
        """Simulation results can be converted to DataStream."""
        from fmd.simulator import simulate, RigidBody6DOF, create_state, result_with_meta
        from fmd.simulator.output import result_to_datastream
        from fmd.analysis.core import DataStream

        # Run a simulation
        body = RigidBody6DOF(mass=1.0, inertia=jnp.array([1.0, 1.0, 1.0]))
        result = simulate(body, create_state(), dt=0.01, duration=1.0)

        # Add metadata for analysis
        rich_result = result_with_meta(body, result)

        # Convert to DataStream
        stream = result_to_datastream(rich_result)

        assert isinstance(stream, DataStream)
        assert len(stream) == len(result.times)
        assert "roll" in stream.columns
        assert "yaw" in stream.columns
        assert stream.source_rate is not None

    def test_datastream_circular_operations_on_simulation(self):
        """DataStream circular operations work on simulation outputs."""
        from fmd.simulator import simulate, RigidBody6DOF, create_state, result_with_meta
        from fmd.simulator.output import result_to_datastream

        # Create a rotating body
        body = RigidBody6DOF(mass=1.0, inertia=jnp.array([1.0, 1.0, 1.0]))
        initial_state = create_state(angular_velocity=jnp.array([0.0, 0.0, 1.0]))  # Spinning about Z

        result = simulate(body, initial_state, dt=0.01, duration=10.0)
        rich_result = result_with_meta(body, result)
        stream = result_to_datastream(rich_result)

        # The yaw angle should be changing - compute circular mean
        yaw_mean = stream.mean("yaw")
        yaw_std = stream.std("yaw")

        # Mean should be defined (not NaN)
        assert not np.isnan(yaw_mean)
        # Std should be significant since we're spinning
        assert yaw_std > 0.1

    def test_simulation_csv_loadable_by_analysis(self):
        """Simulation output CSV is loadable by fmd.analysis."""
        import tempfile
        import os
        from fmd.simulator import simulate, RigidBody6DOF, create_state, LogWriter, result_with_meta
        from fmd.analysis.loaders import load_file

        # Run simulation and write to CSV
        body = RigidBody6DOF(mass=1.0, inertia=jnp.array([1.0, 1.0, 1.0]))
        result = simulate(body, create_state(), dt=0.01, duration=1.0)
        rich_result = result_with_meta(body, result)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LogWriter(tmpdir)
            csv_path = writer.write(rich_result, "test.csv")

            # Load with fmd.analysis
            loaded = load_file(str(csv_path), schema="dynamic_simulator")

            assert loaded.schema_name == "dynamic_simulator"
            assert "time" in loaded.df.columns
            assert "roll" in loaded.df.columns
            assert len(loaded.df) == len(result.times)

    def test_logwriter_raises_on_bad_output_shape(self):
        """LogWriter must fail fast on non-scalar/non-1D outputs (hard contract)."""
        import tempfile
        from fmd.simulator import simulate, RigidBody6DOF, create_state, LogWriter, RichSimulationResult

        body = RigidBody6DOF(mass=1.0, inertia=jnp.array([1.0, 1.0, 1.0]))
        result = simulate(body, create_state(), dt=0.01, duration=0.5)

        # Create rich result with invalid (2D) output
        rich_result = RichSimulationResult(
            times=np.asarray(result.times),
            states=np.asarray(result.states),
            controls=np.asarray(result.controls),
            state_names=tuple(body.state_names),
            control_names=tuple(body.control_names),
            outputs={"bad": np.zeros((len(result.times), 2))},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LogWriter(tmpdir)
            with pytest.raises(ValueError):
                writer.write(rich_result, "bad.csv")


class TestCoreAnalysisIntegration:
    """Test fmd.core + fmd.analysis integration."""

    def test_analysis_uses_core_operations(self):
        """Analysis operations are based on fmd.core."""
        from fmd.analysis.operations import wrap_angle, circular_subtract, circular_mean
        from fmd.core import wrap_angle as core_wrap, circular_subtract as core_sub

        # Verify they produce the same results
        angles = np.array([0.1, 3.5, -3.5])
        assert np.allclose(wrap_angle(angles), core_wrap(angles))

        a = np.radians(10)
        b = np.radians(350)
        assert np.isclose(circular_subtract(a, b), core_sub(a, b))


class TestFullPipelineIntegration:
    """Test complete simulation -> analysis pipeline."""

    def test_pendulum_simulation_analysis(self):
        """Full pipeline: pendulum simulation to analysis."""
        from fmd.simulator import simulate, SimplePendulum, result_with_meta
        from fmd.simulator.output import result_to_datastream
        from fmd.simulator.params import PENDULUM_1M

        # Simulate pendulum
        pendulum = SimplePendulum(PENDULUM_1M)
        initial_angle = np.radians(30)
        initial_state = jnp.array([initial_angle, 0.0])

        result = simulate(pendulum, initial_state, dt=0.001, duration=5.0)
        rich_result = result_with_meta(pendulum, result)

        # Convert to DataStream
        stream = result_to_datastream(rich_result, name="pendulum")

        assert len(stream) > 0
        assert stream.name == "pendulum"

        # Pendulum theta column should exist
        assert "theta" in stream.columns

    def test_rigid_body_drop_test_analysis(self):
        """Full pipeline: drop test simulation to analysis."""
        from fmd.simulator import simulate, RigidBody6DOF, create_state, Gravity, result_with_meta
        from fmd.simulator.output import result_to_datastream

        # Drop a body from height
        mass = 1.0
        body = RigidBody6DOF(
            mass=mass,
            inertia=jnp.array([1.0, 1.0, 1.0]),
            components=[Gravity(mass=mass)],
        )
        initial_state = create_state(position=jnp.array([0.0, 0.0, -100.0]))  # 100m up

        result = simulate(body, initial_state, dt=0.01, duration=2.0)
        rich_result = result_with_meta(body, result)
        stream = result_to_datastream(rich_result)

        # Analyze: velocity should increase linearly
        vel_w = stream.df["vel_w"].values

        # Final velocity should be significant (falling under gravity)
        assert vel_w[-1] > 10  # Should be about 20 m/s after 2 seconds
