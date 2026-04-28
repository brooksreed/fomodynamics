"""Tests for Euler integrator.

Validates the forward Euler integration implementation that matches
safe-control-gym's Physics.DYN mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator import (
    SimplePendulum,
    Cartpole,
    PlanarQuadrotor,
    euler_step,
    rk4_step,
    simulate,
    simulate_euler,
    ConstantControl,
)
from fmd.simulator.params import (
    PENDULUM_1M,
    CARTPOLE_SCG,
    PLANAR_QUAD_SCG,
)


class TestEulerStep:
    """Tests for euler_step function."""

    @pytest.fixture
    def pendulum(self) -> SimplePendulum:
        """Simple pendulum for testing."""
        return SimplePendulum(PENDULUM_1M)

    def test_single_step_updates_state(self, pendulum: SimplePendulum) -> None:
        """Euler step produces a state change."""
        initial = jnp.array([0.5, 0.0])  # theta=0.5, omega=0
        control = jnp.array([])  # No control

        new_state = euler_step(pendulum, initial, control, dt=0.01, t=0.0)

        # State should change (angle unchanged, velocity increased due to gravity)
        assert new_state.shape == initial.shape
        # Angle should be unchanged at first step (omega=0)
        np.testing.assert_allclose(new_state[0], initial[0], atol=1e-10)
        # Velocity should change due to gravity
        assert new_state[1] != initial[1]

    def test_euler_error_decreases_with_dt(self, pendulum: SimplePendulum) -> None:
        """Euler error should decrease as dt decreases.

        For first-order methods, error is O(dt), so smaller dt means smaller error.
        We verify the monotonic decrease rather than exact first-order scaling,
        since nonlinear systems don't show perfect asymptotic behavior at
        practical timesteps.
        """
        initial = jnp.array([0.1, 0.0])  # Small angle for more linear behavior
        control = jnp.array([])
        duration = 0.2

        # Use very small dt as "truth"
        dt_small = 0.00001
        state = initial
        t = 0.0
        while t < duration:
            state = euler_step(pendulum, state, control, dt_small, t)
            t += dt_small
        truth = state

        # Compare errors at different dt values
        dts = [0.02, 0.01, 0.005, 0.002]
        errors = []
        for dt in dts:
            state = initial
            t = 0.0
            while t < duration:
                state = euler_step(pendulum, state, control, dt, t)
                t += dt
            errors.append(float(jnp.linalg.norm(state - truth)))

        # Errors should monotonically decrease when dt decreases
        for i in range(len(dts) - 1):
            assert errors[i] > errors[i + 1], \
                f"Error should decrease: dt={dts[i]} error={errors[i]:.2e} vs dt={dts[i+1]} error={errors[i+1]:.2e}"

        # Verify the finest dt has significantly smaller error than coarsest
        assert errors[-1] < 0.1 * errors[0], \
            "Finest dt should have at least 10x smaller error than coarsest"

    def test_euler_vs_rk4_same_direction(self, pendulum: SimplePendulum) -> None:
        """Euler and RK4 should move state in same direction."""
        initial = jnp.array([0.5, 0.0])
        control = jnp.array([])
        dt = 0.01

        euler_state = euler_step(pendulum, initial, control, dt=dt, t=0.0)
        rk4_state = rk4_step(pendulum, initial, control, dt=dt, t=0.0)

        # Both should have velocity become negative (gravity pulling down)
        assert euler_state[1] < 0
        assert rk4_state[1] < 0

    def test_jit_compilation(self, pendulum: SimplePendulum) -> None:
        """euler_step should be JIT-compilable."""
        initial = jnp.array([0.5, 0.0])
        control = jnp.array([])

        @jax.jit
        def jitted_step(state):
            return euler_step(pendulum, state, control, dt=0.01, t=0.0)

        result = jitted_step(initial)
        assert result.shape == initial.shape


class TestSimulateEuler:
    """Tests for simulate_euler function."""

    @pytest.fixture
    def pendulum(self) -> SimplePendulum:
        """Simple pendulum for testing."""
        return SimplePendulum(PENDULUM_1M)

    @pytest.fixture
    def cartpole(self) -> Cartpole:
        """Cartpole with SCG parameters."""
        return Cartpole(CARTPOLE_SCG)

    def test_basic_simulation(self, pendulum: SimplePendulum) -> None:
        """simulate_euler produces valid trajectory."""
        initial = jnp.array([0.5, 0.0])
        result = simulate_euler(pendulum, initial, dt=0.01, duration=1.0)

        # Check result structure
        assert result.times.shape[0] == result.states.shape[0]
        assert result.states.shape[1] == 2
        assert result.times[0] == 0.0
        assert result.times[-1] == pytest.approx(1.0, abs=0.01)

    def test_matches_manual_loop(self, pendulum: SimplePendulum) -> None:
        """simulate_euler should match manual euler_step loop."""
        initial = jnp.array([0.5, 0.0])
        control = jnp.array([])
        dt = 0.02
        duration = 0.5

        # Using simulate_euler
        result = simulate_euler(pendulum, initial, dt=dt, duration=duration)

        # Manual loop
        state = initial
        t = 0.0
        manual_states = [state]
        while t < duration:
            state = euler_step(pendulum, state, control, dt, t)
            t += dt
            manual_states.append(state)

        manual_states = jnp.array(manual_states)

        # Compare final states
        np.testing.assert_allclose(
            result.states[-1], manual_states[-1], rtol=1e-10
        )

    def test_with_control(self, cartpole: Cartpole) -> None:
        """simulate_euler works with control input."""
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])  # Small angle
        control = ConstantControl(jnp.array([1.0]))  # Push right

        result = simulate_euler(
            cartpole, initial, dt=0.02, duration=1.0, control=control
        )

        # Cart should move right (positive x)
        assert result.states[-1, 0] > 0
        # Controls should be recorded
        assert result.controls.shape[1] == 1
        np.testing.assert_allclose(result.controls[0], jnp.array([1.0]))

    def test_euler_less_accurate_than_rk4(self, pendulum: SimplePendulum) -> None:
        """Euler should be less accurate than RK4 for same dt."""
        initial = jnp.array([0.5, 0.0])
        dt = 0.01
        duration = 2.0

        # Reference: RK4 with very small dt
        result_ref = simulate(pendulum, initial, dt=0.0001, duration=duration)
        truth = result_ref.states[-1]

        # Euler with normal dt
        result_euler = simulate_euler(pendulum, initial, dt=dt, duration=duration)
        error_euler = float(jnp.linalg.norm(result_euler.states[-1] - truth))

        # RK4 with same dt
        result_rk4 = simulate(pendulum, initial, dt=dt, duration=duration)
        error_rk4 = float(jnp.linalg.norm(result_rk4.states[-1] - truth))

        # Euler error should be larger
        assert error_euler > error_rk4

    def test_energy_drift_positive(self, pendulum: SimplePendulum) -> None:
        """Euler should show positive energy drift for pendulum.

        Forward Euler is known to add energy to conservative systems,
        unlike backward Euler which removes energy.
        """
        # Start at horizontal (maximum potential energy)
        initial = jnp.array([jnp.pi / 2, 0.0])
        dt = 0.01
        duration = 10.0

        result = simulate_euler(pendulum, initial, dt=dt, duration=duration)

        # Compute mechanical energy: E = 0.5 * L^2 * omega^2 + g * L * (1 - cos(theta))
        # SimplePendulum assumes unit mass, so m=1
        params = PENDULUM_1M
        L, g = params.length, params.g

        def compute_energy(state):
            theta, omega = state[0], state[1]
            KE = 0.5 * L**2 * omega**2
            PE = g * L * (1 - jnp.cos(theta))
            return KE + PE

        E_initial = float(compute_energy(initial))
        E_final = float(compute_energy(result.states[-1]))

        # Energy should increase (forward Euler adds energy)
        assert E_final > E_initial, "Forward Euler should add energy to pendulum"

    def test_time_grid_correct(self, pendulum: SimplePendulum) -> None:
        """Time grid should match dt and duration."""
        initial = jnp.array([0.5, 0.0])
        dt = 0.05
        duration = 1.0

        result = simulate_euler(pendulum, initial, dt=dt, duration=duration)

        # Check time spacing
        time_diffs = jnp.diff(result.times)
        np.testing.assert_allclose(time_diffs, dt, rtol=1e-10)

        # Check endpoints
        assert result.times[0] == 0.0
        assert result.times[-1] == pytest.approx(duration, abs=dt)


class TestEulerForBenchmarkValidation:
    """Tests specifically for benchmark validation use cases."""

    @pytest.fixture
    def cartpole(self) -> Cartpole:
        """Cartpole with SCG parameters."""
        return Cartpole(CARTPOLE_SCG)

    @pytest.fixture
    def planar_quad(self) -> PlanarQuadrotor:
        """PlanarQuadrotor with SCG parameters."""
        return PlanarQuadrotor(PLANAR_QUAD_SCG)

    def test_cartpole_scg_timestep(self, cartpole: Cartpole) -> None:
        """Test with SCG default timestep (20ms)."""
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])
        dt = 0.02  # SCG default
        duration = 2.0

        result = simulate_euler(cartpole, initial, dt=dt, duration=duration)

        # Should complete without error
        assert result.states.shape[0] > 0
        assert jnp.isfinite(result.states).all()

    def test_quadrotor_scg_timestep(self, planar_quad: PlanarQuadrotor) -> None:
        """Test with SCG-compatible timestep for quadrotor."""
        # Hover state
        hover_thrust = PLANAR_QUAD_SCG.hover_thrust_per_rotor
        initial = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        control = ConstantControl(jnp.array([hover_thrust, hover_thrust]))
        dt = 0.01  # 10ms (conservative for stability)
        duration = 1.0

        result = simulate_euler(
            planar_quad, initial, dt=dt, duration=duration, control=control
        )

        # Should hover in place (z stays at 1.0)
        np.testing.assert_allclose(
            result.states[-1, 1], 1.0, atol=0.1,
            err_msg="Quadrotor should maintain hover altitude"
        )

    def test_euler_discretization_matches(self, cartpole: Cartpole) -> None:
        """Euler integration should match Ad = I + A*dt, Bd = B*dt.

        This is the discrete-time model that SCG uses.
        """
        from fmd.simulator import linearize
        from fmd.simulator.linearize import discretize_euler as disc_euler

        x_eq = jnp.zeros(4)
        u_eq = jnp.zeros(1)
        dt = 0.02

        # Get linearization
        A, B = linearize(cartpole, x_eq, u_eq)

        # Euler discretization
        Ad, Bd = disc_euler(A, B, dt)

        # Check formula: Ad = I + A*dt
        Ad_expected = jnp.eye(4) + A * dt
        np.testing.assert_allclose(Ad, Ad_expected, rtol=1e-10)

        # Check formula: Bd = B*dt
        Bd_expected = B * dt
        np.testing.assert_allclose(Bd, Bd_expected, rtol=1e-10)

    def test_open_loop_trajectory_deterministic(self, cartpole: Cartpole) -> None:
        """Same initial state and control should produce same trajectory."""
        initial = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = ConstantControl(jnp.array([0.0]))
        dt = 0.02
        duration = 2.0

        result1 = simulate_euler(
            cartpole, initial, dt=dt, duration=duration, control=control
        )
        result2 = simulate_euler(
            cartpole, initial, dt=dt, duration=duration, control=control
        )

        np.testing.assert_allclose(result1.states, result2.states, rtol=1e-14)
        np.testing.assert_allclose(result1.times, result2.times, rtol=1e-14)
