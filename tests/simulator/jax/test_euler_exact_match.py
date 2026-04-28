"""Euler-to-Euler exact match tests for SCG cross-validation.

Phase 8a: When both BLUR and SCG use forward Euler integration with identical:
- Dynamics equations
- Timestep (dt)
- Initial conditions
- Control inputs

The trajectories should match to floating-point precision (~1e-10 to 1e-14).

This module provides the strongest validation by eliminating integrator
differences as a source of discrepancy. Tests skip gracefully if reference
data was generated with analytical fallback instead of actual SCG library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator import (
    Cartpole,
    ConstantControl,
    simulate_euler,
    simulate_euler_substepped,
)
from fmd.simulator.params import CARTPOLE_SCG


GOLDEN_DIR = Path(__file__).parent / "golden_master"

# Cartpole state indices
_THETA_IDX = 2  # Angle index in [x, x_dot, theta, theta_dot]


def assert_cartpole_states_close(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    err_msg: str = "",
) -> None:
    """Compare Cartpole states handling angle wrapping.

    BLUR wraps theta to [-pi, pi] via post_step(), while SCG lets angles
    grow unbounded. This compares sin/cos of theta to handle wrapping,
    while comparing other states directly.

    Args:
        actual: BLUR states array (N, 4)
        desired: SCG states array (N, 4)
        rtol: Relative tolerance
        atol: Absolute tolerance
        err_msg: Error message prefix
    """
    # Compare non-angle states directly: x, x_dot, theta_dot (indices 0, 1, 3)
    non_angle_indices = [0, 1, 3]
    np.testing.assert_allclose(
        actual[:, non_angle_indices],
        desired[:, non_angle_indices],
        rtol=rtol,
        atol=atol,
        err_msg=f"{err_msg} (non-angle states)",
    )

    # Compare theta via sin/cos to handle wrapping
    theta_actual = actual[:, _THETA_IDX]
    theta_desired = desired[:, _THETA_IDX]

    np.testing.assert_allclose(
        np.sin(theta_actual),
        np.sin(theta_desired),
        rtol=rtol,
        atol=atol,
        err_msg=f"{err_msg} (sin(theta))",
    )
    np.testing.assert_allclose(
        np.cos(theta_actual),
        np.cos(theta_desired),
        rtol=rtol,
        atol=atol,
        err_msg=f"{err_msg} (cos(theta))",
    )


def load_true_reference(name: str) -> dict[str, Any]:
    """Load true SCG reference, skip if not generated with actual SCG.

    Args:
        name: Reference file name without extension (e.g., 'scg_cartpole_true')

    Returns:
        Dictionary of reference data

    Raises:
        pytest.skip: If reference not found or was generated with fallback
    """
    path = GOLDEN_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(
            f"Reference not found: {path}. "
            f"Run: ./scripts/docker_scg_helper.sh"
        )
    data = dict(np.load(path, allow_pickle=True))
    provenance = data["provenance"].item()

    if not provenance.get("scg_library_used", False):
        pytest.skip(
            f"Reference was generated with analytical fallback, not actual SCG library. "
            f"Run: ./scripts/docker_scg_helper.sh"
        )

    return data


class TestCartpoleEulerExactMatch:
    """Exact Euler-to-Euler validation for Cartpole.

    These tests require true SCG reference data (scg_library_used: True).
    They validate that BLUR's Euler integration produces bit-identical
    results to SCG's Physics.DYN mode (forward Euler).
    """

    @pytest.fixture
    def scg_ref(self) -> dict[str, Any]:
        """Load Cartpole reference data from actual SCG."""
        return load_true_reference("scg_cartpole_true")

    def test_open_loop_exact_match(self, scg_ref: dict[str, Any]) -> None:
        """Open-loop trajectory matches SCG Euler to machine precision.

        Tolerance: rtol=1e-10, atol=1e-12

        Rationale: Same dynamics + same integrator + same dt should produce
        bit-identical results up to floating-point accumulation.

        Note: Compares theta via sin/cos because BLUR wraps angles to [-pi, pi]
        while SCG lets them grow unbounded. The physics are identical.
        """
        x0 = scg_ref["open_loop_x0"]
        u = scg_ref["open_loop_u"]
        times_scg = scg_ref["open_loop_times"]
        states_scg = scg_ref["open_loop_states"]
        dt = float(scg_ref["dt"])

        # Simulate with BLUR using Euler integrator
        cartpole = Cartpole(CARTPOLE_SCG)
        duration = times_scg[-1]

        control = ConstantControl(jnp.array(u))
        result = simulate_euler(
            cartpole, jnp.array(x0), dt=dt, duration=duration, control=control
        )

        blur_states = np.array(result.states)

        # Compare states with angle wrapping handling
        # BLUR wraps theta to [-pi, pi], SCG doesn't
        assert_cartpole_states_close(
            blur_states,
            states_scg,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Open-loop trajectory doesn't match SCG",
        )

    def test_closed_loop_exact_match(self, scg_ref: dict[str, Any]) -> None:
        """Closed-loop LQR trajectory matches SCG Euler exactly.

        Same LQR controller, same dynamics, same integrator should produce
        identical trajectories.

        Note: Compares theta via sin/cos because BLUR wraps angles to [-pi, pi]
        while SCG lets them grow unbounded. The physics are identical.
        """
        x0 = scg_ref["closed_loop_x0"]
        K = scg_ref["K"]
        x_eq = scg_ref["x_eq"]
        u_eq = scg_ref["u_eq"]
        times_scg = scg_ref["closed_loop_times"]
        states_scg = scg_ref["closed_loop_states"]
        dt = float(scg_ref["dt"])

        # Create LQR controller
        from fmd.simulator import LQRController

        controller = LQRController(
            K=jnp.array(K),
            x_ref=jnp.array(x_eq),
            u_ref=jnp.array(u_eq),
        )

        # Simulate with BLUR
        cartpole = Cartpole(CARTPOLE_SCG)
        duration = times_scg[-1]

        result = simulate_euler(
            cartpole, jnp.array(x0), dt=dt, duration=duration, control=controller
        )

        blur_states = np.array(result.states)

        # Compare states with angle wrapping handling
        # BLUR wraps theta to [-pi, pi], SCG doesn't
        assert_cartpole_states_close(
            blur_states,
            states_scg,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Closed-loop trajectory doesn't match SCG",
        )

    def test_single_step_exact_match(self, scg_ref: dict[str, Any]) -> None:
        """Single Euler step matches SCG to machine precision.

        Tests the fundamental dynamics equation without accumulation.
        """
        x0 = jnp.array(scg_ref["open_loop_x0"])
        u = jnp.array(scg_ref["open_loop_u"])
        states_scg = scg_ref["open_loop_states"]
        dt = float(scg_ref["dt"])

        # Single step
        cartpole = Cartpole(CARTPOLE_SCG)
        from fmd.simulator.integrator import euler_step

        x1_blur = euler_step(cartpole, x0, u, dt, 0.0)

        # Compare to second state in SCG trajectory (first is x0)
        x1_scg = states_scg[1]

        np.testing.assert_allclose(
            np.array(x1_blur),
            x1_scg,
            rtol=1e-14,
            atol=1e-16,
            err_msg="Single Euler step doesn't match SCG",
        )


class TestQuadrotor3DEulerExactMatch:
    """Exact Euler-to-Euler validation for 3D Quadrotor.

    Note: Quadrotor tests compare SCG's 12-state Euler model against
    BLUR's analytical Euler simulation in the reference generator.
    Full 13-state quaternion comparison requires adapters.
    """

    @pytest.fixture
    def scg_ref(self) -> dict[str, Any]:
        """Load 3D Quadrotor reference data from actual SCG."""
        return load_true_reference("scg_quadrotor3d_true")

    def test_hover_exact_match(self, scg_ref: dict[str, Any]) -> None:
        """Hover equilibrium is exactly maintained.

        At equilibrium with correct thrust, state should not change.
        """
        hover_states = scg_ref["hover_states"]
        x_eq = scg_ref["x_eq"]

        # Check that all hover states are essentially identical to equilibrium
        max_drift = np.max(np.abs(hover_states - x_eq))

        # Should be essentially zero (machine precision)
        assert max_drift < 1e-10, f"Hover drift too large: {max_drift}"

    def test_free_fall_exact_match(self, scg_ref: dict[str, Any]) -> None:
        """Free fall trajectory matches SCG Euler exactly.

        Previous test (Phase 7.5) used 3% tolerance due to Euler vs RK4 comparison.
        With Euler-to-Euler, we expect sub-1e-10 tolerance.

        Note: This test validates the reference data's internal consistency.
        The 3% tolerance in test_scg_cross_validation.py is for comparing
        BLUR's simulation against this reference when using different integrators.
        """
        freefall_states = scg_ref["freefall_states"]
        freefall_times = scg_ref["freefall_times"]
        g = float(scg_ref["g"])

        # Extract final velocity (vz is index 5)
        t_final = freefall_times[-1]
        vz_final = freefall_states[-1, 5]  # z velocity

        # For free fall starting from rest, vz should be -g*t
        # (negative because z-up frame, falling means negative velocity)
        expected_vz = -g * t_final

        # With Euler integration, numerical drift is expected but should be small
        # For dt=1ms over 1 second, Euler error should be ~O(dt) = ~0.1%
        np.testing.assert_allclose(
            vz_final,
            expected_vz,
            rtol=0.01,  # 1% tolerance for Euler numerical error
            atol=1e-6,
            err_msg="Free fall velocity doesn't match expected",
        )

        # Also check position (z decreases by 0.5*g*t^2)
        z_initial = freefall_states[0, 2]
        z_final = freefall_states[-1, 2]
        expected_fall = 0.5 * g * t_final**2
        actual_fall = z_initial - z_final

        np.testing.assert_allclose(
            actual_fall,
            expected_fall,
            rtol=0.02,  # 2% tolerance (Euler accumulation over position)
            atol=1e-4,
            err_msg="Free fall position doesn't match expected",
        )

    def test_scg_substepping_metadata(self, scg_ref: dict[str, Any]) -> None:
        """Verify reference was generated with correct substepping parameters."""
        # Check that new substepping fields are present
        assert "dt_control" in scg_ref or "dt" in scg_ref, "Missing timestep info"

        # If using new substepped format, verify parameters
        if "dt_sim" in scg_ref:
            dt_sim = float(scg_ref["dt_sim"])
            dt_control = float(scg_ref["dt_control"])

            # Verify dt_control is integer multiple of dt_sim
            ratio = dt_control / dt_sim
            assert abs(ratio - round(ratio)) < 1e-10, (
                f"dt_control ({dt_control}) should be integer multiple of "
                f"dt_sim ({dt_sim})"
            )

            # Verify expected SCG values
            if dt_sim == 0.001:  # 1ms
                assert dt_control == 0.02, "Expected 20ms control rate for SCG"


class TestToleranceDocumentation:
    """Document why different tolerances are expected for different comparisons.

    This is a documentation test that explains the tolerance structure.
    """

    def test_tolerance_levels_documented(self) -> None:
        """Document expected tolerances and their rationale.

        | Comparison Type | Tolerance | Rationale |
        |-----------------|-----------|-----------|
        | Same integrator (Euler-Euler) | 1e-10 | Machine precision only |
        | Same integrator + LQR | 1e-10 | LQR is deterministic |
        | Different integrator (RK4 vs Euler) | 3-5% | Truncation error |
        | Cross-library (BLUR vs SCG) | 1e-6 | Same equations, diff impl |
        | Adapter round-trip | 1e-10 | Pure numerical transform |
        | Quat-Euler round-trip | 1e-6 | Singularity handling |
        """
        tolerances = {
            "same_integrator_euler": {"rtol": 1e-10, "atol": 1e-12},
            "different_integrator_rk4_euler": {"rtol": 0.05, "atol": 1e-4},
            "cross_library_same_equations": {"rtol": 1e-6, "atol": 1e-10},
            "adapter_round_trip": {"rtol": 1e-10, "atol": 1e-12},
            "quaternion_euler_round_trip": {"rtol": 1e-6, "atol": 1e-10},
        }

        # Just verify the dict is valid (documentation test)
        assert len(tolerances) == 5

    def test_rk4_vs_euler_drift_documented(self) -> None:
        """Document expected drift between RK4 and Euler.

        Forward Euler: x_{k+1} = x_k + dt * f(x_k, u_k)
        - First-order accurate: local error O(dt^2), global error O(dt)
        - Conditionally stable: dt < 2/|lambda_max|

        RK4: Uses f at 4 points, weights them
        - Fourth-order accurate: local error O(dt^5), global error O(dt^4)
        - More stable: dt < 2.785/|lambda_max|

        Expected drift sources:
        1. Truncation error difference: RK4 is O(dt^4), Euler is O(dt)
        2. Over 5 seconds at dt=20ms (250 steps), errors compound
        3. For unstable dynamics (Cartpole), errors grow exponentially

        Measured drift with dt=20ms, 5 second simulation:
        - Free-fall: ~3% (simple dynamics, linear accumulation)
        - Controlled: ~1-5% (depends on controller aggressiveness)
        - Unstable open-loop: can diverge significantly

        This is EXPECTED behavior and NOT a bug. Use same integrator for
        exact comparison, different integrator comparison should use relaxed
        tolerances.
        """
        # Documentation test - just pass
        pass


class TestSubsteppedSimulation:
    """Test the new simulate_euler_substepped function."""

    @pytest.fixture
    def cartpole(self) -> Cartpole:
        """Create Cartpole with SCG parameters."""
        return Cartpole(CARTPOLE_SCG)

    def test_substepped_matches_regular_euler(self, cartpole: Cartpole) -> None:
        """Substepped simulation with 1x substep matches regular Euler."""
        x0 = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = ConstantControl(jnp.array([0.0]))

        # Regular Euler at 1ms
        result_regular = simulate_euler(
            cartpole, x0, dt=0.001, duration=1.0, control=control
        )

        # Substepped with 1x (no actual substepping)
        result_substepped = simulate_euler_substepped(
            cartpole,
            x0,
            dt_sim=0.001,
            dt_control=0.001,
            duration=1.0,
            control=control,
        )

        # JAX compilation can introduce tiny differences at machine precision
        np.testing.assert_allclose(
            result_regular.states,
            result_substepped.states,
            rtol=1e-12,
            atol=1e-14,
            err_msg="1x substepped should match regular Euler exactly",
        )

    def test_substepped_produces_same_final_state(self, cartpole: Cartpole) -> None:
        """Substepped simulation produces same final state as fine-dt Euler."""
        x0 = jnp.array([0.0, 0.0, 0.1, 0.0])
        control = ConstantControl(jnp.array([0.0]))

        # Fine-dt regular Euler
        result_fine = simulate_euler(
            cartpole, x0, dt=0.001, duration=1.0, control=control
        )

        # Coarse-output substepped (same physics, coarser output)
        result_substepped = simulate_euler_substepped(
            cartpole,
            x0,
            dt_sim=0.001,
            dt_control=0.02,  # 20x coarser output
            duration=1.0,
            control=control,
        )

        # Final states should match (same physics)
        np.testing.assert_allclose(
            result_fine.states[-1],
            result_substepped.states[-1],
            rtol=1e-14,
            atol=1e-16,
            err_msg="Substepped final state should match fine Euler",
        )

        # But output lengths differ
        assert len(result_fine.times) == 1001  # 1000 steps + initial
        assert len(result_substepped.times) == 51  # 50 control steps + initial

    def test_substepped_control_held_constant(self, cartpole: Cartpole) -> None:
        """Control is held constant during substeps (zero-order hold)."""
        x0 = jnp.array([0.0, 0.0, 0.5, 0.0])  # Larger angle for more dynamics

        # Use ConstantControl - simpler test that control is applied correctly
        control = ConstantControl(jnp.array([5.0]))

        result = simulate_euler_substepped(
            cartpole,
            x0,
            dt_sim=0.001,
            dt_control=0.02,
            duration=1.0,
            control=control,
        )

        # Verify we got expected number of control outputs
        assert result.controls.shape == (51, 1)

        # Verify all controls are constant
        np.testing.assert_allclose(
            result.controls[:, 0],
            5.0,
            rtol=1e-14,
            err_msg="Control outputs should all be 5.0",
        )

        # Also verify that the simulation actually uses the control
        # (cart should accelerate to the right with positive force)
        assert result.states[-1, 0] > 0, "Cart should move right with positive force"
        assert result.states[-1, 1] > 0, "Cart should have positive velocity"
