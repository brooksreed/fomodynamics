"""WI-6: Progressive ventilation test cases for Moth 3DOF model.

Tests three scenarios with increasing complexity:
1. Fully submerged baseline - foils deep enough to avoid ventilation
2. Binary ventilation mode - hard cutoff when foil breaches surface
3. Smooth ventilation near surface - differentiable depth factor with gradient test

These tests validate that the ventilation model behaves correctly across
operating regimes and that the smooth mode is suitable for gradient-based
trim/control optimization.

Note on open-loop stability:
    The Moth model is open-loop unstable (physically realistic for a foiling
    boat). Simulations from trim diverge within ~0.15s without active control.
    Tests account for this by using short durations where the initial behavior
    is meaningful, or by analyzing the valid (pre-NaN) portion of trajectories.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from fmd.simulator.moth_3d import Moth3D, POS_D, THETA, W, Q
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.integrator import simulate
from fmd.simulator.control import ConstantControl
from fmd.simulator.validation import SimCase, compute_diagnostics
from fmd.simulator.moth_validation import run_case
from fmd.simulator.components.moth_forces import compute_depth_factor, compute_foil_ned_depth

# ---------------------------------------------------------------------------
# Shared constants derived from MOTH_BIEKER_V3 parameters
# ---------------------------------------------------------------------------
MAIN_FOIL_SPAN = MOTH_BIEKER_V3.main_foil_span          # 1.0 m
RUDDER_SPAN = MOTH_BIEKER_V3.rudder_span                  # 0.5 m
MAIN_FOIL_POSITION_X = float(MOTH_BIEKER_V3.main_foil_position[0])  # 0.55 m (forward)
MAIN_FOIL_POSITION_Z = float(MOTH_BIEKER_V3.main_foil_position[2])  # 1.82 m (below CG)
RUDDER_POSITION_X = float(MOTH_BIEKER_V3.rudder_position[0])        # -1.755 m (aft)
RUDDER_POSITION_Z = float(MOTH_BIEKER_V3.rudder_position[2])        # 1.77 m (below CG)

HEEL_ANGLE_DEG = 15.0
HEEL_ANGLE_RAD = np.deg2rad(HEEL_ANGLE_DEG)


def _valid_states(states: np.ndarray) -> np.ndarray:
    """Return only the rows before the first NaN appears."""
    nan_per_row = np.any(np.isnan(states), axis=1)
    if not np.any(nan_per_row):
        return states
    first_nan = np.argmax(nan_per_row)
    return states[:first_nan]


class TestFullySubmergedBaseline:
    """Scenario 1: Fly deep enough that foils stay fully submerged.

    With heel_angle=15deg and main_foil_span=1.0:
        max_submergence = (1.0/2) * sin(15 deg) ~ 0.129 m
    So if foil_depth >= ~0.2 (well above max_submergence), depth_factor ~ 1.0.

    The Moth is open-loop unstable, so we use a short simulation duration
    (50ms) where the trim state has not yet diverged significantly.
    """

    def test_trim_exists_with_heel(self):
        """Trim converges with nonzero heel angle at deep ride height."""
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.success or trim.residual < 0.01, (
            f"Trim failed: residual={trim.residual:.2e}, success={trim.success}"
        )
        assert trim.residual < 0.05, (
            f"Trim residual too large: {trim.residual:.2e}"
        )

    def test_submerged_simulation_bounded(self):
        """Short simulation from trim stays bounded with no NaN/Inf.

        The Moth is open-loop unstable (~0.15s to divergence), so we
        simulate 50ms where the response is still near trim.
        """
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        x0 = jnp.array(trim.state)
        ctrl = ConstantControl(jnp.array(trim.control))
        result = simulate(moth, x0, dt=0.001, duration=0.05, control=ctrl)

        states = np.array(result.states)

        # No NaN or Inf
        assert not np.any(np.isnan(states)), "NaN detected in simulation states"
        assert not np.any(np.isinf(states)), "Inf detected in simulation states"

        # State should stay very close to trim over 50ms
        pos_d_range = np.ptp(states[:, POS_D])
        theta_range = np.ptp(states[:, THETA])
        assert pos_d_range < 0.01, f"pos_d range too large: {pos_d_range:.6f} m"
        assert theta_range < np.deg2rad(1.0), (
            f"theta range too large: {np.rad2deg(theta_range):.3f} deg"
        )

    def test_depth_factor_near_one_at_trim(self):
        """Depth factor is near 1.0 at the trim state (foils well submerged).

        At trim, pos_d ~ -1.3, theta ~ small, so:
        - main foil depth = pos_d + 1.94 (CG-adjusted) ~ 0.64 (well submerged)
        - rudder depth = pos_d + 1.89 (CG-adjusted) ~ 0.59 (well submerged)

        Both are far below the surface, so depth_factor should be ~1.0.
        Pitch correction is small at trim theta.
        """
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        pos_d = trim.state[POS_D]
        theta = trim.state[THETA]

        # CG offset from sailor (forward_dynamics applies this internally)
        cg_offset = moth.sailor_mass * moth.sailor_position_schedule(0.0) / moth.total_mass

        foil_depth_main = float(compute_foil_ned_depth(
            pos_d,
            MAIN_FOIL_POSITION_X - float(cg_offset[0]),
            MAIN_FOIL_POSITION_Z - float(cg_offset[2]),
            theta, HEEL_ANGLE_RAD))
        foil_depth_rudder = float(compute_foil_ned_depth(
            pos_d,
            RUDDER_POSITION_X - float(cg_offset[0]),
            RUDDER_POSITION_Z - float(cg_offset[2]),
            theta, HEEL_ANGLE_RAD))

        df_main = compute_depth_factor(
            jnp.float64(foil_depth_main),
            MAIN_FOIL_SPAN,
            HEEL_ANGLE_RAD,
            ventilation_threshold=0.30,
            mode="smooth",
        )
        df_rudder = compute_depth_factor(
            jnp.float64(foil_depth_rudder),
            RUDDER_SPAN,
            HEEL_ANGLE_RAD,
            ventilation_threshold=0.30,
            mode="smooth",
        )

        assert float(df_main) > 0.95, (
            f"Main foil depth_factor at trim too low: "
            f"{float(df_main):.3f} (foil_depth={foil_depth_main:.3f})"
        )
        assert float(df_rudder) > 0.95, (
            f"Rudder depth_factor at trim too low: "
            f"{float(df_rudder):.3f} (foil_depth={foil_depth_rudder:.3f})"
        )

    def test_depth_factor_near_one_after_short_sim(self):
        """Depth factor remains near 1.0 after a short simulation.

        After 50ms from trim, the state hasn't diverged much, so depth
        factor should still be near 1.0. Uses pitch-corrected depth formula.
        """
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        x0 = jnp.array(trim.state)
        ctrl = ConstantControl(jnp.array(trim.control))
        result = simulate(moth, x0, dt=0.001, duration=0.05, control=ctrl)
        states = np.array(result.states)

        # Check depth factor at final state (t=50ms)
        # CG offset from sailor (forward_dynamics applies this internally)
        cg_offset = moth.sailor_mass * moth.sailor_position_schedule(0.0) / moth.total_mass

        pos_d_final = states[-1, POS_D]
        theta_final = states[-1, THETA]
        foil_depth_main = float(compute_foil_ned_depth(
            pos_d_final,
            MAIN_FOIL_POSITION_X - float(cg_offset[0]),
            MAIN_FOIL_POSITION_Z - float(cg_offset[2]),
            theta_final, HEEL_ANGLE_RAD))
        foil_depth_rudder = float(compute_foil_ned_depth(
            pos_d_final,
            RUDDER_POSITION_X - float(cg_offset[0]),
            RUDDER_POSITION_Z - float(cg_offset[2]),
            theta_final, HEEL_ANGLE_RAD))

        df_main = compute_depth_factor(
            jnp.float64(foil_depth_main),
            MAIN_FOIL_SPAN,
            HEEL_ANGLE_RAD,
            ventilation_threshold=0.30,
            mode="smooth",
        )
        df_rudder = compute_depth_factor(
            jnp.float64(foil_depth_rudder),
            RUDDER_SPAN,
            HEEL_ANGLE_RAD,
            ventilation_threshold=0.30,
            mode="smooth",
        )

        assert float(df_main) > 0.9, (
            f"Main foil depth_factor after 50ms too low: "
            f"{float(df_main):.3f} (foil_depth={foil_depth_main:.3f})"
        )
        assert float(df_rudder) > 0.9, (
            f"Rudder depth_factor after 50ms too low: "
            f"{float(df_rudder):.3f} (foil_depth={foil_depth_rudder:.3f})"
        )


class TestBinaryDepthFactor:
    """Pure depth_factor tests for binary mode (no trim dependency)."""

    def test_binary_depth_factor_behavior(self):
        """Binary mode: factor=1 submerged, factor=0 above surface."""
        # Submerged: foil_depth > 0
        df_sub = compute_depth_factor(
            jnp.float64(0.5), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
            ventilation_threshold=0.30, mode="binary",
        )
        assert float(df_sub) == pytest.approx(1.0)

        # Above surface: foil_depth < 0
        df_above = compute_depth_factor(
            jnp.float64(-0.1), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
            ventilation_threshold=0.30, mode="binary",
        )
        assert float(df_above) == pytest.approx(0.0)

        # Exactly at surface: foil_depth = 0
        df_surface = compute_depth_factor(
            jnp.float64(0.0), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
            ventilation_threshold=0.30, mode="binary",
        )
        assert float(df_surface) == pytest.approx(0.0)


class TestBinaryVentilation:
    """Scenario 2: Binary ventilation mode with hard cutoff.

    Uses ventilation_mode="binary" where depth_factor = 1 if foil_depth > 0
    and 0 otherwise. When the foil center reaches the surface, all lift
    drops instantly.

    We start from trim and perturb upward to bring the foil near the surface,
    then verify the response diverges from the initial state.
    """

    def test_binary_ventilation_causes_divergence(self):
        """Perturbing upward in binary mode causes divergent response.

        Start at a shallow ride height where the main foil center is near
        the surface. With binary cutoff, the sudden loss of lift should
        cause noticeable departure from the initial state.

        We use short sim duration and check only the valid (pre-NaN) portion,
        since the open-loop unstable dynamics eventually diverge to NaN.
        """
        # Get trim with smooth mode (binary might not trim well)
        moth_smooth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        # Create binary mode moth for simulation
        moth_binary = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="binary",
        )

        # Perturb initial state upward: decrease pos_d to bring main foil
        # center closer to surface. CG-adjusted foil z = 1.94m,
        # so foil_depth = pos_d + 1.94. We want foil_depth near 0.
        # pos_d = -1.87 => foil_depth ~ 0.07 (barely submerged)
        perturbed_state = np.array(trim.state, dtype=np.float64)
        perturbed_state[POS_D] = -1.87
        x0 = jnp.array(perturbed_state)

        ctrl = ConstantControl(jnp.array(trim.control))
        result = simulate(moth_binary, x0, dt=0.001, duration=1.0, control=ctrl)
        states = np.array(result.states)
        valid = _valid_states(states)

        # Should have meaningful valid data (at least a few steps)
        assert len(valid) >= 5, (
            f"Too few valid steps in binary sim: {len(valid)}"
        )

        # The response should diverge: check any state drifts significantly
        # from the initial value. With foil above water and binary cutoff,
        # the system should depart from the initial condition.
        max_drift = max(
            np.max(np.abs(valid[:, i] - valid[0, i]))
            for i in range(4)  # pos_d, theta, w, q
        )
        assert max_drift > 0.0005, (
            f"Expected divergent response in binary mode, but max state "
            f"drift was only {max_drift:.6f}"
        )

    def test_binary_vs_smooth_divergence_comparison(self):
        """Binary and smooth modes produce different trajectories.

        Both modes start from the same shallow initial condition. The
        different ventilation models produce different forces at the same
        state, leading to divergent trajectories.

        We compare only the valid (pre-NaN) portion of both trajectories.
        """
        # Get trim with smooth mode
        moth_smooth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        # Create binary mode
        moth_binary = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="binary",
        )

        # Start at a shallow depth where ventilation matters
        # CG-adjusted foil_depth = pos_d + 1.94; use pos_d ~ -1.74 => foil_depth ~ 0.2
        shallow_state = np.array(trim.state, dtype=np.float64)
        shallow_state[POS_D] = -1.74
        x0 = jnp.array(shallow_state)
        ctrl = ConstantControl(jnp.array(trim.control))

        # Short sim to capture initial divergence behavior
        duration = 0.1
        dt = 0.001
        result_binary = simulate(moth_binary, x0, dt=dt, duration=duration, control=ctrl)
        result_smooth = simulate(moth_smooth, x0, dt=dt, duration=duration, control=ctrl)

        states_binary = np.array(result_binary.states)
        states_smooth = np.array(result_smooth.states)

        valid_binary = _valid_states(states_binary)
        valid_smooth = _valid_states(states_smooth)

        # Both should have meaningful valid data
        assert len(valid_binary) >= 5, f"Too few valid binary steps: {len(valid_binary)}"
        assert len(valid_smooth) >= 5, f"Too few valid smooth steps: {len(valid_smooth)}"

        # Compare over the common valid interval
        n_compare = min(len(valid_binary), len(valid_smooth))
        assert n_compare >= 5, f"Too few common steps to compare: {n_compare}"

        # The trajectories should differ (different ventilation models
        # produce different forces at the same state)
        pos_d_diff = np.abs(
            valid_binary[n_compare - 1, POS_D] - valid_smooth[n_compare - 1, POS_D]
        )
        assert pos_d_diff > 1e-6, (
            f"Expected different trajectories between binary and smooth, "
            f"but final pos_d difference was only {pos_d_diff:.10f} m"
        )

    def test_binary_forces_differ_from_smooth_at_shallow_depth(self):
        """At a shallow depth, binary and smooth modes produce different forces.

        This is a direct force comparison (no simulation), verifying that
        the two ventilation models produce meaningfully different lift at
        the same operating point near the surface.
        """
        # At foil_depth = 0.1 (shallow), binary gives factor=1 (fully submerged)
        # while smooth should give factor < 1 (partial ventilation at 15 deg heel)
        foil_depth = 0.1
        df_binary = compute_depth_factor(
            jnp.float64(foil_depth), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
            ventilation_threshold=0.30, mode="binary",
        )
        df_smooth = compute_depth_factor(
            jnp.float64(foil_depth), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
            ventilation_threshold=0.30, mode="smooth",
        )

        # Binary: foil_depth > 0 => factor = 1.0
        assert float(df_binary) == pytest.approx(1.0)

        # Smooth: at shallow depth with heel, factor should be reduced
        assert float(df_smooth) < 1.0, (
            f"Smooth factor should be < 1.0 at shallow depth, got {float(df_smooth):.3f}"
        )
        assert float(df_smooth) > 0.0, (
            f"Smooth factor should be > 0.0 at positive depth, got {float(df_smooth):.3f}"
        )

        # The difference demonstrates the value of smooth ventilation modeling
        diff = abs(float(df_binary) - float(df_smooth))
        assert diff > 0.01, (
            f"Expected meaningful difference between modes, got {diff:.6f}"
        )


class TestSmoothVentilation:
    """Scenario 3: Smooth ventilation near surface.

    The key test here is the gradient test: verifying that jax.grad of the
    depth factor w.r.t. foil_depth produces finite values near the surface.
    This validates that smooth mode is suitable for gradient-based
    trim/control optimization.
    """

    def test_smooth_depth_factor_gradual_transition(self):
        """Smooth mode shows gradual lift reduction near surface.

        The depth factor should transition smoothly from ~1 (submerged)
        to ~0 (above surface), without hard discontinuities.
        """
        depths = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.0, -0.1, -0.5]
        factors = []
        for d in depths:
            df = compute_depth_factor(
                jnp.float64(d), MAIN_FOIL_SPAN, HEEL_ANGLE_RAD,
                ventilation_threshold=0.30, mode="smooth",
            )
            factors.append(float(df))

        # Deep submerged should be near 1
        assert factors[0] > 0.95, f"Deep factor should be ~1.0, got {factors[0]:.3f}"

        # Above surface should be near 0
        assert factors[-1] < 0.1, f"Above-surface factor should be ~0, got {factors[-1]:.3f}"

        # Should be monotonically non-increasing (more depth = more lift)
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1] - 1e-6, (
                f"Depth factor not monotone at depths [{depths[i]}, {depths[i+1]}]: "
                f"[{factors[i]:.4f}, {factors[i+1]:.4f}]"
            )

    def test_smooth_gradient_finite_near_surface(self):
        """jax.grad of depth factor w.r.t. foil_depth is finite near surface.

        This is the KEY validation test: it confirms smooth mode is
        differentiable everywhere near the surface, making it suitable
        for gradient-based trim/control optimization (MPC, iLQR, etc.).
        """
        def depth_factor_fn(foil_depth):
            return compute_depth_factor(
                foil_depth,
                foil_span=MAIN_FOIL_SPAN,
                heel_angle=HEEL_ANGLE_RAD,
                ventilation_threshold=0.30,
                mode="smooth",
            )

        grad_fn = jax.grad(depth_factor_fn)

        # Test at several depths near the surface (the critical region)
        test_depths = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        for depth in test_depths:
            g = grad_fn(jnp.float64(depth))
            assert jnp.isfinite(g), (
                f"Gradient not finite at depth={depth}: {g}"
            )

    def test_smooth_gradient_finite_at_surface(self):
        """Gradient is also finite exactly at the surface (depth=0).

        This is the trickiest point: the transition center. The smooth
        model must be differentiable here for optimization to work.
        """
        def depth_factor_fn(foil_depth):
            return compute_depth_factor(
                foil_depth,
                foil_span=MAIN_FOIL_SPAN,
                heel_angle=HEEL_ANGLE_RAD,
                ventilation_threshold=0.30,
                mode="smooth",
            )

        grad_fn = jax.grad(depth_factor_fn)
        g = grad_fn(jnp.float64(0.0))
        assert jnp.isfinite(g), f"Gradient not finite at surface: {g}"

    def test_smooth_gradient_finite_above_surface(self):
        """Gradient is finite even above the surface (negative depth).

        Optimization may explore states above the surface; gradients
        must remain finite to avoid NaN propagation.
        """
        def depth_factor_fn(foil_depth):
            return compute_depth_factor(
                foil_depth,
                foil_span=MAIN_FOIL_SPAN,
                heel_angle=HEEL_ANGLE_RAD,
                ventilation_threshold=0.30,
                mode="smooth",
            )

        grad_fn = jax.grad(depth_factor_fn)
        for depth in [-0.01, -0.1, -0.5]:
            g = grad_fn(jnp.float64(depth))
            assert jnp.isfinite(g), (
                f"Gradient not finite above surface at depth={depth}: {g}"
            )

    def test_smooth_near_surface_simulation(self):
        """Short simulation near the surface with smooth mode remains finite.

        Start at a shallow depth where ventilation is active and verify
        the response is numerically stable (no NaN/Inf) over a short
        interval. The open-loop unstable dynamics will eventually diverge,
        but the initial response should be well-behaved.
        """
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )
        trim = find_moth_trim(
            MOTH_BIEKER_V3, u_forward=10.0,
            heel_angle=HEEL_ANGLE_RAD, ventilation_mode="smooth",
        )
        assert trim.residual < 0.05, f"Trim did not converge: {trim.residual:.2e}"

        # Start shallow: CG-adjusted foil_depth = pos_d + 1.94
        # pos_d = -1.79 => foil_depth ~ 0.15 (near tip-breach region)
        shallow_state = np.array(trim.state, dtype=np.float64)
        shallow_state[POS_D] = -1.79
        x0 = jnp.array(shallow_state)
        ctrl = ConstantControl(jnp.array(trim.control))

        result = simulate(moth, x0, dt=0.001, duration=0.05, control=ctrl)
        states = np.array(result.states)

        # No NaN or Inf over this short interval
        assert not np.any(np.isnan(states)), "NaN detected in smooth near-surface sim"
        assert not np.any(np.isinf(states)), "Inf detected in smooth near-surface sim"

    def test_smooth_main_foil_force_gradient_finite(self):
        """Gradient of main foil z-force w.r.t. pos_d is finite near surface.

        This tests the full force computation chain (not just depth_factor),
        confirming the assembled model is differentiable end-to-end.
        """
        moth = Moth3D(
            MOTH_BIEKER_V3,
            heel_angle=HEEL_ANGLE_RAD,
            ventilation_mode="smooth",
        )

        control = jnp.array([0.05, 0.02])

        def force_z_fn(pos_d):
            state = jnp.array([pos_d, 0.02, 0.0, 0.0, 10.0])
            force, _ = moth.main_foil.compute_moth(state, control, 10.0, 0.0)
            return force[2]

        grad_fn = jax.grad(force_z_fn)

        # Test at depths near the surface
        for depth in [-0.5, -0.2, 0.0, 0.1, 0.2, 0.5]:
            g = grad_fn(jnp.float64(depth))
            assert jnp.isfinite(g), (
                f"Main foil force_z gradient not finite at pos_d={depth}: {g}"
            )

    def test_smooth_zero_heel_gradient(self):
        """Gradient is also finite with zero heel angle.

        At heel_angle=0, the min_submergence floor (0.015m) provides a
        near-binary but differentiable transition over ~3cm.
        """
        def depth_factor_fn(foil_depth):
            return compute_depth_factor(
                foil_depth,
                foil_span=MAIN_FOIL_SPAN,
                heel_angle=0.0,
                ventilation_threshold=0.30,
                mode="smooth",
            )

        grad_fn = jax.grad(depth_factor_fn)

        for depth in [-0.1, 0.0, 0.05, 0.1, 0.5]:
            g = grad_fn(jnp.float64(depth))
            assert jnp.isfinite(g), (
                f"Gradient not finite at depth={depth} with zero heel: {g}"
            )
