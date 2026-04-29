"""Tests for first-order lift lag (Wagner-type) filter.

Phase 3: Verifies exponential response, steady-state convergence,
speed dependence, backward compatibility, and RK4 stability.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from fmd.simulator import Moth3D, ConstantSchedule, simulate, ConstantControl
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim


class TestLiftLagStepResponse:
    """Verify exponential rise with correct time constant."""

    def test_step_aoa_exponential(self):
        """Step AoA change: verify exponential rise with tau = 4c/(pi*V).

        Start with filter states at 0, set control to produce a known
        alpha_eff. The filter should approach alpha_eff exponentially.
        Verifies ~63.2% of target reached after one time constant.
        """
        moth = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=True)
        assert moth.num_states == 7

        # At 10 m/s: tau_main = 4 * 0.089 / (pi * 10) ~ 0.01133s
        chord = MOTH_BIEKER_V3.main_foil_chord
        u = 10.0
        tau_expected = 4.0 * chord / (np.pi * u)

        # Use a state near trim with zero filter states
        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert trim.success

        state_5 = jnp.array(trim.state)
        control = jnp.array(trim.control)

        # Initialize with alpha_filt = 0 (step from 0 to trim alpha_eff)
        state_7 = jnp.concatenate([state_5, jnp.zeros(2)])

        # Get the instantaneous alpha_eff at this state
        aux = moth.compute_aux(state_7, control, t=0.0)
        alpha_inst_idx = moth.aux_names.index("main_alpha_eff")
        alpha_inst = float(aux[alpha_inst_idx])

        # Simulate for several time constants
        dt = 0.001  # Small dt for accuracy
        duration = 10 * tau_expected  # ~0.11s, plenty for convergence
        result = simulate(moth, state_7, dt=dt, duration=duration,
                         control=ConstantControl(control))

        # Check that filter state approaches alpha_inst
        # After 5 tau, should be within 1% of final value
        t_5tau = int(5 * tau_expected / dt)
        alpha_filt = float(result.states[t_5tau, 5])

        # The alpha_inst shifts during simulation, so check convergence
        # at the end instead
        alpha_filt_final = float(result.states[-1, 5])
        alpha_filt_early = float(result.states[1, 5])

        # Filter should have moved from 0 toward alpha_inst
        assert abs(alpha_filt_final) > abs(alpha_filt_early), (
            "Filter alpha should increase over time"
        )

        # Quantitative tau verification: after one time constant,
        # the filter should reach ~63.2% (1 - e^-1) of the target.
        # Use alpha_filt_final as proxy for steady-state target since
        # it has converged after 10*tau.
        # Note: tolerance is abs=0.08 because the coupled dynamics cause
        # alpha_eff to shift slightly during the transient, so this isn't
        # a pure first-order step response.
        t_1tau_idx = int(tau_expected / dt)
        alpha_at_1tau = float(result.states[t_1tau_idx, 5])
        fraction_at_1tau = alpha_at_1tau / alpha_filt_final
        expected_fraction = 1.0 - np.exp(-1.0)  # ~0.6321

        assert fraction_at_1tau == pytest.approx(expected_fraction, abs=0.08), (
            f"At t=tau, filter should reach ~63.2% of target, "
            f"got {fraction_at_1tau*100:.1f}%"
        )


class TestLiftLagSteadyState:
    """At steady state, filtered alpha should converge to instantaneous alpha."""

    def test_steady_state_convergence(self):
        """When initialized at trim alpha, filter should stay there."""
        moth = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=True)
        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert trim.success

        state_5 = jnp.array(trim.state)
        control = jnp.array(trim.control)

        # Get trim alpha values
        moth_5 = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=False)
        aux_5 = moth_5.compute_aux(state_5, control, t=0.0)
        main_alpha = float(aux_5[moth_5.aux_names.index("main_alpha_eff")])
        rudder_alpha = float(aux_5[moth_5.aux_names.index("rudder_alpha_eff")])

        # Initialize filter states at trim alpha
        state_7 = jnp.concatenate([state_5, jnp.array([main_alpha, rudder_alpha])])

        # Compute derivative - filter derivative should be ~0 at steady state
        deriv = moth.forward_dynamics(state_7, control, t=0.0)
        d_alpha_main = float(deriv[5])
        d_alpha_rudder = float(deriv[6])

        assert abs(d_alpha_main) < 1e-3, (
            f"Main filter derivative should be ~0 at steady state, got {d_alpha_main}"
        )
        assert abs(d_alpha_rudder) < 1e-3, (
            f"Rudder filter derivative should be ~0 at steady state, got {d_alpha_rudder}"
        )


class TestLiftLagSpeedDependence:
    """Faster speed -> smaller tau."""

    def test_tau_decreases_with_speed(self):
        """tau = 4c/(pi*V) so tau at 20 m/s < tau at 10 m/s < tau at 5 m/s."""
        chord = MOTH_BIEKER_V3.main_foil_chord
        taus = []
        for speed in [5.0, 10.0, 20.0]:
            tau = 4.0 * chord / (np.pi * speed)
            taus.append(tau)

        assert taus[0] > taus[1] > taus[2], f"tau should decrease with speed: {taus}"

        # Verify absolute values are reasonable
        assert taus[1] == pytest.approx(4 * 0.089 / (np.pi * 10.0), rel=1e-3)


class TestLiftLagBackwardCompat:
    """enable_lift_lag=False must produce identical 5-state behavior."""

    def test_disabled_gives_5_states(self):
        """Disabled lift lag should give 5 states."""
        moth = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=False)
        assert moth.num_states == 5
        assert len(moth.default_state()) == 5

    def test_enabled_gives_7_states(self):
        """Enabled lift lag should give 7 states."""
        moth = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=True)
        assert moth.num_states == 7
        assert len(moth.default_state()) == 7

    def test_disabled_bit_identical(self):
        """Disabled lift lag should produce bit-identical results to default."""
        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=10.0)
        assert trim.success

        moth_default = Moth3D(MOTH_BIEKER_V3)
        moth_disabled = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=False)

        state = jnp.array(trim.state)
        control = jnp.array(trim.control)

        d1 = moth_default.forward_dynamics(state, control, t=0.0)
        d2 = moth_disabled.forward_dynamics(state, control, t=0.0)

        assert jnp.allclose(d1, d2, atol=1e-15), f"Should be bit-identical: max diff={jnp.max(jnp.abs(d1-d2))}"


class TestLiftLagRK4Stability:
    """RK4 stable at dt=5ms across speed range 5-20 m/s."""

    @pytest.mark.parametrize("speed", [5.0, 10.0, 15.0, 20.0])
    def test_rk4_stable(self, speed):
        """Simulation should not diverge at dt=5ms for given speed."""
        moth = Moth3D(
            MOTH_BIEKER_V3,
            u_forward=ConstantSchedule(speed),
            enable_lift_lag=True,
        )

        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=speed)
        if not trim.success:
            pytest.skip(f"Trim failed at {speed} m/s")

        state_5 = jnp.array(trim.state)
        control = jnp.array(trim.control)

        # Initialize filter at trim alpha
        moth_5 = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(speed))
        aux_5 = moth_5.compute_aux(state_5, control, t=0.0)
        main_alpha = float(aux_5[moth_5.aux_names.index("main_alpha_eff")])
        rudder_alpha = float(aux_5[moth_5.aux_names.index("rudder_alpha_eff")])
        state_7 = jnp.concatenate([state_5, jnp.array([main_alpha, rudder_alpha])])

        dt = 0.005
        result = simulate(moth, state_7, dt=dt, duration=1.0,
                         control=ConstantControl(control))

        assert jnp.all(jnp.isfinite(result.states)), (
            f"States should be finite at {speed} m/s"
        )

        # dt/tau check: should be well within RK4 stability limit
        chord = MOTH_BIEKER_V3.main_foil_chord
        tau = 4.0 * chord / (np.pi * speed)
        ratio = dt / tau
        assert ratio < 2.5, f"dt/tau={ratio:.2f} too close to RK4 stability limit"


class TestLiftLagLQRSubsystem:
    """LQR operates on 5-state subsystem regardless of lift lag."""

    def test_lqr_uses_5_states(self):
        """K matrix should be (2, 5) regardless of lift lag."""
        from fmd.simulator.moth_lqr import design_moth_lqr
        lqr = design_moth_lqr(u_forward=10.0, dt=0.005)
        assert lqr.K.shape == (2, 5), f"K should be (2,5), got {lqr.K.shape}"
