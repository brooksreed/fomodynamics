"""Gate tests for ENC-DIST: integrated encounter distance (roadmap chunk C1.C).

Wave phase used to be queried at ``x_ned = u_safe * t + offsets``. With a
varying forward speed, ``u(t) * t`` is not ``∫u dt`` — the encounter frequency
is distorted and past distance is non-physically rewritten every step
(fmd review §4.1). The fix carries an opt-in integrated encounter-distance
state ``x_n`` (``enable_encounter_distance=True``) with
``ẋ_n = u·cosθ + w·sinθ`` (the NED-north velocity of the CG) and queries the
wave field at ``x_n`` instead.

These tests lock the fix:
  * ``x_n`` is an opt-in extra state that stays OUT of default models.
  * ``ẋ_n`` equals the NED-north velocity in the model's own frame convention.
  * In a decelerating run the model's wave phase matches an independent
    ``∫u dt`` reference and diverges from the old ``u·t`` coordinate.
"""
import numpy as np
import pytest
import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

from fmd.simulator import (
    Moth3D,
    Environment,
    ConstantControl,
    ConstantSchedule,
)
from fmd.simulator.params import MOTH_BIEKER_V3, WaveParams
from fmd.simulator import simulate
from fmd.simulator.moth_3d import THETA, W, U


class _LinearSchedule(eqx.Module):
    """Decelerating forward-speed schedule u(t) = u0 + accel * t (accel < 0)."""

    u0: Array
    accel: Array

    def __init__(self, u0, accel):
        self.u0 = jnp.asarray(float(u0))
        self.accel = jnp.asarray(float(accel))

    def __call__(self, t):
        return self.u0 + self.accel * t


def _aux_index(moth, name):
    return moth.aux_names.index(name)


class TestEncounterDistanceState:
    def test_disabled_by_default(self):
        """Default models must NOT carry x_n (keeps trim/LQR/EKF/MPC untouched)."""
        m = Moth3D(MOTH_BIEKER_V3)
        assert m.enable_encounter_distance is False
        assert m.x_n_index is None
        assert m.num_states == 5
        assert "x_n" not in m.state_names

    def test_appended_last_across_configs(self):
        """x_n is appended after the optional lift-lag states."""
        m1 = Moth3D(MOTH_BIEKER_V3, enable_encounter_distance=True)
        assert m1.num_states == 6
        assert m1.x_n_index == 5
        assert m1.state_names[-1] == "x_n"
        assert float(m1.default_state()[5]) == 0.0

        m2 = Moth3D(MOTH_BIEKER_V3, enable_lift_lag=True, enable_encounter_distance=True)
        assert m2.num_states == 8
        assert m2.x_n_index == 7
        assert m2.state_names[-1] == "x_n"

    def test_not_coupled_to_surge(self):
        """Enabling surge must not silently add x_n (would break trim)."""
        m = Moth3D(MOTH_BIEKER_V3, surge_enabled=True)
        assert m.x_n_index is None


class TestEncounterDistanceDerivative:
    def test_xn_dot_is_ned_north_velocity(self):
        """ẋ_n = u·cosθ + w·sinθ in the model's own frame (matches pos_d_dot)."""
        m = Moth3D(MOTH_BIEKER_V3, surge_enabled=True, enable_encounter_distance=True)
        theta, w, u = 0.12, 0.7, 9.3
        # [pos_d, theta, w, q, u, x_n]
        state = jnp.array([-1.3, theta, w, 0.0, u, 42.0])
        control = jnp.zeros(2)
        deriv = m.forward_dynamics(state, control, t=0.0)

        expected = u * np.cos(theta) + w * np.sin(theta)
        assert float(deriv[m.x_n_index]) == pytest.approx(expected, rel=1e-6)

    def test_xn_dot_does_not_depend_on_xn(self):
        """ẋ_n = u·cosθ + w·sinθ carries no x_n term (in calm water or waves)."""
        m = Moth3D(MOTH_BIEKER_V3, surge_enabled=True, enable_encounter_distance=True)
        base = jnp.array([-1.3, 0.05, 0.2, 0.0, 9.0, 0.0])
        d0 = m.forward_dynamics(base, jnp.zeros(2), t=0.0)
        d1 = m.forward_dynamics(base.at[5].set(1000.0), jnp.zeros(2), t=0.0)
        # x_n's own derivative is unchanged by x_n's value.
        assert float(d0[m.x_n_index]) == pytest.approx(float(d1[m.x_n_index]), abs=1e-9)

    def test_xn_is_inert_in_calm_water(self):
        """With no wave field, x_n influences NOTHING (stays out of trim/control)."""
        m = Moth3D(MOTH_BIEKER_V3, surge_enabled=True, enable_encounter_distance=True)
        base = jnp.array([-1.3, 0.05, 0.2, 0.0, 9.0, 0.0])
        d0 = m.forward_dynamics(base, jnp.zeros(2), t=0.0, env=None)
        d1 = m.forward_dynamics(base.at[5].set(1000.0), jnp.zeros(2), t=0.0, env=None)
        assert float(jnp.max(jnp.abs(d0 - d1))) == pytest.approx(0.0, abs=1e-9)

    def test_xn_feeds_forces_only_through_wave_channel(self):
        """With waves, changing x_n shifts wave phase and so DOES move the foil
        forces — confirming x_n enters dynamics *only* via the wave-phase query."""
        m = Moth3D(MOTH_BIEKER_V3, surge_enabled=True, enable_encounter_distance=True)
        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=2.5))
        base = jnp.array([-1.3, 0.02, 0.1, 0.0, 9.0, 0.0])
        control = jnp.zeros(2)
        # Shift x_n by a quarter wavelength so the sampled wave phase changes.
        shifted = base.at[m.x_n_index].set(2.4)
        d0 = m.forward_dynamics(base, control, t=0.0, env=env)
        d1 = m.forward_dynamics(shifted, control, t=0.0, env=env)
        # w_dot / q_dot (heave/pitch accelerations) respond to the wave phase.
        assert float(jnp.max(jnp.abs(d0[:5] - d1[:5]))) > 1e-4


class TestDeceleratingEncounterPhase:
    """The roadmap gate: encounter phase matches ∫u dt, diverges from u·t."""

    def _run(self, enable_encounter_distance):
        # Deterministic decelerating speed via schedule (surge as open-loop input).
        sched = _LinearSchedule(u0=10.0, accel=-1.0)  # 10 -> 7 m/s over 3 s
        moth = Moth3D(
            MOTH_BIEKER_V3,
            u_forward=sched,
            surge_enabled=False,
            enable_encounter_distance=enable_encounter_distance,
        )
        # Short, steep-ish regular wave so a ~4.5 m coordinate error is a large
        # fraction of a wavelength (T=2.5 s -> L≈9.8 m deep water).
        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=2.5))
        control = ConstantControl(jnp.zeros(2))
        state0 = moth.default_state()
        result = simulate(moth, state0, dt=0.005, duration=3.0, env=env, control=control)
        return moth, env, result

    def test_xn_matches_integral_and_diverges_from_ut(self):
        moth, env, result = self._run(enable_encounter_distance=True)
        t = np.asarray(result.times)
        states = np.asarray(result.states)
        u = np.asarray([float(moth.u_forward_schedule(ti)) for ti in t])
        theta = states[:, THETA]
        w = states[:, W]
        x_n = states[:, moth.x_n_index]

        # (1) x_n integrates the NED-north velocity of the CG (trapezoid ref).
        v_north = u * np.cos(theta) + w * np.sin(theta)
        x_ref = np.concatenate([[0.0], np.cumsum(0.5 * (v_north[1:] + v_north[:-1]) * np.diff(t))])
        assert np.max(np.abs(x_n - x_ref)) < 0.02, "x_n must track ∫(u·cosθ+w·sinθ) dt"

        # (2) x_n diverges from the old u·t coordinate as speed bleeds off.
        u_safe = np.maximum(u, 0.1)
        x_ut = u_safe * t
        assert (x_n[-1] - x_ut[-1]) > 1.0, (
            f"decelerating run must separate x_n from u·t; got "
            f"x_n={x_n[-1]:.3f}, u·t={x_ut[-1]:.3f}"
        )
        # Analytic check for level flight: ∫u dt − u·t ≈ 0.5·|accel|·t².
        assert (x_n[-1] - x_ut[-1]) == pytest.approx(0.5 * 1.0 * t[-1] ** 2, rel=0.1)

    def test_model_wave_phase_matches_integral_not_ut(self):
        moth, env, result = self._run(enable_encounter_distance=True)
        t = np.asarray(result.times)
        states = np.asarray(result.states)
        controls = np.asarray(result.controls)

        idx = _aux_index(moth, "wave_eta_main")
        aux = vmap(lambda s, c, ti: moth.compute_aux(s, c, t=ti, env=env))(
            result.states, result.controls, result.times
        )
        eta_model = np.asarray(aux)[:, idx]

        # Reconstruct the two candidate encounter coordinates for the MAIN foil.
        u = np.asarray([float(moth.u_forward_schedule(ti)) for ti in t])
        theta = states[:, THETA]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # main-foil body offset relative to CG (sailor at default => cg_offset≈const);
        # read straight from the model geometry at t=0.
        geom = moth.get_geometry(0.0)
        eff_x, eff_z = float(geom.main_foil_position[0]), float(geom.main_foil_position[2])

        x_n = states[:, moth.x_n_index]
        x_main_int = x_n + eff_x * cos_t + eff_z * sin_t
        u_safe = np.maximum(u, 0.1)
        x_main_ut = u_safe * t + eff_x * cos_t + eff_z * sin_t

        wf = env.wave_field
        eta_int = np.asarray([float(wf.elevation(x_main_int[i], 0.0, t[i])) for i in range(len(t))])
        eta_ut = np.asarray([float(wf.elevation(x_main_ut[i], 0.0, t[i])) for i in range(len(t))])

        # Model wave phase matches the ∫u dt reference (its own x_n), to tol.
        assert np.max(np.abs(eta_model - eta_int)) < 1e-4, (
            "model wave elevation must equal elevation queried at x_n"
        )
        # And it is materially different from the old u·t coordinate.
        rms_diff = np.sqrt(np.mean((eta_int - eta_ut) ** 2))
        assert rms_diff > 0.05, (
            f"∫u dt and u·t encounter phases must diverge; rms diff={rms_diff:.4f}"
        )
