"""Directional (sign) locks for the wave -> foil channel.

These tests pin the *direction* of the wave-to-foil coupling, not just its
magnitude. They are the long-term lock for two independent ~180 deg phase
bugs found in the physics review (roadmap chunk C1.A):

  * ETA-DEPTH (audit H1): ``compute_foil_ned_depth`` composed wave elevation
    with the wrong sign, so the model ventilated under crests instead of
    troughs.
  * WAVE-AOA (fmd review 3.2): wave orbital velocity entered the foil AoA
    (a) added instead of subtracted (relative velocity), and (b) rotated with
    ``Ry(theta)`` instead of ``Ry(theta)^T``.

Pre-existing wave tests only asserted "waves change the dynamics" (magnitude),
which both the buggy and corrected code satisfy. Each test below is written to
FAIL on the pre-fix code in the physically wrong direction.
"""
import numpy as np
import pytest
import jax.numpy as jnp

from fmd.simulator import Moth3D, Environment
from fmd.simulator.params import MOTH_BIEKER_V3, WaveParams
from fmd.simulator.components.moth_forces import (
    compute_foil_ned_depth,
    compute_depth_factor,
)
from fmd.simulator.components.moth_wand import wand_angle_from_state_waves
from fmd.simulator.moth_3d import POS_D, THETA, U


def _aux(moth, state, control, t, env):
    """compute_aux returns a flat array; return a name->float dict."""
    arr = moth.compute_aux(state, control, t=t, env=env)
    return {name: float(arr[i]) for i, name in enumerate(moth.aux_names)}


# ---------------------------------------------------------------------------
# ETA-DEPTH (audit H1): crest => local surface rises => foil deeper
# ---------------------------------------------------------------------------

class TestEtaDepthDirection:
    def test_crest_makes_foil_deeper(self):
        """Gap 11a: compute_foil_ned_depth must be monotone increasing in eta.

        In NED (D positive down) the local surface sits at D = -eta_up, so a
        point's depth below the *local* surface is D_point + eta_up: a crest
        (eta > 0) makes the foil deeper. The pre-fix code did ``- eta`` and
        inverted this.
        """
        kw = dict(pos_d=0.5, eff_pos_x=0.0, eff_pos_z=0.0, theta=0.0, heel_angle=0.0)
        d_calm = float(compute_foil_ned_depth(**kw, eta=0.0))
        d_crest = float(compute_foil_ned_depth(**kw, eta=0.3))
        d_trough = float(compute_foil_ned_depth(**kw, eta=-0.3))

        assert d_crest > d_calm > d_trough, (
            f"crest should be deepest: crest={d_crest}, calm={d_calm}, trough={d_trough}"
        )
        # Quantitative: depth increases by exactly +eta.
        assert d_crest == pytest.approx(d_calm + 0.3, abs=1e-9)
        assert d_trough == pytest.approx(d_calm - 0.3, abs=1e-9)

    def test_depth_factor_ordering_in_ventilation_band(self):
        """Gap 11b: depth_factor(crest) > depth_factor(calm) > depth_factor(trough).

        Crest => foil deeper => less ventilated => factor toward 1. Placed in
        the sensitive band (0 < factor < 1) so the ``>=`` is a real, non-vacuous
        gap rather than a saturated 1 == 1 == 1.
        """
        heel = 0.5235987755982988  # 30 deg (MOTH_BIEKER_V3 main foil)
        span = 0.95
        # Foil center depth ~0.20 m sits mid-transition (df ~ 0.54); +/-0.02 m
        # of wave elevation gives well-separated, non-saturated factors.
        base_pos_d = 0.20
        a = 0.02

        def df_at(eta):
            fd = float(compute_foil_ned_depth(base_pos_d, 0.0, 0.0, 0.0, heel, eta=eta))
            return float(compute_depth_factor(fd, span, heel, 0.30, "smooth"))

        df_crest, df_calm, df_trough = df_at(a), df_at(0.0), df_at(-a)

        # Non-vacuous: calm sits strictly interior (sensitive band).
        assert 0.05 < df_calm < 0.95, f"calm depth_factor not in sensitive band: {df_calm}"
        assert df_crest - df_calm > 0.05, (df_crest, df_calm)
        assert df_calm - df_trough > 0.05, (df_calm, df_trough)


# ---------------------------------------------------------------------------
# WAVE-AOA (a): relative velocity => upwash increases AoA (theta = 0 isolates
# the sign of the orbital-velocity subtraction; the O(theta) rotation term is
# zero here so this locks fix (a) alone).
# ---------------------------------------------------------------------------

class TestWaveAoaRelativeVelocity:
    def _scan(self, moth, env, state, control, period, n=48):
        ts = np.linspace(0.0, period, n, endpoint=False)
        rows = [_aux(moth, state, control, float(t), env) for t in ts]
        return ts, rows

    def test_upwash_increases_aoa_downwash_decreases(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        period = 4.0
        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=period))
        state = moth.default_state()          # level (theta=0), foils submerged ~0.3 m
        control = moth.default_control()

        alpha_calm = _aux(moth, state, control, 1.0, env=None)["main_alpha_geo"]

        ts, rows = self._scan(moth, env, state, control, period)
        w_orb = np.array([r["wave_w_orbital_main"] for r in rows])
        alpha = np.array([r["main_alpha_geo"] for r in rows])

        # NED w positive = downward, so most-negative w_orbital = strongest upwash.
        i_up = int(np.argmin(w_orb))
        i_down = int(np.argmax(w_orb))
        assert w_orb[i_up] < -1e-3 and w_orb[i_down] > 1e-3, "scan did not span up/downwash"

        assert alpha[i_up] > alpha_calm, (
            f"upwash must raise AoA: upwash alpha={alpha[i_up]}, calm={alpha_calm}"
        )
        assert alpha[i_down] < alpha_calm, (
            f"downwash must lower AoA: downwash alpha={alpha[i_down]}, calm={alpha_calm}"
        )

    def test_lift_rises_under_upwash_at_equal_depth(self):
        """AoA increase => lift rise. Compared against the calm reference at the
        upwash instant, chosen where eta ~ 0 so submergence (hence depth_factor)
        matches calm and the lift difference is driven by AoA alone."""
        moth = Moth3D(MOTH_BIEKER_V3)
        period = 4.0
        env = Environment.with_waves(WaveParams.regular(amplitude=0.3, period=period))
        state = moth.default_state()
        control = moth.default_control()

        lift_calm = _aux(moth, state, control, 0.0, env=None)["main_lift_aero"]

        ts, rows = self._scan(moth, env, state, control, period)
        w_orb = np.array([r["wave_w_orbital_main"] for r in rows])
        lift = np.array([r["main_lift_aero"] for r in rows])
        eta = np.array([r["wave_eta_main"] for r in rows])

        i_up = int(np.argmin(w_orb))
        # eta ~ 0 at the upwash instant => same depth/ventilation as calm.
        assert abs(eta[i_up]) < 0.05, f"upwash instant not at a wave face: eta={eta[i_up]}"
        assert lift[i_up] > lift_calm, (
            f"upwash should raise lift vs calm: up={lift[i_up]}, calm={lift_calm}"
        )


# ---------------------------------------------------------------------------
# WAVE-AOA (b): NED->body rotation must be Ry(theta)^T. This is O(theta), so it
# is INVISIBLE at level trim; the test uses theta != 0 and checks the exposed
# body-frame orbital velocity against a hand-computed transpose.
# ---------------------------------------------------------------------------

class TestWaveAoaRotationTranspose:
    def test_orbital_rotation_is_transpose_at_pitch(self):
        moth = Moth3D(MOTH_BIEKER_V3)
        theta = 0.20  # nonzero pitch makes Ry^T distinguishable from Ry
        period = 4.0
        t = 1.3       # off a node so both orbital components are sizeable
        env = Environment.with_waves(WaveParams.regular(amplitude=0.5, period=period))

        state = moth.default_state().at[THETA].set(theta)
        control = moth.default_control()
        aux = _aux(moth, state, control, t, env)

        # Reconstruct the exact NED orbital query the model makes (mirror of
        # Moth3D._compute_step_terms); cg_offset and u_fwd are read from aux so
        # nothing is hard-coded.
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        eff_x = float(moth.main_foil.position_x) - aux["cg_offset_x"]
        eff_z = float(moth.main_foil.position_z) - aux["cg_offset_z"]
        u_safe = max(aux["u_fwd"], 0.1)
        x_ned = u_safe * t + eff_x * cos_t + eff_z * sin_t
        z_ned = float(compute_foil_ned_depth(
            float(state[POS_D]), eff_x, eff_z, theta, float(moth.main_foil.heel_angle)))
        orb = np.asarray(env.wave_field.orbital_velocity(x_ned, 0.0, z_ned, t))

        # Ry(theta)^T applied to NED orbital [n, e, d]:
        u_b_correct = orb[0] * cos_t - orb[2] * sin_t
        w_b_correct = orb[0] * sin_t + orb[2] * cos_t
        # Buggy Ry(theta) (non-transpose) value, for the guard below:
        w_b_bug = -orb[0] * sin_t + orb[2] * cos_t

        # The chosen (theta, phase) must actually exercise the O(theta) term,
        # otherwise the test would be vacuous.
        assert abs(w_b_correct - w_b_bug) > 0.02, "phase/theta does not exercise transpose"

        assert aux["wave_u_orbital_main"] == pytest.approx(u_b_correct, abs=1e-6)
        assert aux["wave_w_orbital_main"] == pytest.approx(w_b_correct, abs=1e-6)


# ---------------------------------------------------------------------------
# Wand / depth-factor coherence (gap 12): the mechanical wand (which composes
# eta correctly) and the foil ventilation model must respond to the SAME
# surface. Pre-fix they were ~half a cycle out of phase (wand up on the crest
# while the foil model believed it had gone shallow).
# ---------------------------------------------------------------------------

class TestWandDepthFactorCoherence:
    def test_wand_and_depth_factor_move_together(self):
        params = MOTH_BIEKER_V3
        moth = Moth3D(params)
        period = 4.0
        env = Environment.with_waves(WaveParams.regular(amplitude=0.15, period=period))

        # Near-static boat with the main foil in the ventilation-sensitive band
        # so depth_factor actually swings across the wave cycle.
        state = moth.default_state().at[U].set(0.0)      # ~static
        # Raise the boat so the foil center sits ~0.13 m deep (in the band).
        state = state.at[POS_D].set(-1.47)
        control = moth.default_control()

        ts = np.linspace(0.0, period, 40, endpoint=False)
        df = []
        wand = []
        for t in ts:
            aux = _aux(moth, state, control, float(t), env)
            df.append(aux["main_df"])
            wand.append(float(wand_angle_from_state_waves(
                state[POS_D], state[THETA], 0.0, float(t), env.wave_field,
                params.wand_pivot_position, params.wand_length,
                moth.main_foil.heel_angle)))
        df = np.array(df)
        wand = np.array(wand)

        # Both must actually vary (non-vacuous).
        assert np.ptp(df) > 0.05, f"depth_factor did not swing: ptp={np.ptp(df)}"
        assert np.ptp(wand) > 0.05, f"wand angle did not swing: ptp={np.ptp(wand)}"

        # Coherent surface response: crest => wand rises AND foil deepens
        # (df toward 1) => positive correlation. Pre-fix eta inversion makes the
        # foil model anti-correlated with the (correct) wand.
        corr = float(np.corrcoef(wand, df)[0, 1])
        assert corr > 0.5, f"wand and depth_factor should co-vary, corr={corr}"
