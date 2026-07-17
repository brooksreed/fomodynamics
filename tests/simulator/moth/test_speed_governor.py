"""Directional + regression tests for the C2.C0 P speed-governor sail.

The governor replaces the calibrated thrust *table* (a required-thrust curve
with zero surge stiffness — the C2.B runaway) with an affine "sailor model"
``F_sail = T0 + Kp*(u_target - u)``. These tests lock:

  (i)   governor sign — u below target => thrust above T0 (and vice-versa);
  (ii)  surge-stiffness regression — the sail's df/du flips from POSITIVE
        (table, the bug) to NEGATIVE (=-Kp, restoring), isolating the
        discovered null-stiffness bug class;
  (iii) a calm-water governed sim actually holds u_target (no droop);
  (iv)  the construction guards (surge required, Kp>0).

Design: docs/private/plans/wand_vs_pid_waves/thrust_governor_design.md (blur).
"""

from fmd.simulator import _config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmd.simulator.closed_loop_pipeline import simulate_closed_loop
from fmd.simulator.moth_3d import Moth3D, ConstantSchedule
from fmd.simulator.moth_lqr import design_moth_lqr
from fmd.simulator.moth_scenarios import (
    apply_speed_governor,
    create_mechanical_wand_config,
    governor_thrust0,
)
from fmd.simulator.params import MOTH_BIEKER_V3

HEEL = np.deg2rad(30.0)
U_TARGET = 10.0
KP = 40.0


@pytest.fixture(scope="module")
def lqr():
    return design_moth_lqr(
        params=MOTH_BIEKER_V3, u_forward=U_TARGET, dt=0.005, heel_angle=HEEL,
    )


@pytest.fixture(scope="module")
def trim_state(lqr):
    return np.asarray(lqr.trim.state)


def _sail_fx(sail, state, u):
    f, _ = sail.compute_moth(state, jnp.array([0.0, 0.0]), u_forward=float(u))
    return float(f[0])


# ---------------------------------------------------------------------------
# (i) Governor sign lock
# ---------------------------------------------------------------------------


def test_governor_sign_below_and_above_target(lqr, trim_state):
    """u below target => F_sail > T0; u above target => F_sail < T0."""
    T0 = float(lqr.trim.thrust)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                  heel_angle=HEEL, surge_enabled=True)
    gov = apply_speed_governor(moth, thrust0=T0, kp=KP, u_target=U_TARGET)
    # theta=0 so body-frame f_x equals the horizontal governor thrust.
    state = jnp.array([trim_state[0], 0.0, 0.0, 0.0, U_TARGET])

    f_at = _sail_fx(gov.sail, state, U_TARGET)
    f_slow = _sail_fx(gov.sail, state, U_TARGET - 1.0)
    f_fast = _sail_fx(gov.sail, state, U_TARGET + 1.0)

    assert f_at == pytest.approx(T0, abs=1e-3)
    assert f_slow > T0          # too slow -> sailor sheets on
    assert f_fast < T0          # too fast -> sailor eases
    # Exact governor increments: +/- Kp per (m/s).
    assert (f_slow - T0) == pytest.approx(KP, abs=1e-3)
    assert (T0 - f_fast) == pytest.approx(KP, abs=1e-3)


# ---------------------------------------------------------------------------
# (ii) Surge-stiffness regression lock (the C2.B null-stiffness bug class)
# ---------------------------------------------------------------------------


def test_surge_stiffness_sign_flip_table_vs_governor(lqr, trim_state):
    """The sail's df/du is POSITIVE for the table (positive feedback — the
    bug) and NEGATIVE (=-Kp, restoring) for the governor.

    Computed directly on the sail element (theta fixed), so it isolates the
    thrust law from the full dynamics and is mirror-independent. This is the
    permanent lock on the discovered null-/negative-stiffness bug class.
    """
    T0 = float(lqr.trim.thrust)
    table_moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                        heel_angle=HEEL, surge_enabled=True)
    gov = apply_speed_governor(table_moth, thrust0=T0, kp=KP, u_target=U_TARGET)
    state = jnp.array([trim_state[0], trim_state[1], 0.0, 0.0, U_TARGET])

    du = 1e-3
    table_slope = (
        _sail_fx(table_moth.sail, state, U_TARGET + du)
        - _sail_fx(table_moth.sail, state, U_TARGET - du)
    ) / (2 * du)
    gov_slope = (
        _sail_fx(gov.sail, state, U_TARGET + du)
        - _sail_fx(gov.sail, state, U_TARGET - du)
    ) / (2 * du)

    assert table_slope > 1.0, (
        f"calibrated table thrust rises with u (df/du={table_slope:.3f}>0) — "
        "positive surge feedback, the root cause of the C2.B runaway"
    )
    assert gov_slope < 0.0, f"governor must restore (df/du<0); got {gov_slope:.3f}"
    assert gov_slope == pytest.approx(-KP, abs=1e-2)


def test_governor_stiffness_at_system_level(lqr, trim_state):
    """Secondary, dynamics-level check: d(u_dot)/du at trim is clearly
    negative under the governor (net restoring surge acceleration)."""
    T0 = float(lqr.trim.thrust)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                  heel_angle=HEEL, surge_enabled=True)
    gov = apply_speed_governor(moth, thrust0=T0, kp=KP, u_target=U_TARGET)
    x = jnp.array([trim_state[0], trim_state[1], trim_state[2],
                   trim_state[3], U_TARGET])
    u_ctrl = jnp.array(lqr.trim.control)
    du = 1e-3
    udot_plus = float(gov.forward_dynamics(x.at[4].set(U_TARGET + du), u_ctrl)[4])
    udot_minus = float(gov.forward_dynamics(x.at[4].set(U_TARGET - du), u_ctrl)[4])
    dudot_du = (udot_plus - udot_minus) / (2 * du)
    assert dudot_du < 0.0, f"governed surge must be restoring; d(u_dot)/du={dudot_du:.4f}"


# ---------------------------------------------------------------------------
# (iii) Calm-water governed sim holds u_target
# ---------------------------------------------------------------------------


def test_calm_governed_sim_holds_u_target(lqr, trim_state):
    """A calm-water (no-wave) governed closed-loop sim settles at u_target
    with no monotonic droop — the property the raw table plant lacks."""
    T0 = float(lqr.trim.thrust)
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                  heel_angle=HEEL, surge_enabled=True)
    gov = apply_speed_governor(moth, thrust0=T0, kp=KP, u_target=U_TARGET)

    sensor, estimator, controller = create_mechanical_wand_config(
        lqr, params=MOTH_BIEKER_V3, heel_angle=HEEL,
    )
    x0 = jnp.array(lqr.trim.state)
    # Start 1 m/s slow: the governor must climb back to target.
    x0 = x0.at[4].set(U_TARGET - 1.0)
    P0 = jnp.eye(gov.num_states) * 0.1

    result = simulate_closed_loop(
        system=gov, sensor=sensor, estimator=estimator, controller=controller,
        x0_true=x0, x0_est=jnp.array(lqr.trim.state), P0=P0,
        dt=0.01, duration=25.0, rng_key=jax.random.PRNGKey(0),
        params=MOTH_BIEKER_V3, env=None,
        trim_state=jnp.array(lqr.trim.state),
        trim_control=jnp.array(lqr.trim.control),
        u_trim=jnp.array(lqr.trim.control),
    )
    u = np.asarray(result.true_states[1:, 4])
    # Settles near target (within the P governor's static tolerance).
    assert abs(float(np.mean(u[-500:])) - U_TARGET) < 0.15
    # Recovered from the initial deficit (last window faster than the start).
    assert float(np.mean(u[-500:])) > float(u[0]) + 0.5


# ---------------------------------------------------------------------------
# (iv) Construction guards + T0 helper
# ---------------------------------------------------------------------------


def test_apply_speed_governor_requires_surge(lqr):
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                  heel_angle=HEEL, surge_enabled=False)
    with pytest.raises(ValueError, match="surge_enabled=True"):
        apply_speed_governor(moth, thrust0=75.5, kp=KP, u_target=U_TARGET)


def test_apply_speed_governor_requires_positive_kp(lqr):
    moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(U_TARGET),
                  heel_angle=HEEL, surge_enabled=True)
    with pytest.raises(ValueError, match="Kp"):
        apply_speed_governor(moth, thrust0=75.5, kp=0.0, u_target=U_TARGET)


def test_governor_thrust0_matches_lqr_trim_at_natural_setpoint(lqr, trim_state):
    """T0 from the pinned solve at the natural setpoint reproduces the LQR
    design point's thrust (single-branch, consistent by construction)."""
    T0 = governor_thrust0(
        MOTH_BIEKER_V3, target_pos_d=float(trim_state[0]),
        u_target=U_TARGET, heel_angle=HEEL,
    )
    assert T0 == pytest.approx(float(lqr.trim.thrust), rel=1e-4)
    # A deeper setpoint needs MORE thrust (more strut immersion drag).
    deeper = float(trim_state[0]) + 0.2
    T0_deep = governor_thrust0(
        MOTH_BIEKER_V3, target_pos_d=deeper, u_target=U_TARGET, heel_angle=HEEL,
    )
    assert T0_deep > T0
