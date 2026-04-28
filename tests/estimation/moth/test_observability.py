"""Observability analysis tests for Moth 3DOF with all sensor configurations.

Tests the observability of the linearized Moth3D system at 5 speeds (6-22 kts)
with 4 measurement configurations:
- full_state: All 5 states observed directly (H=I)
- vakaros: SOG, pitch, ride height (3 outputs)
- ardupilot_base: SOG, pitch, pitch rate, ride height (4 outputs)
- ardupilot_accel: SOG, pitch, pitch rate, ride height, vertical accel (5 outputs)

Observability is checked via the rank of the observability matrix O = [C; CA; ...; CA^{n-1}].

Marked slow because the module-scoped obs_data fixture computes 5 trim+linearize
points, taking ~65s on CPU.
"""

from fmd.simulator import _config  # noqa: F401

import numpy as np
import jax.numpy as jnp
import pytest

pytestmark = [
    pytest.mark.slow,
]

from fmd.simulator import Moth3D, ConstantSchedule
from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.trim_casadi import find_moth_trim
from fmd.simulator.linearize import linearize, observability_matrix, is_observable
from fmd.estimation import create_moth_measurement


# Speed points matching LQR analysis (6-22 kts in m/s)
SPEEDS = [3.09, 5.14, 7.20, 9.26, 11.32]
SPEED_LABELS = ["6 kt", "10 kt", "14 kt", "18 kt", "22 kt"]

VARIANTS = ["full_state", "speed_pitch_height", "speed_pitch_rate_height", "speed_pitch_rate_height_accel"]
VARIANT_SIZES = {"full_state": 5, "speed_pitch_height": 3, "speed_pitch_rate_height": 4, "speed_pitch_rate_height_accel": 5}

N_STATES = 5


@pytest.fixture(scope="module")
def obs_data():
    """Compute A matrices and measurement Jacobians at all speeds."""
    results = []
    for spd in SPEEDS:
        moth = Moth3D(MOTH_BIEKER_V3, u_forward=ConstantSchedule(spd))
        trim = find_moth_trim(MOTH_BIEKER_V3, u_forward=spd)
        x_trim = jnp.array(trim.state)
        u_trim = jnp.array(trim.control)
        A, B = linearize(moth, x_trim, u_trim)
        A_np = np.asarray(A)

        measurements = {}
        for variant in VARIANTS:
            bp = MOTH_BIEKER_V3.bowsprit_position if variant != "full_state" else None
            size = VARIANT_SIZES[variant]
            meas = create_moth_measurement(
                variant,
                bowsprit_position=bp,
                R=np.eye(size) * 0.01,
            )
            H = np.asarray(meas.get_measurement_jacobian(x_trim, u_trim))
            O = np.asarray(observability_matrix(jnp.array(A_np), jnp.array(H)))
            rank = int(np.linalg.matrix_rank(O))
            svs = np.linalg.svd(O, compute_uv=False)
            cond = float(svs[0] / svs[-1]) if svs[-1] > 1e-15 else float("inf")
            measurements[variant] = {
                "model": meas,
                "H": H,
                "O": O,
                "rank": rank,
                "singular_values": svs,
                "condition_number": cond,
            }

        results.append({
            "speed": spd,
            "A": A_np,
            "trim": trim,
            "x_trim": np.asarray(x_trim),
            "u_trim": np.asarray(u_trim),
            "measurements": measurements,
        })
    return results


class TestFullStateObservability:
    """Full state measurement (H=I) should always be trivially observable."""

    def test_full_state_always_observable(self, obs_data):
        """Rank = 5 (full) at all speeds with H=I."""
        for entry in obs_data:
            info = entry["measurements"]["full_state"]
            assert info["rank"] == N_STATES, (
                f"full_state not observable at {entry['speed']:.2f} m/s: "
                f"rank={info['rank']}, expected {N_STATES}"
            )

    def test_full_state_identity_H(self, obs_data):
        """Verify H is identity for full_state."""
        for entry in obs_data:
            H = entry["measurements"]["full_state"]["H"]
            np.testing.assert_allclose(H, np.eye(N_STATES), atol=1e-14)


class TestVakarosObservability:
    """Vakaros (3 measurements: SOG, pitch, ride_height)."""

    def test_vakaros_observability(self, obs_data):
        """Check rank at each speed and report."""
        for i, entry in enumerate(obs_data):
            info = entry["measurements"]["speed_pitch_height"]
            spd = entry["speed"]
            # Report rank and singular values for diagnostic purposes
            print(
                f"Vakaros @ {spd:.2f} m/s ({SPEED_LABELS[i]}): "
                f"rank={info['rank']}/{N_STATES}, "
                f"cond={info['condition_number']:.2e}, "
                f"svs={np.array2string(info['singular_values'][:N_STATES], precision=4)}"
            )
            # Vakaros has 3 outputs, so maximum possible rank contribution
            # per block is 3, but with n-1=4 powers we can reach rank 5
            # if the dynamics couple unobserved states into observed ones.
            # The test documents the actual rank.

    def test_vakaros_full_rank(self, obs_data):
        """Vakaros achieves full observability (rank 5) at all speeds."""
        for entry in obs_data:
            info = entry["measurements"]["speed_pitch_height"]
            assert info["rank"] == N_STATES, (
                f"Vakaros not fully observable at {entry['speed']:.2f} m/s: "
                f"rank={info['rank']}, expected {N_STATES}"
            )


class TestArduPilotBaseObservability:
    """ArduPilot base (4 measurements: SOG, pitch, pitch_rate, ride_height)."""

    def test_ardupilot_base_observability(self, obs_data):
        """Check rank at each speed and report."""
        for i, entry in enumerate(obs_data):
            info = entry["measurements"]["speed_pitch_rate_height"]
            spd = entry["speed"]
            print(
                f"ArduPilot base @ {spd:.2f} m/s ({SPEED_LABELS[i]}): "
                f"rank={info['rank']}/{N_STATES}, "
                f"cond={info['condition_number']:.2e}, "
                f"svs={np.array2string(info['singular_values'][:N_STATES], precision=4)}"
            )

    def test_ardupilot_base_full_rank(self, obs_data):
        """ArduPilot base achieves full observability (rank 5) at all speeds."""
        for entry in obs_data:
            info = entry["measurements"]["speed_pitch_rate_height"]
            assert info["rank"] == N_STATES, (
                f"ArduPilot base not fully observable at {entry['speed']:.2f} m/s: "
                f"rank={info['rank']}, expected {N_STATES}"
            )


class TestArduPilotAccelObservability:
    """ArduPilot with accel (5 measurements: SOG, pitch, pitch_rate, ride_height, vertical_accel)."""

    def test_ardupilot_accel_observability(self, obs_data):
        """Check rank at each speed and report."""
        for i, entry in enumerate(obs_data):
            info = entry["measurements"]["speed_pitch_rate_height_accel"]
            spd = entry["speed"]
            print(
                f"ArduPilot accel @ {spd:.2f} m/s ({SPEED_LABELS[i]}): "
                f"rank={info['rank']}/{N_STATES}, "
                f"cond={info['condition_number']:.2e}, "
                f"svs={np.array2string(info['singular_values'][:N_STATES], precision=4)}"
            )

    def test_ardupilot_accel_full_rank(self, obs_data):
        """ArduPilot accel achieves full observability (rank 5) at all speeds."""
        for entry in obs_data:
            info = entry["measurements"]["speed_pitch_rate_height_accel"]
            assert info["rank"] == N_STATES, (
                f"ArduPilot accel not fully observable at {entry['speed']:.2f} m/s: "
                f"rank={info['rank']}, expected {N_STATES}"
            )


class TestRankMonotonicity:
    """More measurements should not decrease observability rank."""

    def test_more_measurements_no_rank_decrease(self, obs_data):
        """Rank ordering: full_state >= ardupilot_accel >= ardupilot_base >= vakaros."""
        for entry in obs_data:
            spd = entry["speed"]
            ranks = {v: entry["measurements"][v]["rank"] for v in VARIANTS}

            assert ranks["full_state"] >= ranks["speed_pitch_rate_height_accel"], (
                f"At {spd:.2f} m/s: full_state rank ({ranks['full_state']}) < "
                f"ardupilot_accel rank ({ranks['ardupilot_accel']})"
            )
            assert ranks["speed_pitch_rate_height_accel"] >= ranks["speed_pitch_rate_height"], (
                f"At {spd:.2f} m/s: ardupilot_accel rank ({ranks['ardupilot_accel']}) < "
                f"ardupilot_base rank ({ranks['ardupilot_base']})"
            )
            assert ranks["speed_pitch_rate_height"] >= ranks["speed_pitch_height"], (
                f"At {spd:.2f} m/s: ardupilot_base rank ({ranks['ardupilot_base']}) < "
                f"vakaros rank ({ranks['vakaros']})"
            )


class TestObservabilityConsistency:
    """Cross-checks between is_observable() and matrix_rank."""

    def test_is_observable_matches_rank(self, obs_data):
        """is_observable(A, H) should agree with rank(O) == n."""
        for entry in obs_data:
            A = jnp.array(entry["A"])
            for variant in VARIANTS:
                H = jnp.array(entry["measurements"][variant]["H"])
                rank = entry["measurements"][variant]["rank"]
                observable = is_observable(A, H)
                expected = rank == N_STATES
                assert observable == expected, (
                    f"is_observable disagrees with rank at {entry['speed']:.2f} m/s, "
                    f"{variant}: is_observable={observable}, rank={rank}"
                )


class TestUnobservableSubspace:
    """When rank < n, verify the unobservable subspace is properly identified."""

    def test_all_variants_fully_observable(self, obs_data, artifact_saver):
        """All variants at all speeds should be fully observable (rank = n).

        This test explicitly asserts that no unobservable subspace exists,
        which is a stronger statement than the individual rank tests: it
        covers every speed x variant combination in a single assertion.
        """
        for entry in obs_data:
            spd = entry["speed"]
            for variant in VARIANTS:
                info = entry["measurements"][variant]
                assert info["rank"] == N_STATES, (
                    f"{variant} @ {spd:.2f} m/s: rank={info['rank']}, "
                    f"expected {N_STATES} (fully observable)"
                )

        # Save observability summary
        save_data = {
            "speeds": np.array(SPEEDS),
        }
        variant_names = VARIANTS  # ["full_state", "speed_pitch_height", "speed_pitch_rate_height", "speed_pitch_rate_height_accel"]
        for variant in variant_names:
            ranks = np.array([entry["measurements"][variant]["rank"] for entry in obs_data])
            conds = np.array([entry["measurements"][variant]["condition_number"] for entry in obs_data])
            save_data[f"rank_{variant}"] = ranks
            save_data[f"cond_{variant}"] = conds

        artifact_saver.save("test_observability_summary", save_data, metadata={
            "variants": ",".join(variant_names),
        })
