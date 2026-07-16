"""Tests for fmd_commit/install_mode/params_hash provenance stamping (C1.H)."""

import attrs

from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator.provenance import (
    _classify_direct_url,
    get_fmd_provenance,
    params_hash,
    provenance_stamp,
)


def test_classify_direct_url_pin():
    direct_url = {
        "url": "https://github.com/brooksreed/fomodynamics.git",
        "vcs_info": {"vcs": "git", "commit_id": "8f4a01c123ba776cdf9909247f0c2cac6b64930a"},
    }
    commit, mode = _classify_direct_url(direct_url)
    assert commit == "8f4a01c123ba776cdf9909247f0c2cac6b64930a"
    assert mode == "pin"


def test_classify_direct_url_editable(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "commit",
         "--allow-empty", "-q", "-m", "init"],
        cwd=tmp_path, check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    direct_url = {"url": f"file://{tmp_path}", "dir_info": {"editable": True}}
    commit, mode = _classify_direct_url(direct_url)
    assert mode == "editable"
    assert commit == head


def test_classify_direct_url_unknown():
    commit, mode = _classify_direct_url({})
    assert commit is None
    assert mode == "unknown"


def test_get_fmd_provenance_missing_package():
    result = get_fmd_provenance(package="not-a-real-package-xyz")
    assert result == {"fmd_commit": None, "fmd_install_mode": "unknown"}


def test_get_fmd_provenance_installed():
    """The installed fomodynamics dist should classify as pin or editable, not unknown."""
    result = get_fmd_provenance()
    assert result["fmd_install_mode"] in ("pin", "editable")


def test_params_hash_deterministic():
    assert params_hash(MOTH_BIEKER_V3) == params_hash(MOTH_BIEKER_V3)


def test_params_hash_changes_with_field():
    changed = attrs.evolve(MOTH_BIEKER_V3, hull_mass=MOTH_BIEKER_V3.hull_mass + 1.0)
    assert params_hash(changed) != params_hash(MOTH_BIEKER_V3)


def test_provenance_stamp_has_all_three_fields():
    stamp = provenance_stamp(MOTH_BIEKER_V3)
    assert set(stamp.keys()) == {"fmd_commit", "fmd_install_mode", "params_hash"}
