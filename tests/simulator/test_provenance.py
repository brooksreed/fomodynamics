"""Tests for fmd_commit/install_mode/params_hash provenance stamping."""

import json

import attrs

from fmd.simulator.params import MOTH_BIEKER_V3
from fmd.simulator import provenance as _prov
from fmd.simulator.provenance import (
    _classify_direct_url,
    get_fmd_provenance,
    params_hash,
    provenance_stamp,
)


class _FakeDist:
    """Minimal Distribution stand-in: only ``read_text`` is exercised."""

    def __init__(self, direct_url_text):
        self._direct_url_text = direct_url_text

    def read_text(self, name):
        return self._direct_url_text if name == "direct_url.json" else None


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


def test_egg_info_does_not_shadow_install_record(monkeypatch):
    """A bare src-tree egg-info (no direct_url.json) must not mask the real
    install record. Report scripts insert ``src/`` on ``sys.path``, so an
    egg-info without ``direct_url.json`` sorts ahead of the venv dist-info;
    the resolver must skip it and find the record that carries the commit."""
    egg_info = _FakeDist(None)  # source-tree metadata, no install record
    dist_info = _FakeDist(json.dumps({
        "url": "https://github.com/brooksreed/fomodynamics.git",
        "vcs_info": {"vcs": "git", "commit_id": "abc1234def5678"},
    }))
    monkeypatch.setattr(
        _prov._metadata, "distributions", lambda **kw: iter([egg_info, dist_info])
    )
    result = get_fmd_provenance()
    assert result == {"fmd_commit": "abc1234def5678", "fmd_install_mode": "pin"}


def test_get_fmd_provenance_unknown_when_no_install_record(monkeypatch):
    """No distribution carries a direct_url.json → unknown (not a crash)."""
    monkeypatch.setattr(
        _prov._metadata, "distributions", lambda **kw: iter([_FakeDist(None)])
    )
    assert get_fmd_provenance() == {"fmd_commit": None, "fmd_install_mode": "unknown"}


def test_params_hash_deterministic():
    assert params_hash(MOTH_BIEKER_V3) == params_hash(MOTH_BIEKER_V3)


def test_params_hash_changes_with_field():
    changed = attrs.evolve(MOTH_BIEKER_V3, hull_mass=MOTH_BIEKER_V3.hull_mass + 1.0)
    assert params_hash(changed) != params_hash(MOTH_BIEKER_V3)


def test_provenance_stamp_has_all_three_fields():
    stamp = provenance_stamp(MOTH_BIEKER_V3)
    assert set(stamp.keys()) == {"fmd_commit", "fmd_install_mode", "params_hash"}
