"""Provenance stamping: fmd commit + install mode + params hash.

Answers "which fmd produced this artifact" (audit A-M1/A-M3, gap 22): the
same downstream command can run against a pinned fmd install or a local
editable override at a different commit, silently changing metric semantics
with nothing recorded to tell the two vintages apart.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata as _metadata
from typing import Any


def _classify_direct_url(direct_url: dict[str, Any]) -> tuple[str | None, str]:
    """Pure classifier: parsed ``direct_url.json`` -> ``(commit, install_mode)``.

    Split out from :func:`get_fmd_provenance` so the editable/pin/unknown
    branches are unit-testable against synthetic dicts, without a live
    install in each state.
    """
    dir_info = direct_url.get("dir_info") or {}
    if dir_info.get("editable"):
        return _git_head_commit(direct_url.get("url", "")), "editable"

    vcs_info = direct_url.get("vcs_info") or {}
    commit = vcs_info.get("commit_id")
    if commit:
        return commit, "pin"

    return None, "unknown"


def _git_head_commit(file_url: str) -> str | None:
    """Best-effort HEAD commit for an editable install's source checkout."""
    if not file_url.startswith("file://"):
        return None
    path = file_url[len("file://"):]
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_fmd_provenance(package: str = "fomodynamics") -> dict[str, str | None]:
    """Return ``{"fmd_commit": <sha or None>, "fmd_install_mode": <mode>}``.

    ``fmd_install_mode`` is one of ``"pin"``, ``"editable"``, or ``"unknown"``
    (package not installed via pip/uv metadata, e.g. a dev checkout on
    ``PYTHONPATH``).
    """
    try:
        dist = _metadata.distribution(package)
        direct_url_text = dist.read_text("direct_url.json")
    except _metadata.PackageNotFoundError:
        direct_url_text = None

    if direct_url_text is None:
        return {"fmd_commit": None, "fmd_install_mode": "unknown"}

    commit, mode = _classify_direct_url(json.loads(direct_url_text))
    return {"fmd_commit": commit, "fmd_install_mode": mode}


def params_hash(params: Any) -> str:
    """Short deterministic hash of a params object's field values.

    Works for any ``attrs``-defined params object (e.g. ``MothParams``);
    changing any field value changes the hash.
    """
    import attrs

    def _default(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    payload = json.dumps(attrs.asdict(params), sort_keys=True, default=_default)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def provenance_stamp(params: Any) -> dict[str, str | None]:
    """Combine fmd commit + install mode + params hash into one stamp dict."""
    stamp = get_fmd_provenance()
    stamp["params_hash"] = params_hash(params)
    return stamp
