"""I/O utilities for validation sweep results.

Save/load sweep results as JSON metadata + NPZ trajectories for
reproducible report and plot generation.
"""

# NOTE — moth-coupling seam (Phase 1 OSS split, code review M1 + M3):
# This module is moth-coupled despite living in generic `fmd/simulator/`. It
# imports `SweepResult` (dataclass) from `moth_validation.py`, and the cat1-cat5
# categories in `trajectories_to_npz` are moth-specific. Future cleanup: rename
# to `moth_validation_io.py`, or refactor `SweepResult` into a generic shape
# with moth specifics moved to `MothSweepResult`.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from fmd.simulator.moth_validation import SweepResult


def _numpy_to_list(obj):
    """Convert numpy arrays/scalars to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sweep_to_json(sweep: SweepResult, test_matrix: dict) -> dict:
    """Serialize SweepResult metadata + test_matrix to JSON-compatible dict.

    Converts numpy arrays to lists and handles None values. Does not
    include trajectory data (use trajectories_to_npz for that).

    Args:
        sweep: SweepResult with sweep metadata and results.
        test_matrix: Test matrix dict (should already be JSON-serializable,
            but numpy values will be converted).

    Returns:
        JSON-serializable dict with version, timestamp, parameters,
        baseline_trim, test_matrix, and summary.
    """
    now = datetime.now(timezone.utc)

    # Extract summary from test_matrix if present, else compute from results
    if "summary" in test_matrix:
        summary = test_matrix["summary"]
    else:
        summary = {"total": 0, "pass": 0, "fail": 0}

    # Build baseline_trim section
    baseline_trim = {
        "state": _numpy_to_list(sweep.baseline_trim_state)
        if sweep.baseline_trim_state is not None
        else None,
        "control": _numpy_to_list(sweep.baseline_trim_control)
        if sweep.baseline_trim_control is not None
        else None,
        "residual": None,
    }

    # Build parameters section
    parameters = {
        "default_duration": sweep.duration,
        "dt": sweep.dt,
        "baseline_speed": sweep.u_forward,
        "divergence_threshold_deg": test_matrix.get("divergence_threshold_deg", 30.0),
        "params_preset": test_matrix.get("params_preset", "MOTH_BIEKER_V3"),
    }

    doc = {
        "version": test_matrix.get("version", "4c_v1"),
        "timestamp": now.isoformat(),
        "parameters": parameters,
        "baseline_trim": baseline_trim,
        "test_matrix": _deep_convert(test_matrix),
        "summary": _deep_convert(summary),
    }
    return doc


def _deep_convert(obj):
    """Recursively convert numpy types in a nested structure to Python types."""
    if isinstance(obj, dict):
        return {k: _deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_convert(v) for v in obj]
    return _numpy_to_list(obj)


def trajectories_to_npz(sweep: SweepResult) -> dict:
    """Extract all trajectories from SweepResult as {key: np.array} dict.

    Iterates through each result list, generating keys with index-based
    test IDs and category prefixes:
      - cat1: perturbation_results (SimulationResult in result.result)
      - cat2: configuration_results (SimulationResult in result.simulation_result)
      - cat3: transient_results (SimulationResult in result.result)
      - cat4: damping_comparison_results (SimulationResult in
              result.perturbation_result.result)
      - cat5: speed_variation_results (SimulationResult in result.result)

    Only saves trajectories where the SimulationResult is not None.

    Args:
        sweep: SweepResult with result lists.

    Returns:
        Dict mapping string keys like "cat1_0_times" to numpy arrays.
    """
    arrays = {}

    # Category mappings: (prefix, result_list, accessor_for_sim_result)
    categories = [
        ("cat1", sweep.perturbation_results, lambda r: r.result),
        ("cat2", sweep.configuration_results, lambda r: r.simulation_result),
        ("cat3", sweep.transient_results, lambda r: r.result),
        (
            "cat4",
            sweep.damping_comparison_results,
            lambda r: r.perturbation_result.result
            if r.perturbation_result is not None
            else None,
        ),
        ("cat5", sweep.speed_variation_results, lambda r: r.result),
    ]

    for prefix, result_list, get_sim in categories:
        for idx, res in enumerate(result_list):
            sim_result = get_sim(res)
            if sim_result is None:
                continue
            key_base = f"{prefix}_{idx}"
            arrays[f"{key_base}_times"] = np.asarray(sim_result.times)
            arrays[f"{key_base}_states"] = np.asarray(sim_result.states)
            arrays[f"{key_base}_controls"] = np.asarray(sim_result.controls)

    return arrays


def save_sweep_results(
    sweep: SweepResult, test_matrix: dict, output_dir: str
) -> str:
    """Save sweep results as JSON metadata + compressed NPZ trajectories.

    Writes directly to output_dir:
      - metadata.json: version, parameters, baseline trim, test matrix, summary
      - trajectories.npz: all simulation trajectories keyed by category and index

    Args:
        sweep: SweepResult to save.
        test_matrix: Test matrix dict with version, categories, summary, etc.
        output_dir: Directory to write results into.

    Returns:
        Path to the output directory (str).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = sweep_to_json(sweep, test_matrix)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save trajectories
    traj_arrays = trajectories_to_npz(sweep)
    traj_path = os.path.join(output_dir, "trajectories.npz")
    if traj_arrays:
        np.savez_compressed(traj_path, **traj_arrays)
    else:
        # Save empty npz so load always finds the file
        np.savez_compressed(traj_path)

    return output_dir


def load_sweep_results(run_dir: str) -> Tuple[dict, dict]:
    """Load sweep results from a run directory.

    Args:
        run_dir: Path to the run directory containing metadata.json
            and trajectories.npz.

    Returns:
        Tuple of (metadata_dict, trajectories_dict) where:
          - metadata_dict: parsed JSON metadata
          - trajectories_dict: {str: np.ndarray} from the npz file

    Raises:
        FileNotFoundError: If metadata.json or trajectories.npz is missing.
    """
    metadata_path = os.path.join(run_dir, "metadata.json")
    traj_path = os.path.join(run_dir, "trajectories.npz")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    npz = np.load(traj_path)
    trajectories = dict(npz)
    npz.close()

    return metadata, trajectories
