#!/usr/bin/env python3
"""Rebuild one corrected P0-4 Hamiltonian and propagate the published pulse.

Requires the pinned atomic databases. No optimization, checkpoint reuse, or
archive writes. Runtime is hardware dependent; this is not the full audit.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/manuscript_claims_2026_09_10"


def compare(actual, expected):
    """Compare fresh amplitudes and populations, not only a rounded fidelity."""
    import numpy as np

    for key in ("basis_size", "active_modes"):
        if actual[key] != expected[key]:
            raise ValueError(f"Quick-check {key}: {actual[key]} != archived {expected[key]}")
    for key in (
        "nominal_rabi_fixed_z_overlap",
        "nominal_rabi_mean_survival",
        "kraus_diagonal_re_im",
        "amplitude_vertex_overlaps",
    ):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-8, err_msg=key)
    for key, value in expected["population"].items():
        np.testing.assert_allclose(actual["population"][key], value, rtol=0, atol=1e-8, err_msg=key)


def main():
    # This CLI runs in a separate process so archived imports cannot collide
    # with the adapted historical modules used by the test suite.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "2"
    from verify_pairinteraction_databases import verify_database_manifest

    sys.path.insert(0, str(ARCHIVE / "simulations"))
    import evaluate_forster_control_decay as corrected

    started = time.monotonic()
    verify_database_manifest()
    baseline = json.loads((ARCHIVE / "simulations/forster_control_decay_recheck.json").read_text())
    expected = baseline["rows"]["0.000000/0.000000"]
    # evaluate builds the pair model, prepares connected bright modes, propagates
    # nominal and amplitude vertices, and computes a fresh population trajectory.
    actual = corrected.evaluate((0.0, 0.0), tuple(expected["fixed_local_z_rad"]))
    compare(actual, expected)
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "fresh nominal P0-4 Hamiltonian and corrected-decay propagation only",
                "basis_size": actual["basis_size"],
                "active_modes": actual["active_modes"],
                "nominal_overlap": actual["nominal_rabi_fixed_z_overlap"],
                "absolute_tolerance": 1e-8,
                "elapsed_seconds": time.monotonic() - started,
                "archive_written": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
