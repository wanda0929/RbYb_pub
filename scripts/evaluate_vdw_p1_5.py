#!/usr/bin/env python3
"""Evaluate one-at-a-time basis convergence of the static vdW candidate.

The addressed branch is Rb 66S + Yb S(nu=62.682292758), M=+1 at B=25 G,
R=3.3 micrometers, and theta=0.  The calculation independently varies the Rb
and Yb radial ranges, maximum orbital angular momentum, and pair-energy window
while retaining the published target and dipole-dipole-only interaction.  It
does not construct or assess a vdW gate.

Output:
  data/vdw_p1_5_basis_convergence.json
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import resource
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import reproduce_vdw_dense as vdw
import scipy
from verify_pairinteraction_databases import verify_database_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vdw_p1_5_basis_convergence.json"
REFERENCE_DATA = ROOT / "data" / "vdw_dense_data.json"
REFERENCE_LABEL = "numerical reference"
REFERENCE_TOLERANCE = 1e-6

# The 0.3 offset keeps the Yb MQDT boundary between neighboring roots.  Rb and
# Yb radial ranges are varied separately so every non-reference row changes one
# numerical basis axis only.
CONFIGURATIONS = (
    (
        "Rb radial contracted: n=66+/-2",
        "rb_radial_range",
        "contracted",
        vdw.BasisConfig(rb_n_half_width=2),
    ),
    (REFERENCE_LABEL, "reference", "reference", vdw.REFERENCE_BASIS),
    (
        "Rb radial expanded: n=66+/-4",
        "rb_radial_range",
        "expanded",
        vdw.BasisConfig(rb_n_half_width=4),
    ),
    (
        "l_max = 4",
        "l_max",
        "expanded",
        vdw.BasisConfig(l_max=4),
    ),
    (
        "Yb radial contracted: nu=target+/-2.3",
        "yb_radial_range",
        "contracted",
        vdw.BasisConfig(yb_nu_half_width=2.3),
    ),
    (
        "Yb radial expanded: nu=target+/-4.3",
        "yb_radial_range",
        "expanded",
        vdw.BasisConfig(yb_nu_half_width=4.3),
    ),
    (
        "l_max = 2",
        "l_max",
        "contracted",
        vdw.BasisConfig(l_max=2),
    ),
    (
        "pair window +/-40 GHz",
        "pair_energy_window",
        "contracted",
        vdw.BasisConfig(pair_window_ghz=40.0),
    ),
    (
        "pair window +/-120 GHz",
        "pair_energy_window",
        "expanded",
        vdw.BasisConfig(pair_window_ghz=120.0),
    ),
)

METRICS = (
    "c6_ghz_um6",
    "working_point_shift_mhz",
    "bare_pair_weight",
)


def _maximum_change(
    rows: list[dict[str, object]], reference: dict[str, object], key: str
) -> dict[str, float | str]:
    row = max(rows, key=lambda item: abs(float(item[key]) - float(reference[key])))
    difference = float(row[key]) - float(reference[key])
    return {
        "row": str(row["label"]),
        "signed_change": difference,
        "absolute_change": abs(difference),
        "relative_change": abs(difference / float(reference[key])),
    }


def add_reference_changes(rows: list[dict[str, object]]) -> None:
    """Annotate every row with signed changes from the published basis."""
    reference = next(row for row in rows if row["label"] == REFERENCE_LABEL)
    for row in rows:
        row["change_from_reference"] = {
            key: float(row[key]) - float(reference[key]) for key in METRICS
        }
        row["change_from_reference"]["pair_basis_size"] = int(row["pair_basis_size"]) - int(
            reference["pair_basis_size"]
        )


def convergence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return maxima over expanded one-axis checks, excluding contractions."""
    reference = next(row for row in rows if row["label"] == REFERENCE_LABEL)
    expanded = [row for row in rows if row["comparison"] == "expanded"]
    largest_basis = max(expanded, key=lambda row: int(row["pair_basis_size"]))
    return {
        "reference_label": REFERENCE_LABEL,
        "expanded_rows": [str(row["label"]) for row in expanded],
        "maximum_changes_over_expanded_rows": {
            key: _maximum_change(expanded, reference, key) for key in METRICS
        },
        "maximum_expanded_pair_basis": {
            "row": largest_basis["label"],
            "pair_basis_size": largest_basis["pair_basis_size"],
        },
    }


def evaluate_configuration(
    label: str,
    axis: str,
    comparison: str,
    config: vdw.BasisConfig,
    sigma_shift_mhz: float,
) -> dict[str, object]:
    print(f"evaluating P1-5 basis row: {label}", flush=True)
    started = time.perf_counter()
    c6 = vdw.zero_field_c6_channel(1.0, [(0.5, 0.5)], config)
    sector = vdw.make_sector(vdw.B_FIXED_G, 0.5, 0.5, config)
    point = vdw.solve_point(sector, 3.3, sigma_shift_mhz, target_component_only=True)
    row = {
        "label": label,
        "axis": axis,
        "comparison": comparison,
        "basis": {
            **asdict(config),
            "rb_n_bounds": [
                vdw.N_RB - config.rb_n_half_width,
                vdw.N_RB + config.rb_n_half_width,
            ],
            "yb_nu_bounds": [
                vdw.NU_YB - config.yb_nu_half_width,
                vdw.NU_YB + config.yb_nu_half_width,
            ],
        },
        "zero_field_pair_basis_size": c6["n_states"],
        "pair_basis_size": point["n_states"],
        "finite_field_target_component_size": point["solver_n_states"],
        "finite_field_number_of_components": point["number_of_components"],
        "target_norm_in_solver_component": point["target_norm_in_solver_component"],
        "c6_ghz_um6": float(c6["c6_matrix_ghz_um6"][0][0]),
        "nearest_dipole_coupled_detuning_mhz": c6["nearest_dipole_coupled_detuning_mhz"],
        "c3_squared_mhz2_um6": c6["c3_squared_mhz2_um6"],
        "perturbative_delta_eff_mhz": c6["perturbative_delta_eff_mhz"],
        "working_point_shift_mhz": point["u_mhz"],
        "bare_pair_weight": point["overlap"],
        "row_wall_seconds": time.perf_counter() - started,
        "process_peak_rss_kib_after_row": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(
        f"  N={row['pair_basis_size']}, C6={row['c6_ghz_um6']:.12g}, "
        f"U={row['working_point_shift_mhz']:.12g}, "
        f"w={row['bare_pair_weight']:.12g}",
        flush=True,
    )
    del sector, c6, point
    gc.collect()
    return row


def reference_reproduction(
    rows: list[dict[str, object]], reference_data: dict[str, object]
) -> dict[str, object]:
    reference = next(row for row in rows if row["label"] == REFERENCE_LABEL)
    stored_point = next(
        point
        for point in reference_data["distance_track_b25_pp"]
        if np.isclose(point["distance_um"], 3.3)
    )
    stored_c6 = float(
        reference_data["configuration"]["zero_field_c6"]["channels"]["M=+1"]["c6_matrix_ghz_um6"][
            0
        ][0]
    )
    expected = {
        "c6_ghz_um6": stored_c6,
        "working_point_shift_mhz": float(stored_point["u_mhz"]),
        "bare_pair_weight": float(stored_point["overlap"]),
        "pair_basis_size": int(stored_point["n_states"]),
    }
    differences = {key: float(reference[key]) - expected[key] for key in METRICS}
    passed = (
        max(abs(value) for value in differences.values()) <= REFERENCE_TOLERANCE
        and int(reference["pair_basis_size"]) == expected["pair_basis_size"]
    )
    result = {
        "source": "data/vdw_dense_data.json",
        "absolute_tolerance": REFERENCE_TOLERANCE,
        "expected": expected,
        "differences": differences,
        "pair_basis_size_difference": int(reference["pair_basis_size"])
        - expected["pair_basis_size"],
        "passed": passed,
    }
    if not passed:
        raise AssertionError("reference row does not reproduce vdw_dense_data.json")
    return result


def main() -> None:
    started = time.perf_counter()
    verify_database_manifest()
    vdw.initialize_database()
    reference_data = json.loads(REFERENCE_DATA.read_text())
    stored_working_point = next(
        point
        for point in reference_data["distance_track_b25_pp"]
        if np.isclose(point["distance_um"], 3.3)
    )
    sigma_shift_mhz = float(stored_working_point["u_mhz"])

    rows = []
    for label, axis, comparison, config in CONFIGURATIONS:
        rows.append(evaluate_configuration(label, axis, comparison, config, sigma_shift_mhz))
    add_reference_changes(rows)
    reproduction = reference_reproduction(rows, reference_data)
    summary = convergence_summary(rows)

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": importlib.metadata.version("pairinteraction"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "database_versions": {"Rb": "v1.2", "Yb171_mqdt": "v1.4"},
            "database_hash_manifest": "provenance/pairinteraction_database_manifest.json",
        },
        "status": "static vdW one-at-a-time basis convergence complete",
        "scope": {
            "included": (
                "zero-field M=+1 second-order C6 and the B=25 G, R=3.3 "
                "micrometer target-overlap branch for the specified bare pair"
            ),
            "method": (
                "one numerical basis axis changed per row; Rb and Yb radial "
                "ranges varied independently; all magnetic sublevels retained "
                "and only total pair M=+1 fixed; finite-field eigensolves use "
                "the exact connected Hamiltonian component containing unit "
                "bare-target norm"
            ),
            "excluded": [
                "driven dynamics",
                "vdW gate construction",
                "process fidelity",
                "multipoles beyond electric dipole-dipole",
                "combined multi-axis basis expansions",
            ],
        },
        "fixed_inputs": {
            "rb_state": "66S_1/2, m=+1/2",
            "yb_state": "S(nu=62.682292758), F=1/2, m=+1/2",
            "field_gauss_for_working_point": vdw.B_FIXED_G,
            "distance_um": 3.3,
            "theta_degree": vdw.ANGLE_DEGREE,
            "interaction_order": vdw.INTERACTION_ORDER,
            "interaction": "electric dipole-dipole only",
            "zero_field_c6_method": ("second-order A^dagger V (E0-H0)^-1 V A at R=1 micrometer"),
            "branch_selection": (
                "largest bare-pair weight among 24 shift-invert eigenpairs "
                "centered on the stored reference shift, after exact removal "
                "of disconnected zero-target-overlap Hamiltonian components"
            ),
            "reference_basis": asdict(vdw.REFERENCE_BASIS),
            "radial_range_convention": (
                "reference Rb n=66+/-3 and Yb nu=62.682292758+/-3.3; "
                "contracted/expanded Yb widths use 2.3/4.3 so boundaries stay "
                "between adjacent MQDT roots"
            ),
            "serial_execution": (
                "rows evaluated sequentially in one process; heavy basis and "
                "Hamiltonian objects released before the next row"
            ),
        },
        "one_at_a_time_rows": rows,
        "reference_reproduction": reproduction,
        "convergence": summary,
        "resource_usage": {
            "wall_seconds_before_output_write": time.perf_counter() - started,
            "process_peak_rss_kib_before_output_write": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "logical_cpu_count": os.cpu_count(),
        },
        "caveat": (
            "The expanded checks bound one-axis truncation sensitivity only; "
            "they do not test cross-terms from simultaneous radial, angular, "
            "and pair-window expansion and do not establish a driven gate model."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
