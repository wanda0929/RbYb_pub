#!/usr/bin/env python3
"""Reproduce Appendix B's reduced two-level vdW feasibility estimate.

The calculation maps the static branch shift and overlap to a two-state pair
surrogate and reports screening metrics. It is not a driven multichannel gate
simulation and its reported reduced-model fidelity is not a gate result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "vdw_dense_data.json"
DEFAULT_OUTPUT = ROOT / "data" / "vdw_feasibility_results.json"
DEFAULT_DEVIATION_OUTPUT = ROOT / "data" / "vdw_r6_deviation.csv"

EFFECTIVE_RABI_MHZ = 2.0
RB_LIFETIME_US = 323.60
YB_LIFETIME_US = 87.56
WORKING_DISTANCE_UM = 3.3
COMPUTATIONAL_INDICES = np.array([0, 1, 3, 5])
IDEAL_CZ_DIAGONAL = np.array([1, 1, 1, -1], dtype=complex)


def pair_parameters(u_mhz: float, overlap: float) -> tuple[float, float]:
    """Map static shift and bare overlap to Delta_eff and V in cyclic MHz."""
    delta_eff = u_mhz * (2 * overlap - 1) / (1 - overlap)
    coupling = np.sqrt(u_mhz * (u_mhz + delta_eff))
    return float(delta_eff), float(coupling)


def hamiltonian(
    u_mhz: float,
    overlap: float,
    rb_rabi_mhz: float = 0.0,
    yb_rabi_mhz: float = 0.0,
    include_decay: bool = True,
) -> np.ndarray:
    """Ten-state square-pulse surrogate Hamiltonian H/h in cyclic MHz."""
    delta_eff, coupling = pair_parameters(u_mhz, overlap)
    matrix = np.zeros((10, 10), dtype=complex)
    for first, second, rabi in (
        (1, 2, yb_rabi_mhz),
        (3, 4, rb_rabi_mhz),
        (5, 6, yb_rabi_mhz),
        (5, 7, rb_rabi_mhz),
        (6, 8, rb_rabi_mhz),
        (7, 8, yb_rabi_mhz),
    ):
        matrix[first, second] = matrix[second, first] = rabi / 2
    matrix[8, 9] = matrix[9, 8] = coupling
    matrix[9, 9] = -delta_eff

    if include_decay:
        rates = {
            2: 1 / YB_LIFETIME_US,
            4: 1 / RB_LIFETIME_US,
            6: 1 / YB_LIFETIME_US,
            7: 1 / RB_LIFETIME_US,
            8: 1 / RB_LIFETIME_US + 1 / YB_LIFETIME_US,
            9: 1 / RB_LIFETIME_US + 1 / YB_LIFETIME_US,
        }
        for index, rate in rates.items():
            matrix[index, index] -= 0.5j * rate / (2 * np.pi)
    return matrix


def returning_amplitudes(
    u_mhz: float,
    overlap: float,
    effective_rabi_mhz: float = EFFECTIVE_RABI_MHZ,
    include_decay: bool = True,
) -> np.ndarray:
    """Return the surviving computational amplitudes of the screening sequence."""
    rb_pi = expm(
        -2j
        * np.pi
        * hamiltonian(
            u_mhz,
            overlap,
            rb_rabi_mhz=effective_rabi_mhz,
            include_decay=include_decay,
        )
        * (0.5 / effective_rabi_mhz)
    )
    yb_2pi = expm(
        -2j
        * np.pi
        * hamiltonian(
            u_mhz,
            overlap,
            yb_rabi_mhz=effective_rabi_mhz,
            include_decay=include_decay,
        )
        * (1.0 / effective_rabi_mhz)
    )
    propagator = rb_pi @ yb_2pi @ rb_pi
    return np.diag(propagator)[COMPUTATIONAL_INDICES]


def local_z_metrics(amplitudes: np.ndarray) -> dict[str, object]:
    """Calculate the local-Z-corrected reduced-model screening metrics."""

    def negative_overlap(alpha: float) -> float:
        first = amplitudes[0] + np.exp(1j * alpha) * amplitudes[2]
        second = amplitudes[1] - np.exp(1j * alpha) * amplitudes[3]
        return -(abs(first) + abs(second))

    alpha_grid = np.linspace(-np.pi, np.pi, 2049)
    alpha0 = float(alpha_grid[np.argmin([negative_overlap(a) for a in alpha_grid])])
    step = float(alpha_grid[1] - alpha_grid[0])
    alpha = float(
        minimize_scalar(
            negative_overlap,
            bounds=(alpha0 - 2 * step, alpha0 + 2 * step),
            method="bounded",
            options={"xatol": 1e-13},
        ).x
    )
    first = amplitudes[0] + np.exp(1j * alpha) * amplitudes[2]
    second = amplitudes[1] - np.exp(1j * alpha) * amplitudes[3]
    beta = float(np.angle(first) - np.angle(second))
    correction = np.array([1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))])
    overlap = complex(np.vdot(IDEAL_CZ_DIAGONAL, correction * amplitudes))
    mean_survival = float(np.vdot(amplitudes, amplitudes).real / 4)
    conditional_phase = float(
        np.angle(amplitudes[3] * amplitudes[0] / (amplitudes[2] * amplitudes[1]))
    )
    return {
        "reduced_model_average_fidelity": float((4 * mean_survival + abs(overlap) ** 2) / 20),
        "mean_computational_survival": mean_survival,
        "conditional_phase_rad": conditional_phase,
        "conditional_phase_error_rad": float(np.angle(np.exp(1j * (conditional_phase - np.pi)))),
        "optimal_local_z_alpha_rad": alpha,
        "optimal_local_z_beta_rad": beta,
        "returning_amplitudes_re_im": [
            [float(value.real), float(value.imag)] for value in amplitudes
        ],
    }


def evaluate(distance_um: float, u_mhz: float, overlap: float) -> dict[str, object]:
    delta_eff, coupling = pair_parameters(u_mhz, overlap)
    decaying = returning_amplitudes(u_mhz, overlap, include_decay=True)
    lossless = returning_amplitudes(u_mhz, overlap, include_decay=False)
    metrics = local_z_metrics(decaying)
    lossless_metrics = local_z_metrics(lossless)
    return {
        "distance_um": float(distance_um),
        "u_mhz": float(u_mhz),
        "overlap": float(overlap),
        "blockade_ratio": float(u_mhz / EFFECTIVE_RABI_MHZ),
        "delta_eff_mhz": delta_eff,
        "pair_coupling_mhz": coupling,
        **metrics,
        "minimum_basis_return": float(np.min(np.abs(decaying) ** 2)),
        "lossless_reduced_model_average_fidelity": lossless_metrics[
            "reduced_model_average_fidelity"
        ],
        "lossless_minimum_basis_return": float(np.min(np.abs(lossless) ** 2)),
    }


def generate_data(input_path: Path, output_path: Path, deviation_path: Path) -> None:
    dense_data = json.loads(input_path.read_text())
    c6_ghz_um6 = float(dense_data["configuration"]["zero_field_c6_ghz_um6"])
    pair_track = np.array(
        [
            [row["distance_um"], row["u_mhz"], row["overlap"]]
            for row in dense_data["distance_track_b25_pp"]
        ],
        dtype=float,
    )

    exact = []
    for source, (distance_um, u_mhz, overlap) in zip(
        dense_data["distance_track_b25_pp"], pair_track, strict=True
    ):
        row = evaluate(distance_um, u_mhz, overlap)
        r6_guide_mhz = c6_ghz_um6 * 1000 / distance_um**6
        deviation_mhz = u_mhz - r6_guide_mhz
        row.update(
            {
                "n_states": source["n_states"],
                "u_r6_guide_mhz": float(r6_guide_mhz),
                "dense_minus_r6_mhz": float(deviation_mhz),
                "relative_r6_deviation_percent": float(100 * deviation_mhz / r6_guide_mhz),
                "effective_c6_ghz_um6": float(u_mhz * distance_um**6 / 1000),
            }
        )
        exact.append(row)

    u_interpolator = PchipInterpolator(pair_track[:, 0], pair_track[:, 1])
    overlap_interpolator = PchipInterpolator(pair_track[:, 0], pair_track[:, 2])
    distance_sweep = []
    for distance in np.linspace(3.0, 5.0, 401):
        row = evaluate(
            float(distance),
            float(u_interpolator(distance)),
            float(overlap_interpolator(distance)),
        )
        distance_sweep.append(
            {
                key: row[key]
                for key in (
                    "distance_um",
                    "u_mhz",
                    "overlap",
                    "blockade_ratio",
                    "reduced_model_average_fidelity",
                    "mean_computational_survival",
                    "minimum_basis_return",
                    "lossless_reduced_model_average_fidelity",
                    "conditional_phase_error_rad",
                )
            }
        )

    sectors = []
    for source in dense_data["sector_track_b25"]:
        row = evaluate(WORKING_DISTANCE_UM, source["u_mhz"], source["overlap"])
        row.update(
            {
                "m_rb": source["m_rb"],
                "m_yb": source["m_yb"],
                "n_states": source["n_states"],
            }
        )
        sectors.append(row)

    field_scan = []
    for source in dense_data["magnetic_field_track_r3p3_pp"]:
        row = evaluate(WORKING_DISTANCE_UM, source["u_mhz"], source["overlap"])
        row.update({"b_gauss": source["b_gauss"], "n_states": source["n_states"]})
        field_scan.append(row)

    selected = next(row for row in sectors if row["m_rb"] == 0.5 and row["m_yb"] == 0.5)
    if abs(selected["u_mhz"] - 57.072053) > 1e-5:
        raise RuntimeError("selected static branch shift does not match Appendix B")
    if abs(selected["reduced_model_average_fidelity"] - 0.997699) > 1e-6:
        raise RuntimeError("selected reduced-model estimate does not match Appendix B")

    threshold_mhz = 0.3
    threshold_start_um = next(
        row["distance_um"]
        for index, row in enumerate(exact)
        if all(abs(later["dense_minus_r6_mhz"]) < threshold_mhz for later in exact[index:])
    )
    working_deviation = next(
        row for row in exact if np.isclose(row["distance_um"], WORKING_DISTANCE_UM)
    )
    deviation_summary = {
        "definition": "U_dense - C6/R^6 using the zero-field C6 guide",
        "threshold_mhz": threshold_mhz,
        "first_sampled_distance_with_all_later_samples_below_threshold_um": threshold_start_um,
        "maximum_absolute_deviation_mhz": float(
            max(abs(row["dense_minus_r6_mhz"]) for row in exact)
        ),
        "maximum_deviation_distance_um": float(
            max(exact, key=lambda row: abs(row["dense_minus_r6_mhz"]))["distance_um"]
        ),
        "working_point_deviation_mhz": working_deviation["dense_minus_r6_mhz"],
        "working_point_relative_deviation_percent": working_deviation[
            "relative_r6_deviation_percent"
        ],
    }

    deviation_path.parent.mkdir(parents=True, exist_ok=True)
    with deviation_path.open("w", newline="") as stream:
        columns = [
            "distance_um",
            "u_mhz",
            "u_r6_guide_mhz",
            "dense_minus_r6_mhz",
            "relative_r6_deviation_percent",
            "effective_c6_ghz_um6",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in exact)

    field_u = np.array([row["u_mhz"] for row in field_scan])
    field_overlap = np.array([row["overlap"] for row in field_scan])
    field_estimate = np.array([row["reduced_model_average_fidelity"] for row in field_scan])
    result = {
        "scope": (
            "Appendix B reduced two-level feasibility estimate only. This is not a "
            "driven multichannel gate simulation, gate design, or process-fidelity result."
        ),
        "model": {
            "pair_model": "two-level branch-composition surrogate",
            "screening_sequence": "square Rb pi, Yb 2pi, Rb pi",
            "effective_rabi_mhz": EFFECTIVE_RABI_MHZ,
            "sequence_duration_us": 2 / EFFECTIVE_RABI_MHZ,
            "rb_lifetime_0k_us": RB_LIFETIME_US,
            "yb_lifetime_0k_us": YB_LIFETIME_US,
            "screening_metric": (
                "local-Z-corrected loss-aware Haar-average formula applied only to the "
                "surviving computational block of the reduced surrogate"
            ),
            "distance_interpolation": "PCHIP through calculated U(R), w(R); guide only",
        },
        "dense_input": {
            "path": str(input_path.relative_to(ROOT)),
            "provenance": dense_data["provenance"],
            "configuration": dense_data["configuration"],
        },
        "c6_ghz_um6": c6_ghz_um6,
        "r6_deviation_summary": deviation_summary,
        "working_point": selected,
        "pair_track": exact,
        "four_magnetic_sectors": sectors,
        "magnetic_field_scan": field_scan,
        "magnetic_field_summary": {
            "range_gauss": [
                min(row["b_gauss"] for row in field_scan),
                max(row["b_gauss"] for row in field_scan),
            ],
            "u_span_mhz": float(np.ptp(field_u)),
            "u_relative_span_percent": float(100 * np.ptp(field_u) / np.mean(field_u)),
            "overlap_range": [float(np.min(field_overlap)), float(np.max(field_overlap))],
            "reduced_model_average_fidelity_range": [
                float(np.min(field_estimate)),
                float(np.max(field_estimate)),
            ],
        },
        "distance_sweep": distance_sweep,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")
    print(f"wrote {deviation_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deviation-output", type=Path, default=DEFAULT_DEVIATION_OUTPUT)
    args = parser.parse_args()
    generate_data(args.input, args.output, args.deviation_output)


if __name__ == "__main__":
    main()
