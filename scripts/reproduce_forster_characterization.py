#!/usr/bin/env python3
"""Reproduce the numerical characterization behind Figure 1.

No figure is generated. The output contains the state identities and transition
directions, bright-state composition, fixed-m distance/field/angle scans, and
the all-magnetic-sublevel field scan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pairinteraction as pi
from forster_model import SCAN_BASIS, build_pair_model, forster_kets
from verify_pairinteraction_databases import verify_database_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "forster_characterization.json"
PRIMARY_MANIFEST = ROOT / "provenance" / "pairinteraction_database_manifest.json"
FIXED_M_MANIFEST = ROOT / "provenance" / "pairinteraction_database_manifest_figure1_fixed_m.json"

M = 0.5
DISTANCE_GRID_UM = (3.0, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.5, 5.0)
FIXED_M_FIELD_GRID_G = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
ANGLE_GRID_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
ALL_M_COARSE_FIELD_GRID_G = tuple(np.arange(0.0, 6.0 + 0.25, 0.5))
ALL_M_FINE_FIELD_GRID_G = tuple(np.arange(3.7, 4.3 + 0.0125, 0.025))
ALL_M_EXTENSION_FIELD_GRID_G = tuple(np.arange(6.5, 10.0 + 0.25, 0.5))


def dense(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def transition_data() -> dict[str, object]:
    rb_pp, rb_ss, yb_pp, yb_ss = forster_kets()
    rb_absorption = float(rb_pp.get_energy("GHz") - rb_ss.get_energy("GHz"))
    yb_release = float(yb_ss.get_energy("GHz") - yb_pp.get_energy("GHz"))
    defect = 1000 * (yb_release - rb_absorption)
    return {
        "initial_pair": "SS",
        "final_pair": "PP",
        "states": {
            "rb_ss": {"selector": "Rb n=56 l=0 j=1/2 m=1/2", "label": str(rb_ss)},
            "rb_pp": {"selector": "Rb n=56 l=1 j=1/2 m=1/2", "label": str(rb_pp)},
            "yb_ss": {
                "selector": "Yb171_mqdt n=53 l=0 s=1 f=1/2 m=1/2",
                "label": str(yb_ss),
                "effective_principal_quantum_number": float(yb_ss.nu),
            },
            "yb_pp": {
                "selector": "Yb171_mqdt n=52 l=1 s=0 f=1/2 m=1/2",
                "label": str(yb_pp),
                "effective_principal_quantum_number": float(yb_pp.nu),
            },
        },
        "transitions": {
            "rb": {
                "direction": "56S_1/2 -> 56P_1/2",
                "process": "absorbs",
                "frequency_ghz": rb_absorption,
            },
            "yb": {
                "direction": "S(nu=48.369927) -> P(nu=48.014048)",
                "process": "releases",
                "frequency_ghz": yb_release,
            },
        },
        "signed_defect_definition": "E(SS)-E(PP)",
        "signed_defect_mhz": defect,
        "absolute_mismatch_mhz": abs(defect),
    }


def fixed_m_basis(field_gauss: float = 0.0, archived_database: bool = False):
    rb_pp, _, yb_pp, _ = forster_kets()
    if archived_database:
        # In Yb171_mqdt v1.2 this physical P series is selected by its
        # triplet-character mixing value rather than the v1.4 singlet label.
        yb_pp = pi.KetAtom("Yb171_mqdt", n=52, l=1, s=0.8, f=0.5, m=0.5)
    pp_energy_mhz = float(rb_pp.get_energy("MHz") + yb_pp.get_energy("MHz"))
    rb_system = pi.SystemAtom(pi.BasisAtom("Rb", n=(53, 59), l=(0, 3), m=(M, M)))
    yb_system = pi.SystemAtom(pi.BasisAtom("Yb171_mqdt", n=(49, 55), l=(0, 3), m=(M, M)))
    if field_gauss:
        rb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
        yb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        m=(2 * M, 2 * M),
        energy=(pp_energy_mhz / 1000 - 80, pp_energy_mhz / 1000 + 80),
        energy_unit="GHz",
    )
    return pair_basis, (rb_pp, yb_pp), pp_energy_mhz


def fixed_m_point(
    distance_um: float,
    angle_deg: float = 0.0,
    field_gauss: float = 0.0,
    archived_database: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    pair_basis, pp_pair, energy_reference = fixed_m_basis(field_gauss, archived_database)
    hamiltonian = dense(
        pi.SystemPair(pair_basis)
        .set_distance(distance_um, angle_degree=angle_deg, unit="micrometer")
        .set_interaction_order(3)
        .get_hamiltonian(unit="MHz")
    ).astype(complex, copy=False)
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=1e-8, atol=1e-10):
        raise RuntimeError("fixed-m pair Hamiltonian is not Hermitian")
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    pp_vector = dense(pair_basis.get_overlaps(pp_pair)).astype(complex).reshape(-1)
    pp_overlap = eigenvectors.conj().T @ pp_vector
    pp_weights = np.abs(pp_overlap) ** 2

    first_bright = int(np.argmax(pp_weights))
    first_bright_composition = np.abs(eigenvectors[:, first_bright]) ** 2
    pp_basis_index = int(np.argmax(np.abs(pp_vector) ** 2))
    ss_basis_index = int(
        next(index for index in np.argsort(-first_bright_composition) if index != pp_basis_index)
    )
    ss_vector = np.zeros_like(pp_vector)
    ss_vector[ss_basis_index] = 1.0
    ss_overlap = eigenvectors.conj().T @ ss_vector
    ss_weights = np.abs(ss_overlap) ** 2
    second_bright = int(next(index for index in np.argsort(-ss_weights) if index != first_bright))
    splitting = float(abs(energies[first_bright] - energies[second_bright]))
    times_us = np.arange(0, max(3 / (2 * splitting), 0.25), 5e-5)
    phase = np.exp(-2j * np.pi * np.outer(energies - energy_reference, times_us))
    pp_population = np.abs((np.abs(pp_overlap) ** 2) @ phase) ** 2
    ss_population = np.abs((np.conj(ss_overlap) * pp_overlap) @ phase) ** 2
    global_maximum = int(np.argmax(ss_population))
    half_global_maximum = 0.5 * ss_population[global_maximum]
    maximum = next(
        (
            index
            for index in range(1, len(times_us) - 1)
            if ss_population[index] > half_global_maximum
            and ss_population[index] >= ss_population[index - 1]
            and ss_population[index] >= ss_population[index + 1]
        ),
        global_maximum,
    )
    metrics = {
        "distance_um": float(distance_um),
        "theta_deg": float(angle_deg),
        "field_gauss": float(field_gauss),
        "splitting_mhz": splitting,
        "first_maximum_transfer": float(ss_population[maximum]),
        "coherent_spectator_leakage": float(1 - pp_population[maximum] - ss_population[maximum]),
        "first_maximum_time_ns": float(1000 * times_us[maximum]),
        "basis_size": int(pair_basis.number_of_states),
    }
    bright_states = [
        {
            "energy_relative_to_pp_mhz": float(energies[index] - energy_reference),
            "pp_weight": float(pp_weights[index]),
            "ss_weight": float(ss_weights[index]),
            "spectator_weight": float(max(0, 1 - pp_weights[index] - ss_weights[index])),
        }
        for index in (first_bright, second_bright)
    ]
    return metrics, bright_states


def all_m_point(field_gauss: float) -> dict[str, float | int]:
    model = build_pair_model(field_gauss, SCAN_BASIS, 3.4, 0.0)
    return {
        "field_gauss": float(field_gauss),
        "maximum_transfer": model.static_transfer["maximum_ss_population"],
        "first_maximum_time_ns": 1000 * model.static_transfer["time_us"],
        "pp_population_at_maximum": model.static_transfer["pp_population_at_maximum"],
        "spectator_population_at_maximum": model.static_transfer["spectator_population_at_maximum"],
        "pair_basis_size": model.pair_basis_size,
        "symmetry_component_size": model.symmetry_component_size,
    }


def fixed_m_scan_data(quick: bool = False) -> dict[str, object]:
    if quick:
        _, bright_states_at_3p0 = fixed_m_point(3.0, archived_database=True)
        metrics, bright_states_at_3p4 = fixed_m_point(3.4, archived_database=True)
        return {
            "bright_eigenstates_at_3p0_um": bright_states_at_3p0,
            "bright_eigenstates_at_3p4_um": bright_states_at_3p4,
            "fixed_m_distance_scan": [metrics],
        }

    print("fixed-m distance scan", flush=True)
    distance_rows = []
    bright_states_at_3p0: list[dict[str, float]] = []
    bright_states_at_3p4: list[dict[str, float]] = []
    for distance in DISTANCE_GRID_UM:
        print(f"  R={distance:.3f} um", flush=True)
        row, bright_states = fixed_m_point(distance, archived_database=True)
        distance_rows.append(row)
        if distance == 3.0:
            bright_states_at_3p0 = bright_states
        if distance == 3.4:
            bright_states_at_3p4 = bright_states

    print("fixed-m field scan", flush=True)
    field_rows = []
    for field in FIXED_M_FIELD_GRID_G:
        print(f"  B={field:.3f} G", flush=True)
        field_rows.append(fixed_m_point(3.4, field_gauss=field, archived_database=True)[0])

    print("fixed-m angle scan", flush=True)
    angle_rows = []
    for angle in ANGLE_GRID_DEG:
        print(f"  theta={angle:.3f} deg", flush=True)
        angle_rows.append(fixed_m_point(3.4, angle_deg=angle, archived_database=True)[0])
    return {
        "bright_eigenstates_at_3p0_um": bright_states_at_3p0,
        "bright_eigenstates_at_3p4_um": bright_states_at_3p4,
        "fixed_m_distance_scan": distance_rows,
        "fixed_m_field_scan": field_rows,
        "fixed_m_angle_scan": angle_rows,
    }


def fixed_m_worker(database_dir: Path, output: Path, quick: bool) -> None:
    """Run the archived Figure 1 calculation in an isolated v1.2 process."""
    verify_database_manifest(FIXED_M_MANIFEST, database_dir)
    pi.Database.initialize_global_database(
        download_missing=False,
        database_dir=database_dir,
    )
    payload = {
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "database_manifest": str(FIXED_M_MANIFEST.relative_to(ROOT)),
        **fixed_m_scan_data(quick),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def run_fixed_m_worker(database_dir: Path, quick: bool) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rbyb-fixed-m-") as temporary_dir:
        output = Path(temporary_dir) / "fixed_m.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--fixed-m-worker",
            "--fixed-m-database-dir",
            str(database_dir),
            "--output",
            str(output),
        ]
        if quick:
            command.append("--quick-check")
        subprocess.run(command, check=True)
        return json.loads(output.read_text())


def quick_check(fixed_m_database_dir: Path) -> None:
    verify_database_manifest(PRIMARY_MANIFEST)
    archived = run_fixed_m_worker(fixed_m_database_dir, quick=True)
    transition = transition_data()
    if transition["transitions"]["rb"]["process"] != "absorbs":
        raise RuntimeError("Rb transition direction is not encoded as absorption")
    if transition["transitions"]["yb"]["process"] != "releases":
        raise RuntimeError("Yb transition direction is not encoded as release")
    archived_metrics = archived["fixed_m_distance_scan"][0]
    if abs(archived_metrics["first_maximum_transfer"] - 0.996892700) > 5e-5:
        raise RuntimeError(f"archived fixed-m operating point drifted: {archived_metrics}")
    current_metrics, _ = fixed_m_point(3.4)
    if abs(current_metrics["first_maximum_transfer"] - 0.97286139) > 5e-5:
        raise RuntimeError(f"current-v1.4 fixed-m operating point drifted: {current_metrics}")
    print(
        "quick check passed: "
        f"defect={transition['signed_defect_mhz']:.6f} MHz, "
        f"archived-v1.2 fixed-m transfer="
        f"{100 * archived_metrics['first_maximum_transfer']:.6f}%, "
        f"current-v1.4 fixed-m transfer="
        f"{100 * current_metrics['first_maximum_transfer']:.6f}%"
    )


def generate_data(output: Path, fixed_m_database_dir: Path) -> None:
    verify_database_manifest(PRIMARY_MANIFEST)
    archived_fixed_m = run_fixed_m_worker(fixed_m_database_dir, quick=False)
    current_fixed_m, _ = fixed_m_point(3.4)

    all_m_by_field: dict[float, dict[str, float | int]] = {}
    for label, fields in (
        ("coarse", ALL_M_COARSE_FIELD_GRID_G),
        ("fine", ALL_M_FINE_FIELD_GRID_G),
        ("extension", ALL_M_EXTENSION_FIELD_GRID_G),
    ):
        print(f"all-m {label} field scan", flush=True)
        for field in fields:
            key = round(float(field), 12)
            if key not in all_m_by_field:
                print(f"  B={field:.3f} G", flush=True)
                all_m_by_field[key] = all_m_point(float(field))

    result = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        **transition_data(),
        "database_provenance": {
            "transition_states_and_all_m": str(PRIMARY_MANIFEST.relative_to(ROOT)),
            "figure1_fixed_m_archived": archived_fixed_m["database_manifest"],
            "note": (
                "The archived fixed-m Figure 1 scans are exactly reproduced with "
                "Yb171_mqdt v1.2. Transition identities, all-m calculations, and all "
                "gate calculations use v1.4. The current-v1.4 fixed-m operating-point "
                "recalculation is retained below to make the database drift explicit. "
                "The weights printed in Figure 1(b) came from the R=3.0 um member of "
                "the archived scan although the caption assigns them to R=3.4 um; both "
                "recalculated compositions and the displayed values are retained below."
            ),
        },
        "figure1b_manuscript_reported_weights": {
            "caption_distance_um": 3.4,
            "reconstructed_source_distance_um": 3.0,
            "values": [
                {"pp_weight": 0.544, "ss_weight": 0.452, "spectator_weight": 0.004},
                {"pp_weight": 0.454, "ss_weight": 0.543, "spectator_weight": 0.003},
            ],
            "note": (
                "These are the values printed in the manuscript and hard-coded in its "
                "plotting script. The original raw eigensystem was not archived. An "
                "independent reconstruction with the versioned databases reproduces "
                "the PP and SS weights at R=3.0 um, not at the captioned R=3.4 um."
            ),
        },
        "bright_eigenstates_at_3p0_um": archived_fixed_m["bright_eigenstates_at_3p0_um"],
        "bright_eigenstates_at_3p4_um": archived_fixed_m["bright_eigenstates_at_3p4_um"],
        "fixed_m_distance_scan": archived_fixed_m["fixed_m_distance_scan"],
        "fixed_m_field_scan": archived_fixed_m["fixed_m_field_scan"],
        "fixed_m_angle_scan": archived_fixed_m["fixed_m_angle_scan"],
        "fixed_m_v1p4_recalculation_at_3p4_um": current_fixed_m,
        "all_m_field_scan": {
            label: [all_m_by_field[round(float(field), 12)] for field in fields]
            for label, fields in (
                ("coarse", ALL_M_COARSE_FIELD_GRID_G),
                ("fine", ALL_M_FINE_FIELD_GRID_G),
                ("extension", ALL_M_EXTENSION_FIELD_GRID_G),
            )
        },
        "calculation_scope": {
            "fixed_m": "m_Rb=m_Yb=1/2 and total M=1",
            "all_m": (
                "all magnetic sublevels followed by exact connected-component reduction at theta=0"
            ),
            "pairinteraction_order": 3,
            "first_maximum_time_step_ns": 0.05,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick-check", action="store_true")
    parser.add_argument(
        "--fixed-m-database-dir",
        type=Path,
        required=True,
        help=("isolated PairInteraction database root containing Rb v1.2 and Yb171_mqdt v1.2"),
    )
    parser.add_argument("--fixed-m-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixed_m_worker:
        fixed_m_worker(args.fixed_m_database_dir, args.output, args.quick_check)
    elif args.quick_check:
        quick_check(args.fixed_m_database_dir)
    else:
        generate_data(args.output, args.fixed_m_database_dir)


if __name__ == "__main__":
    main()
