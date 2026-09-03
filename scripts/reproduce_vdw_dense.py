#!/usr/bin/env python3
"""Reproduce the paper's all-magnetic-sublevel static vdW data.

This is a static interaction calculation, not a driven gate simulation. It
computes the perturbative C6 matrix, follows the selected finite-field branch
over distance, and samples magnetic sectors and bias fields.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pairinteraction as pi
from scipy.sparse.linalg import eigsh
from verify_pairinteraction_databases import verify_database_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "vdw_dense_data.json"

N_RB = 66
NU_YB = 62.682292758
FIXED_FIELD_G = 25.0
DELTA_N = 3
L_MAX = 3
PAIR_WINDOW_GHZ = 80.0
INTERACTION_ORDER = 3
ANGLE_DEG = 0.0
NUMBER_EIGENPAIRS = 24
DISTANCES_UM = (3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.25, 4.5, 5.0)
SECTORS = ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
FIELD_SCAN_G = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0)


def dense_vector(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=complex).reshape(-1)


def initialize_database() -> None:
    if pi.Database.get_global_database() is None:
        pi.Database.initialize_global_database(download_missing=False)


def atomic_systems(field_gauss: float) -> tuple[pi.SystemAtom, pi.SystemAtom]:
    rb_system = pi.SystemAtom(pi.BasisAtom("Rb", n=(N_RB - DELTA_N, N_RB + DELTA_N), l=(0, L_MAX)))
    yb_system = pi.SystemAtom(
        pi.BasisAtom("Yb171_mqdt", nu=(NU_YB - 3.3, NU_YB + 3.3), l=(0, L_MAX))
    )
    rb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    yb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    return rb_system, yb_system


def make_sector(field_gauss: float, m_rb: float, m_yb: float) -> dict[str, object]:
    rb_target = pi.KetAtom("Rb", n=N_RB, l=0, j=0.5, m=m_rb)
    yb_target = pi.KetAtom("Yb171_mqdt", nu=NU_YB, l=0, f=0.5, m=m_yb)
    rb_system, yb_system = atomic_systems(field_gauss)
    asymptote_mhz = float(
        rb_system.get_corresponding_energy(rb_target, unit="MHz")
        + yb_system.get_corresponding_energy(yb_target, unit="MHz")
    )
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        m=(m_rb + m_yb, m_rb + m_yb),
        energy=(
            asymptote_mhz / 1000 - PAIR_WINDOW_GHZ,
            asymptote_mhz / 1000 + PAIR_WINDOW_GHZ,
        ),
        energy_unit="GHz",
    )
    target_amplitudes = dense_vector(pair_basis.get_amplitudes((rb_target, yb_target)))
    target_norm = float(np.vdot(target_amplitudes, target_amplitudes).real)
    if abs(target_norm - 1.0) > 1e-8:
        raise RuntimeError(f"truncated target norm {target_norm:.12f}")
    return {
        "pair_basis": pair_basis,
        "asymptote_mhz": asymptote_mhz,
        "target_amplitudes": target_amplitudes,
    }


def solve_point(
    sector: dict[str, object], distance_um: float, sigma_shift_mhz: float
) -> dict[str, float | int]:
    pair_basis = sector["pair_basis"]
    asymptote_mhz = float(sector["asymptote_mhz"])
    target_amplitudes = np.asarray(sector["target_amplitudes"])
    hamiltonian = (
        pi.SystemPair(pair_basis)
        .set_distance(distance_um, angle_degree=ANGLE_DEG, unit="micrometer")
        .set_interaction_order(INTERACTION_ORDER)
        .get_hamiltonian(unit="MHz")
    )
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=NUMBER_EIGENPAIRS,
        sigma=asymptote_mhz + sigma_shift_mhz,
        which="LM",
        tol=1e-10,
        maxiter=5000,
        v0=target_amplitudes,
    )
    weights = np.abs(eigenvectors.conj().T @ target_amplitudes) ** 2
    selected = int(np.argmax(weights))
    residual = (
        hamiltonian @ eigenvectors[:, selected] - eigenvalues[selected] * eigenvectors[:, selected]
    )
    maximum_residual_mhz = float(np.max(np.abs(residual)))
    if maximum_residual_mhz > 1e-4:
        raise RuntimeError(f"eigenpair residual {maximum_residual_mhz:.3e} MHz")
    return {
        "u_mhz": float(eigenvalues[selected] - asymptote_mhz),
        "overlap": float(weights[selected]),
        "n_states": int(pair_basis.number_of_states),
    }


def distance_point(
    sector: dict[str, object], distance_um: float, sigma_shift_mhz: float
) -> dict[str, float | int]:
    return {
        "distance_um": float(distance_um),
        **solve_point(sector, distance_um, sigma_shift_mhz),
    }


def zero_field_c6_channel(
    total_m: float, target_sectors: tuple[tuple[float, float], ...]
) -> dict[str, object]:
    rb_system, yb_system = atomic_systems(0.0)
    target_kets = [
        (
            pi.KetAtom("Rb", n=N_RB, l=0, j=0.5, m=m_rb),
            pi.KetAtom("Yb171_mqdt", nu=NU_YB, l=0, f=0.5, m=m_yb),
        )
        for m_rb, m_yb in target_sectors
    ]
    target_energies_mhz = np.array(
        [
            rb_system.get_corresponding_energy(rb, unit="MHz")
            + yb_system.get_corresponding_energy(yb, unit="MHz")
            for rb, yb in target_kets
        ],
        dtype=float,
    )
    if np.ptp(target_energies_mhz) > 1e-6:
        raise RuntimeError("zero-field target subspace is not degenerate")
    asymptote_mhz = float(target_energies_mhz[0])
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        m=(total_m, total_m),
        energy=(
            asymptote_mhz / 1000 - PAIR_WINDOW_GHZ,
            asymptote_mhz / 1000 + PAIR_WINDOW_GHZ,
        ),
        energy_unit="GHz",
    )
    amplitudes = np.column_stack(
        [dense_vector(pair_basis.get_amplitudes(pair)) for pair in target_kets]
    )
    gram = amplitudes.conj().T @ amplitudes
    if not np.allclose(gram, np.eye(len(target_sectors)), atol=1e-8):
        raise RuntimeError(f"truncated zero-field target Gram matrix: {gram}")

    h0 = pi.SystemPair(pair_basis).get_hamiltonian(unit="MHz")
    h1 = (
        pi.SystemPair(pair_basis)
        .set_distance(1.0, angle_degree=ANGLE_DEG, unit="micrometer")
        .set_interaction_order(INTERACTION_ORDER)
        .get_hamiltonian(unit="MHz")
    )
    interaction = h1 - h0
    coupled = interaction @ amplitudes
    denominators_mhz = asymptote_mhz - np.asarray(h0.diagonal()).real
    inverse_denominators = np.zeros_like(denominators_mhz)
    nondegenerate = np.abs(denominators_mhz) > 1e-6
    inverse_denominators[nondegenerate] = 1 / denominators_mhz[nondegenerate]
    c6_mhz_um6 = coupled.conj().T @ (inverse_denominators[:, None] * coupled)
    if np.max(np.abs(c6_mhz_um6.imag)) > 1e-6:
        raise RuntimeError("unexpected complex zero-field C6 matrix")
    c6_ghz_um6 = c6_mhz_um6.real / 1000

    coupling_strengths = np.sum(np.abs(coupled) ** 2, axis=1)
    coupled_nondegenerate = nondegenerate & (coupling_strengths > 1e-16)
    nearest_index = int(
        np.argmin(np.where(coupled_nondegenerate, np.abs(denominators_mhz), np.inf))
    )
    result: dict[str, object] = {
        "target_m_pairs": [[m_rb, m_yb] for m_rb, m_yb in target_sectors],
        "c6_matrix_ghz_um6": c6_ghz_um6.tolist(),
        "c6_eigenvalues_ghz_um6": np.linalg.eigvalsh(c6_ghz_um6).tolist(),
        "n_states": int(pair_basis.number_of_states),
        "nearest_dipole_coupled_detuning_mhz": float(denominators_mhz[nearest_index]),
    }
    if len(target_sectors) == 1:
        c3_squared_mhz2_um6 = float(coupling_strengths.sum())
        result["c3_squared_mhz2_um6"] = c3_squared_mhz2_um6
        result["perturbative_delta_eff_mhz"] = float(c3_squared_mhz2_um6 / c6_mhz_um6.real[0, 0])
    return result


def zero_field_c6() -> dict[str, object]:
    channels = {
        "M=-1": zero_field_c6_channel(-1.0, ((-0.5, -0.5),)),
        "M=0": zero_field_c6_channel(0.0, ((-0.5, 0.5), (0.5, -0.5))),
        "M=+1": zero_field_c6_channel(1.0, ((0.5, 0.5),)),
    }
    c6_minus = float(channels["M=-1"]["c6_matrix_ghz_um6"][0][0])
    c6_plus = float(channels["M=+1"]["c6_matrix_ghz_um6"][0][0])
    if abs(c6_minus - c6_plus) > 1e-6:
        raise RuntimeError("zero-field stretched-channel C6 values disagree")
    return {
        "method": (
            "second-order A^dagger V (E0-H0)^-1 V A at R=1 micrometer; "
            "all individual magnetic sublevels retained and only total M fixed"
        ),
        "channels": channels,
    }


def build_data() -> dict[str, object]:
    initialize_database()
    c6 = zero_field_c6()
    addressed_c6 = float(c6["channels"]["M=+1"]["c6_matrix_ghz_um6"][0][0])
    addressed = make_sector(FIXED_FIELD_G, 0.5, 0.5)
    descending_track = []
    sigma_shift_mhz = 0.0
    for distance_um in sorted(DISTANCES_UM, reverse=True):
        print(f"distance track: R={distance_um:.2f} um", flush=True)
        point = distance_point(addressed, distance_um, sigma_shift_mhz)
        descending_track.append(point)
        sigma_shift_mhz = float(point["u_mhz"])
    distance_track = sorted(descending_track, key=lambda point: point["distance_um"])
    addressed_by_distance = {point["distance_um"]: point for point in distance_track}

    sector_track = []
    for m_rb, m_yb in SECTORS:
        print(f"sector track: m_Rb={m_rb:+.1f}, m_Yb={m_yb:+.1f}", flush=True)
        sector = addressed if (m_rb, m_yb) == (0.5, 0.5) else make_sector(FIXED_FIELD_G, m_rb, m_yb)
        point = (
            addressed_by_distance[3.3]
            if (m_rb, m_yb) == (0.5, 0.5)
            else distance_point(sector, 3.3, float(addressed_by_distance[3.3]["u_mhz"]))
        )
        sector_track.append({"m_rb": m_rb, "m_yb": m_yb, **point})

    field_track = []
    for field_gauss in FIELD_SCAN_G:
        print(f"field track: B={field_gauss:.1f} G", flush=True)
        if field_gauss == FIXED_FIELD_G:
            existing = addressed_by_distance[3.3]
            point = {key: existing[key] for key in ("u_mhz", "overlap", "n_states")}
        else:
            point = solve_point(
                make_sector(field_gauss, 0.5, 0.5),
                3.3,
                float(addressed_by_distance[3.3]["u_mhz"]),
            )
        field_track.append({"b_gauss": field_gauss, **point})

    all_points = distance_track + sector_track + field_track
    state_counts = {int(point["n_states"]) for point in all_points}
    if min(state_counts) < 15000:
        raise RuntimeError(f"all-m pair basis unexpectedly small: {sorted(state_counts)}")
    return {
        "schema_version": 2,
        "scope": (
            "Static blockade-compatible vdW interaction resource. This calculation is not "
            "a driven multichannel gate design or process-fidelity result."
        ),
        "provenance": {
            "pairinteraction_version": importlib.metadata.version("pairinteraction"),
            "database_versions": {"Rb": "v1.2", "Yb171_mqdt": "v1.4"},
            "database_hashes": "provenance/pairinteraction_database_manifest.json",
            "solver": (
                "scipy.sparse.linalg.eigsh, 24 eigenpairs around the previous branch "
                "shift while stepping from R=5 to 3 micrometers"
            ),
            "target_projection": "BasisPair.get_amplitudes(target_pair)",
            "magnetic_basis": ("all individual atomic magnetic sublevels; only total pair M fixed"),
        },
        "configuration": {
            "rb_state": {"species": "Rb", "n": 66, "l": 0, "j": 0.5},
            "yb_state": {
                "species": "Yb171_mqdt",
                "nu": NU_YB,
                "l": 0,
                "f": 0.5,
            },
            "rb_basis": {
                "n": [N_RB - DELTA_N, N_RB + DELTA_N],
                "l": [0, L_MAX],
                "m": "all sublevels",
            },
            "yb_basis": {
                "nu": [NU_YB - 3.3, NU_YB + 3.3],
                "l": [0, L_MAX],
                "m": "all sublevels",
            },
            "pair_energy_window_ghz_about_dressed_asymptote": PAIR_WINDOW_GHZ,
            "theta_degree": ANGLE_DEG,
            "interaction_order": INTERACTION_ORDER,
            "zero_field_c6_ghz_um6": addressed_c6,
            "zero_field_c6": c6,
            "branch_selection": (
                "largest bare-product weight among eigenpairs centered on the preceding "
                "branch shift"
            ),
            "shift_definition": "selected eigenenergy minus same-field dressed pair asymptote",
            "finite_field_state_counts": sorted(state_counts),
        },
        "distance_track_b25_pp": distance_track,
        "sector_track_b25": sector_track,
        "magnetic_field_track_r3p3_pp": field_track,
    }


def quick_check() -> None:
    verify_database_manifest()
    initialize_database()
    c6 = zero_field_c6_channel(1.0, ((0.5, 0.5),))
    value = float(c6["c6_matrix_ghz_um6"][0][0])
    if abs(value - 73.5794485) > 1e-5:
        raise RuntimeError(f"C6 quick check failed: {value:.9f} GHz um^6")
    point = solve_point(make_sector(FIXED_FIELD_G, 0.5, 0.5), 3.3, 57.0)
    if abs(float(point["u_mhz"]) - 57.072053) > 1e-5:
        raise RuntimeError(f"vdW working-point quick check failed: {point}")
    print(f"quick check passed: C6={value:.9f} GHz um^6, U(3.3 um)={point['u_mhz']:.6f} MHz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()
    if args.quick_check:
        quick_check()
        return
    verify_database_manifest()
    result = build_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
