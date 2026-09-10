#!/usr/bin/env python3
"""Reproduce the Rb--Yb Foerster characterization's explicitly distinct models.

Every reported eigenstate weight, splitting, and transfer probability is
derived from the same PairInteraction Hamiltonian and the explicit |PP> and
|SS> target amplitudes.  The output is the sole numerical input for Fig. 1.

Default execution preserves the existing zero-field fixed-m characterization
and updates/checkpoints only the axial HFS field curve on the final P0-4 basis.
--rebuild-fixed-m explicitly regenerates the historical fixed-m inputs first.
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; --workers 4 bounds model concurrency.

Required database assets:
  Rb v1.2, Yb171_mqdt v1.4, and misc v1.4.

Output:
  simulations/forster_characterization.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy
from pairinteraction._backend import get_cache_directory
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulations" / "forster_characterization.json"
sys.path.insert(0, str(ROOT / "simulations"))

import simulate_forster_gate as gate_model  # noqa: E402


M = 0.5
INTERACTION_ORDER = 3
PRIMARY_DELTA_N = 3
PRIMARY_L_MAX = 3
PRIMARY_PAIR_WINDOW_GHZ = 80.0
R_GRID_UM = [3.0, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.5, 5.0]
B_GRID_G = [0.0, 0.5, 1.0, 2.0, 3.0, 3.1, 4.0, 5.0, 7.0, 10.0]
ANGLE_GRID_DEG = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
OPERATING_DISTANCE_UM = 3.4


def dense_vector(value: object) -> np.ndarray:
    """Flatten a PairInteraction vector without discarding complex phases."""

    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=complex).reshape(-1)


def target_kets() -> tuple[pi.KetAtom, pi.KetAtom, pi.KetAtom, pi.KetAtom]:
    rb_pp = pi.KetAtom("Rb", n=56, l=1, j=0.5, m=M)
    rb_ss = pi.KetAtom("Rb", n=56, l=0, j=0.5, m=M)
    yb_pp = pi.KetAtom("Yb171_mqdt", n=52, l=1, s=0, f=0.5, m=M)
    yb_ss = pi.KetAtom("Yb171_mqdt", n=53, l=0, s=1, f=0.5, m=M)
    return rb_pp, rb_ss, yb_pp, yb_ss


def make_fixed_m_basis(
    field_gauss: float,
    delta_n: int = PRIMARY_DELTA_N,
    l_max: int = PRIMARY_L_MAX,
    pair_window_ghz: float = PRIMARY_PAIR_WINDOW_GHZ,
) -> dict[str, object]:
    rb_pp, rb_ss, yb_pp, yb_ss = target_kets()
    rb_system = pi.SystemAtom(
        pi.BasisAtom(
            "Rb",
            n=(56 - delta_n, 56 + delta_n),
            l=(0, l_max),
            m=(M, M),
        )
    )
    yb_system = pi.SystemAtom(
        pi.BasisAtom(
            "Yb171_mqdt",
            n=(52 - delta_n, 52 + delta_n),
            l=(0, l_max),
            m=(M, M),
        )
    )
    if field_gauss:
        rb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
        yb_system.set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()

    pp_energy_mhz = float(
        rb_system.get_corresponding_energy(rb_pp, unit="MHz")
        + yb_system.get_corresponding_energy(yb_pp, unit="MHz")
    )
    ss_energy_mhz = float(
        rb_system.get_corresponding_energy(rb_ss, unit="MHz")
        + yb_system.get_corresponding_energy(yb_ss, unit="MHz")
    )
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        m=(2 * M, 2 * M),
        energy=(
            pp_energy_mhz / 1000 - pair_window_ghz,
            pp_energy_mhz / 1000 + pair_window_ghz,
        ),
        energy_unit="GHz",
    )

    pp_raw = dense_vector(pair_basis.get_amplitudes((rb_pp, yb_pp)))
    ss_raw = dense_vector(pair_basis.get_amplitudes((rb_ss, yb_ss)))
    pp_norm = float(np.vdot(pp_raw, pp_raw).real)
    ss_norm = float(np.vdot(ss_raw, ss_raw).real)
    target_gram = complex(np.vdot(pp_raw, ss_raw))
    if min(pp_norm, ss_norm) < 0.999 or abs(target_gram) > 1e-10:
        raise RuntimeError(
            "Target states are not faithfully represented: "
            f"PP norm={pp_norm}, SS norm={ss_norm}, Gram={target_gram}"
        )

    return {
        "field_gauss": field_gauss,
        "delta_n": delta_n,
        "l_max": l_max,
        "pair_window_ghz": pair_window_ghz,
        "pair_basis": pair_basis,
        "pair_basis_size": int(pair_basis.number_of_states),
        "pp_energy_mhz": pp_energy_mhz,
        "ss_energy_mhz": ss_energy_mhz,
        "pp_norm_before_normalization": pp_norm,
        "ss_norm_before_normalization": ss_norm,
        "pp": pp_raw / np.sqrt(pp_norm),
        "ss": ss_raw / np.sqrt(ss_norm),
    }


def transfer_populations(
    time_us: float,
    energies_mhz: np.ndarray,
    pp_overlap: np.ndarray,
    ss_overlap: np.ndarray,
) -> tuple[float, float, float]:
    phases = np.exp(-2j * np.pi * energies_mhz * time_us)
    pp_amplitude = np.sum(phases * np.conj(pp_overlap) * ss_overlap)
    ss_amplitude = np.sum(phases * np.abs(ss_overlap) ** 2)
    pp_population = float(abs(pp_amplitude) ** 2)
    ss_population = float(abs(ss_amplitude) ** 2)
    spectator_population = float(max(0.0, 1 - pp_population - ss_population))
    return pp_population, ss_population, spectator_population


def analyze_spectrum(
    energies_mhz: np.ndarray,
    pp_overlap: np.ndarray,
    ss_overlap: np.ndarray,
    include_trajectory: bool,
    time_samples: int = 801,
) -> dict[str, object]:
    pp_weight = np.abs(pp_overlap) ** 2
    ss_weight = np.abs(ss_overlap) ** 2
    bright = np.argsort(-(pp_weight + ss_weight))[:2]
    bright = bright[np.argsort(energies_mhz[bright])]
    full_splitting_mhz = float(energies_mhz[bright[1]] - energies_mhz[bright[0]])

    # Locate the largest transfer in the first bright-state oscillation and
    # refine it continuously.  This avoids mistaking tiny high-frequency
    # spectator ripples for the physically relevant first exchange maximum.
    period_us = 1 / full_splitting_mhz
    coarse_times = np.linspace(0, period_us, time_samples)
    coarse_transfer = np.array(
        [
            transfer_populations(t, energies_mhz, pp_overlap, ss_overlap)[0]
            for t in coarse_times
        ]
    )
    maximum_index = int(np.argmax(coarse_transfer))
    lower = coarse_times[max(0, maximum_index - 2)]
    upper = coarse_times[min(len(coarse_times) - 1, maximum_index + 2)]
    optimum = minimize_scalar(
        lambda t: -transfer_populations(
            float(t), energies_mhz, pp_overlap, ss_overlap
        )[0],
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-15},
    )
    pp_population, ss_population, spectator_population = transfer_populations(
        float(optimum.x), energies_mhz, pp_overlap, ss_overlap
    )

    bright_states = []
    for label, index in zip(("lower", "upper"), bright):
        coefficient = complex(np.conj(pp_overlap[index]) * ss_overlap[index])
        bright_states.append(
            {
                "label": label,
                "energy_relative_to_pp_mhz": float(energies_mhz[index]),
                "pp_weight": float(pp_weight[index]),
                "ss_weight": float(ss_weight[index]),
                "other_weight": float(1 - pp_weight[index] - ss_weight[index]),
                "transfer_amplitude_coefficient_re_im": [
                    float(coefficient.real),
                    float(coefficient.imag),
                ],
            }
        )

    result: dict[str, object] = {
        "full_bright_splitting_mhz": full_splitting_mhz,
        "bright_states": bright_states,
        "first_exchange_maximum": {
            "time_ns": float(1000 * optimum.x),
            "pp_population": pp_population,
            "residual_ss_population": ss_population,
            "spectator_population": spectator_population,
        },
        "unitarity_transfer_upper_bound": float(
            np.sum(np.abs(np.conj(pp_overlap) * ss_overlap)) ** 2
        ),
        "completeness": {
            "sum_pp_weights": float(np.sum(pp_weight)),
            "sum_ss_weights": float(np.sum(ss_weight)),
        },
    }
    if include_trajectory:
        trajectory_times = np.linspace(0, period_us, 401)
        trajectory = np.array(
            [
                transfer_populations(t, energies_mhz, pp_overlap, ss_overlap)
                for t in trajectory_times
            ]
        )
        result["trajectory"] = {
            "time_ns": (1000 * trajectory_times).tolist(),
            "pp_population": trajectory[:, 0].tolist(),
            "ss_population": trajectory[:, 1].tolist(),
            "spectator_population": trajectory[:, 2].tolist(),
        }
    return result


def solve_fixed_m_point(
    model: dict[str, object],
    distance_um: float,
    angle_deg: float,
    include_trajectory: bool = False,
) -> dict[str, object]:
    pair_basis = model["pair_basis"]
    pp = np.asarray(model["pp"], dtype=complex)
    ss = np.asarray(model["ss"], dtype=complex)
    pp_energy_mhz = float(model["pp_energy_mhz"])
    hamiltonian = (
        pi.SystemPair(pair_basis)
        .set_distance(distance_um, angle_degree=angle_deg, unit="micrometer")
        .set_interaction_order(INTERACTION_ORDER)
        .get_hamiltonian(unit="MHz")
        .toarray()
        .astype(complex)
    )
    hamiltonian -= pp_energy_mhz * np.eye(len(hamiltonian))
    scale = max(float(np.max(np.abs(hamiltonian))), 1.0)
    hermiticity_error_mhz = float(
        np.max(np.abs(hamiltonian - hamiltonian.conj().T))
    )
    if hermiticity_error_mhz / scale > 1e-12:
        raise RuntimeError(f"Pair Hamiltonian is not Hermitian: {hermiticity_error_mhz}")

    projected_pp_mhz = float(np.vdot(pp, hamiltonian @ pp).real)
    projected_ss_mhz = float(np.vdot(ss, hamiltonian @ ss).real)
    coupling = complex(np.vdot(pp, hamiltonian @ ss))
    projected_defect_mhz = projected_ss_mhz - projected_pp_mhz
    projected_splitting_mhz = float(
        np.sqrt(projected_defect_mhz**2 + 4 * abs(coupling) ** 2)
    )
    two_state = {
        "pp_diagonal_mhz": projected_pp_mhz,
        "ss_diagonal_mhz": projected_ss_mhz,
        "defect_ss_minus_pp_mhz": projected_defect_mhz,
        "coupling_re_im_mhz": [float(coupling.real), float(coupling.imag)],
        "generalized_splitting_mhz": projected_splitting_mhz,
        "maximum_transfer_probability": float(
            4 * abs(coupling) ** 2 / projected_splitting_mhz**2
        ),
        "first_maximum_time_ns": float(1000 / (2 * projected_splitting_mhz)),
    }

    energies_mhz, eigenvectors = eigh(
        hamiltonian, driver="evr", check_finite=False
    )
    pp_overlap = eigenvectors.conj().T @ pp
    ss_overlap = eigenvectors.conj().T @ ss
    result = analyze_spectrum(
        energies_mhz, pp_overlap, ss_overlap, include_trajectory
    )
    result.update(
        {
            "distance_um": distance_um,
            "theta_deg": angle_deg,
            "field_gauss": float(model["field_gauss"]),
            "pair_basis_size": int(model["pair_basis_size"]),
            "hermiticity_error_mhz": hermiticity_error_mhz,
            "projected_two_state_model": two_state,
        }
    )
    return result


def summarize_all_m_model(model: gate_model.PairModel) -> dict[str, object]:
    result = analyze_spectrum(
        model.energies_mhz,
        model.pp_overlap,
        model.ss_overlap,
        include_trajectory=False,
    )
    return {
        "field_gauss": model.field_gauss,
        "distance_um": model.distance_um,
        "theta_deg": model.theta_deg,
        "pair_basis_size": model.pair_basis_size,
        "symmetry_component_size": model.symmetry_component_size,
        "pp_component_norm": model.pp_component_norm,
        "ss_component_norm": model.ss_component_norm,
        **result,
    }


def database_manifest() -> dict[str, object]:
    tables = get_cache_directory() / "database" / "tables"
    assets = {
        "Rb": "Rb_v1.2",
        "Yb171_mqdt": "Yb171_mqdt_v1.4",
        "misc": "misc_v1.4",
    }
    files = []
    for species, directory in assets.items():
        asset_dir = tables / directory
        if not asset_dir.is_dir():
            raise RuntimeError(f"Missing required database asset: {asset_dir}")
        for path in sorted(asset_dir.glob("*.parquet")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files.append(
                {
                    "species": species,
                    "relative_path": f"{directory}/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return {
        "versions": {"Rb": "v1.2", "Yb171_mqdt": "v1.4", "misc": "v1.4"},
        "files": files,
    }


def compact_point(point: dict[str, object]) -> dict[str, object]:
    maximum = point["first_exchange_maximum"]
    return {
        "distance_um": point["distance_um"],
        "theta_deg": point["theta_deg"],
        "field_gauss": point["field_gauss"],
        "pair_basis_size": point["pair_basis_size"],
        "full_bright_splitting_mhz": point["full_bright_splitting_mhz"],
        "first_exchange_maximum": maximum,
        "projected_two_state_model": point["projected_two_state_model"],
    }


def convergence_row(
    label: str,
    point: dict[str, object],
    primary: dict[str, object],
    config: dict[str, float | int],
) -> dict[str, object]:
    point_max = point["first_exchange_maximum"]
    primary_max = primary["first_exchange_maximum"]
    upper = point["bright_states"][1]
    primary_upper = primary["bright_states"][1]
    return {
        "label": label,
        "configuration": config,
        "pair_basis_size": point["pair_basis_size"],
        "full_bright_splitting_mhz": point["full_bright_splitting_mhz"],
        "first_exchange_maximum": point_max,
        "upper_bright_pp_weight": upper["pp_weight"],
        "absolute_change_from_primary": {
            "splitting_mhz": abs(
                point["full_bright_splitting_mhz"]
                - primary["full_bright_splitting_mhz"]
            ),
            "transfer_probability": abs(
                point_max["pp_population"] - primary_max["pp_population"]
            ),
            "spectator_probability": abs(
                point_max["spectator_population"]
                - primary_max["spectator_population"]
            ),
            "upper_bright_pp_weight": abs(
                upper["pp_weight"] - primary_upper["pp_weight"]
            ),
        },
    }


def _rebuild_fixed_m_record() -> None:
    if pi.Database.get_global_database() is None:
        pi.Database.initialize_global_database(download_missing=False)

    rb_pp, rb_ss, yb_pp, yb_ss = target_kets()
    rb_interval_ghz = float(
        (rb_pp.get_energy(unit="MHz") - rb_ss.get_energy(unit="MHz")) / 1000
    )
    yb_release_ghz = float(
        (yb_ss.get_energy(unit="MHz") - yb_pp.get_energy(unit="MHz")) / 1000
    )
    asymptotic_defect_mhz = float(1000 * (yb_release_ghz - rb_interval_ghz))
    if abs(asymptotic_defect_mhz + 0.763732) > 1e-3:
        raise RuntimeError(
            "State-identity guard failed: expected SS-PP defect near -0.763732 MHz, "
            f"got {asymptotic_defect_mhz:.9f} MHz"
        )

    print("Building primary fixed-m basis ...", flush=True)
    primary_model = make_fixed_m_basis(0.0)
    distance_scan = []
    operating_point = None
    for distance_um in R_GRID_UM:
        print(f"  R={distance_um:.1f} um", flush=True)
        point = solve_fixed_m_point(
            primary_model,
            distance_um,
            0.0,
            include_trajectory=distance_um == OPERATING_DISTANCE_UM,
        )
        distance_scan.append(compact_point(point))
        if distance_um == OPERATING_DISTANCE_UM:
            operating_point = point
    assert operating_point is not None

    print("Scanning fixed-m magnetic field ...", flush=True)
    fixed_m_field_scan = []
    for field_gauss in B_GRID_G:
        print(f"  B={field_gauss:.1f} G", flush=True)
        field_model = primary_model if field_gauss == 0 else make_fixed_m_basis(field_gauss)
        fixed_m_field_scan.append(
            compact_point(
                solve_fixed_m_point(
                    field_model, OPERATING_DISTANCE_UM, 0.0
                )
            )
        )

    print("Scanning fixed-m angle ...", flush=True)
    fixed_m_angle_scan = []
    for angle_deg in ANGLE_GRID_DEG:
        print(f"  theta={angle_deg:.0f} deg", flush=True)
        fixed_m_angle_scan.append(
            compact_point(
                solve_fixed_m_point(
                    primary_model, OPERATING_DISTANCE_UM, angle_deg
                )
            )
        )

    # build_pair_model now uses get_amplitudes directly.  The all-m scan has a
    # smaller, explicitly recorded basis because it is a contextual diagnostic
    # rather than the fixed-m source of the operating-point weights.
    print("Scanning all-m magnetic field ...", flush=True)
    all_m_field_scan = []
    for field_gauss in B_GRID_G:
        print(f"  B={field_gauss:.1f} G", flush=True)
        model = gate_model.build_pair_model(field_gauss, gate_model.SCAN_BASIS)
        all_m_field_scan.append(summarize_all_m_model(model))

    print("Running operating-point convergence checks ...", flush=True)
    convergence_specs = [
        (
            "pair window 100 GHz",
            {"delta_n": 3, "l_max": 3, "pair_window_ghz": 100.0},
        ),
        (
            "delta_n 4",
            {"delta_n": 4, "l_max": 3, "pair_window_ghz": 80.0},
        ),
        (
            "l_max 4",
            {"delta_n": 3, "l_max": 4, "pair_window_ghz": 80.0},
        ),
    ]
    convergence = []
    for label, config in convergence_specs:
        print(f"  {label}", flush=True)
        check_model = make_fixed_m_basis(0.0, **config)
        check = solve_fixed_m_point(
            check_model, OPERATING_DISTANCE_UM, 0.0
        )
        convergence.append(
            convergence_row(label, check, operating_point, config)
        )

    thresholds = {
        "splitting_mhz": 0.05,
        "transfer_probability": 5e-4,
        "spectator_probability": 5e-4,
        "upper_bright_pp_weight": 5e-4,
    }
    convergence_passed = all(
        row["absolute_change_from_primary"][name] < tolerance
        for row in convergence
        for name, tolerance in thresholds.items()
    )
    if not convergence_passed:
        raise RuntimeError("Operating-point convergence thresholds were not met")

    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": sys.version.split()[0],
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "database": database_manifest(),
        "target_states": {
            "pp": {
                "Rb": "56P_1/2, m_J=+1/2",
                "Yb171_mqdt": "nu=48.014048, L=1, F=1/2, m_F=+1/2; selector n=52, s=0",
            },
            "ss": {
                "Rb": "56S_1/2, m_J=+1/2",
                "Yb171_mqdt": "nu=48.369927, L=0, F=1/2, m_F=+1/2; selector n=53, s=1",
            },
            "energy_exchange": {
                "direction": "SS to PP",
                "rb_absorbed_ghz": rb_interval_ghz,
                "yb_released_ghz": yb_release_ghz,
                "defect_ss_minus_pp_mhz": asymptotic_defect_mhz,
            },
        },
        "fixed_m_model": {
            "m_rb": M,
            "m_yb": M,
            "total_m": 2 * M,
            "field_gauss": 0.0,
            "theta_deg": 0.0,
            "delta_n": PRIMARY_DELTA_N,
            "l_max": PRIMARY_L_MAX,
            "pair_energy_window_ghz": [
                -PRIMARY_PAIR_WINDOW_GHZ,
                PRIMARY_PAIR_WINDOW_GHZ,
            ],
            "interaction": "electric dipole-dipole",
            "interaction_order": INTERACTION_ORDER,
            "target_projection": "complex PairInteraction get_amplitudes vectors",
            "pair_basis_size": primary_model["pair_basis_size"],
            "pp_norm_before_normalization": primary_model[
                "pp_norm_before_normalization"
            ],
            "ss_norm_before_normalization": primary_model[
                "ss_norm_before_normalization"
            ],
        },
        "operating_point": operating_point,
        "distance_scan": distance_scan,
        "fixed_m_field_scan": fixed_m_field_scan,
        "fixed_m_angle_scan": fixed_m_angle_scan,
        "all_m_field_scan": {
            "basis": asdict(gate_model.SCAN_BASIS),
            "theta_deg": 0.0,
            "distance_um": OPERATING_DISTANCE_UM,
            "points": all_m_field_scan,
        },
        "convergence": {
            "thresholds": thresholds,
            "checks": convergence,
            "passed": convergence_passed,
            "reporting_precision_supported": {
                "splitting_mhz": 0.1,
                "transfer_percent": 0.1,
                "bright_weight_percent": 0.1,
            },
        },
        "validation": {
            "same_hamiltonian_for_operating_weights_splitting_and_transfer": True,
            "explicit_ss_and_pp_targets": True,
            "complex_amplitudes_used": True,
            "probability_completeness_passed": abs(
                operating_point["completeness"]["sum_pp_weights"] - 1
            )
            < 1e-10
            and abs(operating_point["completeness"]["sum_ss_weights"] - 1)
            < 1e-10,
            "full_transfer_below_unitarity_bound": operating_point[
                "first_exchange_maximum"
            ]["pp_population"]
            <= operating_point["unitarity_transfer_upper_bound"] + 1e-12,
            "convergence_passed": convergence_passed,
        },
    }
    if not all(result["validation"].values()):
        raise RuntimeError(f"Validation failed: {result['validation']}")

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    maximum = operating_point["first_exchange_maximum"]
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(
        "Operating point: "
        f"splitting={operating_point['full_bright_splitting_mhz']:.6f} MHz, "
        f"P(SS->PP)={100 * maximum['pp_population']:.6f}%, "
        f"t={maximum['time_ns']:.6f} ns, "
        f"spectators={100 * maximum['spectator_population']:.6f}%"
    )


def final_hfs_field_point(field_gauss):
    import evaluate_forster_p0_4 as p0

    start = time.monotonic()
    model = gate_model.build_pair_model(float(field_gauss), p0.REFERENCE_BASIS,
        distance_um=OPERATING_DISTANCE_UM, theta_deg=0.)
    # All eigenmodes, no bright cutoff, drive or decay. Do not use static_transfer:
    # that helper starts from PP and searches the whole 0--80 ns interval.
    energies = model.energies_mhz - model.pp_asymptote_mhz
    bright = np.argsort(np.abs(model.pp_overlap)**2 + np.abs(model.ss_overlap)**2)[-2:]
    splitting = float(np.ptp(energies[bright]))
    # At least 16 samples across the fastest retained spectral beat, followed
    # by continuous local refinement. The historical fixed-m record is untouched.
    samples = max(801, int(np.ceil(16*np.ptp(energies)/splitting))+1)
    result = analyze_spectrum(energies, model.pp_overlap, model.ss_overlap,
        include_trajectory=field_gauss == 3.1, time_samples=samples)
    result.update(field_gauss=float(field_gauss), distance_um=OPERATING_DISTANCE_UM,
        theta_deg=0., basis=asdict(p0.REFERENCE_BASIS), pair_basis_size=model.pair_basis_size,
        symmetry_component_size=model.symmetry_component_size,
        pp_asymptote_mhz=model.pp_asymptote_mhz, ss_asymptote_mhz=model.ss_asymptote_mhz,
        forster_defect_mhz=model.forster_defect_mhz,
        hermiticity_error_mhz=model.hermiticity_error_mhz,
        peak_search_samples=samples, peak_search_step_ns=1000/(splitting*(samples-1)))
    for state in result['bright_states']:
        state['energy_relative_to_ss_mhz'] = state['energy_relative_to_pp_mhz']-model.forster_defect_mhz
    if field_gauss == 3.1:
        result['peak_search_convergence'] = [
            {'samples': n, **analyze_spectrum(energies, model.pp_overlap, model.ss_overlap,
                False, time_samples=n)['first_exchange_maximum']}
            for n in (801, 3201, samples, 2*samples-1)]
        spectrum_path = ROOT / 'simulations/forster_p1_4_reference_spectrum.json'
        spectrum = json.loads(spectrum_path.read_text())
        assert spectrum['fixed_inputs']['basis'] == asdict(p0.REFERENCE_BASIS)
        target = spectrum['spectral_diagnostics']['target_eigenstates']
        np.testing.assert_allclose(result['full_bright_splitting_mhz'],
            target[1]['energy_mhz']-target[0]['energy_mhz'], rtol=0, atol=1e-6)
        for actual, expected in zip(result['bright_states'], target):
            np.testing.assert_allclose([actual['pp_weight'],actual['ss_weight']],
                [expected['pp_weight'],expected['ss_weight']], rtol=0, atol=1e-7)
        result['p1_4_crosscheck'] = {'passed': True, 'source': str(spectrum_path.relative_to(ROOT))}
    assert max(abs(result['completeness'][k]-1) for k in ('sum_pp_weights','sum_ss_weights')) < 1e-10
    assert result['first_exchange_maximum']['pp_population'] <= result['unitarity_transfer_upper_bound']+1e-12
    result['resources'] = {'wall_seconds': time.monotonic()-start,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    del model
    gc.collect()
    return result


def main():
    import evaluate_forster_p0_4 as p0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-fixed-m', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1,5), default=1)
    args = parser.parse_args()
    if args.rebuild_fixed_m:
        _rebuild_fixed_m_record()
    data = json.loads(OUT.read_text())
    start = time.monotonic()
    if 'historical_all_m_field_scan' not in data:
        data['historical_all_m_field_scan'] = data['all_m_field_scan']
        data['all_m_field_scan'] = {'basis': asdict(p0.REFERENCE_BASIS),
            'distance_um': OPERATING_DISTANCE_UM, 'theta_deg': 0., 'points': [],
            'status': 'in progress',
            'observable': 'SS-initial unitary free exchange; largest PP population within the first bright-state period, not the earliest tiny spectator ripple or an 80-ns global maximum; spectator population is simultaneous',
            'model_scope': 'full retained axial M_tot=5/2 HFS block through R^-4, all eigenmodes, no optical drive or decay; not a full-angular model',
            'fixed_m_comparison': 'fixed_m_field_scan uses the same first-exchange convention but its own historical fixed-m dipole basis',
            'basis_convergence_scope': 'root convergence checks concern only the zero-field fixed-m model; no new HFS first-exchange basis-convergence claim',
            'software': {'pairinteraction': pi.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
            'runs': []}
    scan = data['all_m_field_scan']
    assert scan['basis'] == asdict(p0.REFERENCE_BASIS)
    completed = {point['field_gauss'] for point in scan['points']}
    missing = [b for b in B_GRID_G if b not in completed]
    if not missing:
        print('Final HFS field scan already complete; no models rebuilt.')
        return
    run = {'workers': args.workers, 'fields_gauss': missing}
    scan['runs'].append(run)

    def checkpoint():
        scan['points'].sort(key=lambda row: row['field_gauss'])
        run.update(wall_seconds=time.monotonic()-start,
            peak_parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maximum_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        temporary = OUT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(OUT)

    checkpoint()
    if args.workers == 1:
        for field in missing:
            scan['points'].append(final_hfs_field_point(field))
            checkpoint()
            print('saved HFS field',field,flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for point in pool.map(final_hfs_field_point, missing):
                scan['points'].append(point)
                checkpoint()
                print('saved HFS field',point['field_gauss'],flush=True)
    scan['status'] = 'complete'
    checkpoint()


if __name__ == "__main__":
    main()
