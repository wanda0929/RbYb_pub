#!/usr/bin/env python3
"""Audit the projected driven Hamiltonian for P1-2.

The stored P0-4 pulse and local-Z correction are held fixed.  At the nominal
point and the P0-4 limiting axial vertex, this script compares the published
10^-6 bright-mode projection with 10^-8 and with every eigenmode in the 3684-
state numerical-reference block.  A sparse star Hamiltonian makes the all-mode
propagation practical without changing the modeled decay convention.

Output:
  data/forster_p1_2_projection_audit.json
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_forster_p0_4 as p0  # noqa: E402
import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import reproduce_forster_gate as minimax  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from simulate_forster_gate import PairModel, query_lifetimes  # noqa: E402
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402

OUT = ROOT / "data" / "forster_p1_2_projection_audit.json"
P0_RESULT = ROOT / "data" / "forster_p0_4_uncertainty_convergence.json"
MODE_CUTOFFS: tuple[float | None, ...] = (1e-6, 1e-8, None)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _sparse_target_returns(
    model: PairModel,
    pulse,
    lifetimes,
    mode_cutoff: float | None,
    *,
    target_amplitude_scale: float,
) -> dict[str, object]:
    ss_weight_full = np.abs(model.ss_overlap) ** 2
    mask = (
        np.ones(len(model.energies_mhz), dtype=bool)
        if mode_cutoff is None
        else ss_weight_full > mode_cutoff
    )
    energies = model.energies_mhz[mask] - model.ss_asymptote_mhz
    ss_amplitudes = model.ss_overlap[mask]
    pp_amplitudes = model.pp_overlap[mask]
    ss_weights = np.abs(ss_amplitudes) ** 2
    pp_weights = np.abs(pp_amplitudes) ** 2
    retained_ss_weight = float(np.sum(ss_weights))

    pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
    ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
    spectator_weights = np.maximum(0.0, 1 - pp_weights - ss_weights)
    mode_rates = (
        pp_weights * pp_rate + ss_weights * ss_rate + spectator_weights * max(pp_rate, ss_rate)
    )

    unblocked = np.array([1.0, 0.0], dtype=complex)
    blocked = np.zeros(len(energies) + 1, dtype=complex)
    blocked[0] = 1.0
    mode_indices = np.arange(1, len(energies) + 1)
    rows = np.r_[
        np.zeros(len(energies), dtype=int),
        mode_indices,
        np.arange(len(energies) + 1),
    ]
    columns = np.r_[
        mode_indices,
        np.zeros(len(energies), dtype=int),
        np.arange(len(energies) + 1),
    ]

    maximum_spectator = 0.0
    maximum_pair_population = 0.0
    maximum_pp = 0.0
    maximum_ss = 0.0
    final_population = None
    for segment in pulse:
        omega = target_amplitude_scale * segment.omega_mhz
        detuning = segment.detuning_mhz

        h_unblocked = np.array([[0.0, omega / 2], [omega / 2, -detuning]], dtype=complex)
        h_unblocked[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        unblocked = expm(-2j * np.pi * h_unblocked * segment.duration_us) @ unblocked

        diagonal = energies - detuning - 0.5j * mode_rates / (2 * np.pi)
        data = np.r_[
            omega * np.conj(ss_amplitudes) / 2,
            omega * ss_amplitudes / 2,
            0.0,
            diagonal,
        ]
        h_blocked = csr_matrix(
            (data, (rows, columns)),
            shape=(len(energies) + 1, len(energies) + 1),
        )
        generator = -2j * np.pi * h_blocked * segment.duration_us
        blocked = expm_multiply(
            generator,
            blocked,
            traceA=complex(-2j * np.pi * segment.duration_us * np.sum(diagonal)),
        )

        pair_population = float(np.vdot(blocked[1:], blocked[1:]).real)
        pp_population = float(abs(np.vdot(pp_amplitudes, blocked[1:])) ** 2)
        ss_population = float(abs(np.vdot(ss_amplitudes, blocked[1:])) ** 2)
        spectator_population = max(0.0, pair_population - pp_population - ss_population)
        maximum_pair_population = max(maximum_pair_population, pair_population)
        maximum_pp = max(maximum_pp, pp_population)
        maximum_ss = max(maximum_ss, ss_population)
        maximum_spectator = max(maximum_spectator, spectator_population)
        final_population = {
            "computational": float(abs(blocked[0]) ** 2),
            "pair": pair_population,
            "pp": pp_population,
            "ss": ss_population,
            "spectator": spectator_population,
            "loss": max(0.0, 1 - float(abs(blocked[0]) ** 2) - pair_population),
        }

    return {
        "active_modes": int(np.count_nonzero(mask)),
        "retained_ss_weight": retained_ss_weight,
        "omitted_ss_weight": max(0.0, 1 - retained_ss_weight),
        "unblocked_return": complex(unblocked[0]),
        "blocked_return": complex(blocked[0]),
        "population": {
            "maximum_pair": maximum_pair_population,
            "maximum_pp": maximum_pp,
            "maximum_ss": maximum_ss,
            "maximum_spectator": maximum_spectator,
            "final": final_population,
        },
    }


def _gate_row(
    model: PairModel,
    pulse,
    lifetimes,
    correction: tuple[float, float],
    mode_cutoff: float | None,
    *,
    target_amplitude_scale: float,
    control_amplitude_scale: float,
) -> dict[str, object]:
    target = _sparse_target_returns(
        model,
        pulse,
        lifetimes,
        mode_cutoff,
        target_amplitude_scale=target_amplitude_scale,
    )
    duration = sum(segment.duration_us for segment in pulse)
    control_pi = shaped._control_pi(lifetimes.rb_56s_us, amplitude_scale=control_amplitude_scale)
    idle = np.exp(-duration / (2 * lifetimes.rb_56s_us))
    control_only = complex((control_pi @ np.diag([1.0, idle]) @ control_pi)[0, 0])
    both = complex(
        (control_pi @ np.diag([target["unblocked_return"], target["blocked_return"]]) @ control_pi)[
            0, 0
        ]
    )
    kraus = np.array([1.0, target["unblocked_return"], control_only, both], dtype=complex)
    fixed_fidelity = shaped._fixed_correction_fidelity(kraus, correction)
    recalibrated = shaped._local_z_metrics(kraus)
    mean_survival = float(np.vdot(kraus, kraus).real / 4)
    return {
        "bright_mode_cutoff": mode_cutoff,
        "active_modes": target["active_modes"],
        "retained_ss_weight": target["retained_ss_weight"],
        "omitted_ss_weight": target["omitted_ss_weight"],
        "kraus_diagonal_re_im": [_complex_pair(value) for value in kraus],
        "fixed_reference_local_z_fidelity": fixed_fidelity,
        "mean_computational_survival": mean_survival,
        "success_weighted_conditional_overlap": fixed_fidelity / mean_survival,
        "conditional_phase_error_rad": recalibrated["conditional_phase_error_rad"],
        "phase_recalibrated_fidelity": recalibrated["average_gate_fidelity"],
        "target_population": target["population"],
    }


def _evaluate_point(
    label: str,
    model: PairModel,
    pulse,
    lifetimes,
    correction,
    target_amplitude_scale: float,
    control_amplitude_scale: float,
) -> dict[str, object]:
    print(f"evaluating P1-2 projection audit: {label}", flush=True)
    rows = [
        _gate_row(
            model,
            pulse,
            lifetimes,
            correction,
            cutoff,
            target_amplitude_scale=target_amplitude_scale,
            control_amplitude_scale=control_amplitude_scale,
        )
        for cutoff in MODE_CUTOFFS
    ]
    projected = rows[0]
    all_modes = rows[-1]

    projected_modes = shaped._prepare_modes(model, 1e-6)
    legacy_kraus = hardware._kraus(
        projected_modes,
        pulse,
        lifetimes,
        target_amplitude_scale=target_amplitude_scale,
        control_amplitude_scale=control_amplitude_scale,
    )
    sparse_kraus = np.array([complex(*value) for value in projected["kraus_diagonal_re_im"]])
    maximum_kraus_difference = float(np.max(np.abs(sparse_kraus - legacy_kraus)))
    fixed_fidelity_difference = float(
        projected["fixed_reference_local_z_fidelity"]
        - shaped._fixed_correction_fidelity(legacy_kraus, correction)
    )
    if maximum_kraus_difference > 1e-10 or abs(fixed_fidelity_difference) > 1e-11:
        raise AssertionError("sparse propagator does not match the P0-4 implementation")
    return {
        "label": label,
        "distance_um": model.distance_um,
        "theta_deg": model.theta_deg,
        "yb_rabi_scale": target_amplitude_scale,
        "rb_rabi_scale": control_amplitude_scale,
        "pair_basis_size": model.pair_basis_size,
        "connected_component_size": model.symmetry_component_size,
        "cutoff_rows": rows,
        "sparse_implementation_crosscheck": {
            "maximum_absolute_kraus_difference": maximum_kraus_difference,
            "fixed_fidelity_difference": fixed_fidelity_difference,
        },
        "all_mode_minus_published_projection": {
            "signed_fidelity_difference": float(
                all_modes["fixed_reference_local_z_fidelity"]
                - projected["fixed_reference_local_z_fidelity"]
            ),
            "absolute_fidelity_difference": float(
                abs(
                    all_modes["fixed_reference_local_z_fidelity"]
                    - projected["fixed_reference_local_z_fidelity"]
                )
            ),
            "maximum_absolute_kraus_difference": float(
                np.max(
                    np.abs(
                        np.array([complex(*value) for value in all_modes["kraus_diagonal_re_im"]])
                        - sparse_kraus
                    )
                )
            ),
            "maximum_spectator_population_difference": float(
                all_modes["target_population"]["maximum_spectator"]
                - projected["target_population"]["maximum_spectator"]
            ),
        },
    }


def main() -> None:
    verify_database_manifest()
    p0_result = json.loads(P0_RESULT.read_text())
    fixed = p0_result["fixed_inputs"]
    correction = (
        fixed["fixed_reference_local_z_alpha_rad"],
        fixed["fixed_reference_local_z_beta_rad"],
    )
    pulse = p0._pulse()
    lifetimes = query_lifetimes(0.0)

    nominal_model = p0._build(p0.REFERENCE_BASIS)
    nominal = _evaluate_point(
        "nominal",
        nominal_model,
        pulse,
        lifetimes,
        correction,
        1.0,
        1.0,
    )
    del nominal_model
    gc.collect()

    worst_definition = p0_result["reference_model_bounded_validation"]["worst_case"]
    worst_model = minimax._build_geometry(
        worst_definition["radial_displacement_nm"] / 1000,
        worst_definition["direction_cosine"],
        p0.REFERENCE_BASIS,
    ).model
    worst = _evaluate_point(
        "P0-4 limiting axial vertex",
        worst_model,
        pulse,
        lifetimes,
        correction,
        worst_definition["yb_rabi_scale"],
        worst_definition["rb_rabi_scale"],
    )
    del worst_model
    gc.collect()

    maximum_projection_error = max(
        row["all_mode_minus_published_projection"]["absolute_fidelity_difference"]
        for row in (nominal, worst)
    )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "status": (
            "axial gate-level bright-mode projection audit complete at the "
            "nominal and limiting points; hyperfine-resolved full-angular "
            "transverse propagation remains outside this audit"
        ),
        "purpose": (
            "gate-level bright-mode projection audit using the fixed P0-4 "
            "Hamiltonian, pulse, propagation grid, decay convention, and local Z"
        ),
        "fixed_p0_4_inputs": fixed,
        "mode_cutoffs": [1e-6, 1e-8, "all modes"],
        "points": [nominal, worst],
        "maximum_absolute_all_mode_projection_fidelity_difference": (maximum_projection_error),
        "phase_and_frame_convention": {
            "hamiltonian_units": "cyclic frequencies H/h in MHz",
            "propagator": "exp[-2 pi i (H/h) t]",
            "target_rabi_frequency": (
                "real field-amplitude envelope; common optical phase fixed to zero"
            ),
            "target_detuning": (
                "rotating-frame diagonal -Delta; modeled as instantaneous optical "
                "frequency, with optical phase continuous rather than reset at segments"
            ),
            "rb_pulses": ("two positive 5 MHz pi pulses with the same optical phase phi=0"),
            "aom": (
                "independent first-order field-amplitude and detuning responses with "
                "10 ns 10-to-90 rise time; an effective model, not a measured transfer function"
            ),
            "local_z": {
                "basis_order": ["00", "01", "10", "11"],
                "alpha_on_rb_rad": correction[0],
                "beta_on_yb_rad": correction[1],
                "held_fixed": True,
            },
        },
        "angular_scope": {
            "axial_points": (
                "hyperfine-resolved exact M_tot=5/2 block; no angular projection error"
            ),
            "transverse_reference_model": (
                "retains angle-dependent Delta-M=0 terms but omits Delta-M=+/-1,+/-2"
            ),
            "existing_full_angle_control": {
                "model": "electronic-only dipole-dipole pulse-search model",
                "geometries": 19,
                "gate_values": 76,
                "maximum_absolute_fidelity_difference": 4.97e-6,
                "limiting_vertices_axial_in_both_models": True,
            },
            "claim_limit": (
                "the all-mode audit closes bright-mode projection at the axial headline "
                "points; it does not upgrade the existing transverse control to a "
                "hyperfine-resolved full-angular P0-4 calculation"
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
