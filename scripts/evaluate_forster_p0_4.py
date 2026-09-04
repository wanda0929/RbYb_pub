#!/usr/bin/env python3
"""Evaluate atomic-structure sensitivity and numerical convergence of the gate.

This is a post-optimization audit of the stored bounded-minimax pulse.  The
pulse command is never reoptimized.  A single numerical-reference local-Z
correction is used for the fixed-calibration columns; phase-recalibrated
columns are diagnostics that change only the two virtual local phases.

The audit deliberately separates three questions:

* one-at-a-time pair-basis and interaction-order convergence;
* a deterministic spectroscopy-motivated sensitivity envelope, not a
  confidence interval; and
* axial dc-field sensitivity with either carrier tracking or fixed zero-field
  laser frequencies.

Output:
  data/forster_p0_4_uncertainty_convergence.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import reproduce_forster_gate as minimax  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from simulate_forster_gate import (  # noqa: E402
    BasisConfig,
    PairModel,
    build_pair_model,
    query_lifetimes,
)
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402

OUT = ROOT / "data" / "forster_p0_4_uncertainty_convergence.json"
MODE_CUTOFF = 1e-6
REFERENCE_BASIS = BasisConfig(
    delta_n=3,
    atomic_window_ghz=80.0,
    pair_window_ghz=40.0,
    delta_l=2,
    interaction_order=4,
)
PAIR_WINDOW_CONVERGENCE_BASIS = replace(REFERENCE_BASIS, pair_window_ghz=60.0)


def _pulse(step_ns: float = hardware.FINAL_STEP_NS) -> tuple[shaped.Segment, ...]:
    return hardware._filtered_pulse(
        minimax.SELECTED_PARAMETERS,
        hardware.AOM_RISE_TIME_NS,
        step_ns,
    )


def _gate_metrics(
    model: PairModel,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    fixed_correction: tuple[float, float],
    mode_cutoff: float = MODE_CUTOFF,
    **errors: float,
) -> dict[str, object]:
    modes = shaped._prepare_modes(model, mode_cutoff)
    kraus = hardware._kraus(modes, pulse, lifetimes, **errors)
    recalibrated = shaped._local_z_metrics(kraus)
    return {
        "active_modes": len(modes.energies_rel_mhz),
        "retained_ss_weight": modes.retained_ss_weight,
        "fixed_reference_local_z_fidelity": shaped._fixed_correction_fidelity(
            kraus, fixed_correction
        ),
        "phase_recalibrated_fidelity": recalibrated["average_gate_fidelity"],
        "phase_recalibrated_local_z_alpha_rad": recalibrated["optimal_local_z_alpha_rad"],
        "phase_recalibrated_local_z_beta_rad": recalibrated["optimal_local_z_beta_rad"],
        "mean_computational_survival": recalibrated["mean_computational_survival"],
        "conditional_phase_error_rad": recalibrated["conditional_phase_error_rad"],
    }


def _spectrum_summary(model: PairModel) -> dict[str, float]:
    target = model.spectral_diagnostics["target_eigenstates"]
    energies = [float(state["energy_mhz"]) for state in target]
    target_weights = [float(state["ss_weight"] + state["pp_weight"]) for state in target]
    strongest_spectators = model.spectral_diagnostics["largest_target_overlap_spectators"]
    largest_spectator_weight = max(
        float(state["ss_weight"] + state["pp_weight"]) for state in strongest_spectators
    )
    return {
        "field_dressed_asymptotic_defect_mhz": model.forster_defect_mhz,
        "target_bright_splitting_mhz": abs(energies[1] - energies[0]),
        "minimum_target_subspace_weight": min(target_weights),
        "nearest_spectator_gap_mhz": model.spectral_diagnostics["nearest_spectator_gap_mhz"],
        "largest_spectator_target_subspace_weight": largest_spectator_weight,
    }


def _population_summary(
    model: PairModel,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    mode_cutoff: float = MODE_CUTOFF,
) -> dict[str, float]:
    mask = np.abs(model.ss_overlap) ** 2 > mode_cutoff
    modes = shaped._prepare_modes(model, mode_cutoff)
    trajectory = minimax._population_trajectory(
        modes,
        model.pp_overlap[mask],
        pulse,
        lifetimes,
    )
    return {
        "maximum_transient_spectator_population": max(trajectory["blocked_spectator"]),
        "final_spectator_population": trajectory["blocked_spectator"][-1],
        "final_computational_population": trajectory["blocked_computational"][-1],
    }


def _model_row(
    label: str,
    model: PairModel,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    fixed_correction: tuple[float, float],
) -> dict[str, object]:
    return {
        "label": label,
        "basis": asdict(model.basis_config),
        "pair_basis_size": model.pair_basis_size,
        "connected_component_size": model.symmetry_component_size,
        **_spectrum_summary(model),
        "static_transfer": model.static_transfer,
        "gate": _gate_metrics(model, pulse, lifetimes, fixed_correction),
        "fixed_pulse_population": _population_summary(model, pulse, lifetimes),
    }


def _shift_target_pair_defect(model: PairModel, offset_mhz: float) -> PairModel:
    """Apply the same target-PP projector shift without rebuilding the basis."""

    hamiltonian = np.diag(model.energies_mhz) - offset_mhz * np.outer(
        model.pp_overlap, model.pp_overlap.conj()
    )
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    pp_overlap = eigenvectors.conj().T @ model.pp_overlap
    ss_overlap = eigenvectors.conj().T @ model.ss_overlap

    target_weight = np.abs(pp_overlap) ** 2 + np.abs(ss_overlap) ** 2
    target_modes = np.argsort(target_weight)[-2:]
    target_modes = target_modes[np.argsort(energies[target_modes])]
    spectator_modes = np.setdiff1d(np.arange(len(energies)), target_modes, assume_unique=True)
    distances = np.min(np.abs(energies[spectator_modes, None] - energies[target_modes]), axis=1)
    strongest_spectators = spectator_modes[np.argsort(target_weight[spectator_modes])[-5:][::-1]]

    def state_summary(index: int) -> dict[str, object]:
        return {
            "mode_index": int(index),
            "energy_mhz": float(energies[index]),
            "ss_weight": float(abs(ss_overlap[index]) ** 2),
            "pp_weight": float(abs(pp_overlap[index]) ** 2),
            "residual_weight": float(max(0.0, 1 - target_weight[index])),
            "top_pair_components": [],
        }

    times = np.linspace(0, 0.080, 4001)
    phase = np.exp(-2j * np.pi * np.outer(times, energies))
    ss_amplitude = phase @ (np.conj(ss_overlap) * pp_overlap)
    pp_amplitude = phase @ (np.abs(pp_overlap) ** 2)
    ss_population = np.abs(ss_amplitude) ** 2
    maximum_index = int(np.argmax(ss_population))
    pp_population = float(abs(pp_amplitude[maximum_index]) ** 2)

    return replace(
        model,
        defect_offset_mhz=model.defect_offset_mhz + offset_mhz,
        energies_mhz=energies,
        pp_overlap=pp_overlap,
        ss_overlap=ss_overlap,
        pp_asymptote_mhz=model.pp_asymptote_mhz - offset_mhz,
        forster_defect_mhz=model.forster_defect_mhz + offset_mhz,
        static_transfer={
            "maximum_ss_population": float(ss_population[maximum_index]),
            "time_us": float(times[maximum_index]),
            "pp_population_at_maximum": pp_population,
            "spectator_population_at_maximum": float(
                max(0.0, 1 - ss_population[maximum_index] - pp_population)
            ),
        },
        spectral_diagnostics={
            "target_eigenstates": [state_summary(int(index)) for index in target_modes],
            "nearest_spectator_gap_mhz": float(np.min(distances)),
            "nearest_spectator_eigenstate": state_summary(
                int(spectator_modes[np.argmin(distances)])
            ),
            "largest_target_overlap_spectators": [
                state_summary(int(index)) for index in strongest_spectators
            ],
        },
    )


def _build(config: BasisConfig, **kwargs: float) -> PairModel:
    return build_pair_model(
        shaped.B_GAUSS,
        config,
        distance_um=shaped.R0_UM,
        theta_deg=0.0,
        **kwargs,
    )


def main() -> None:
    verify_database_manifest()
    lifetimes = query_lifetimes(0.0)
    pulse = _pulse()

    print("building numerical-reference model", flush=True)
    reference_model = _build(REFERENCE_BASIS)
    reference_mode = shaped._prepare_modes(reference_model, MODE_CUTOFF)
    reference_correction = hardware._correction(reference_mode, pulse, lifetimes)

    variant_configs = [
        ("pair window +/-20 GHz", replace(REFERENCE_BASIS, pair_window_ghz=20.0)),
        ("numerical reference", REFERENCE_BASIS),
        ("pair window +/-60 GHz", PAIR_WINDOW_CONVERGENCE_BASIS),
        ("Delta n = 4", replace(REFERENCE_BASIS, delta_n=4)),
        (
            "atomic window +/-160 GHz",
            replace(REFERENCE_BASIS, atomic_window_ghz=160.0),
        ),
        ("l_max = 4", replace(REFERENCE_BASIS, delta_l=3)),
        ("dipole-dipole only (order 3)", replace(REFERENCE_BASIS, interaction_order=3)),
        (
            "partial order 5",
            replace(REFERENCE_BASIS, interaction_order=5),
        ),
    ]
    convergence_rows = []
    for label, config in variant_configs:
        print(f"basis audit: {label}", flush=True)
        model = reference_model if config == REFERENCE_BASIS else _build(config)
        convergence_rows.append(_model_row(label, model, pulse, lifetimes, reference_correction))

    print("checking target-projector shift implementation", flush=True)
    shifted_models: dict[float, PairModel] = {0.0: reference_model}

    def shifted_model(offset_mhz: float) -> PairModel:
        if offset_mhz not in shifted_models:
            shifted_models[offset_mhz] = _shift_target_pair_defect(reference_model, offset_mhz)
        return shifted_models[offset_mhz]

    shifted_fast = shifted_model(1.0)
    shifted_direct = _build(REFERENCE_BASIS, defect_offset_mhz=1.0)
    direct_mode = shaped._prepare_modes(shifted_direct, MODE_CUTOFF)
    fast_mode = shaped._prepare_modes(shifted_fast, MODE_CUTOFF)
    projector_shift_check = {
        "offset_mhz": 1.0,
        "maximum_absolute_eigenenergy_difference_mhz": float(
            np.max(np.abs(shifted_fast.energies_mhz - shifted_direct.energies_mhz))
        ),
        "maximum_absolute_ss_weight_difference": float(
            np.max(
                np.abs(
                    np.abs(shifted_fast.ss_overlap) ** 2 - np.abs(shifted_direct.ss_overlap) ** 2
                )
            )
        ),
        "phase_recalibrated_fidelity_difference": float(
            hardware._fidelity(
                fast_mode,
                pulse,
                lifetimes,
                hardware._correction(fast_mode, pulse, lifetimes),
            )
            - hardware._fidelity(
                direct_mode,
                pulse,
                lifetimes,
                hardware._correction(direct_mode, pulse, lifetimes),
            )
        ),
    }
    assert projector_shift_check["maximum_absolute_eigenenergy_difference_mhz"] < 1e-8
    assert abs(projector_shift_check["phase_recalibrated_fidelity_difference"]) < 1e-10

    print("evaluating bright-mode cutoff convergence", flush=True)
    cutoff_rows = []
    for cutoff in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        cutoff_rows.append(
            {
                "bright_mode_cutoff": cutoff,
                **_gate_metrics(
                    reference_model,
                    pulse,
                    lifetimes,
                    reference_correction,
                    mode_cutoff=cutoff,
                ),
            }
        )

    print("evaluating propagation-step convergence", flush=True)
    step_rows = []
    for step_ns in (1.0, 0.5, 0.25, hardware.FINAL_STEP_NS):
        step_pulse = _pulse(step_ns)
        step_rows.append(
            {
                "maximum_step_ns": step_ns,
                "filtered_target_duration_us": sum(segment.duration_us for segment in step_pulse),
                **_gate_metrics(
                    reference_model,
                    step_pulse,
                    lifetimes,
                    reference_correction,
                ),
            }
        )

    print("re-evaluating the 19-geometry bounded set", flush=True)
    reference_geometries = [
        minimax.Geometry(
            radial_displacement_um=0.0,
            direction_cosine=0.0,
            delta_z_um=0.0,
            transverse_um=0.0,
            distance_um=shaped.R0_UM,
            theta_deg=0.0,
            model=reference_model,
        )
    ]
    for radius_um in minimax.VALIDATION_RADII_UM[1:]:
        reference_geometries.extend(
            minimax._build_geometry(radius_um, cosine, REFERENCE_BASIS)
            for cosine in minimax.VALIDATION_COSINES
        )
    reference_modes = [
        shaped._prepare_modes(geometry.model, MODE_CUTOFF) for geometry in reference_geometries
    ]
    reference_vertex_grid = minimax._scenario_fidelities(
        reference_modes,
        pulse,
        lifetimes,
        reference_correction,
    )
    reference_position_only = np.array(
        [
            hardware._fidelity(
                modes,
                pulse,
                lifetimes,
                reference_correction,
            )
            for modes in reference_modes
        ]
    )
    worst_flat = int(np.argmin(reference_vertex_grid))
    worst_geometry_index, worst_yb_index, worst_rb_index = (
        int(index) for index in np.unravel_index(worst_flat, reference_vertex_grid.shape)
    )
    worst_geometry = reference_geometries[worst_geometry_index]
    reference_bounded_validation = {
        "pulse_reoptimized": False,
        "number_of_geometries": len(reference_geometries),
        "radii_um": list(minimax.VALIDATION_RADII_UM),
        "direction_cosines": list(minimax.VALIDATION_COSINES),
        "independent_yb_and_rb_rabi_scales": list(minimax.AMPLITUDE_VERTICES),
        "nominal_fidelity": float(reference_position_only[0]),
        "position_only_minimum_fidelity": float(np.min(reference_position_only)),
        "position_and_amplitude_vertex_minimum_fidelity": float(np.min(reference_vertex_grid)),
        "worst_case": {
            "geometry_index": worst_geometry_index,
            "radial_displacement_nm": (1000 * worst_geometry.radial_displacement_um),
            "direction_cosine": worst_geometry.direction_cosine,
            "delta_z_nm": 1000 * worst_geometry.delta_z_um,
            "transverse_nm": 1000 * worst_geometry.transverse_um,
            "theta_deg": worst_geometry.theta_deg,
            "yb_rabi_scale": minimax.AMPLITUDE_VERTICES[worst_yb_index],
            "rb_rabi_scale": minimax.AMPLITUDE_VERTICES[worst_rb_index],
            "fidelity": float(
                reference_vertex_grid[worst_geometry_index, worst_yb_index, worst_rb_index]
            ),
        },
        "geometry": [
            {
                "radial_displacement_nm": 1000 * geometry.radial_displacement_um,
                "direction_cosine": geometry.direction_cosine,
                "distance_um": geometry.distance_um,
                "theta_deg": geometry.theta_deg,
                "connected_component_size": geometry.model.symmetry_component_size,
                "position_only_fidelity": float(position_fidelity),
                "amplitude_vertex_minimum_fidelity": float(np.min(reference_vertex_grid[index])),
            }
            for index, (geometry, position_fidelity) in enumerate(
                zip(reference_geometries, reference_position_only, strict=True)
            )
        ],
    }

    # Peper et al. report a 2.3 MHz RMS residual for the fitted S series.  The
    # target-neighboring P-series table entry differs from its fitted value by
    # 3.16 MHz; 3.2 MHz is used below as a rounded deterministic scale.  These
    # are neither independent standard deviations nor a confidence region.
    print("evaluating spectroscopy-motivated sensitivity vertices", flush=True)
    spectroscopy_vertices = []
    for s_shift_mhz in (-2.3, 2.3):
        for p_shift_mhz in (-3.2, 3.2):
            defect_offset_mhz = s_shift_mhz - p_shift_mhz
            shifted_pair_model = shifted_model(defect_offset_mhz)
            tracked_gate = _gate_metrics(shifted_pair_model, pulse, lifetimes, reference_correction)
            fixed_laser_errors = {
                "detuning_offset_mhz": -s_shift_mhz,
            }
            fixed_laser_gate = _gate_metrics(
                shifted_pair_model,
                pulse,
                lifetimes,
                reference_correction,
                **fixed_laser_errors,
            )
            spectroscopy_vertices.append(
                {
                    "yb_ss_energy_shift_mhz": s_shift_mhz,
                    "yb_pp_energy_shift_mhz": p_shift_mhz,
                    "pair_defect_offset_mhz": defect_offset_mhz,
                    "carrier_tracked": tracked_gate,
                    "fixed_yb_laser": fixed_laser_gate,
                }
            )

    print("evaluating pure pair-defect scan", flush=True)
    defect_rows = []
    for offset_mhz in (-5.5, -3.2, -2.0, -1.0, 0.0, 1.0, 2.0, 3.2, 5.5):
        model = shifted_model(offset_mhz)
        defect_rows.append(
            {
                "pair_defect_offset_mhz": offset_mhz,
                **_gate_metrics(model, pulse, lifetimes, reference_correction),
            }
        )

    print("evaluating axial electric-field scan", flush=True)
    electric_rows = []
    for electric_field_v_cm in (
        -0.1,
        -0.05,
        -0.01,
        -0.003,
        -0.001,
        0.0,
        0.001,
        0.003,
        0.01,
        0.05,
        0.1,
    ):
        print(f"  E_z={electric_field_v_cm:g} V/cm", flush=True)
        model = (
            reference_model
            if electric_field_v_cm == 0.0
            else _build(
                REFERENCE_BASIS,
                electric_field_v_cm=electric_field_v_cm,
            )
        )
        carrier_tracked = _gate_metrics(model, pulse, lifetimes, reference_correction)
        fixed_laser = _gate_metrics(
            model,
            pulse,
            lifetimes,
            reference_correction,
            detuning_offset_mhz=-model.yb_ss_carrier_shift_mhz,
            control_detuning_offset_mhz=-model.rb_ss_carrier_shift_mhz,
        )
        electric_rows.append(
            {
                "electric_field_v_cm": electric_field_v_cm,
                "rb_ss_carrier_shift_mhz": model.rb_ss_carrier_shift_mhz,
                "yb_ss_carrier_shift_mhz": model.yb_ss_carrier_shift_mhz,
                "field_dressed_pair_defect_mhz": model.forster_defect_mhz,
                "carrier_tracked": carrier_tracked,
                "fixed_zero_field_lasers": fixed_laser,
            }
        )

    print("evaluating 3 mV/cm pair-window check", flush=True)
    electric_pair_window_model = _build(
        PAIR_WINDOW_CONVERGENCE_BASIS,
        electric_field_v_cm=0.003,
    )
    electric_pair_window_check = {
        "electric_field_v_cm": 0.003,
        "comparison_basis": asdict(PAIR_WINDOW_CONVERGENCE_BASIS),
        **_spectrum_summary(electric_pair_window_model),
        "carrier_tracked": _gate_metrics(
            electric_pair_window_model,
            pulse,
            lifetimes,
            reference_correction,
        ),
        "fixed_zero_field_lasers": _gate_metrics(
            electric_pair_window_model,
            pulse,
            lifetimes,
            reference_correction,
            detuning_offset_mhz=-electric_pair_window_model.yb_ss_carrier_shift_mhz,
            control_detuning_offset_mhz=-electric_pair_window_model.rb_ss_carrier_shift_mhz,
        ),
    }

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "purpose": (
            "post-optimization numerical-convergence and deterministic "
            "atomic-structure sensitivity audit of the stored pulse"
        ),
        "fixed_inputs": {
            "pulse_parameters": minimax.SELECTED_PARAMETERS.tolist(),
            "pulse_reoptimized": False,
            "temperature_k": 0.0,
            "mode_cutoff": MODE_CUTOFF,
            "propagation_step_ns": hardware.FINAL_STEP_NS,
            "field_gauss": shaped.B_GAUSS,
            "distance_um": shaped.R0_UM,
            "theta_deg": 0.0,
            "numerical_reference_basis": asdict(REFERENCE_BASIS),
            "fixed_reference_local_z_alpha_rad": reference_correction[0],
            "fixed_reference_local_z_beta_rad": reference_correction[1],
        },
        "interaction_order_scope": {
            "order_3": "electric dipole-dipole, R^-3",
            "order_4": (
                "cumulative through R^-4: dipole-dipole plus "
                "dipole-quadrupole and quadrupole-dipole"
            ),
            "order_5": (
                "partial R^-5 sensitivity: adds quadrupole-quadrupole but the "
                "PairInteraction implementation does not include "
                "dipole-octupole or octupole-dipole"
            ),
            "source": (
                "PairInteraction SystemPair.cpp setInteractionOrder and "
                "getInteraction implementation"
            ),
        },
        "one_at_a_time_numerical_convergence": convergence_rows,
        "target_projector_shift_equivalence_check": projector_shift_check,
        "bright_mode_cutoff_convergence": cutoff_rows,
        "propagation_step_convergence": step_rows,
        "reference_model_bounded_validation": reference_bounded_validation,
        "spectroscopy_sensitivity_envelope": {
            "source": {
                "citation": "Peper et al., Phys. Rev. X 15, 011009 (2025)",
                "doi": "10.1103/PhysRevX.15.011009",
                "arxiv": "2406.01482v2",
                "yb_s_series_rms_residual_mhz": 2.3,
                "target_neighboring_p_entry_residual_mhz": 3.16,
                "rounded_p_sensitivity_scale_mhz": 3.2,
                "parameter_covariance": (
                    "no covariance matrix is reported in the article or its tabulated supplement"
                ),
            },
            "interpretation": (
                "four signed deterministic vertices used as a sensitivity "
                "envelope, not independent standard deviations, a confidence "
                "interval, or a refit of the PairInteraction v1.4 database"
            ),
            "model_scope": (
                "shifts only the addressed Yb SS and PP energies; coupling "
                "matrix elements, wave functions, all other levels, and Rb "
                "structure remain fixed"
            ),
            "vertices": spectroscopy_vertices,
        },
        "pure_target_pair_defect_scan": {
            "definition": (
                "offset added to E_SS-E_PP by lowering the target-PP projector; "
                "fixed pulse with no wave-function or coupling changes"
            ),
            "points": defect_rows,
        },
        "axial_dc_electric_field_scan": {
            "field_direction": "parallel to quantization and internuclear axes",
            "experimental_stability_context": (
                "Peper et al. report day-to-day compensation variation below "
                "0.003 V/cm; this is context, not a distribution"
            ),
            "fixed_laser_scope": (
                "includes isolated Rb 56S and Yb SS carrier shifts relative to "
                "zero field; ground/metastable-state Stark shifts are omitted"
            ),
            "limitations": (
                "axial fields only; no transverse-field scan or covariance; "
                "large-field rows are stress tests rather than a converged "
                "laboratory robustness guarantee"
            ),
            "points": electric_rows,
            "pair_window_check_at_experimental_scale": electric_pair_window_check,
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
