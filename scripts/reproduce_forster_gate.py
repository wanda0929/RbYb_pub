#!/usr/bin/env python3
"""Reproduce the paper's hardware-aware composite Förster CZ data.

This script writes numerical data only. It includes the selected pulse, the
first-order control-response model, bounded geometry and effective-Rabi
samples, the phase-recalibrated field scan, and nominal no-jump trajectories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy
from forster_model import (
    PRIMARY_BASIS,
    SCAN_BASIS,
    PairModel,
    build_pair_model,
    local_z_metrics,
    query_lifetimes,
    ss_origin_offset_mhz,
)
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from verify_pairinteraction_databases import verify_database_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "forster_gate_results.json"

B_GAUSS = 3.10
R0_UM = 3.40
OMEGA_RB_MHZ = 5.0
RB_PI_DURATION_US = 1 / (2 * OMEGA_RB_MHZ)
AOM_RISE_TIME_NS = 10.0
FILTER_TAIL_CONSTANTS = 7.0
SEARCH_STEP_NS = 1.0
FINAL_STEP_NS = 0.125
MODE_CUTOFF = 1e-6
MAX_RELATIVE_DISPLACEMENT_UM = 0.05
AMPLITUDE_VERTICES = (0.99, 1.01)
VALIDATION_RADII_UM = (0.0, 0.025, 0.05)
VALIDATION_COSINES = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

SELECTED_PARAMETERS = np.array(
    [
        4.410416083986085,
        9.418791104832165,
        11.734642228670836,
        -1.1443728579899664,
        -0.9263577893268974,
        1.744859450470417,
        0.025634115265641792,
    ],
    dtype=float,
)
PARAMETER_BOUNDS = [
    (0.0, 15.0),
    (0.0, 15.0),
    (0.0, 15.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (0.016, 0.026),
]


@dataclass(frozen=True)
class Segment:
    omega_mhz: float
    detuning_mhz: float
    duration_us: float


@dataclass(frozen=True)
class ModeData:
    distance_um: float
    energies_rel_mhz: np.ndarray
    ss_amplitudes: np.ndarray
    pp_weights: np.ndarray
    ss_weights: np.ndarray
    retained_ss_weight: float


@dataclass(frozen=True)
class Geometry:
    radial_displacement_um: float
    direction_cosine: float
    delta_z_um: float
    transverse_um: float
    distance_um: float
    theta_deg: float
    model: PairModel


def command_pulse(parameters: np.ndarray) -> tuple[Segment, ...]:
    omega_a, omega_b, omega_c, det_a, det_b, det_c, duration = parameters
    half = (
        Segment(float(omega_a), float(det_a), float(duration)),
        Segment(float(omega_b), float(det_b), float(duration)),
        Segment(float(omega_c), float(det_c), float(duration)),
    )
    return half + (half[1], half[0])


def filtered_pulse(
    parameters: np.ndarray,
    rise_time_ns: float = AOM_RISE_TIME_NS,
    step_ns: float = FINAL_STEP_NS,
) -> tuple[Segment, ...]:
    """Apply the first-order amplitude and detuning response used in the paper."""
    command = command_pulse(parameters)
    tau_us = rise_time_ns / (1000 * np.log(9))
    intervals = [
        (segment.omega_mhz, segment.detuning_mhz, segment.duration_us) for segment in command
    ]
    intervals.append((0.0, command[-1].detuning_mhz, FILTER_TAIL_CONSTANTS * tau_us))

    amplitude = 0.0
    detuning = command[0].detuning_mhz
    filtered: list[Segment] = []
    for amplitude_command, detuning_command, duration_us in intervals:
        number_steps = int(np.ceil(1000 * duration_us / step_ns))
        dt = duration_us / number_steps
        for _ in range(number_steps):
            ratio = dt / tau_us
            decay = np.exp(-ratio)
            average_factor = (1 - decay) / ratio
            average_amplitude = amplitude_command + (amplitude - amplitude_command) * average_factor
            average_detuning = detuning_command + (detuning - detuning_command) * average_factor
            amplitude = amplitude_command + (amplitude - amplitude_command) * decay
            detuning = detuning_command + (detuning - detuning_command) * decay
            filtered.append(
                Segment(
                    omega_mhz=float(average_amplitude),
                    detuning_mhz=float(average_detuning),
                    duration_us=float(dt),
                )
            )
    return tuple(filtered)


def prepare_modes(model: PairModel, threshold: float = MODE_CUTOFF) -> ModeData:
    mask = np.abs(model.ss_overlap) ** 2 > threshold
    ss_amplitudes = model.ss_overlap[mask]
    pp_weights = np.abs(model.pp_overlap[mask]) ** 2
    ss_weights = np.abs(ss_amplitudes) ** 2
    return ModeData(
        distance_um=model.distance_um,
        energies_rel_mhz=model.energies_mhz[mask] - ss_origin_offset_mhz(model.field_gauss),
        ss_amplitudes=ss_amplitudes,
        pp_weights=pp_weights,
        ss_weights=ss_weights,
        retained_ss_weight=float(np.sum(ss_weights)),
    )


def control_pi(lifetime_us: float | None, amplitude_scale: float = 1.0) -> np.ndarray:
    hamiltonian = np.array(
        [
            [0.0, amplitude_scale * OMEGA_RB_MHZ / 2],
            [amplitude_scale * OMEGA_RB_MHZ / 2, 0.0],
        ],
        dtype=complex,
    )
    if lifetime_us is not None:
        hamiltonian[1, 1] -= 0.5j / (2 * np.pi * lifetime_us)
    return expm(-2j * np.pi * hamiltonian * RB_PI_DURATION_US)


def target_returns(
    modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes,
    amplitude_scale: float = 1.0,
) -> tuple[complex, complex]:
    unblocked = np.array([1.0, 0.0], dtype=complex)
    blocked = np.zeros(len(modes.energies_rel_mhz) + 1, dtype=complex)
    blocked[0] = 1.0

    if lifetimes is None:
        mode_rates = None
    else:
        pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
        ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
        spectator = np.maximum(0.0, 1 - modes.pp_weights - modes.ss_weights)
        mode_rates = (
            modes.pp_weights * pp_rate
            + modes.ss_weights * ss_rate
            + spectator * max(pp_rate, ss_rate)
        )

    for segment in pulse:
        omega = amplitude_scale * segment.omega_mhz
        h_unblocked = np.array(
            [
                [0.0, omega / 2],
                [omega / 2, -segment.detuning_mhz],
            ],
            dtype=complex,
        )
        if lifetimes is not None:
            h_unblocked[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        unblocked = expm(-2j * np.pi * h_unblocked * segment.duration_us) @ unblocked

        h_blocked = np.zeros((len(modes.energies_rel_mhz) + 1,) * 2, dtype=complex)
        h_blocked[0, 1:] = omega * np.conj(modes.ss_amplitudes) / 2
        h_blocked[1:, 0] = omega * modes.ss_amplitudes / 2
        diagonal = modes.energies_rel_mhz - segment.detuning_mhz
        if mode_rates is not None:
            diagonal = diagonal - 0.5j * mode_rates / (2 * np.pi)
        h_blocked[1:, 1:] = np.diag(diagonal)
        blocked = expm_multiply(-2j * np.pi * h_blocked * segment.duration_us, blocked)
    return complex(unblocked[0]), complex(blocked[0])


def kraus_diagonal(
    modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes,
    target_amplitude_scale: float = 1.0,
    control_amplitude_scale: float = 1.0,
) -> np.ndarray:
    unblocked, blocked = target_returns(
        modes, pulse, lifetimes, amplitude_scale=target_amplitude_scale
    )
    duration = sum(segment.duration_us for segment in pulse)
    control_lifetime = None if lifetimes is None else lifetimes.rb_56s_us
    pi_pulse = control_pi(control_lifetime, control_amplitude_scale)
    idle = 1.0 if control_lifetime is None else np.exp(-duration / (2 * control_lifetime))
    control_only = complex((pi_pulse @ np.diag([1.0, idle]) @ pi_pulse)[0, 0])
    both = complex((pi_pulse @ np.diag([unblocked, blocked]) @ pi_pulse)[0, 0])
    return np.array([1.0, unblocked, control_only, both], dtype=complex)


def correction(modes: ModeData, pulse: tuple[Segment, ...], lifetimes) -> tuple[float, float]:
    metrics = local_z_metrics(kraus_diagonal(modes, pulse, lifetimes))
    return (
        float(metrics["optimal_local_z_alpha_rad"]),
        float(metrics["optimal_local_z_beta_rad"]),
    )


def fixed_correction_fidelity(kraus: np.ndarray, phases: tuple[float, float]) -> float:
    alpha, beta = phases
    ideal = np.array([1, 1, 1, -1], dtype=complex)
    local_z = np.array([1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))])
    survival = float(np.vdot(kraus, kraus).real / 4)
    overlap = complex(np.vdot(ideal, local_z * kraus))
    return float((4 * survival + abs(overlap) ** 2) / 20)


def fidelity(
    modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes,
    phases: tuple[float, float],
    target_amplitude_scale: float = 1.0,
    control_amplitude_scale: float = 1.0,
) -> float:
    return fixed_correction_fidelity(
        kraus_diagonal(
            modes,
            pulse,
            lifetimes,
            target_amplitude_scale,
            control_amplitude_scale,
        ),
        phases,
    )


def build_geometry(radius_um: float, cosine: float) -> Geometry:
    delta_z = radius_um * cosine
    transverse = radius_um * np.sqrt(max(0.0, 1 - cosine**2))
    axial = R0_UM + delta_z
    distance = float(np.hypot(axial, transverse))
    theta = float(np.degrees(np.arctan2(transverse, axial)))
    print(
        f"geometry |dr|={1000 * radius_um:.1f} nm, cos={cosine:+.2f}: "
        f"R={distance:.6f} um, theta={theta:.4f} deg",
        flush=True,
    )
    return Geometry(
        radial_displacement_um=float(radius_um),
        direction_cosine=float(cosine),
        delta_z_um=float(delta_z),
        transverse_um=float(transverse),
        distance_um=distance,
        theta_deg=theta,
        model=build_pair_model(B_GAUSS, SCAN_BASIS, distance, theta),
    )


def validation_geometries() -> list[Geometry]:
    geometries = [build_geometry(0.0, 0.0)]
    for radius in VALIDATION_RADII_UM[1:]:
        geometries.extend(build_geometry(radius, cosine) for cosine in VALIDATION_COSINES)
    return geometries


def scenario_kraus(
    modes: list[ModeData],
    pulse: tuple[Segment, ...],
    lifetimes,
    target_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
    control_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
) -> np.ndarray:
    duration = sum(segment.duration_us for segment in pulse)
    control_lifetime = None if lifetimes is None else lifetimes.rb_56s_us
    idle = 1.0 if control_lifetime is None else np.exp(-duration / (2 * control_lifetime))
    control_data = []
    for control_scale in control_scales:
        pi_pulse = control_pi(control_lifetime, control_scale)
        control_only = complex((pi_pulse @ np.diag([1.0, idle]) @ pi_pulse)[0, 0])
        control_data.append((pi_pulse, control_only))

    values = np.empty((len(modes), len(target_scales), len(control_scales), 4), dtype=complex)
    for mode_index, mode in enumerate(modes):
        for target_index, target_scale in enumerate(target_scales):
            unblocked, blocked = target_returns(
                mode, pulse, lifetimes, amplitude_scale=target_scale
            )
            for control_index, (pi_pulse, control_only) in enumerate(control_data):
                both = complex((pi_pulse @ np.diag([unblocked, blocked]) @ pi_pulse)[0, 0])
                values[mode_index, target_index, control_index] = np.array(
                    [1.0, unblocked, control_only, both], dtype=complex
                )
    return values


def fidelities_from_kraus(values: np.ndarray, phases: tuple[float, float]) -> np.ndarray:
    fidelities = np.empty(values.shape[:-1], dtype=float)
    for index in np.ndindex(fidelities.shape):
        fidelities[index] = fixed_correction_fidelity(values[index], phases)
    return fidelities


def scenario_fidelities(
    modes: list[ModeData],
    pulse: tuple[Segment, ...],
    lifetimes,
    phases: tuple[float, float],
    target_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
    control_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
) -> np.ndarray:
    return fidelities_from_kraus(
        scenario_kraus(modes, pulse, lifetimes, target_scales, control_scales), phases
    )


def optimize_pulse(
    seed: np.ndarray,
    modes: list[ModeData],
    nominal_mode: ModeData,
    lifetimes,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    evaluations = 0
    fixed_duration = float(seed[-1])

    def objective(free_parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        parameters = np.r_[free_parameters, fixed_duration]
        pulse = filtered_pulse(parameters)
        phases = correction(nominal_mode, pulse, lifetimes)
        scenario = scenario_fidelities(modes, pulse, lifetimes, phases)
        nominal = fidelity(nominal_mode, pulse, lifetimes, phases)
        infidelities = 1 - np.r_[scenario.ravel(), nominal]
        return float(np.max(infidelities) + 0.02 * np.mean(infidelities))

    result = minimize(
        objective,
        seed[:-1],
        method="Nelder-Mead",
        bounds=PARAMETER_BOUNDS[:-1],
        options={
            "maxiter": max_iterations,
            "xatol": 2e-8,
            "fatol": 2e-11,
            "adaptive": True,
        },
    )
    parameters = np.r_[np.asarray(result.x), fixed_duration]
    return parameters, {
        "performed": True,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": evaluations,
        "objective": float(result.fun),
        "bright_mode_cutoff": MODE_CUTOFF,
        "propagation_step_ns": FINAL_STEP_NS,
        "fixed_segment_duration_us": fixed_duration,
    }


def validate(
    parameters: np.ndarray,
    modes: list[ModeData],
    lifetimes,
) -> tuple[tuple[Segment, ...], tuple[float, float], np.ndarray, dict[str, object]]:
    pulse = filtered_pulse(parameters)
    phases = correction(modes[0], pulse, lifetimes)
    vertex_grid = scenario_fidelities(modes, pulse, lifetimes, phases)
    nominal_curve = np.array([fidelity(mode, pulse, lifetimes, phases) for mode in modes])
    flat_index = int(np.argmin(vertex_grid))
    geometry_index, target_index, control_index = np.unravel_index(flat_index, vertex_grid.shape)
    result = {
        "target_duration_us": float(sum(segment.duration_us for segment in pulse)),
        "total_gate_time_us": float(
            2 * RB_PI_DURATION_US + sum(segment.duration_us for segment in pulse)
        ),
        "nominal_fidelity": float(nominal_curve[0]),
        "position_only_worst_fidelity": float(np.min(nominal_curve)),
        "amplitude_vertex_worst_fidelity": float(np.min(vertex_grid)),
        "worst_case": {
            "geometry_index": int(geometry_index),
            "target_amplitude_scale": AMPLITUDE_VERTICES[target_index],
            "control_amplitude_scale": AMPLITUDE_VERTICES[control_index],
            "fidelity": float(vertex_grid[geometry_index, target_index, control_index]),
        },
        "local_z_alpha_rad": phases[0],
        "local_z_beta_rad": phases[1],
        "vertex_fidelity_grid": vertex_grid.tolist(),
    }
    return pulse, phases, nominal_curve, result


def population_trajectory(
    modes: ModeData,
    pp_amplitudes: np.ndarray,
    pulse: tuple[Segment, ...],
    lifetimes,
) -> dict[str, list[float]]:
    unblocked = np.array([1.0, 0.0], dtype=complex)
    blocked = np.zeros(len(modes.energies_rel_mhz) + 1, dtype=complex)
    blocked[0] = 1.0
    pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
    ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
    spectator = np.maximum(0.0, 1 - modes.pp_weights - modes.ss_weights)
    mode_rates = (
        modes.pp_weights * pp_rate + modes.ss_weights * ss_rate + spectator * max(pp_rate, ss_rate)
    )
    result = {
        "time_us": [0.0],
        "unblocked_yb_rydberg": [0.0],
        "blocked_computational": [1.0],
        "blocked_pp": [0.0],
        "blocked_ss": [0.0],
    }
    for segment in pulse:
        h_unblocked = np.array(
            [
                [0.0, segment.omega_mhz / 2],
                [segment.omega_mhz / 2, -segment.detuning_mhz],
            ],
            dtype=complex,
        )
        h_unblocked[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        unblocked = expm(-2j * np.pi * h_unblocked * segment.duration_us) @ unblocked

        h_blocked = np.zeros((len(modes.energies_rel_mhz) + 1,) * 2, dtype=complex)
        h_blocked[0, 1:] = segment.omega_mhz * np.conj(modes.ss_amplitudes) / 2
        h_blocked[1:, 0] = segment.omega_mhz * modes.ss_amplitudes / 2
        h_blocked[1:, 1:] = np.diag(
            modes.energies_rel_mhz - segment.detuning_mhz - 0.5j * mode_rates / (2 * np.pi)
        )
        blocked = expm_multiply(-2j * np.pi * h_blocked * segment.duration_us, blocked)

        result["time_us"].append(result["time_us"][-1] + segment.duration_us)
        result["unblocked_yb_rydberg"].append(float(abs(unblocked[1]) ** 2))
        result["blocked_computational"].append(float(abs(blocked[0]) ** 2))
        result["blocked_pp"].append(float(abs(np.vdot(pp_amplitudes, blocked[1:])) ** 2))
        result["blocked_ss"].append(float(abs(np.vdot(modes.ss_amplitudes, blocked[1:])) ** 2))
    return result


def quick_check() -> None:
    verify_database_manifest()
    defect = ss_origin_offset_mhz(0.0)
    if abs(defect + 0.763732) > 0.05:
        raise RuntimeError(f"Förster state-identity check failed: defect={defect:.6f} MHz")
    model = build_pair_model(B_GAUSS, SCAN_BASIS)
    modes = prepare_modes(model)
    pulse = filtered_pulse(SELECTED_PARAMETERS)
    phases = correction(modes, pulse, query_lifetimes(0.0))
    nominal = fidelity(modes, pulse, query_lifetimes(0.0), phases)
    if nominal < 0.999:
        raise RuntimeError(f"nominal fidelity smoke check failed: {nominal:.9f}")
    print(f"quick check passed: defect={defect:.6f} MHz, F_avg={nominal:.9f}")


def generate_data(output: Path, do_optimize: bool, max_iterations: int) -> None:
    verify_database_manifest()
    lifetimes = query_lifetimes(0.0)
    geometries = validation_geometries()
    modes = [prepare_modes(geometry.model) for geometry in geometries]
    parameters = SELECTED_PARAMETERS.copy()
    optimization: dict[str, object] = {
        "performed": False,
        "note": "stored pulse evaluated without further refinement",
        "stored_search": {
            "method": "two-sided active-set adaptive Nelder-Mead minimax",
            "iterations": 100,
            "evaluations": 195,
            "bright_mode_cutoff": MODE_CUTOFF,
            "propagation_step_ns": SEARCH_STEP_NS,
            "fixed_segment_duration_us": float(SELECTED_PARAMETERS[-1]),
        },
    }
    if do_optimize:
        active_modes = [
            mode
            for geometry, mode in zip(geometries, modes, strict=True)
            if geometry.radial_displacement_um == MAX_RELATIVE_DISPLACEMENT_UM
            and geometry.direction_cosine in (-1.0, 1.0)
        ]
        parameters, optimization = optimize_pulse(
            parameters, active_modes, modes[0], lifetimes, max_iterations
        )

    pulse, phases, _, selected = validate(parameters, modes, lifetimes)

    endpoint_indices = [
        index
        for index, geometry in enumerate(geometries)
        if geometry.radial_displacement_um == MAX_RELATIVE_DISPLACEMENT_UM
        and abs(geometry.direction_cosine) == 1.0
    ]
    propagation_convergence = []
    for step_ns in (1.0, 0.5, 0.25, FINAL_STEP_NS):
        convergence_pulse = filtered_pulse(parameters, step_ns=step_ns)
        convergence_phases = correction(modes[0], convergence_pulse, lifetimes)
        endpoint_grid = scenario_fidelities(
            [modes[index] for index in endpoint_indices],
            convergence_pulse,
            lifetimes,
            convergence_phases,
        )
        flat_index = int(np.argmin(endpoint_grid))
        endpoint_index, target_index, control_index = np.unravel_index(
            flat_index, endpoint_grid.shape
        )
        geometry = geometries[endpoint_indices[endpoint_index]]
        propagation_convergence.append(
            {
                "step_ns": step_ns,
                "nominal_fidelity": fidelity(
                    modes[0], convergence_pulse, lifetimes, convergence_phases
                ),
                "axial_endpoint_minimum_fidelity": float(
                    endpoint_grid[endpoint_index, target_index, control_index]
                ),
                "worst_endpoint_delta_z_nm": 1000 * geometry.delta_z_um,
                "target_amplitude_scale": AMPLITUDE_VERTICES[target_index],
                "control_amplitude_scale": AMPLITUDE_VERTICES[control_index],
            }
        )

    primary_model = build_pair_model(B_GAUSS, PRIMARY_BASIS)
    primary_mode = prepare_modes(primary_model, threshold=1e-8)
    selected["primary_basis_nominal_fidelity"] = fidelity(primary_mode, pulse, lifetimes, phases)

    response_scan = []
    for rise_time_ns in (2.0, 5.0, 10.0, 15.0, 20.0):
        response_pulse = filtered_pulse(parameters, rise_time_ns)
        response_phases = correction(modes[0], response_pulse, lifetimes)
        response_values = scenario_kraus(modes, response_pulse, lifetimes)
        response_scan.append(
            {
                "rise_time_ns": rise_time_ns,
                "target_duration_us": float(sum(segment.duration_us for segment in response_pulse)),
                "fixed_10ns_correction_nominal_fidelity": fidelity(
                    modes[0], response_pulse, lifetimes, phases
                ),
                "fixed_10ns_correction_vertex_worst_fidelity": float(
                    np.min(fidelities_from_kraus(response_values, phases))
                ),
                "recalibrated_nominal_fidelity": fidelity(
                    modes[0], response_pulse, lifetimes, response_phases
                ),
                "recalibrated_vertex_worst_fidelity": float(
                    np.min(fidelities_from_kraus(response_values, response_phases))
                ),
            }
        )

    axial_displacements_nm = np.linspace(-50.0, 50.0, 11)
    existing_axial_modes = {
        round(1000 * geometry.delta_z_um, 9): mode
        for geometry, mode in zip(geometries, modes, strict=True)
        if abs(geometry.transverse_um) < 1e-12
    }
    axial_modes: list[ModeData] = []
    for displacement_nm in axial_displacements_nm:
        key = round(float(displacement_nm), 9)
        if key not in existing_axial_modes:
            geometry = build_geometry(
                abs(float(displacement_nm)) / 1000, float(np.sign(displacement_nm))
            )
            existing_axial_modes[key] = prepare_modes(geometry.model)
        axial_modes.append(existing_axial_modes[key])
    axial_grid = scenario_fidelities(axial_modes, pulse, lifetimes, phases)
    axial_nominal = np.array([fidelity(mode, pulse, lifetimes, phases) for mode in axial_modes])

    target_errors_pct = np.linspace(-1.0, 1.0, 9)
    target_scales = tuple(float(1 + value / 100) for value in target_errors_pct)
    target_grid = scenario_fidelities(
        modes,
        pulse,
        lifetimes,
        phases,
        target_scales=target_scales,
        control_scales=(0.99, 1.0, 1.01),
    )
    rb_errors_pct = np.linspace(-1.0, 1.0, 9)
    rb_scales = tuple(float(1 + value / 100) for value in rb_errors_pct)
    rb_grid = scenario_fidelities(
        [modes[0]],
        pulse,
        lifetimes,
        phases,
        target_scales=(1.0,),
        control_scales=rb_scales,
    )

    coarse_fields = np.linspace(0.0, 5.0, 11)
    fine_fields = np.linspace(2.8, 3.4, 13)
    field_values = sorted(set(np.round(np.r_[coarse_fields, fine_fields], 12)))
    by_field: dict[float, float] = {}
    for field in field_values:
        field_mode = (
            modes[0]
            if abs(field - B_GAUSS) < 1e-12
            else prepare_modes(build_pair_model(float(field), SCAN_BASIS))
        )
        field_phases = correction(field_mode, pulse, lifetimes)
        by_field[field] = fidelity(field_mode, pulse, lifetimes, field_phases)

    primary_mask = np.abs(primary_model.ss_overlap) ** 2 > 1e-8
    result = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "database_manifest": "provenance/pairinteraction_database_manifest.json",
        "operating_point": {
            "distance_um": R0_UM,
            "theta_deg": 0.0,
            "magnetic_field_gauss": B_GAUSS,
            "rb_effective_rabi_mhz": OMEGA_RB_MHZ,
            "rb_pi_duration_ns": 1000 * RB_PI_DURATION_US,
        },
        "lifetimes_0k_us": asdict(lifetimes),
        "assumptions": {
            "temperature_k": 0.0,
            "relative_position_uncertainty": "bounded three-dimensional ball",
            "maximum_relative_displacement_um": MAX_RELATIVE_DISPLACEMENT_UM,
            "amplitude_error_box": {
                "yb_effective_rabi_scale": list(AMPLITUDE_VERTICES),
                "rb_effective_rabi_scale": list(AMPLITUDE_VERTICES),
            },
            "aom_model": "independent first-order field-amplitude and detuning response",
            "aom_10_to_90_rise_time_ns": AOM_RISE_TIME_NS,
            "rb_excitation": (
                "effective two-level 5 MHz pi pulse; intermediate-state scattering omitted"
            ),
            "decay": "zero-temperature lifetime-weighted no-jump loss mapped to orthogonal leakage",
            "local_z": "one correction calibrated at the nominal point and held fixed",
        },
        "optimization": optimization,
        "parameters": parameters.tolist(),
        "command_segments": [asdict(segment) for segment in command_pulse(parameters)],
        "short_minimax": selected,
        "propagation_convergence": propagation_convergence,
        "response_time_scan": response_scan,
        "axial_position_scan": {
            "delta_z_nm": axial_displacements_nm.tolist(),
            "nominal_amplitudes_fidelity": axial_nominal.tolist(),
            "amplitude_vertex_worst_fidelity": np.min(axial_grid, axis=(1, 2)).tolist(),
        },
        "target_amplitude_scan": {
            "yb_rabi_error_pct": target_errors_pct.tolist(),
            "nominal_position_rb_nominal_fidelity": target_grid[0, :, 1].tolist(),
            "position_and_rb_scale_worst_fidelity": np.min(target_grid, axis=(0, 2)).tolist(),
            "rb_rabi_error_pct": rb_errors_pct.tolist(),
            "nominal_position_yb_nominal_fidelity": rb_grid[0, 0].tolist(),
        },
        "magnetic_field_scan": {
            "coarse_field_gauss": coarse_fields.tolist(),
            "coarse_phase_recalibrated_fidelity": [
                by_field[round(float(field), 12)] for field in coarse_fields
            ],
            "fine_field_gauss": fine_fields.tolist(),
            "fine_phase_recalibrated_fidelity": [
                by_field[round(float(field), 12)] for field in fine_fields
            ],
        },
        "population_trajectory": population_trajectory(
            primary_mode, primary_model.pp_overlap[primary_mask], pulse, lifetimes
        ),
        "validation": {
            "mode_cutoff": MODE_CUTOFF,
            "propagation_step_ns": FINAL_STEP_NS,
            "radii_um": list(VALIDATION_RADII_UM),
            "direction_cosines": list(VALIDATION_COSINES),
            "number_of_geometries": len(geometries),
            "geometry": [
                {
                    "radial_displacement_um": geometry.radial_displacement_um,
                    "direction_cosine": geometry.direction_cosine,
                    "delta_z_um": geometry.delta_z_um,
                    "transverse_um": geometry.transverse_um,
                    "distance_um": geometry.distance_um,
                    "theta_deg": geometry.theta_deg,
                    "basis_component_size": geometry.model.symmetry_component_size,
                }
                for geometry in geometries
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick-check", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--maxiter", type=int, default=400)
    args = parser.parse_args()
    if args.quick_check:
        quick_check()
        return
    generate_data(args.output, args.optimize, args.maxiter)


if __name__ == "__main__":
    main()
