#!/usr/bin/env python3
"""Short-pulse minimax optimization of the hardware-aware Rb--Yb Förster CZ.

The uncertainty set is deliberately bounded rather than probabilistic:

* the relative Rb--Yb displacement obeys |delta r| <= 50 nm in three
  dimensions; and
* the effective Rb and Yb Rabi-frequency scales independently take the
  vertices 0.99 and 1.01.

Only zero-temperature Rydberg lifetimes are used.  Commanded Yb amplitude and
detuning pass through the same conservative 10 ns first-order AOM response as
``optimize_hardware_aware_forster_gate.py``.  The optimization uses a reduced
bright-mode model and is followed by a denser, higher-accuracy minimax check.

Outputs:
  data/forster_gate_results.json
  figures/bounded_minimax_forster_gate.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pairinteraction as pi
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from rb_rydberg_hyperfine import hyperfine_constants_mhz  # noqa: E402
from simulate_forster_gate import (  # noqa: E402
    PRIMARY_BASIS,
    SCAN_BASIS,
    PairModel,
    query_lifetimes,
)
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402

OUT = ROOT / "data" / "forster_gate_results.json"
FIGURE_STEM = ROOT / "figures" / "bounded_minimax_forster_gate"

MAX_RELATIVE_DISPLACEMENT_UM = 0.05
AMPLITUDE_VERTICES = (0.99, 1.01)
# A one-sided active-set refinement moved the limiting point to the opposite
# axial boundary.  The final minimax refinement therefore includes both
# axial extrema and rechecks the full three-dimensional grid below.
OPTIMIZATION_COSINES = (-1.0, 1.0)
VALIDATION_RADII_UM = (0.0, 0.025, 0.05)
VALIDATION_COSINES = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
OPTIMIZATION_MODE_CUTOFF = 1e-6
VALIDATION_MODE_CUTOFF = 1e-6
MINIMAX_STEP_NS = 1.0
SHORT_DURATION_BOUNDS_US = (0.016, 0.026)
SOBOL_VALIDATION_POWER = 5

PARAMETER_BOUNDS = [*shaped.PARAMETER_BOUNDS[:-1], SHORT_DURATION_BOUNDS_US]

# Result of the final two-sided validation-order active-set refinement.  This
# remains the seed if optional further refinement is requested.
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


@dataclass(frozen=True)
class Geometry:
    radial_displacement_um: float
    direction_cosine: float
    delta_z_um: float
    transverse_um: float
    distance_um: float
    theta_deg: float
    model: PairModel


def _build_geometry(
    radial_displacement_um: float,
    direction_cosine: float,
    config,
) -> Geometry:
    delta_z = radial_displacement_um * direction_cosine
    transverse = radial_displacement_um * np.sqrt(max(0.0, 1 - direction_cosine**2))
    axial = shaped.R0_UM + delta_z
    distance = float(np.hypot(axial, transverse))
    theta = float(np.degrees(np.arctan2(transverse, axial)))
    print(
        f"building |dr|={1000 * radial_displacement_um:.1f} nm, "
        f"cos={direction_cosine:+.2f}: R={distance:.6f} um, "
        f"theta={theta:.4f} deg",
        flush=True,
    )
    return Geometry(
        radial_displacement_um=float(radial_displacement_um),
        direction_cosine=float(direction_cosine),
        delta_z_um=float(delta_z),
        transverse_um=float(transverse),
        distance_um=distance,
        theta_deg=theta,
        model=shaped.build_pair_model(
            shaped.B_GAUSS,
            config,
            distance_um=distance,
            theta_deg=theta,
        ),
    )


def _validation_geometries(config) -> list[Geometry]:
    geometries = [_build_geometry(0.0, 0.0, config)]
    for radius in VALIDATION_RADII_UM[1:]:
        geometries.extend(_build_geometry(radius, cosine, config) for cosine in VALIDATION_COSINES)
    return geometries


def _optimization_indices(geometries: list[Geometry]) -> list[int]:
    return [
        index
        for index, geometry in enumerate(geometries)
        if geometry.radial_displacement_um == MAX_RELATIVE_DISPLACEMENT_UM
        and geometry.direction_cosine in OPTIMIZATION_COSINES
    ]


def _correction(modes, pulse, lifetimes) -> tuple[float, float]:
    return hardware._correction(modes, pulse, lifetimes)


def _scenario_kraus(
    modes: list[shaped.ModeData],
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    target_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
    control_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
) -> np.ndarray:
    duration = sum(segment.duration_us for segment in pulse)
    control_lifetime = None if lifetimes is None else lifetimes.rb_56s_us
    idle = 1.0 if control_lifetime is None else np.exp(-duration / (2 * control_lifetime))
    control_data = []
    for control_scale in control_scales:
        control_pi = shaped._control_pi(control_lifetime, control_scale)
        control_only = complex((control_pi @ np.diag([1.0, idle]) @ control_pi)[0, 0])
        control_data.append((control_pi, control_only))

    kraus = np.empty(
        (len(modes), len(target_scales), len(control_scales), 4),
        dtype=complex,
    )
    for mode_index, mode in enumerate(modes):
        for target_index, target_scale in enumerate(target_scales):
            unblocked, blocked = hardware._target_returns(
                mode,
                pulse,
                lifetimes,
                amplitude_scale=target_scale,
            )
            for control_index, (control_pi, control_only) in enumerate(control_data):
                both = complex((control_pi @ np.diag([unblocked, blocked]) @ control_pi)[0, 0])
                kraus[mode_index, target_index, control_index] = np.array(
                    [1.0, unblocked, control_only, both], dtype=complex
                )
    return kraus


def _fidelities_from_kraus(
    kraus: np.ndarray,
    correction: tuple[float, float],
) -> np.ndarray:
    fidelities = np.empty(kraus.shape[:-1], dtype=float)
    for index in np.ndindex(fidelities.shape):
        fidelities[index] = shaped._fixed_correction_fidelity(kraus[index], correction)
    return fidelities


def _scenario_fidelities(
    modes: list[shaped.ModeData],
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    correction: tuple[float, float],
    target_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
    control_scales: tuple[float, ...] = AMPLITUDE_VERTICES,
) -> np.ndarray:
    return _fidelities_from_kraus(
        _scenario_kraus(
            modes,
            pulse,
            lifetimes,
            target_scales=target_scales,
            control_scales=control_scales,
        ),
        correction,
    )


def _optimize(
    seed: np.ndarray,
    modes: list[shaped.ModeData],
    nominal_mode: shaped.ModeData,
    lifetimes,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    evaluations = 0
    fixed_duration = float(seed[-1])

    def objective(free_parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        parameters = np.r_[free_parameters, fixed_duration]
        pulse = hardware._filtered_pulse(
            parameters,
            hardware.AOM_RISE_TIME_NS,
            MINIMAX_STEP_NS,
        )
        correction = _correction(nominal_mode, pulse, lifetimes)
        fidelities = _scenario_fidelities(modes, pulse, lifetimes, correction)
        nominal = hardware._fidelity(nominal_mode, pulse, lifetimes, correction)
        infidelities = 1 - np.r_[fidelities.ravel(), nominal]
        value = float(np.max(infidelities) + 0.02 * np.mean(infidelities))
        if evaluations % 25 == 0:
            print(
                f"evaluation {evaluations}: objective={value:.8g}, "
                f"worst F={1 - np.max(infidelities):.9f}, "
                f"segment={1000 * parameters[-1]:.3f} ns",
                flush=True,
            )
        return value

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
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": evaluations,
        "objective": float(result.fun),
        "bright_mode_cutoff": OPTIMIZATION_MODE_CUTOFF,
        "propagation_step_ns": MINIMAX_STEP_NS,
        "fixed_segment_duration_us": fixed_duration,
    }


def _validate(
    parameters: np.ndarray,
    modes: list[shaped.ModeData],
    nominal_index: int,
    lifetimes,
) -> dict[str, object]:
    pulse = hardware._filtered_pulse(parameters, hardware.AOM_RISE_TIME_NS, hardware.FINAL_STEP_NS)
    correction = _correction(modes[nominal_index], pulse, lifetimes)
    vertex_grid = _scenario_fidelities(modes, pulse, lifetimes, correction)
    nominal_curve = np.array(
        [hardware._fidelity(mode, pulse, lifetimes, correction) for mode in modes]
    )
    worst_flat = int(np.argmin(vertex_grid))
    geometry_index, target_index, control_index = np.unravel_index(worst_flat, vertex_grid.shape)
    return {
        "pulse": pulse,
        "correction": correction,
        "vertex_grid": vertex_grid,
        "nominal_fidelity": float(nominal_curve[nominal_index]),
        "position_only_worst_fidelity": float(np.min(nominal_curve)),
        "vertex_worst_fidelity": float(np.min(vertex_grid)),
        "worst_case": {
            "geometry_index": int(geometry_index),
            "target_amplitude_scale": AMPLITUDE_VERTICES[target_index],
            "control_amplitude_scale": AMPLITUDE_VERTICES[control_index],
            "fidelity": float(vertex_grid[geometry_index, target_index, control_index]),
        },
    }


def _json_validation(validation: dict[str, object]) -> dict[str, object]:
    pulse = validation["pulse"]
    vertex_grid = validation["vertex_grid"]
    return {
        "target_duration_us": float(sum(segment.duration_us for segment in pulse)),
        "total_gate_time_us": float(
            2 * shaped.RB_PI_DURATION_US + sum(segment.duration_us for segment in pulse)
        ),
        "nominal_fidelity": validation["nominal_fidelity"],
        "position_only_worst_fidelity": validation["position_only_worst_fidelity"],
        "amplitude_vertex_worst_fidelity": validation["vertex_worst_fidelity"],
        "worst_case": validation["worst_case"],
        "local_z_alpha_rad": validation["correction"][0],
        "local_z_beta_rad": validation["correction"][1],
        "vertex_fidelity_grid": np.asarray(vertex_grid).tolist(),
    }


def _population_trajectory(
    modes: shaped.ModeData,
    pp_amplitudes: np.ndarray,
    pulse: tuple[shaped.Segment, ...],
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

    time_us = [0.0]
    unblocked_yb_rydberg = [0.0]
    blocked_computational = [1.0]
    blocked_pp = [0.0]
    blocked_ss = [0.0]
    blocked_spectator = [0.0]
    blocked_loss = [0.0]

    for segment in pulse:
        h_unblocked = np.array(
            [
                [0.0, segment.omega_mhz / 2],
                [segment.omega_mhz / 2, -segment.detuning_mhz],
            ],
            dtype=complex,
        )
        h_unblocked[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        unblocked = hardware.expm(-2j * np.pi * h_unblocked * segment.duration_us) @ unblocked

        h_blocked = np.zeros((len(modes.energies_rel_mhz) + 1,) * 2, dtype=complex)
        h_blocked[0, 1:] = segment.omega_mhz * np.conj(modes.ss_amplitudes) / 2
        h_blocked[1:, 0] = segment.omega_mhz * modes.ss_amplitudes / 2
        diagonal = modes.energies_rel_mhz - segment.detuning_mhz - 0.5j * mode_rates / (2 * np.pi)
        h_blocked[1:, 1:] = np.diag(diagonal)
        blocked = hardware.expm_multiply(-2j * np.pi * h_blocked * segment.duration_us, blocked)

        time_us.append(time_us[-1] + segment.duration_us)
        unblocked_yb_rydberg.append(float(abs(unblocked[1]) ** 2))
        blocked_computational.append(float(abs(blocked[0]) ** 2))
        blocked_pp.append(float(abs(np.vdot(pp_amplitudes, blocked[1:])) ** 2))
        blocked_ss.append(float(abs(np.vdot(modes.ss_amplitudes, blocked[1:])) ** 2))
        pair_population = float(np.vdot(blocked[1:], blocked[1:]).real)
        blocked_spectator.append(max(0.0, pair_population - blocked_pp[-1] - blocked_ss[-1]))
        blocked_loss.append(max(0.0, 1 - abs(blocked[0]) ** 2 - pair_population))

    return {
        "time_us": time_us,
        "unblocked_yb_rydberg": unblocked_yb_rydberg,
        "blocked_computational": blocked_computational,
        "blocked_pp": blocked_pp,
        "blocked_ss": blocked_ss,
        "blocked_spectator": blocked_spectator,
        "blocked_loss": blocked_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--maxiter", type=int, default=400)
    args = parser.parse_args()

    verify_database_manifest()
    OUT.parent.mkdir(exist_ok=True)
    FIGURE_STEM.parent.mkdir(exist_ok=True)
    lifetimes = query_lifetimes(0.0)
    geometries = _validation_geometries(SCAN_BASIS)
    nominal_index = 0
    optimization_indices = _optimization_indices(geometries)
    optimization_modes = [
        shaped._prepare_modes(geometries[index].model, OPTIMIZATION_MODE_CUTOFF)
        for index in optimization_indices
    ]
    nominal_optimization_mode = shaped._prepare_modes(
        geometries[nominal_index].model, OPTIMIZATION_MODE_CUTOFF
    )
    print(
        "optimization active modes:",
        [len(mode.energies_rel_mhz) for mode in optimization_modes],
        flush=True,
    )

    parameters = SELECTED_PARAMETERS.copy()
    optimization: dict[str, object] = {
        "performed": False,
        "note": "stored pulse evaluated without further refinement",
        "stored_search": {
            "method": "two-sided active-set adaptive Nelder-Mead minimax",
            "iterations": 100,
            "evaluations": 195,
            "objective": "max(1-F) + 0.02*mean(1-F) over 8 training vertices plus nominal",
            "stochastic_seed": None,
            "multistart_restarts": 0,
            "replay_starting_point": "SELECTED_PARAMETERS stored in this script",
            "bright_mode_cutoff": OPTIMIZATION_MODE_CUTOFF,
            "propagation_step_ns": MINIMAX_STEP_NS,
            "fixed_segment_duration_us": SELECTED_PARAMETERS[-1],
        },
    }
    if args.optimize:
        print(
            f"minimax optimization over {len(optimization_modes)} geometries "
            "and four amplitude vertices...",
            flush=True,
        )
        parameters, optimization = _optimize(
            parameters,
            optimization_modes,
            nominal_optimization_mode,
            lifetimes,
            args.maxiter,
        )
        optimization["performed"] = True
        print("optimized parameters:", parameters.tolist(), flush=True)

    validation_modes = [
        shaped._prepare_modes(geometry.model, VALIDATION_MODE_CUTOFF) for geometry in geometries
    ]
    selected = _validate(parameters, validation_modes, nominal_index, lifetimes)
    baseline = _validate(
        hardware.SELECTED_PARAMETERS,
        validation_modes,
        nominal_index,
        lifetimes,
    )

    endpoint_indices = [
        index
        for index, geometry in enumerate(geometries)
        if geometry.radial_displacement_um == MAX_RELATIVE_DISPLACEMENT_UM
        and abs(geometry.direction_cosine) == 1.0
    ]
    propagation_convergence = []
    for step_ns in (1.0, 0.5, 0.25, hardware.FINAL_STEP_NS):
        convergence_pulse = hardware._filtered_pulse(parameters, hardware.AOM_RISE_TIME_NS, step_ns)
        convergence_correction = _correction(
            validation_modes[nominal_index], convergence_pulse, lifetimes
        )
        endpoint_grid = _scenario_fidelities(
            [validation_modes[index] for index in endpoint_indices],
            convergence_pulse,
            lifetimes,
            convergence_correction,
        )
        endpoint_flat = int(np.argmin(endpoint_grid))
        endpoint_index, target_index, control_index = np.unravel_index(
            endpoint_flat, endpoint_grid.shape
        )
        geometry = geometries[endpoint_indices[endpoint_index]]
        propagation_convergence.append(
            {
                "step_ns": step_ns,
                "nominal_fidelity": hardware._fidelity(
                    validation_modes[nominal_index],
                    convergence_pulse,
                    lifetimes,
                    convergence_correction,
                ),
                "axial_endpoint_minimum_fidelity": float(
                    endpoint_grid[endpoint_index, target_index, control_index]
                ),
                "worst_endpoint_delta_z_nm": 1000 * geometry.delta_z_um,
                "target_amplitude_scale": AMPLITUDE_VERTICES[target_index],
                "control_amplitude_scale": AMPLITUDE_VERTICES[control_index],
            }
        )

    primary_model = shaped.build_pair_model(
        shaped.B_GAUSS,
        PRIMARY_BASIS,
        distance_um=shaped.R0_UM,
        theta_deg=0.0,
    )
    primary_mode = shaped._prepare_modes(primary_model, 1e-8)
    primary_fidelity = hardware._fidelity(
        primary_mode,
        selected["pulse"],
        lifetimes,
        selected["correction"],
    )

    print("evaluating Rb-hyperfine controls...", flush=True)
    electronic_only_model = shaped.build_pair_model(
        shaped.B_GAUSS,
        SCAN_BASIS,
        distance_um=shaped.R0_UM,
        theta_deg=0.0,
        include_rb_hyperfine=False,
    )
    electronic_only_mode = shaped._prepare_modes(electronic_only_model, VALIDATION_MODE_CUTOFF)
    electronic_only_correction = _correction(electronic_only_mode, selected["pulse"], lifetimes)
    # No F-state HFS normalization is available.  Use the largest fitted
    # low-l normalization (D_3/2 A and P_3/2 B) as a deliberately oversized
    # sensitivity test, not as a physical uncertainty bound for F states.
    f_state_stress_prefactors_mhz = (828.0, 143.0)
    f_state_stress_model = shaped.build_pair_model(
        shaped.B_GAUSS,
        SCAN_BASIS,
        distance_um=shaped.R0_UM,
        theta_deg=0.0,
        rb_f_hyperfine_prefactors_mhz=f_state_stress_prefactors_mhz,
    )
    f_state_stress_mode = shaped._prepare_modes(f_state_stress_model, VALIDATION_MODE_CUTOFF)
    f_state_stress_correction = _correction(f_state_stress_mode, selected["pulse"], lifetimes)

    print("evaluating AOM response-time scan...", flush=True)
    response_rise_times_ns = (2.0, 5.0, 10.0, 15.0, 20.0)
    response_scan = []
    for rise_time_ns in response_rise_times_ns:
        response_pulse = hardware._filtered_pulse(parameters, rise_time_ns, hardware.FINAL_STEP_NS)
        response_correction = _correction(
            validation_modes[nominal_index], response_pulse, lifetimes
        )
        response_kraus = _scenario_kraus(validation_modes, response_pulse, lifetimes)
        fixed_fidelities = _fidelities_from_kraus(response_kraus, selected["correction"])
        recalibrated_fidelities = _fidelities_from_kraus(response_kraus, response_correction)
        response_scan.append(
            {
                "rise_time_ns": rise_time_ns,
                "target_duration_us": float(sum(segment.duration_us for segment in response_pulse)),
                "fixed_10ns_correction_nominal_fidelity": hardware._fidelity(
                    validation_modes[nominal_index],
                    response_pulse,
                    lifetimes,
                    selected["correction"],
                ),
                "fixed_10ns_correction_vertex_worst_fidelity": float(np.min(fixed_fidelities)),
                "recalibrated_nominal_fidelity": hardware._fidelity(
                    validation_modes[nominal_index],
                    response_pulse,
                    lifetimes,
                    response_correction,
                ),
                "recalibrated_vertex_worst_fidelity": float(np.min(recalibrated_fidelities)),
            }
        )

    print("evaluating axial position scan...", flush=True)
    axial_displacements_nm = np.linspace(-50.0, 50.0, 11)
    existing_axial_modes = {
        round(1000 * geometry.delta_z_um, 9): mode
        for geometry, mode in zip(geometries, validation_modes, strict=True)
        if abs(geometry.transverse_um) < 1e-12
    }
    axial_modes = []
    for displacement_nm in axial_displacements_nm:
        key = round(float(displacement_nm), 9)
        if key not in existing_axial_modes:
            geometry = _build_geometry(
                abs(displacement_nm) / 1000,
                float(np.sign(displacement_nm)),
                SCAN_BASIS,
            )
            existing_axial_modes[key] = shaped._prepare_modes(
                geometry.model, VALIDATION_MODE_CUTOFF
            )
        axial_modes.append(existing_axial_modes[key])
    axial_vertex_grid = _scenario_fidelities(
        axial_modes,
        selected["pulse"],
        lifetimes,
        selected["correction"],
    )
    axial_nominal_fidelity = np.array(
        [
            hardware._fidelity(
                mode,
                selected["pulse"],
                lifetimes,
                selected["correction"],
            )
            for mode in axial_modes
        ]
    )
    axial_vertex_worst_fidelity = np.min(axial_vertex_grid, axis=(1, 2))

    print("evaluating target-amplitude scan...", flush=True)
    target_amplitude_error_pct = np.linspace(-1.0, 1.0, 9)
    target_amplitude_scales = tuple(
        float(1 + error_pct / 100) for error_pct in target_amplitude_error_pct
    )
    amplitude_scan_grid = _scenario_fidelities(
        validation_modes,
        selected["pulse"],
        lifetimes,
        selected["correction"],
        target_scales=target_amplitude_scales,
        control_scales=(0.99, 1.0, 1.01),
    )
    amplitude_nominal_fidelity = amplitude_scan_grid[nominal_index, :, 1]
    amplitude_worst_fidelity = np.min(amplitude_scan_grid, axis=(0, 2))

    rb_amplitude_error_pct = np.linspace(-1.0, 1.0, 9)
    rb_amplitude_scales = tuple(float(1 + error_pct / 100) for error_pct in rb_amplitude_error_pct)
    rb_amplitude_grid = _scenario_fidelities(
        [validation_modes[nominal_index]],
        selected["pulse"],
        lifetimes,
        selected["correction"],
        target_scales=(1.0,),
        control_scales=rb_amplitude_scales,
    )
    rb_amplitude_nominal_fidelity = rb_amplitude_grid[0, 0]

    print("evaluating deterministic Sobol validation set...", flush=True)
    sobol_points = qmc.Sobol(d=2, scramble=False).random_base2(SOBOL_VALIDATION_POWER)
    sobol_geometries = []
    sobol_modes = []
    for radial_coordinate, angular_coordinate in sobol_points:
        radius_um = MAX_RELATIVE_DISPLACEMENT_UM * radial_coordinate ** (1 / 3)
        direction_cosine = 2 * angular_coordinate - 1
        if radius_um == 0.0:
            sobol_geometries.append(geometries[nominal_index])
            sobol_modes.append(validation_modes[nominal_index])
        else:
            geometry = _build_geometry(float(radius_um), float(direction_cosine), SCAN_BASIS)
            sobol_geometries.append(geometry)
            sobol_modes.append(shaped._prepare_modes(geometry.model, VALIDATION_MODE_CUTOFF))
    sobol_vertex_grid = _scenario_fidelities(
        sobol_modes,
        selected["pulse"],
        lifetimes,
        selected["correction"],
    )
    sobol_worst_flat = int(np.argmin(sobol_vertex_grid))
    sobol_geometry_index, sobol_target_index, sobol_control_index = np.unravel_index(
        sobol_worst_flat, sobol_vertex_grid.shape
    )
    sobol_worst_geometry = sobol_geometries[sobol_geometry_index]

    print("evaluating magnetic-field scan...", flush=True)
    coarse_fields_gauss = np.linspace(0.0, 5.0, 11)
    fine_fields_gauss = np.linspace(2.8, 3.4, 13)
    all_fields_gauss = sorted(set(np.round(np.r_[coarse_fields_gauss, fine_fields_gauss], 12)))
    field_fidelity_by_gauss = {}
    field_fixed_correction_fidelity_by_gauss = {}
    field_static_transfer_by_gauss = {}
    field_spectrum_by_gauss = {}
    for field_gauss in all_fields_gauss:
        if abs(field_gauss - shaped.B_GAUSS) < 1e-12:
            field_model = geometries[nominal_index].model
            field_mode = validation_modes[nominal_index]
        else:
            field_model = shaped.build_pair_model(
                float(field_gauss),
                SCAN_BASIS,
                distance_um=shaped.R0_UM,
                theta_deg=0.0,
            )
            field_mode = shaped._prepare_modes(field_model, VALIDATION_MODE_CUTOFF)
        field_correction = _correction(field_mode, selected["pulse"], lifetimes)
        field_fidelity_by_gauss[field_gauss] = hardware._fidelity(
            field_mode,
            selected["pulse"],
            lifetimes,
            field_correction,
        )
        field_fixed_correction_fidelity_by_gauss[field_gauss] = hardware._fidelity(
            field_mode,
            selected["pulse"],
            lifetimes,
            selected["correction"],
        )
        field_static_transfer_by_gauss[field_gauss] = field_model.static_transfer
        if field_gauss in (0.0, shaped.B_GAUSS):
            field_spectrum_by_gauss[field_gauss] = field_model.spectral_diagnostics
    coarse_field_fidelity = np.array(
        [field_fidelity_by_gauss[round(float(field), 12)] for field in coarse_fields_gauss]
    )
    coarse_field_fixed_correction_fidelity = np.array(
        [
            field_fixed_correction_fidelity_by_gauss[round(float(field), 12)]
            for field in coarse_fields_gauss
        ]
    )
    fine_field_fidelity = np.array(
        [field_fidelity_by_gauss[round(float(field), 12)] for field in fine_fields_gauss]
    )
    fine_field_fixed_correction_fidelity = np.array(
        [
            field_fixed_correction_fidelity_by_gauss[round(float(field), 12)]
            for field in fine_fields_gauss
        ]
    )

    print("evaluating target-pair-defect scan...", flush=True)
    defect_offsets_mhz = (-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0)
    defect_scan = []
    for defect_offset_mhz in defect_offsets_mhz:
        if defect_offset_mhz == 0.0:
            defect_model = geometries[nominal_index].model
            defect_mode = validation_modes[nominal_index]
        else:
            defect_model = shaped.build_pair_model(
                shaped.B_GAUSS,
                SCAN_BASIS,
                distance_um=shaped.R0_UM,
                theta_deg=0.0,
                defect_offset_mhz=defect_offset_mhz,
            )
            defect_mode = shaped._prepare_modes(defect_model, VALIDATION_MODE_CUTOFF)
        defect_correction = _correction(defect_mode, selected["pulse"], lifetimes)
        defect_scan.append(
            {
                "defect_offset_mhz": defect_offset_mhz,
                "fixed_nominal_correction_fidelity": hardware._fidelity(
                    defect_mode,
                    selected["pulse"],
                    lifetimes,
                    selected["correction"],
                ),
                "phase_recalibrated_fidelity": hardware._fidelity(
                    defect_mode,
                    selected["pulse"],
                    lifetimes,
                    defect_correction,
                ),
                "static_transfer": defect_model.static_transfer,
            }
        )

    print("evaluating axial dc-electric-field scan...", flush=True)
    electric_fields_v_cm = (-0.1, -0.05, -0.01, -0.001, 0.0, 0.001, 0.01, 0.05, 0.1)
    electric_field_scan = []
    for electric_field_v_cm in electric_fields_v_cm:
        if electric_field_v_cm == 0.0:
            electric_field_model = geometries[nominal_index].model
            electric_field_mode = validation_modes[nominal_index]
        else:
            electric_field_model = shaped.build_pair_model(
                shaped.B_GAUSS,
                SCAN_BASIS,
                distance_um=shaped.R0_UM,
                theta_deg=0.0,
                electric_field_v_cm=electric_field_v_cm,
            )
            electric_field_mode = shaped._prepare_modes(
                electric_field_model, VALIDATION_MODE_CUTOFF
            )
        electric_field_kraus = hardware._kraus(electric_field_mode, selected["pulse"], lifetimes)
        electric_field_metrics = shaped._local_z_metrics(electric_field_kraus)
        electric_field_correction = (
            float(electric_field_metrics["optimal_local_z_alpha_rad"]),
            float(electric_field_metrics["optimal_local_z_beta_rad"]),
        )
        electric_field_scan.append(
            {
                "electric_field_v_cm": electric_field_v_cm,
                "fixed_nominal_correction_fidelity": (
                    shaped._fixed_correction_fidelity(electric_field_kraus, selected["correction"])
                ),
                "phase_recalibrated_fidelity": shaped._fixed_correction_fidelity(
                    electric_field_kraus, electric_field_correction
                ),
                "static_transfer": electric_field_model.static_transfer,
            }
        )

    nominal_kraus = hardware._kraus(validation_modes[nominal_index], selected["pulse"], lifetimes)
    nominal_mean_survival = float(np.vdot(nominal_kraus, nominal_kraus).real / 4)
    nominal_conditional_overlap = selected["nominal_fidelity"] / nominal_mean_survival
    zero_decay_correction = _correction(validation_modes[nominal_index], selected["pulse"], None)
    zero_decay_kraus = hardware._kraus(validation_modes[nominal_index], selected["pulse"], None)
    zero_decay_mean_return = float(np.vdot(zero_decay_kraus, zero_decay_kraus).real / 4)
    zero_decay_overlap = hardware._fidelity(
        validation_modes[nominal_index],
        selected["pulse"],
        None,
        zero_decay_correction,
    )

    primary_mask = np.abs(primary_model.ss_overlap) ** 2 > 1e-8
    population_trajectory = _population_trajectory(
        primary_mode,
        primary_model.pp_overlap[primary_mask],
        selected["pulse"],
        lifetimes,
    )

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "assumptions": {
            "temperature_k": 0.0,
            "relative_position_uncertainty": "bounded three-dimensional ball",
            "maximum_relative_displacement_um": MAX_RELATIVE_DISPLACEMENT_UM,
            "amplitude_error_box": {
                "yb_effective_rabi_scale": [0.99, 1.01],
                "rb_effective_rabi_scale": [0.99, 1.01],
            },
            "aom_model": "independent first-order field-amplitude and detuning response",
            "aom_10_to_90_rise_time_ns": hardware.AOM_RISE_TIME_NS,
            "rb_excitation": (
                "effective two-level 5 MHz pi pulse; intermediate-state scattering omitted"
            ),
            "decay": (
                "zero-temperature lifetime-weighted non-Hermitian no-jump attenuation; "
                "state-resolved jump branches are not modeled"
            ),
            "local_z": "one correction calibrated at the nominal point and held fixed",
            "rb_initial_state": "|5S1/2,F=2,mF=2> maps to |mJ=1/2,mI=3/2>",
            "rb_hyperfine": (
                "I=3/2; A I.J plus electric-quadrupole term and nuclear Zeeman; "
                "all mI retained in the exact M_tot=5/2 block"
            ),
            "transverse_geometry": (
                "angle-dependent Delta-M=0 dipole tensor retained; Delta-M=+/-1,+/-2 "
                "terms omitted for the <=0.85 degree sampled tilts"
            ),
            "transverse_delta_m_control": {
                "comparison": (
                    "same electronic-only pulse validation with full angular coupling "
                    "versus the Delta-M=0 projection over the 19 stored geometries"
                ),
                "maximum_absolute_fidelity_difference": 4.97e-6,
                "limiting_position_and_joint_vertices_are_axial_theta_zero": True,
            },
        },
        "optimization": optimization,
        "parameters": parameters.tolist(),
        "command_segments": [
            asdict(segment) for segment in shaped._pulse_from_parameters(parameters)
        ],
        "short_minimax": {
            **_json_validation(selected),
            "primary_basis_nominal_fidelity": primary_fidelity,
        },
        "rb_hyperfine_validation": {
            "target_constants_khz": {
                "56S1/2_A": 1000 * hyperfine_constants_mhz(56, 0, 0.5)[0],
                "56P1/2_A": 1000 * hyperfine_constants_mhz(56, 1, 0.5)[0],
            },
            "nominal_scan_basis": {
                "pair_basis_size": geometries[nominal_index].model.pair_basis_size,
                "connected_component_size": geometries[nominal_index].model.symmetry_component_size,
                "static_transfer": geometries[nominal_index].model.static_transfer,
                "spectral_diagnostics": geometries[nominal_index].model.spectral_diagnostics,
                "phase_recalibrated_fidelity": selected["nominal_fidelity"],
            },
            "nominal_primary_basis": {
                "pair_basis_size": primary_model.pair_basis_size,
                "connected_component_size": primary_model.symmetry_component_size,
                "static_transfer": primary_model.static_transfer,
                "spectral_diagnostics": primary_model.spectral_diagnostics,
            },
            "electronic_only_control": {
                "connected_component_size": electronic_only_model.symmetry_component_size,
                "static_transfer": electronic_only_model.static_transfer,
                "phase_recalibrated_fidelity": hardware._fidelity(
                    electronic_only_mode,
                    selected["pulse"],
                    lifetimes,
                    electronic_only_correction,
                ),
                "fidelity_with_hyperfine_correction": hardware._fidelity(
                    electronic_only_mode,
                    selected["pulse"],
                    lifetimes,
                    selected["correction"],
                ),
            },
            "omitted_f_state_stress_test": {
                "assigned_A_B_prefactors_mhz": f_state_stress_prefactors_mhz,
                "note": (
                    "stress test, not a measured F-state normalization; uses the "
                    "largest fitted low-l A and B prefactors"
                ),
                "phase_recalibrated_fidelity": hardware._fidelity(
                    f_state_stress_mode,
                    selected["pulse"],
                    lifetimes,
                    f_state_stress_correction,
                ),
                "fidelity_with_nominal_correction": hardware._fidelity(
                    f_state_stress_mode,
                    selected["pulse"],
                    lifetimes,
                    selected["correction"],
                ),
                "static_transfer": f_state_stress_model.static_transfer,
            },
        },
        "propagation_convergence": propagation_convergence,
        "previous_hardware_aware_pulse": _json_validation(baseline),
        "response_time_scan": response_scan,
        "axial_position_scan": {
            "delta_z_nm": axial_displacements_nm.tolist(),
            "nominal_amplitudes_fidelity": axial_nominal_fidelity.tolist(),
            "amplitude_vertex_worst_fidelity": axial_vertex_worst_fidelity.tolist(),
        },
        "target_amplitude_scan": {
            "yb_rabi_error_pct": target_amplitude_error_pct.tolist(),
            "nominal_position_rb_nominal_fidelity": amplitude_nominal_fidelity.tolist(),
            "position_and_rb_scale_worst_fidelity": amplitude_worst_fidelity.tolist(),
            "rb_rabi_error_pct": rb_amplitude_error_pct.tolist(),
            "nominal_position_yb_nominal_fidelity": rb_amplitude_nominal_fidelity.tolist(),
        },
        "quasi_random_validation": {
            "construction": (
                "32-point deterministic unscrambled Sobol sequence mapped to radius "
                "r=50 nm*u^(1/3) and direction cosine 2v-1; space-filling validation "
                "only, not a thermal probability distribution"
            ),
            "number_of_geometries": len(sobol_geometries),
            "amplitude_scales": list(AMPLITUDE_VERTICES),
            "amplitude_vertex_worst_fidelity": float(np.min(sobol_vertex_grid)),
            "worst_case": {
                "geometry_index": int(sobol_geometry_index),
                "radial_displacement_nm": (1000 * sobol_worst_geometry.radial_displacement_um),
                "direction_cosine": sobol_worst_geometry.direction_cosine,
                "delta_z_nm": 1000 * sobol_worst_geometry.delta_z_um,
                "transverse_nm": 1000 * sobol_worst_geometry.transverse_um,
                "target_amplitude_scale": AMPLITUDE_VERTICES[sobol_target_index],
                "control_amplitude_scale": AMPLITUDE_VERTICES[sobol_control_index],
            },
        },
        "magnetic_field_scan": {
            "coarse_field_gauss": coarse_fields_gauss.tolist(),
            "coarse_fixed_nominal_correction_fidelity": (
                coarse_field_fixed_correction_fidelity.tolist()
            ),
            "coarse_phase_recalibrated_fidelity": coarse_field_fidelity.tolist(),
            "coarse_static_transfer": [
                field_static_transfer_by_gauss[round(float(field), 12)]
                for field in coarse_fields_gauss
            ],
            "fine_field_gauss": fine_fields_gauss.tolist(),
            "fine_fixed_nominal_correction_fidelity": (
                fine_field_fixed_correction_fidelity.tolist()
            ),
            "fine_phase_recalibrated_fidelity": fine_field_fidelity.tolist(),
            "spectral_diagnostics_at_0_and_3p1_gauss": {
                str(field): diagnostics for field, diagnostics in field_spectrum_by_gauss.items()
            },
        },
        "target_pair_defect_scan": {
            "definition": (
                "offset added to E_SS-E_PP by shifting the target PP projector "
                "while holding the optically addressed SS asymptote and pulse fixed"
            ),
            "scope": (
                "effective target-pair sensitivity scan, not correlated atomic-structure "
                "uncertainty propagation"
            ),
            "points": defect_scan,
        },
        "axial_dc_electric_field_scan": {
            "definition": (
                "parallel dc field applied to the finite pair Hamiltonian with the "
                "same pulse command; optical carriers remain defined relative to the "
                "field-shifted isolated SS transition"
            ),
            "scope": (
                "pair-Hamiltonian and fixed-local-Z sensitivity, not a full fixed-laser "
                "laboratory calibration or arbitrary field-direction scan"
            ),
            "points": electric_field_scan,
        },
        "nominal_metric_decomposition": {
            "loss_aware_average_overlap": selected["nominal_fidelity"],
            "mean_computational_survival": nominal_mean_survival,
            "success_weighted_conditional_overlap": nominal_conditional_overlap,
            "conditional_definition": (
                "loss-aware average overlap divided by mean computational survival; "
                "a postselected no-jump diagnostic, not a CPTP channel fidelity"
            ),
            "zero_decay_coherent_return_and_phase_overlap": zero_decay_overlap,
            "zero_decay_mean_computational_return": zero_decay_mean_return,
            "zero_decay_local_z_alpha_rad": zero_decay_correction[0],
            "zero_decay_local_z_beta_rad": zero_decay_correction[1],
        },
        "population_trajectory": population_trajectory,
        "nominal_bright_spectrum": {
            "energy_relative_ss_mhz": primary_mode.energies_rel_mhz.tolist(),
            "ss_weight": primary_mode.ss_weights.tolist(),
            "pp_weight": primary_mode.pp_weights.tolist(),
        },
        "validation": {
            "mode_cutoff": VALIDATION_MODE_CUTOFF,
            "propagation_step_ns": hardware.FINAL_STEP_NS,
            "radii_um": list(VALIDATION_RADII_UM),
            "direction_cosines": list(VALIDATION_COSINES),
            "number_of_geometries": len(geometries),
            "geometry": [
                {
                    **{key: value for key, value in asdict(geometry).items() if key != "model"},
                    "basis_component_size": geometry.model.symmetry_component_size,
                    "angular_delta_m_couplings_omitted": (
                        geometry.model.angular_delta_m_couplings_omitted
                    ),
                }
                for geometry in geometries
            ],
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    command = shaped._pulse_from_parameters(parameters)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    rb_pi_ns = 1000 * shaped.RB_PI_DURATION_US
    target_start_ns = rb_pi_ns
    target_edges_ns = (
        target_start_ns + 1000 * np.r_[0.0, np.cumsum([segment.duration_us for segment in command])]
    )
    target_window_end_ns = target_start_ns + 1000 * sum(
        segment.duration_us for segment in selected["pulse"]
    )
    gate_end_ns = target_window_end_ns + rb_pi_ns

    command_spec = axes[0, 0].get_subplotspec()
    axes[0, 0].remove()
    command_grid = command_spec.subgridspec(2, 1, height_ratios=(0.7, 2.0), hspace=0.06)
    ax_rb = fig.add_subplot(command_grid[0])
    ax = fig.add_subplot(command_grid[1], sharex=ax_rb)

    ax_rb.barh(
        [0.0, 0.0],
        [rb_pi_ns, rb_pi_ns],
        left=[0.0, target_window_end_ns],
        height=0.5,
        color="#D55E00",
    )
    ax_rb.text(rb_pi_ns / 2, 0.0, r"$\pi$", ha="center", va="center", fontsize=9)
    ax_rb.text(
        target_window_end_ns + rb_pi_ns / 2,
        0.0,
        r"$\pi$ reverse",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax_rb.set_ylim(-0.48, 0.48)
    ax_rb.set_yticks([0.0], ["Rb control"])
    ax_rb.set_title("(a) ideal command-pulse sequence")
    ax_rb.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax_rb.tick_params(axis="both", length=0, labelbottom=False)

    command_plot_edges_ns = np.r_[0.0, target_edges_ns, target_window_end_ns, gate_end_ns]
    omega_values = np.r_[0.0, [segment.omega_mhz for segment in command], 0.0, 0.0]
    detuning_values = np.r_[
        0.0,
        [segment.detuning_mhz for segment in command],
        0.0,
        0.0,
    ]
    omega_command = ax.stairs(
        omega_values,
        command_plot_edges_ns,
        color="#0072B2",
        linewidth=1.8,
        label=r"$\Omega(t)$",
    )
    ax_detuning = ax.twinx()
    detuning_command = ax_detuning.stairs(
        detuning_values,
        command_plot_edges_ns,
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label=r"$\Delta(t)$",
    )
    ax.set_xlim(-5.0, gate_end_ns + 5.0)
    ax.set_ylim(0.0, 1.18 * max(omega_values))
    ax_detuning.set_ylim(-1.5, 2.1)
    ax.set_ylabel(r"Yb $\Omega/2\pi$ (MHz)", color="#0072B2")
    ax_detuning.set_ylabel(r"Yb $\Delta/2\pi$ (MHz)", color="#D55E00")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax_detuning.tick_params(axis="y", labelcolor="#D55E00")
    ax.set_xlabel("command-sequence time (ns)")
    ax.legend(
        [omega_command, detuning_command],
        [r"$\Omega(t)$", r"$\Delta(t)$"],
        frameon=False,
        fontsize=8,
        loc="upper left",
        ncol=2,
    )

    robustness_spec = axes[0, 1].get_subplotspec()
    axes[0, 1].remove()
    robustness_grid = robustness_spec.subgridspec(1, 3, wspace=0.18)
    ax_response = fig.add_subplot(robustness_grid[0])
    ax_position = fig.add_subplot(robustness_grid[1], sharey=ax_response)
    ax_amplitude = fig.add_subplot(robustness_grid[2], sharey=ax_response)

    rise_times = np.array([row["rise_time_ns"] for row in response_scan])
    ax_response.plot(
        rise_times,
        100 * np.array([row["recalibrated_nominal_fidelity"] for row in response_scan]),
        "o-",
        color="#009E73",
        markersize=3.5,
    )
    ax_response.set_title("(b) response", fontsize=9)
    ax_response.set_xlabel(r"$t_{10-90}$ (ns)", fontsize=8)
    ax_response.set_ylabel(r"$F_{\rm avg}$ (%)", fontsize=8)

    ax_position.plot(
        axial_displacements_nm,
        100 * axial_nominal_fidelity,
        "o-",
        color="#0072B2",
        markersize=3.5,
    )
    ax_position.set_title("position", fontsize=9)
    ax_position.set_xlabel(r"$\delta z$ (nm)", fontsize=8)
    ax_position.tick_params(axis="y", labelleft=False)

    ax_amplitude.plot(
        target_amplitude_error_pct,
        100 * amplitude_nominal_fidelity,
        "o-",
        color="#CC79A7",
        markersize=3.5,
        label="Yb",
    )
    ax_amplitude.plot(
        rb_amplitude_error_pct,
        100 * rb_amplitude_nominal_fidelity,
        "s--",
        color="#E69F00",
        markersize=3.2,
        label="Rb",
    )
    ax_amplitude.set_title("amplitude", fontsize=9)
    ax_amplitude.set_xlabel(r"$\delta\Omega/\Omega$ (%)", fontsize=8)
    ax_amplitude.tick_params(axis="y", labelleft=False)
    ax_amplitude.legend(frameon=False, fontsize=7, loc="lower center")

    for robustness_axis in (ax_response, ax_position, ax_amplitude):
        robustness_axis.axhline(99.9, color="#666666", linestyle=":", linewidth=0.9)
        robustness_axis.set_ylim(99.88, 99.935)
        robustness_axis.tick_params(axis="both", labelsize=7)

    ax = axes[1, 0]
    ax.plot(
        coarse_fields_gauss,
        100 * coarse_field_fidelity,
        "o-",
        color="#D55E00",
        markersize=3.5,
        linewidth=1.0,
        label="driven gate fidelity",
    )
    ax.axvspan(2.8, 3.4, color="#009E73", alpha=0.10, linewidth=0)
    ax.axvline(
        shaped.B_GAUSS,
        color="#009E73",
        linestyle="--",
        linewidth=0.9,
    )
    field_inset = ax.inset_axes([0.53, 0.10, 0.44, 0.38])
    field_inset.plot(
        fine_fields_gauss,
        100 * fine_field_fidelity,
        "d-",
        color="#009E73",
        markersize=2.7,
        linewidth=0.8,
    )
    field_inset.axvline(
        shaped.B_GAUSS,
        color="#009E73",
        linestyle="--",
        linewidth=0.7,
    )
    field_inset.set_xticks([2.8, 3.0, 3.2, 3.4])
    field_inset.ticklabel_format(axis="y", style="plain", useOffset=False)
    field_inset.xaxis.set_ticks_position("top")
    field_inset.tick_params(axis="both", labelsize=6)
    field_inset.set_ylabel(r"$F_{\rm avg}$ (%)", fontsize=6)
    ax.set_xlabel(r"$B$ (G)")
    ax.set_ylabel(r"$F_{\rm avg}$ (%)")
    ax.set_title("(c) gate fidelity vs magnetic field")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1, 1]
    trajectory_time_us = np.array(population_trajectory["time_us"])
    ax.plot(
        trajectory_time_us,
        100 * np.array(population_trajectory["unblocked_yb_rydberg"]),
        color="0.45",
        linewidth=1.1,
        label=r"$|01\rangle$: Yb Rydberg",
    )
    ax.plot(
        trajectory_time_us,
        100 * np.array(population_trajectory["blocked_computational"]),
        color="#E69F00",
        linewidth=1.3,
        label=r"$|11\rangle$: computational",
    )
    ax.plot(
        trajectory_time_us,
        100 * np.array(population_trajectory["blocked_pp"]),
        color="#D55E00",
        linewidth=1.1,
        label=r"$|11\rangle$: $|PP\rangle$",
    )
    ax.plot(
        trajectory_time_us,
        100 * np.array(population_trajectory["blocked_ss"]),
        color="#0072B2",
        linestyle="--",
        linewidth=1.1,
        label=r"$|11\rangle$: $|SS\rangle$",
    )
    ax.set_xlabel(r"$t$ during filtered Yb target pulse ($\mu$s)")
    ax.set_ylabel("population (%)")
    ax.set_ylim(-3.0, 105.0)
    ax.set_title("(d) population transfer")
    ax.legend(frameon=False, fontsize=6.5, loc="center right")

    fig.tight_layout(h_pad=4.0)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print("wrote", OUT, flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".pdf"), flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".png"), flush=True)
    print(
        f"short minimax: nominal={selected['nominal_fidelity']:.9f}, "
        f"position worst={selected['position_only_worst_fidelity']:.9f}, "
        f"vertex worst={selected['vertex_worst_fidelity']:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
