#!/usr/bin/env python3
"""Reoptimize the five-segment Förster CZ on the final P0-4 reference model.

Optimization constructs only the nominal and axial +/-50 nm PairInteraction
models.  Their bright modes are cached before either objective is evaluated.
After optimization, the baseline and selected candidate are independently
validated on the full 19-geometry bounded set at 0.125 ns resolution.

Output:
  data/forster_p0_4_reoptimization.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pairinteraction as pi
import scipy
from pairinteraction._backend import get_cache_directory
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_forster_p0_4 as p0_4  # noqa: E402
import reproduce_forster_gate as minimax  # noqa: E402
import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from simulate_forster_gate import query_lifetimes  # noqa: E402
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402


OUT = ROOT / "data" / "forster_p0_4_reoptimization.json"
MODE_CUTOFF = 1e-6
OPTIMIZATION_STEP_NS = 1.0
FINAL_STEP_NS = 0.125
OBJECTIVE_MEAN_WEIGHT = 0.02
NOMINAL_TARGET = 0.999
ACCEPTANCE_ATOL = 1e-12
OPTIMIZATION_SPECS = ((0.0, 0.0), (0.05, -1.0), (0.05, 1.0))
# Keep the historical P0-4 reoptimization seed literal here.  In particular,
# do not source it from minimax.SELECTED_PARAMETERS: that constant is the
# accepted downstream pulse and can change without altering replay semantics.
REOPTIMIZATION_SEED_PARAMETERS = np.array(
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
EXPECTED_BASELINE_NOMINAL = 0.9989445403730188
EXPECTED_BASELINE_VERTEX_MINIMUM = 0.9980009924062412
DATABASE_ASSETS = {
    "Rb": "Rb_v1.2",
    "Yb171_mqdt": "Yb171_mqdt_v1.4",
    "misc": "misc_v1.4",
}


@dataclass(frozen=True)
class PreparedGeometry:
    geometry: minimax.Geometry
    modes: shaped.ModeData


def validation_specs() -> tuple[tuple[float, float], ...]:
    """Return nominal plus the existing 25/50 nm, nine-cosine grid."""

    return ((0.0, 0.0),) + tuple(
        (radius, cosine)
        for radius in minimax.VALIDATION_RADII_UM[1:]
        for cosine in minimax.VALIDATION_COSINES
    )


def robust_objective(
    endpoint_vertex_fidelities: np.ndarray,
    nominal_fidelity: float,
) -> float:
    """Minimax-plus-mean objective over eight vertices and nominal."""

    endpoint = np.asarray(endpoint_vertex_fidelities, dtype=float)
    if endpoint.size != 8:
        raise ValueError("the training objective requires exactly eight endpoint vertices")
    infidelities = 1 - np.r_[endpoint.ravel(), float(nominal_fidelity)]
    return float(
        np.max(infidelities) + OBJECTIVE_MEAN_WEIGHT * np.mean(infidelities)
    )


def acceptance_checks(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    include_full_grid: bool,
    atol: float = ACCEPTANCE_ATOL,
) -> dict[str, bool]:
    """Apply the predeclared nominal and robustness acceptance gates."""

    checks = {
        "nominal_at_least_0_999": (
            candidate["nominal_fidelity"] >= NOMINAL_TARGET - atol
        ),
        "endpoint_objective_no_worse_than_baseline": (
            candidate["endpoint_objective"] <= baseline["endpoint_objective"] + atol
        ),
        "endpoint_vertex_minimum_no_worse_than_baseline": (
            candidate["endpoint_vertex_minimum"]
            >= baseline["endpoint_vertex_minimum"] - atol
        ),
    }
    if include_full_grid:
        checks["full_grid_vertex_minimum_no_worse_than_baseline"] = (
            candidate["full_grid_vertex_minimum"]
            >= baseline["full_grid_vertex_minimum"] - atol
        )
    checks["accepted"] = all(checks.values())
    return checks


def _key(spec: tuple[float, float]) -> tuple[float, float]:
    return tuple(round(float(value), 12) for value in spec)


def _prepare_geometry(spec: tuple[float, float]) -> PreparedGeometry:
    geometry = minimax._build_geometry(*spec, p0_4.REFERENCE_BASIS)
    return PreparedGeometry(
        geometry=geometry,
        modes=shaped._prepare_modes(geometry.model, MODE_CUTOFF),
    )


def _get_prepared(
    specs: Sequence[tuple[float, float]],
    cache: dict[tuple[float, float], PreparedGeometry],
) -> tuple[list[PreparedGeometry], int]:
    prepared = []
    builds = 0
    for spec in specs:
        key = _key(spec)
        if key not in cache:
            cache[key] = _prepare_geometry(spec)
            builds += 1
        prepared.append(cache[key])
    return prepared, builds


def _pulse(parameters: np.ndarray, step_ns: float) -> tuple[shaped.Segment, ...]:
    return hardware._filtered_pulse(
        parameters,
        hardware.AOM_RISE_TIME_NS,
        step_ns,
    )


def _training_metrics(
    parameters: np.ndarray,
    optimization_geometries: Sequence[PreparedGeometry],
    lifetimes,
    step_ns: float,
) -> dict[str, object]:
    pulse = _pulse(parameters, step_ns)
    nominal_mode = optimization_geometries[0].modes
    endpoint_modes = [item.modes for item in optimization_geometries[1:]]
    correction = hardware._correction(nominal_mode, pulse, lifetimes)
    endpoint_grid = minimax._scenario_fidelities(
        endpoint_modes,
        pulse,
        lifetimes,
        correction,
    )
    nominal = hardware._fidelity(nominal_mode, pulse, lifetimes, correction)
    return {
        "nominal_fidelity": float(nominal),
        "endpoint_vertex_minimum": float(np.min(endpoint_grid)),
        "endpoint_objective": robust_objective(endpoint_grid, nominal),
        "local_z_alpha_rad": correction[0],
        "local_z_beta_rad": correction[1],
        "endpoint_vertex_fidelities": endpoint_grid.tolist(),
    }


def _optimize_stage(
    seed: np.ndarray,
    optimization_geometries: Sequence[PreparedGeometry],
    lifetimes,
    *,
    vary_duration: bool,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    evaluations = 0
    fixed_duration = float(seed[-1])
    free_seed = seed if vary_duration else seed[:-1]
    bounds = minimax.PARAMETER_BOUNDS if vary_duration else minimax.PARAMETER_BOUNDS[:-1]
    initial_metrics = _training_metrics(
        seed, optimization_geometries, lifetimes, OPTIMIZATION_STEP_NS
    )

    def objective(free_parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        parameters = (
            np.asarray(free_parameters, dtype=float)
            if vary_duration
            else np.r_[free_parameters, fixed_duration]
        )
        metrics = _training_metrics(
            parameters,
            optimization_geometries,
            lifetimes,
            OPTIMIZATION_STEP_NS,
        )
        if evaluations % 25 == 0:
            print(
                f"  evaluation {evaluations}: "
                f"objective={metrics['endpoint_objective']:.9g}, "
                f"nominal={metrics['nominal_fidelity']:.9f}, "
                f"endpoint min={metrics['endpoint_vertex_minimum']:.9f}",
                flush=True,
            )
        return float(metrics["endpoint_objective"])

    stage_start = time.perf_counter()
    result = minimize(
        objective,
        free_seed,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": max_iterations,
            "xatol": 2e-8,
            "fatol": 2e-11,
            "adaptive": True,
        },
    )
    wall_seconds = time.perf_counter() - stage_start
    parameters = (
        np.asarray(result.x, dtype=float)
        if vary_duration
        else np.r_[result.x, fixed_duration]
    )
    final_metrics = _training_metrics(
        parameters, optimization_geometries, lifetimes, OPTIMIZATION_STEP_NS
    )
    return parameters, {
        "method": "bounded adaptive Nelder-Mead",
        "varied_parameters": (
            "six amplitudes/detunings plus common segment duration"
            if vary_duration
            else "six amplitudes/detunings at fixed common segment duration"
        ),
        "bounds": [list(bound) for bound in bounds],
        "maximum_iterations": max_iterations,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": evaluations,
        "scipy_reported_evaluations": int(result.nfev),
        "wall_seconds": wall_seconds,
        "seed_parameters": seed.tolist(),
        "result_parameters": parameters.tolist(),
        "initial_1ns_metrics": initial_metrics,
        "final_1ns_metrics": final_metrics,
    }


def _geometry_json(item: PreparedGeometry) -> dict[str, object]:
    geometry = item.geometry
    return {
        "radial_displacement_nm": 1000 * geometry.radial_displacement_um,
        "direction_cosine": geometry.direction_cosine,
        "delta_z_nm": 1000 * geometry.delta_z_um,
        "transverse_nm": 1000 * geometry.transverse_um,
        "distance_um": geometry.distance_um,
        "theta_deg": geometry.theta_deg,
        "pair_basis_size": geometry.model.pair_basis_size,
        "connected_component_size": geometry.model.symmetry_component_size,
        "angular_delta_m_couplings_omitted": (
            geometry.model.angular_delta_m_couplings_omitted
        ),
        "active_modes": len(item.modes.energies_rel_mhz),
        "retained_ss_weight": item.modes.retained_ss_weight,
    }


def _validation_summary(
    parameters: np.ndarray,
    validation_geometries: Sequence[PreparedGeometry],
    lifetimes,
) -> dict[str, object]:
    pulse = _pulse(parameters, FINAL_STEP_NS)
    modes = [item.modes for item in validation_geometries]
    geometry_metadata = [_geometry_json(item) for item in validation_geometries]
    correction = hardware._correction(modes[0], pulse, lifetimes)
    vertex_grid = minimax._scenario_fidelities(
        modes, pulse, lifetimes, correction
    )
    position_only = np.array(
        [hardware._fidelity(mode, pulse, lifetimes, correction) for mode in modes]
    )
    nominal = float(position_only[0])

    endpoint_indices = [
        index
        for index, item in enumerate(validation_geometries)
        if item.geometry.radial_displacement_um
        == minimax.MAX_RELATIVE_DISPLACEMENT_UM
        and abs(item.geometry.direction_cosine) == 1.0
    ]
    endpoint_grid = vertex_grid[endpoint_indices]

    vertex_flat = int(np.argmin(vertex_grid))
    geometry_index, yb_index, rb_index = (
        int(value) for value in np.unravel_index(vertex_flat, vertex_grid.shape)
    )
    position_index = int(np.argmin(position_only))

    nominal_kraus = hardware._kraus(modes[0], pulse, lifetimes)
    nominal_metrics = shaped._local_z_metrics(nominal_kraus)
    mean_survival = float(nominal_metrics["mean_computational_survival"])
    zero_decay_correction = hardware._correction(modes[0], pulse, None)
    zero_decay_kraus = hardware._kraus(modes[0], pulse, None)
    zero_decay_mean_return = float(np.vdot(zero_decay_kraus, zero_decay_kraus).real / 4)
    zero_decay_overlap = hardware._fidelity(
        modes[0], pulse, None, zero_decay_correction
    )

    return {
        "pulse_parameters": parameters.tolist(),
        "command_segments": [
            asdict(segment) for segment in shaped._pulse_from_parameters(parameters)
        ],
        "filtered_target_duration_us": float(
            sum(segment.duration_us for segment in pulse)
        ),
        "total_gate_time_us": float(
            2 * shaped.RB_PI_DURATION_US
            + sum(segment.duration_us for segment in pulse)
        ),
        "local_z_alpha_rad": correction[0],
        "local_z_beta_rad": correction[1],
        "nominal_fidelity": nominal,
        "mean_computational_survival": mean_survival,
        "success_weighted_conditional_no_jump_overlap": nominal / mean_survival,
        "conditional_phase_error_rad": nominal_metrics[
            "conditional_phase_error_rad"
        ],
        "position_only_minimum_fidelity": float(np.min(position_only)),
        "full_grid_vertex_minimum": float(np.min(vertex_grid)),
        "endpoint_vertex_minimum": float(np.min(endpoint_grid)),
        "endpoint_objective": robust_objective(endpoint_grid, nominal),
        "position_only_worst_case": {
            **geometry_metadata[position_index],
            "fidelity": float(position_only[position_index]),
        },
        "joint_vertex_worst_case": {
            **geometry_metadata[geometry_index],
            "yb_rabi_scale": minimax.AMPLITUDE_VERTICES[yb_index],
            "rb_rabi_scale": minimax.AMPLITUDE_VERTICES[rb_index],
            "fidelity": float(vertex_grid[geometry_index, yb_index, rb_index]),
        },
        "zero_decay_diagnostic": {
            "coherent_return_and_phase_overlap": zero_decay_overlap,
            "mean_computational_return": zero_decay_mean_return,
            "local_z_alpha_rad": zero_decay_correction[0],
            "local_z_beta_rad": zero_decay_correction[1],
        },
        "geometry": [
            {
                **geometry_metadata[index],
                "position_only_fidelity": float(position_only[index]),
                "vertex_minimum_fidelity": float(np.min(vertex_grid[index])),
                "vertex_fidelities": vertex_grid[index].tolist(),
            }
            for index in range(len(validation_geometries))
        ],
    }


def _convergence_summary(
    parameters: np.ndarray,
    optimization_geometries: Sequence[PreparedGeometry],
    lifetimes,
) -> list[dict[str, object]]:
    rows = []
    for step_ns in (1.0, 0.5, 0.25, FINAL_STEP_NS):
        metrics = _training_metrics(
            parameters, optimization_geometries, lifetimes, step_ns
        )
        endpoint_grid = np.asarray(metrics.pop("endpoint_vertex_fidelities"))
        rows.append(
            {
                "maximum_step_ns": step_ns,
                **metrics,
                "axial_minus_50nm_vertex_minimum": float(
                    np.min(endpoint_grid[0])
                ),
                "axial_plus_50nm_vertex_minimum": float(
                    np.min(endpoint_grid[1])
                ),
                "endpoint_vertex_fidelities": endpoint_grid.tolist(),
            }
        )
    return rows


def _optimization_metric_view(summary: dict[str, object]) -> dict[str, float]:
    return {
        key: float(summary[key])
        for key in (
            "nominal_fidelity",
            "endpoint_objective",
            "endpoint_vertex_minimum",
        )
    }


def _full_metric_view(summary: dict[str, object]) -> dict[str, float]:
    return {
        **_optimization_metric_view(summary),
        "full_grid_vertex_minimum": float(summary["full_grid_vertex_minimum"]),
    }


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_database_assets() -> None:
    tables = get_cache_directory() / "database" / "tables"
    mismatches = []
    for species, expected in DATABASE_ASSETS.items():
        available = sorted(path.name for path in tables.glob(f"{species}_v*"))
        if available != [expected]:
            mismatches.append(
                f"{species}: expected only {expected}, found {available or 'none'}"
            )
    if mismatches:
        raise RuntimeError(
            "PairInteraction database mismatch; " + "; ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-maxiter", type=int, default=400)
    parser.add_argument("--stage2-maxiter", type=int, default=400)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage1_maxiter <= 0 or args.stage2_maxiter <= 0:
        parser.error("optimizer iteration limits must be positive")

    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    verify_database_manifest()
    _validate_database_assets()
    lifetimes = query_lifetimes(0.0)
    cache: dict[tuple[float, float], PreparedGeometry] = {}

    print("building and caching the three final-model optimization geometries", flush=True)
    model_start = time.perf_counter()
    optimization_geometries, optimization_builds = _get_prepared(
        OPTIMIZATION_SPECS, cache
    )
    optimization_model_wall_seconds = time.perf_counter() - model_start
    print(
        "optimization active modes:",
        [len(item.modes.energies_rel_mhz) for item in optimization_geometries],
        flush=True,
    )

    baseline_parameters = REOPTIMIZATION_SEED_PARAMETERS.copy()
    baseline_endpoint = _training_metrics(
        baseline_parameters,
        optimization_geometries,
        lifetimes,
        FINAL_STEP_NS,
    )

    print("stage 1: refining six amplitudes/detunings at fixed duration", flush=True)
    stage1_parameters, stage1 = _optimize_stage(
        baseline_parameters,
        optimization_geometries,
        lifetimes,
        vary_duration=False,
        max_iterations=args.stage1_maxiter,
    )
    stage1_exact = _training_metrics(
        stage1_parameters,
        optimization_geometries,
        lifetimes,
        FINAL_STEP_NS,
    )
    stage1["exact_0_125ns_metrics"] = stage1_exact
    stage1["exact_acceptance"] = acceptance_checks(
        _optimization_metric_view(baseline_endpoint),
        _optimization_metric_view(stage1_exact),
        include_full_grid=False,
    )

    stage2 = None
    candidate_parameters = stage1_parameters
    stage2_triggered = float(stage1_exact["nominal_fidelity"]) < NOMINAL_TARGET
    if stage2_triggered:
        print(
            "stage 1 exact nominal is below 0.999; stage 2 also varies duration",
            flush=True,
        )
        stage2_parameters, stage2 = _optimize_stage(
            stage1_parameters,
            optimization_geometries,
            lifetimes,
            vary_duration=True,
            max_iterations=args.stage2_maxiter,
        )
        stage2_exact = _training_metrics(
            stage2_parameters,
            optimization_geometries,
            lifetimes,
            FINAL_STEP_NS,
        )
        stage2["exact_0_125ns_metrics"] = stage2_exact
        stage2["exact_acceptance"] = acceptance_checks(
            _optimization_metric_view(baseline_endpoint),
            _optimization_metric_view(stage2_exact),
            include_full_grid=False,
        )
        candidate_parameters = stage2_parameters
    else:
        print("stage 1 crossed exact nominal 0.999; stage 2 not run", flush=True)

    print("building the remaining final-model validation geometries", flush=True)
    validation_model_start = time.perf_counter()
    validation_geometries, validation_builds = _get_prepared(
        validation_specs(), cache
    )
    validation_model_wall_seconds = time.perf_counter() - validation_model_start

    print("running exact 0.125 ns baseline and candidate validation", flush=True)
    baseline = _validation_summary(
        baseline_parameters, validation_geometries, lifetimes
    )
    candidate = _validation_summary(
        candidate_parameters, validation_geometries, lifetimes
    )
    final_acceptance = acceptance_checks(
        _full_metric_view(baseline),
        _full_metric_view(candidate),
        include_full_grid=True,
    )

    print("running propagation convergence checks", flush=True)
    convergence = {
        "baseline": _convergence_summary(
            baseline_parameters, optimization_geometries, lifetimes
        ),
        "candidate": _convergence_summary(
            candidate_parameters, optimization_geometries, lifetimes
        ),
    }

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": _git_revision(),
        "purpose": (
            "exploratory local reoptimization of the existing five-segment pulse "
            "against the final P0-4 numerical-reference model"
        ),
        "software": {
            "python": platform.python_version(),
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pairinteraction_database_assets": DATABASE_ASSETS,
        },
        "model": {
            "basis": asdict(p0_4.REFERENCE_BASIS),
            "temperature_k": 0.0,
            "field_gauss": shaped.B_GAUSS,
            "nominal_distance_um": shaped.R0_UM,
            "mode_cutoff": MODE_CUTOFF,
            "optimization_geometry_specs": [list(spec) for spec in OPTIMIZATION_SPECS],
            "validation_radii_um": list(minimax.VALIDATION_RADII_UM),
            "validation_direction_cosines": list(minimax.VALIDATION_COSINES),
            "independent_yb_and_rb_rabi_scales": list(
                minimax.AMPLITUDE_VERTICES
            ),
            "active_mode_counts": [
                len(item.modes.energies_rel_mhz) for item in validation_geometries
            ],
            "geometry_metadata": [
                _geometry_json(item) for item in validation_geometries
            ],
        },
        "metric_definition": {
            "reported_fidelity": (
                "loss-aware average computational-basis CZ overlap from the "
                "non-Hermitian no-jump return diagonal, with lost norm counted "
                "as orthogonal erasure"
            ),
            "local_z": (
                "one nominal correction calibrated separately for each pulse and "
                "propagation resolution, then fixed over its geometry/amplitude set"
            ),
            "objective": (
                "max(1-F) + 0.02*mean(1-F) over the eight axial-endpoint/"
                "independent-amplitude vertices plus nominal"
            ),
            "conditional_no_jump_diagnostic": (
                "loss-aware overlap divided by mean computational survival; "
                "a postselected diagnostic, not a CPTP process fidelity"
            ),
            "cptp_scope": (
                "state-resolved jump branches are not modeled, so neither the "
                "loss-aware overlap nor the conditional diagnostic is a "
                "reconstructed CPTP process fidelity"
            ),
        },
        "claim_limitations": {
            "task_scope": "theory-paper numerical reoptimization, not experimental validation",
            "decay": (
                "zero-temperature lifetime-weighted non-Hermitian no-jump "
                "attenuation; state-resolved jump destinations omitted"
            ),
            "rb_excitation": (
                "effective two-level 5 MHz pi pulse; intermediate-state "
                "scattering omitted"
            ),
            "position_domain": "bounded static relative displacement ball of radius 50 nm",
            "amplitude_domain": "only the four independent +/-1% Rabi vertices",
            "transverse_geometry": (
                "angle-dependent Delta-M=0 tensor term retained; Delta-M=+/-1,+/-2 "
                "terms omitted for the sampled sub-degree tilts, as in the existing model"
            ),
            "exploratory_status": (
                "deterministic accepted local candidate; the optimizer exhausted its "
                "budget and did not report convergence; no global optimum is claimed"
            ),
        },
        "optimization": {
            "seed": "optimize_forster_p0_4_reference.REOPTIMIZATION_SEED_PARAMETERS",
            "seed_parameters": baseline_parameters.tolist(),
            "propagation_step_ns": OPTIMIZATION_STEP_NS,
            "models_built_before_objective_evaluation": optimization_builds,
            "pairinteraction_rebuilds_inside_objective": 0,
            "optimization_model_build_wall_seconds": optimization_model_wall_seconds,
            "stage_1": stage1,
            "stage_2_triggered": stage2_triggered,
            "stage_2_trigger_condition": (
                "stage-1 exact 0.125 ns nominal fidelity < 0.999"
            ),
            "stage_2": stage2,
            "selected_stage": 2 if stage2_triggered else 1,
            "stopping_condition": (
                "stage 2 reached its recorded optimizer termination after stage 1 "
                "missed exact nominal 0.999"
                if stage2_triggered
                else "stage 1 reached its recorded optimizer termination; exact "
                "validation then crossed nominal 0.999, so stage 2 was not triggered"
            ),
        },
        "baseline_expected_reproduction": {
            "expected_nominal_fidelity": EXPECTED_BASELINE_NOMINAL,
            "actual_nominal_fidelity": baseline["nominal_fidelity"],
            "nominal_difference": (
                float(baseline["nominal_fidelity"]) - EXPECTED_BASELINE_NOMINAL
            ),
            "expected_full_grid_vertex_minimum": (
                EXPECTED_BASELINE_VERTEX_MINIMUM
            ),
            "actual_full_grid_vertex_minimum": baseline[
                "full_grid_vertex_minimum"
            ],
            "vertex_minimum_difference": (
                float(baseline["full_grid_vertex_minimum"])
                - EXPECTED_BASELINE_VERTEX_MINIMUM
            ),
        },
        "exact_0_125ns_validation": {
            "number_of_geometries": len(validation_geometries),
            "baseline": baseline,
            "candidate": candidate,
            "candidate_minus_baseline": {
                "nominal_fidelity": (
                    float(candidate["nominal_fidelity"])
                    - float(baseline["nominal_fidelity"])
                ),
                "mean_computational_survival": (
                    float(candidate["mean_computational_survival"])
                    - float(baseline["mean_computational_survival"])
                ),
                "position_only_minimum_fidelity": (
                    float(candidate["position_only_minimum_fidelity"])
                    - float(baseline["position_only_minimum_fidelity"])
                ),
                "full_grid_vertex_minimum": (
                    float(candidate["full_grid_vertex_minimum"])
                    - float(baseline["full_grid_vertex_minimum"])
                ),
                "endpoint_objective": (
                    float(candidate["endpoint_objective"])
                    - float(baseline["endpoint_objective"])
                ),
            },
            "acceptance": final_acceptance,
        },
        "propagation_convergence": convergence,
        "resources": {
            "started_at_utc": started_at.isoformat(),
            "wall_seconds": time.perf_counter() - wall_start,
            "user_cpu_seconds": usage_end.ru_utime - usage_start.ru_utime,
            "system_cpu_seconds": usage_end.ru_stime - usage_start.ru_stime,
            "peak_rss_kib": int(usage_end.ru_maxrss),
            "validation_models_newly_built_after_optimization": validation_builds,
            "validation_model_build_wall_seconds": validation_model_wall_seconds,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output}", flush=True)
    print("candidate parameters:", candidate_parameters.tolist(), flush=True)
    print(
        f"baseline: nominal={baseline['nominal_fidelity']:.12f}, "
        f"vertex min={baseline['full_grid_vertex_minimum']:.12f}",
        flush=True,
    )
    print(
        f"candidate: nominal={candidate['nominal_fidelity']:.12f}, "
        f"vertex min={candidate['full_grid_vertex_minimum']:.12f}, "
        f"accepted={final_acceptance['accepted']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
