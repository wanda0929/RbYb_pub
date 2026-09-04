#!/usr/bin/env python3
"""Evaluate P1-1 bounded robustness and literature-based motion scales.

The expensive PairInteraction calculation is performed on a regular response
grid.  A cubic tensor interpolant is then used for the nested
128/256/512/1024-point Sobol check and local adversarial search.  Independent
off-grid points and the returned worst candidates are re-evaluated with the
direct P0-4 retained-block numerical-reference model; the interpolation is
never presented as a certificate.

The thermal section intentionally reports a cross-paper benchmark rather than
an apparatus prediction.  No publication currently supplies a complete,
same-condition Rb--Yb trap-frequency/temperature/axis data set for this gate.

Output:
  data/forster_p1_1_robustness.json
"""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_forster_p0_4 as p0  # noqa: E402
import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import reproduce_forster_gate as minimax  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from p1_robustness import (  # noqa: E402
    HarmonicThermalState,
    RobustnessDomain,
    RobustnessScenario,
    ballistic_relative_trajectory,
    doppler_detuning_mhz,
    harmonic_phase_space_widths,
    local_adversarial_search,
    nested_minimum_trace,
    sample_two_species_phase_space,
    sobol_scenarios,
    worst_sample_seeds,
)
from simulate_forster_gate import query_lifetimes  # noqa: E402
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402

OUT = ROOT / "data" / "forster_p1_1_robustness.json"
P0_RESULT = ROOT / "data" / "forster_p0_4_uncertainty_convergence.json"

DOMAIN = RobustnessDomain()
RADII_UM = np.array([0.0, 0.0125, 0.025, 0.0375, 0.05])
DIRECTION_COSINES = np.linspace(-1.0, 1.0, 9)
AMPLITUDE_SCALES = np.linspace(0.99, 1.01, 5)
SOBOL_POWER = 10
HOLDOUT_INDICES = (37, 101, 173, 269, 389, 503)
LOCAL_SEED_COUNT = 12
THERMAL_SAMPLE_COUNT = 2**18
THERMAL_SEED = 20260904


def _scenario_dict(scenario: RobustnessScenario) -> dict[str, float]:
    radius = scenario.displacement_radius_um
    return {
        **asdict(scenario),
        "displacement_radius_um": radius,
        "direction_cosine": 0.0 if radius == 0 else scenario.delta_z_um / radius,
        "transverse_um": float(np.hypot(scenario.delta_x_um, scenario.delta_y_um)),
    }


def _scenario_coordinates(scenarios: list[RobustnessScenario]) -> np.ndarray:
    coordinates = np.empty((len(scenarios), 4), dtype=float)
    for index, scenario in enumerate(scenarios):
        radius = scenario.displacement_radius_um
        coordinates[index] = (
            radius,
            0.0 if radius == 0 else scenario.delta_z_um / radius,
            scenario.yb_amplitude_scale,
            scenario.rb_amplitude_scale,
        )
    return coordinates


def _build_mode(radius_um: float, direction_cosine: float):
    if radius_um == 0:
        model = p0._build(p0.REFERENCE_BASIS)
    else:
        model = minimax._build_geometry(radius_um, direction_cosine, p0.REFERENCE_BASIS).model
    mode = shaped._prepare_modes(model, p0.MODE_CUTOFF)
    metadata = {
        "distance_um": model.distance_um,
        "theta_deg": model.theta_deg,
        "connected_component_size": model.symmetry_component_size,
        "active_modes": len(mode.energies_rel_mhz),
        "retained_ss_weight": mode.retained_ss_weight,
    }
    del model
    gc.collect()
    return mode, metadata


def _amplitude_grid(mode, pulse, lifetimes, correction) -> np.ndarray:
    return minimax._scenario_fidelities(
        [mode],
        pulse,
        lifetimes,
        correction,
        target_scales=tuple(float(value) for value in AMPLITUDE_SCALES),
        control_scales=tuple(float(value) for value in AMPLITUDE_SCALES),
    )[0]


def _direct_fidelity(mode, pulse, lifetimes, correction, scenario) -> float:
    return hardware._fidelity(
        mode,
        pulse,
        lifetimes,
        correction,
        target_amplitude_scale=scenario.yb_amplitude_scale,
        control_amplitude_scale=scenario.rb_amplitude_scale,
    )


def _thermal_summary(response) -> dict[str, object]:
    # Direct Yb gate parameters from Peper et al. are combined with the complete
    # but cross-platform Rb benchmark of Kaufman et al.  This is a scale study,
    # not a coherent apparatus parameter set.
    yb_state = HarmonicThermalState(171.0, 2.9, (60.0, 60.0, 10.0))
    rb_state = HarmonicThermalState(87.0, 13.0, (154.0, 150.0, 30.0))
    phase_space = sample_two_species_phase_space(
        yb_state,
        rb_state,
        THERMAL_SAMPLE_COUNT,
        seed=THERMAL_SEED,
    )
    target_duration_us = sum(segment.duration_us for segment in p0._pulse())
    target_start_us = shaped.RB_PI_DURATION_US
    target_midpoint_us = target_start_us + target_duration_us / 2
    target_end_us = target_start_us + target_duration_us
    total_gate_us = target_end_us + shaped.RB_PI_DURATION_US
    times_us = np.array([0.0, target_start_us, target_midpoint_us, target_end_us, total_gate_us])
    trajectory = ballistic_relative_trajectory((0.0, 0.0, shaped.R0_UM), phase_space, times_us)
    offsets = trajectory - np.array([0.0, 0.0, shaped.R0_UM])[None, None, :]

    trajectory_rows = []
    for index, time_us in enumerate(times_us):
        radius = np.linalg.norm(offsets[:, index], axis=1)
        trajectory_rows.append(
            {
                "time_us": float(time_us),
                "axis_sigma_nm": (1000 * np.std(offsets[:, index], axis=0, ddof=1)).tolist(),
                "radial_offset_percentiles_nm": {
                    str(percentile): float(1000 * np.percentile(radius, percentile))
                    for percentile in (50, 90, 95, 99)
                },
                "probability_inside_50nm_ball": float(np.mean(radius <= 0.05)),
            }
        )

    midpoint_offset = offsets[:, 2]
    midpoint_radius = np.linalg.norm(midpoint_offset, axis=1)
    inside = midpoint_radius <= DOMAIN.max_displacement_um
    bounded_coordinates = np.column_stack(
        (
            midpoint_radius[inside],
            np.divide(
                midpoint_offset[inside, 2],
                midpoint_radius[inside],
                out=np.zeros(np.count_nonzero(inside)),
                where=midpoint_radius[inside] > 0,
            ),
            np.ones(np.count_nonzero(inside)),
            np.ones(np.count_nonzero(inside)),
        )
    )
    bounded_fidelity = np.asarray(response(bounded_coordinates), dtype=float)

    yb_position_width_um, yb_velocity_width = harmonic_phase_space_widths(yb_state)
    rb_position_width_um, rb_velocity_width = harmonic_phase_space_widths(rb_state)
    k_yb = 2 * np.pi / 0.302043
    k_rb_780 = 2 * np.pi / 0.780
    k_rb_480 = 2 * np.pi / 0.480

    def shift_sigma(velocity, wavevector) -> float:
        return float(
            np.std(
                doppler_detuning_mhz(velocity, (0.0, 0.0, wavevector)),
                ddof=1,
            )
        )

    return {
        "interpretation": (
            "cross-paper thermal scale benchmark; not a same-apparatus gate "
            "prediction and not a substitute for the bounded stress test"
        ),
        "sample_count": THERMAL_SAMPLE_COUNT,
        "seed": THERMAL_SEED,
        "trap_axis_convention": (
            "the two weak trap axes are aligned with the pair/quantization z axis"
        ),
        "inputs": {
            "yb": {
                **asdict(yb_state),
                "position_sigma_nm": (1000 * yb_position_width_um).tolist(),
                "velocity_sigma_um_per_us": yb_velocity_width.tolist(),
                "source": "Peper et al., Phys. Rev. X 15, 011009 (2025)",
                "doi": "10.1103/PhysRevX.15.011009",
            },
            "rb": {
                **asdict(rb_state),
                "position_sigma_nm": (1000 * rb_position_width_um).tolist(),
                "velocity_sigma_um_per_us": rb_velocity_width.tolist(),
                "source": "Kaufman et al., Phys. Rev. X 2, 041014 (2012)",
                "doi": "10.1103/PhysRevX.2.041014",
                "caveat": "complete three-axis cross-platform benchmark",
            },
        },
        "ballistic_relative_motion": trajectory_rows,
        "doppler_sigma_mhz": {
            "yb_302p043nm_along_weak_axis": shift_sigma(phase_space.yb.velocity_um_per_us, k_yb),
            "rb_780_plus_480_counterpropagating_along_weak_axis": shift_sigma(
                phase_space.rb.velocity_um_per_us, abs(k_rb_480 - k_rb_780)
            ),
            "rb_780_plus_480_copropagating_along_weak_axis": shift_sigma(
                phase_space.rb.velocity_um_per_us, k_rb_480 + k_rb_780
            ),
        },
        "bounded_subset_geometry_only_diagnostic": {
            "definition": (
                "static midpoint geometry, nominal amplitudes, no Doppler; "
                "conditioned on the literature benchmark lying inside 50 nm"
            ),
            "conditional_sample_count": int(np.count_nonzero(inside)),
            "conditional_probability": float(np.mean(inside)),
            "mean_interpolated_fidelity": float(np.mean(bounded_fidelity)),
            "minimum_interpolated_fidelity": float(np.min(bounded_fidelity)),
            "fidelity_percentiles": {
                str(percentile): float(np.percentile(bounded_fidelity, percentile))
                for percentile in (1, 5, 50, 95, 99)
            },
        },
        "missing_for_apparatus_prediction": [
            "same-apparatus Rb and Yb release-depth trap frequencies",
            "gate-time temperatures",
            "both trap-axis rotations relative to the pair axis",
            "effective Rb and Yb wavevector directions",
            "exact trap-off/on timing",
            "time-dependent full-angular pair-Hamiltonian propagation",
        ],
    }


def main() -> None:
    verify_database_manifest()
    p0_result = json.loads(P0_RESULT.read_text())
    fixed = p0_result["fixed_inputs"]
    pulse = p0._pulse()
    lifetimes = query_lifetimes(0.0)
    correction = (
        fixed["fixed_reference_local_z_alpha_rad"],
        fixed["fixed_reference_local_z_beta_rad"],
    )

    grid = np.empty(
        (
            len(RADII_UM),
            len(DIRECTION_COSINES),
            len(AMPLITUDE_SCALES),
            len(AMPLITUDE_SCALES),
        ),
        dtype=float,
    )
    modes_by_geometry = {}
    geometry_metadata = []
    for radius_index, radius_um in enumerate(RADII_UM):
        if radius_um == 0:
            print("building P0-4 response node: nominal", flush=True)
            mode, metadata = _build_mode(0.0, 0.0)
            values = _amplitude_grid(mode, pulse, lifetimes, correction)
            grid[radius_index, :, :, :] = values[None, :, :]
            modes_by_geometry[(0.0, 0.0)] = mode
            geometry_metadata.append({"radius_um": 0.0, "direction_cosine": 0.0, **metadata})
            continue
        for cosine_index, direction_cosine in enumerate(DIRECTION_COSINES):
            print(
                "building P0-4 response node: "
                f"r={1000 * radius_um:.1f} nm, cos={direction_cosine:+.2f}",
                flush=True,
            )
            mode, metadata = _build_mode(float(radius_um), float(direction_cosine))
            grid[radius_index, cosine_index] = _amplitude_grid(mode, pulse, lifetimes, correction)
            modes_by_geometry[(float(radius_um), float(direction_cosine))] = mode
            geometry_metadata.append(
                {
                    "radius_um": float(radius_um),
                    "direction_cosine": float(direction_cosine),
                    **metadata,
                }
            )

    response = RegularGridInterpolator(
        (RADII_UM, DIRECTION_COSINES, AMPLITUDE_SCALES, AMPLITUDE_SCALES),
        grid,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )
    scenarios = list(sobol_scenarios(SOBOL_POWER, DOMAIN))
    interpolated_values = np.asarray(response(_scenario_coordinates(scenarios)), dtype=float)
    trace = nested_minimum_trace(interpolated_values, powers=(7, 8, 9, 10))
    trace_rows = []
    for point in trace:
        trace_rows.append(
            {
                **asdict(point),
                "scenario": _scenario_dict(scenarios[point.index]),
            }
        )

    print("re-evaluating independent Sobol holdout points", flush=True)
    holdout_rows = []
    for scenario_index in HOLDOUT_INDICES:
        scenario = scenarios[scenario_index]
        radius = scenario.displacement_radius_um
        cosine = scenario.delta_z_um / radius
        mode, metadata = _build_mode(radius, cosine)
        direct = _direct_fidelity(mode, pulse, lifetimes, correction, scenario)
        interpolated = float(response(_scenario_coordinates([scenario])).item())
        holdout_rows.append(
            {
                "sobol_index": scenario_index,
                "scenario": _scenario_dict(scenario),
                "interpolated_fidelity": interpolated,
                "direct_fidelity": direct,
                "interpolation_error": interpolated - direct,
                "full_model": metadata,
            }
        )
        del mode
        gc.collect()

    def interpolated_objective(scenario: RobustnessScenario) -> float:
        if not DOMAIN.contains(scenario):
            return 1.0
        return float(response(_scenario_coordinates([scenario])).item())

    print("running local adversarial refinement on the response surface", flush=True)
    local_results = local_adversarial_search(
        interpolated_objective,
        worst_sample_seeds(scenarios, interpolated_values, LOCAL_SEED_COUNT),
        DOMAIN,
    )
    local_results = tuple(sorted(local_results, key=lambda result: result.value))

    direct_candidates = []
    seen = set()
    for local_result in sorted(
        local_results, key=lambda result: (not result.success, result.value)
    ):
        scenario = local_result.scenario
        radius = scenario.displacement_radius_um
        cosine = 0.0 if radius == 0 else scenario.delta_z_um / radius
        key = tuple(np.round([radius, cosine], 8))
        if key in seen:
            continue
        seen.add(key)
        cached_key = next(
            (
                existing
                for existing in modes_by_geometry
                if np.allclose(existing, key, atol=1e-8, rtol=0)
            ),
            None,
        )
        if cached_key is None:
            print(
                "building retained-block adversarial candidate: "
                f"r={1000 * radius:.3f} nm, cos={cosine:+.6f}",
                flush=True,
            )
            mode, metadata = _build_mode(radius, cosine)
        else:
            mode = modes_by_geometry[cached_key]
            metadata = next(
                row
                for row in geometry_metadata
                if np.allclose(
                    [row["radius_um"], row["direction_cosine"]],
                    cached_key,
                    atol=1e-12,
                    rtol=0,
                )
            )
        direct = _direct_fidelity(mode, pulse, lifetimes, correction, scenario)
        direct_candidates.append(
            {
                "scenario": _scenario_dict(scenario),
                "interpolated_fidelity": local_result.value,
                "direct_fidelity": direct,
                "interpolation_error": local_result.value - direct,
                "optimizer_success": local_result.success,
                "optimizer_evaluations": local_result.evaluations,
                "optimizer_message": local_result.message,
                "full_model": metadata,
            }
        )
        if len(direct_candidates) == 3:
            break

    vertex_indices = (0, len(AMPLITUDE_SCALES) - 1)
    training_vertex_minimum = float(np.min(grid[:, :, vertex_indices, :][:, :, :, vertex_indices]))
    continuous_grid_minimum = float(np.min(grid))
    minimum_grid_index = np.unravel_index(int(np.argmin(grid)), grid.shape)
    grid_minimum_scenario = {
        "radius_um": float(RADII_UM[minimum_grid_index[0]]),
        "direction_cosine": float(DIRECTION_COSINES[minimum_grid_index[1]]),
        "yb_amplitude_scale": float(AMPLITUDE_SCALES[minimum_grid_index[2]]),
        "rb_amplitude_scale": float(AMPLITUDE_SCALES[minimum_grid_index[3]]),
    }

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "status": (
            "bounded sampled-and-locally-refined evidence complete; no global "
            "certificate; thermal section is a literature-based scale benchmark "
            "pending apparatus inputs"
        ),
        "fixed_p0_4_inputs": fixed,
        "bounded_domain": asdict(DOMAIN),
        "response_grid": {
            "basis": asdict(p0.REFERENCE_BASIS),
            "mode_cutoff": p0.MODE_CUTOFF,
            "propagation_step_ns": hardware.FINAL_STEP_NS,
            "radii_um": RADII_UM.tolist(),
            "direction_cosines": DIRECTION_COSINES.tolist(),
            "yb_amplitude_scales": AMPLITUDE_SCALES.tolist(),
            "rb_amplitude_scales": AMPLITUDE_SCALES.tolist(),
            "number_of_direct_retained_block_geometries": len(geometry_metadata),
            "geometry": geometry_metadata,
            "fidelity_grid": grid.tolist(),
            "amplitude_vertex_minimum_fidelity": training_vertex_minimum,
            "five_by_five_amplitude_grid_minimum_fidelity": continuous_grid_minimum,
            "minimum_scenario": grid_minimum_scenario,
        },
        "interpolation": {
            "method": "cubic tensor-product regular-grid spline",
            "role": (
                "search/coverage surrogate only; independent points and final "
                "candidates use the direct P0-4 retained-block numerical-reference model"
            ),
            "holdout": holdout_rows,
            "maximum_absolute_holdout_error": float(
                max(abs(row["interpolation_error"]) for row in holdout_rows)
            ),
            "rms_holdout_error": float(
                np.sqrt(np.mean([row["interpolation_error"] ** 2 for row in holdout_rows]))
            ),
        },
        "nested_sobol": {
            "construction": (
                "deterministic unscrambled 5D Sobol sequence; position uniform "
                "in ball volume and both Rabi scales continuous"
            ),
            "largest_sample_count": len(scenarios),
            "minimum_trace": trace_rows,
        },
        "local_adversarial_refinement": {
            "method": "12 lowest Sobol values as SLSQP starts on the response surface",
            "claim_scope": "local adversarial search, not a global certificate",
            "surface_results": [
                {
                    **asdict(local_result),
                    "scenario": _scenario_dict(local_result.scenario),
                }
                for local_result in local_results
            ],
            "direct_retained_block_rechecks": direct_candidates,
            "minimum_direct_recheck_fidelity": float(
                min(row["direct_fidelity"] for row in direct_candidates)
            ),
        },
        "thermal_motion": _thermal_summary(response),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
