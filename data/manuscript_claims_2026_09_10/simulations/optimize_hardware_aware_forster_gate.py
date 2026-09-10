#!/usr/bin/env python3
"""Hardware-aware optimization of the shaped Rb--Yb Förster CZ.

This extends ``simulate_robust_shaped_forster_gate.py`` in two ways:

* commanded Yb amplitude and detuning pass through first-order AOM response
  functions with an explicit 10--90% rise time and ring-down; and
* robustness is optimized over an isotropic three-dimensional Gaussian
  distribution of the relative atom position.  A three-node Gauss--Hermite
  rule per Cartesian coordinate reduces by cylindrical symmetry to nine
  distinct PairInteraction Hamiltonians.

Rb preparation remains the effective two-level model requested for this
stage.  The result is therefore a hardware-bandwidth and geometry test, not a
complete laboratory error budget.

Outputs:
  simulations/hardware_aware_forster_gate_results.json
  figures/hardware_aware_forster_gate.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pairinteraction as pi
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))

from simulate_forster_gate import PRIMARY_BASIS, SCAN_BASIS, PairModel, query_lifetimes  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402


OUT = ROOT / "simulations" / "hardware_aware_forster_gate_results.json"
FIGURE_STEM = ROOT / "figures" / "hardware_aware_forster_gate"

AOM_RISE_TIME_NS = 10.0
OPTIMIZATION_STEP_NS = 2.0
# Boundary-aligned convergence checks stabilize the reported nominal and
# limiting axial-vertex fidelities to four decimal places in percent here.
FINAL_STEP_NS = 0.125
FILTER_TAIL_CONSTANTS = 7.0

# This point is updated after a hardware-aware optimization and is also the
# seed for optional further refinement.
SELECTED_PARAMETERS = np.array(
    [
        3.4685139845100377,
        8.785348135595783,
        9.820348794207916,
        -3.0799080113108808,
        -0.6438764613592791,
        1.203329658157223,
        0.029628467958173407,
    ],
    dtype=float,
)
PARAMETER_BOUNDS = shaped.PARAMETER_BOUNDS


@dataclass(frozen=True)
class Geometry:
    delta_z_um: float
    transverse_um: float
    distance_um: float
    theta_deg: float
    weight: float
    model: PairModel


def _geometry_grid(config) -> list[Geometry]:
    """Three-point Gauss--Hermite rule in each relative Cartesian axis."""

    displacement = np.sqrt(3) * shaped.SIGMA_R_UM
    axial_values = (-displacement, 0.0, displacement)
    axial_weights = (1 / 6, 2 / 3, 1 / 6)

    # Combining x and y nodes produces three transverse radii.  Their summed
    # weights are 4/9, 4/9, and 1/9, respectively.
    transverse_values = (0.0, displacement, np.sqrt(2) * displacement)
    transverse_weights = (4 / 9, 4 / 9, 1 / 9)

    geometries = []
    for delta_z, axial_weight in zip(axial_values, axial_weights):
        for transverse, transverse_weight in zip(
            transverse_values, transverse_weights
        ):
            axial = shaped.R0_UM + delta_z
            distance = float(np.hypot(axial, transverse))
            theta = float(np.degrees(np.arctan2(transverse, axial)))
            print(
                f"building geometry R={distance:.6f} um, theta={theta:.4f} deg",
                flush=True,
            )
            model = shaped.build_pair_model(
                shaped.B_GAUSS,
                config,
                distance_um=distance,
                theta_deg=theta,
            )
            geometries.append(
                Geometry(
                    delta_z_um=float(delta_z),
                    transverse_um=float(transverse),
                    distance_um=distance,
                    theta_deg=theta,
                    weight=float(axial_weight * transverse_weight),
                    model=model,
                )
            )
    return geometries


def _filtered_pulse(
    parameters: np.ndarray,
    rise_time_ns: float,
    step_ns: float,
) -> tuple[shaped.Segment, ...]:
    """Return the filtered optical field on a command-boundary-aligned grid."""

    command = shaped._pulse_from_parameters(parameters)
    tau_us = rise_time_ns / (1000 * np.log(9))
    intervals = [
        (segment.omega_mhz, segment.detuning_mhz, segment.duration_us)
        for segment in command
    ]
    intervals.append(
        (0.0, command[-1].detuning_mhz, FILTER_TAIL_CONSTANTS * tau_us)
    )

    amplitude = 0.0
    detuning = command[0].detuning_mhz
    filtered = []
    for amplitude_command, detuning_command, duration_us in intervals:
        number_steps = int(np.ceil(1000 * duration_us / step_ns))
        dt = duration_us / number_steps
        for _ in range(number_steps):
            ratio = dt / tau_us
            decay = np.exp(-ratio)
            average_factor = (1 - decay) / ratio
            average_amplitude = amplitude_command + (
                amplitude - amplitude_command
            ) * average_factor
            average_detuning = detuning_command + (
                detuning - detuning_command
            ) * average_factor
            amplitude = amplitude_command + (
                amplitude - amplitude_command
            ) * decay
            detuning = detuning_command + (
                detuning - detuning_command
            ) * decay
            filtered.append(
                shaped.Segment(
                    omega_mhz=float(average_amplitude),
                    detuning_mhz=float(average_detuning),
                    duration_us=float(dt),
                )
            )
    return tuple(filtered)


def _target_returns(
    modes: shaped.ModeData,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    amplitude_scale: float = 1.0,
    detuning_offset_mhz: float = 0.0,
) -> tuple[complex, complex]:
    unblocked = np.array([1.0, 0.0], dtype=complex)
    blocked = np.zeros(len(modes.energies_rel_mhz) + 1, dtype=complex)
    blocked[0] = 1

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
        detuning = segment.detuning_mhz + detuning_offset_mhz
        h_unblocked = np.array(
            [[0.0, omega / 2], [omega / 2, -detuning]], dtype=complex
        )
        if lifetimes is not None:
            h_unblocked[1, 1] -= 0.5j / (
                2 * np.pi * lifetimes.yb_53s_us
            )
        unblocked = expm(
            -2j * np.pi * h_unblocked * segment.duration_us
        ) @ unblocked

        h_blocked = np.zeros(
            (len(modes.energies_rel_mhz) + 1,) * 2, dtype=complex
        )
        h_blocked[0, 1:] = omega * np.conj(modes.ss_amplitudes) / 2
        h_blocked[1:, 0] = omega * modes.ss_amplitudes / 2
        diagonal = modes.energies_rel_mhz - detuning
        if mode_rates is not None:
            # The control is already in 56S while Yb is still computational.
            h_blocked[0, 0] = -0.5j / (2 * np.pi * lifetimes.rb_56s_us)
            diagonal = diagonal - 0.5j * mode_rates / (2 * np.pi)
        h_blocked[1:, 1:] = np.diag(diagonal)
        blocked = expm_multiply(
            -2j * np.pi * h_blocked * segment.duration_us, blocked
        )

    return complex(unblocked[0]), complex(blocked[0])


def _kraus(
    modes: shaped.ModeData,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    target_amplitude_scale: float = 1.0,
    detuning_offset_mhz: float = 0.0,
    control_amplitude_scale: float = 1.0,
    control_detuning_offset_mhz: float = 0.0,
) -> np.ndarray:
    unblocked, blocked = _target_returns(
        modes,
        pulse,
        lifetimes,
        amplitude_scale=target_amplitude_scale,
        detuning_offset_mhz=detuning_offset_mhz,
    )
    duration = sum(segment.duration_us for segment in pulse)
    control_lifetime = None if lifetimes is None else lifetimes.rb_56s_us
    control_pi = shaped._control_pi(
        control_lifetime,
        amplitude_scale=control_amplitude_scale,
        detuning_offset_mhz=control_detuning_offset_mhz,
    )
    idle = (
        1.0
        if control_lifetime is None
        else np.exp(-duration / (2 * control_lifetime))
    )
    control_only = complex(
        (control_pi @ np.diag([1.0, idle]) @ control_pi)[0, 0]
    )
    both = complex(
        (control_pi @ np.diag([unblocked, blocked]) @ control_pi)[0, 0]
    )
    return np.array([1.0, unblocked, control_only, both], dtype=complex)


def _correction(modes, pulse, lifetimes) -> tuple[float, float]:
    metrics = shaped._local_z_metrics(_kraus(modes, pulse, lifetimes))
    return (
        float(metrics["optimal_local_z_alpha_rad"]),
        float(metrics["optimal_local_z_beta_rad"]),
    )


def _fidelity(modes, pulse, lifetimes, correction, **errors) -> float:
    return shaped._fixed_correction_fidelity(
        _kraus(modes, pulse, lifetimes, **errors), correction
    )


def _evaluate_ensemble(
    modes: list[shaped.ModeData],
    weights: np.ndarray,
    pulse: tuple[shaped.Segment, ...],
    lifetimes,
    correction: tuple[float, float],
    **errors,
) -> tuple[float, np.ndarray]:
    fidelities = np.array(
        [
            _fidelity(mode, pulse, lifetimes, correction, **errors)
            for mode in modes
        ]
    )
    return float(np.sum(weights * fidelities)), fidelities


def _optimize(
    seed: np.ndarray,
    modes: list[shaped.ModeData],
    weights: np.ndarray,
    nominal_index: int,
    lifetimes,
) -> tuple[np.ndarray, dict[str, object]]:
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        pulse = _filtered_pulse(
            parameters, AOM_RISE_TIME_NS, OPTIMIZATION_STEP_NS
        )
        correction = _correction(modes[nominal_index], pulse, lifetimes)
        ensemble_fidelities = []
        for amplitude_scale in (0.99, 1.0, 1.01):
            ensemble, _ = _evaluate_ensemble(
                modes,
                weights,
                pulse,
                lifetimes,
                correction,
                target_amplitude_scale=amplitude_scale,
            )
            ensemble_fidelities.append(ensemble)
        nominal = _fidelity(
            modes[nominal_index], pulse, lifetimes, correction
        )
        infidelities = 1 - np.array([*ensemble_fidelities, nominal])
        value = float(np.mean(infidelities**8) ** (1 / 8))
        if evaluations % 50 == 0:
            print(
                f"evaluation {evaluations}: objective={value:.8g}, "
                f"ensemble={ensemble_fidelities[1]:.9f}",
                flush=True,
            )
        return value

    result = minimize(
        objective,
        seed,
        method="Nelder-Mead",
        bounds=PARAMETER_BOUNDS,
        options={"maxiter": 300, "xatol": 2e-8, "fatol": 2e-11},
    )
    return np.asarray(result.x), {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": evaluations,
        "objective": float(result.fun),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="refine the stored command pulse for a 10 ns AOM rise time",
    )
    args = parser.parse_args()

    lt0 = query_lifetimes(0.0)
    lt300 = query_lifetimes(300.0)
    geometries = _geometry_grid(SCAN_BASIS)
    weights = np.array([geometry.weight for geometry in geometries])
    nominal_index = next(
        index
        for index, geometry in enumerate(geometries)
        if geometry.delta_z_um == 0 and geometry.transverse_um == 0
    )

    optimization_modes = [
        shaped._prepare_modes(geometry.model, 1e-3) for geometry in geometries
    ]
    parameters = SELECTED_PARAMETERS.copy()
    optimization: dict[str, object] = {
        "performed": False,
        "note": "stored hardware-aware optimum evaluated without refinement",
        "stored_search": {
            "method": "Nelder-Mead",
            "evaluations": 769,
            "bright_mode_cutoff": 1e-3,
            "propagation_step_ns": OPTIMIZATION_STEP_NS,
        },
    }
    if args.optimize:
        print("optimizing the filtered command pulse...", flush=True)
        parameters, optimization = _optimize(
            parameters, optimization_modes, weights, nominal_index, lt0
        )
        optimization["performed"] = True
        print("optimized parameters:", parameters.tolist(), flush=True)

    final_modes = [
        shaped._prepare_modes(geometry.model, 1e-6) for geometry in geometries
    ]
    pulse = _filtered_pulse(parameters, AOM_RISE_TIME_NS, FINAL_STEP_NS)
    correction = _correction(final_modes[nominal_index], pulse, lt0)
    ensemble_0k, geometry_fidelities = _evaluate_ensemble(
        final_modes, weights, pulse, lt0, correction
    )
    nominal_0k = _fidelity(
        final_modes[nominal_index], pulse, lt0, correction
    )

    original_pulse = _filtered_pulse(
        shaped.SELECTED_PARAMETERS, AOM_RISE_TIME_NS, FINAL_STEP_NS
    )
    original_correction = _correction(
        final_modes[nominal_index], original_pulse, lt0
    )
    original_ensemble, original_geometry_fidelities = _evaluate_ensemble(
        final_modes, weights, original_pulse, lt0, original_correction
    )

    robustness = {}
    for label, errors in {
        "target_amplitude_minus_1pct": {"target_amplitude_scale": 0.99},
        "target_amplitude_plus_1pct": {"target_amplitude_scale": 1.01},
        "target_detuning_minus_50khz": {"detuning_offset_mhz": -0.05},
        "target_detuning_plus_50khz": {"detuning_offset_mhz": 0.05},
        "control_amplitude_minus_1pct": {"control_amplitude_scale": 0.99},
        "control_amplitude_plus_1pct": {"control_amplitude_scale": 1.01},
    }.items():
        ensemble, curve = _evaluate_ensemble(
            final_modes, weights, pulse, lt0, correction, **errors
        )
        robustness[label] = {
            "ensemble_fidelity": ensemble,
            "nominal_fidelity": float(curve[nominal_index]),
            "worst_quadrature_fidelity": float(np.min(curve)),
        }

    rise_time_scan = {}
    for rise_time in (2.0, 5.0, 10.0):
        filtered = _filtered_pulse(parameters, rise_time, FINAL_STEP_NS)
        filtered_correction = _correction(
            final_modes[nominal_index], filtered, lt0
        )
        ensemble, curve = _evaluate_ensemble(
            final_modes, weights, filtered, lt0, filtered_correction
        )
        rise_time_scan[str(rise_time)] = {
            "target_duration_us": float(
                sum(segment.duration_us for segment in filtered)
            ),
            "nominal_fidelity": float(curve[nominal_index]),
            "ensemble_fidelity": ensemble,
            "worst_quadrature_fidelity": float(np.min(curve)),
        }

    temperature = {}
    for label, lifetimes in (("300k", lt300), ("coherent", None)):
        ensemble, curve = _evaluate_ensemble(
            final_modes, weights, pulse, lifetimes, correction
        )
        temperature[label] = {
            "nominal_fidelity": float(curve[nominal_index]),
            "ensemble_fidelity": ensemble,
            "worst_quadrature_fidelity": float(np.min(curve)),
        }

    # The off-axis primary basis is unnecessarily expensive.  At theta=0,
    # compare the primary and scan bases directly at the selected waveform.
    primary_model = shaped.build_pair_model(
        shaped.B_GAUSS,
        PRIMARY_BASIS,
        distance_um=shaped.R0_UM,
        theta_deg=0.0,
    )
    primary_mode = shaped._prepare_modes(primary_model, 1e-8)
    primary_fidelity = _fidelity(primary_mode, pulse, lt0, correction)

    command = shaped._pulse_from_parameters(parameters)
    output_times = np.cumsum([segment.duration_us for segment in pulse])
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "assumptions": {
            "aom_model": "independent first-order field-amplitude and detuning response",
            "aom_10_to_90_rise_time_ns": AOM_RISE_TIME_NS,
            "filter_tail_time_constants": FILTER_TAIL_CONSTANTS,
            "relative_position_distribution": "independent Gaussian x,y,z coordinates",
            "relative_position_sigma_each_axis_um": shaped.SIGMA_R_UM,
            "geometry_quadrature": "three-point Gauss-Hermite per Cartesian axis; nine unique cylindrical points",
            "rb_excitation": "effective two-level 5 MHz pi pulse; intermediate-state scattering omitted",
            "decay": "PairInteraction lifetime-weighted no-jump loss counted as erasure",
            "local_z": "one correction calibrated at nominal geometry and held fixed",
        },
        "optimization": optimization,
        "parameters": parameters.tolist(),
        "command_segments": [asdict(segment) for segment in command],
        "filtered_waveform": {
            "time_us": output_times.tolist(),
            "omega_mhz": [segment.omega_mhz for segment in pulse],
            "detuning_mhz": [segment.detuning_mhz for segment in pulse],
            "step_us": pulse[0].duration_us,
        },
        "selected_10ns": {
            "target_duration_us": float(
                sum(segment.duration_us for segment in pulse)
            ),
            "total_gate_time_us": float(
                2 * shaped.RB_PI_DURATION_US
                + sum(segment.duration_us for segment in pulse)
            ),
            "nominal_fidelity_0k": nominal_0k,
            "ensemble_fidelity_0k": ensemble_0k,
            "worst_quadrature_fidelity_0k": float(
                np.min(geometry_fidelities)
            ),
            "primary_basis_nominal_fidelity_0k": primary_fidelity,
            "local_z_alpha_rad": correction[0],
            "local_z_beta_rad": correction[1],
        },
        "unrefined_previous_pulse_10ns": {
            "ensemble_fidelity_0k": original_ensemble,
            "geometry_fidelity": original_geometry_fidelities.tolist(),
        },
        "temperature": temperature,
        "robustness": robustness,
        "rise_time_scan": rise_time_scan,
        "geometry": [
            {
                "delta_z_um": geometry.delta_z_um,
                "transverse_um": geometry.transverse_um,
                "distance_um": geometry.distance_um,
                "theta_deg": geometry.theta_deg,
                "weight": geometry.weight,
                "fidelity_0k": float(fidelity),
                "basis_component_size": geometry.model.symmetry_component_size,
            }
            for geometry, fidelity in zip(geometries, geometry_fidelities)
        ],
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45))
    ax = axes[0]
    command_edges = np.r_[
        0, np.cumsum([segment.duration_us for segment in command])
    ]
    ax.stairs(
        [segment.omega_mhz for segment in command],
        1000 * command_edges,
        color="0.6",
        ls="--",
        label="command",
    )
    ax.plot(
        1000 * output_times,
        [segment.omega_mhz for segment in pulse],
        color="#0072B2",
        drawstyle="steps-post",
        label="filtered",
    )
    ax.set_xlabel("target-pulse time (ns)")
    ax.set_ylabel(r"$\Omega/2\pi$ (MHz)")
    ax.set_title("(a) 10 ns AOM response")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    labels = [
        f"{geometry.delta_z_um * 1000:+.0f}\n{geometry.theta_deg:.1f}°"
        for geometry in geometries
    ]
    ax.plot(
        np.arange(len(geometries)),
        1e4 * (1 - geometry_fidelities),
        "o-",
        color="#0072B2",
        label="optimized",
    )
    ax.plot(
        np.arange(len(geometries)),
        1e4 * (1 - original_geometry_fidelities),
        "o--",
        color="#CC79A7",
        label="previous pulse",
    )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel(r"$\delta z$ (nm) / $\theta$")
    ax.set_ylabel(r"$10^4(1-F_{\rm avg})$")
    ax.set_title("(b) 3D geometry quadrature")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    bar_labels = ["nominal", "3D ensemble"] + [
        r"$\Omega_t-1\%$",
        r"$\Omega_t+1\%$",
        r"$\Delta-50$ kHz",
        r"$\Delta+50$ kHz",
        r"$\Omega_c-1\%$",
        r"$\Omega_c+1\%$",
    ]
    bar_fidelities = [nominal_0k, ensemble_0k] + [
        row["ensemble_fidelity"] for row in robustness.values()
    ]
    bars = ax.bar(
        np.arange(len(bar_fidelities)),
        1e4 * (1 - np.array(bar_fidelities)),
        color="#009E73",
    )
    ax.axhline(10.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.text(
        -0.45,
        10.15,
        r"99.9\% fidelity",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(
        np.arange(len(bar_labels)), bar_labels, rotation=48, ha="right"
    )
    ax.set_ylabel(r"$10^4(1-F_{\rm avg})$")
    ax.set_title("(c) fixed-waveform robustness")

    fig.tight_layout()
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print("wrote", OUT, flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".pdf"), flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".png"), flush=True)
    print(
        f"10 ns hardware-aware result: nominal={nominal_0k:.9f}, "
        f"3D ensemble={ensemble_0k:.9f}, primary={primary_fidelity:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
