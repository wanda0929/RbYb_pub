#!/usr/bin/env python3
"""Robust static-field Förster CZ with a shaped Yb target pulse.

The protocol keeps the experimentally simple Rb-control sequence used by
``simulate_ss_scheme.py`` but replaces the square Yb target pulse by a
palindromic five-segment pulse.  The pulse is optimized against pair-distance
variation in the projected all-(m_J,m_I) PairInteraction Hamiltonian.  One local-Z
correction is calibrated at the nominal point and then held fixed for every
robustness calculation.

The calculation uses complex bare-state amplitudes (``get_amplitudes``), not
their squared overlaps, when projecting the optical target into the pair
basis.  Loss is the same non-Hermitian no-jump/orthogonal-erasure model as in
the square-pulse calculation.

Outputs:
  simulations/robust_shaped_forster_gate_results.json
  figures/robust_shaped_forster_gate.{pdf,png}
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
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))

from simulate_forster_gate import (  # noqa: E402
    PRIMARY_BASIS,
    SCAN_BASIS,
    BasisConfig,
    Lifetimes,
    PairModel,
    _local_z_metrics,
    build_pair_model,
    query_lifetimes,
)
import simulate_ss_scheme as ss  # noqa: E402

OUT = ROOT / "simulations" / "robust_shaped_forster_gate_results.json"
FIGURE_STEM = ROOT / "figures" / "robust_shaped_forster_gate"

B_GAUSS = 3.10
R0_UM = 3.40
SIGMA_R_UM = 0.05
OMEGA_RB_MHZ = ss.OMEGA_RB_TWOPhoton_MHZ
RB_PI_DURATION_US = 1 / (2 * OMEGA_RB_MHZ)

# Existing square-revival pulse, reevaluated below with complex target-state
# amplitudes so that the comparison uses exactly the same pair model.
SQUARE_OMEGA_MHZ = 6.306228240400818
SQUARE_DETUNING_MHZ = 0.12638923156839477
SQUARE_DURATION_US = 0.15854155322462685

# Selected result of a bounded five-segment search.  A common optical phase was
# removed because it is only a calibratable one-qubit phase here.  The same
# point is the seed when optional refinement is requested.
SELECTED_PARAMETERS = np.array(
    [
        3.305844440673641,
        8.705322098787446,
        10.06493542238902,
        -2.9557334827111355,
        -0.5817518908977706,
        1.0774636147276295,
        0.02979991180278341,
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
    (0.012, 0.050),
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


def _pulse_from_parameters(parameters: np.ndarray) -> tuple[Segment, ...]:
    omega_a, omega_b, omega_c, det_a, det_b, det_c, duration = parameters
    half = (
        Segment(float(omega_a), float(det_a), float(duration)),
        Segment(float(omega_b), float(det_b), float(duration)),
        Segment(float(omega_c), float(det_c), float(duration)),
    )
    return half + (half[1], half[0])


def _square_pulse() -> tuple[Segment, ...]:
    return (
        Segment(SQUARE_OMEGA_MHZ, SQUARE_DETUNING_MHZ, SQUARE_DURATION_US),
    )


def _prepare_modes(
    model: PairModel,
    threshold: float,
) -> ModeData:
    mask = np.abs(model.ss_overlap) ** 2 > threshold
    energies = model.energies_mhz[mask] - model.ss_asymptote_mhz
    pp = model.pp_overlap[mask]
    ss_amplitudes = model.ss_overlap[mask]
    pp_weight = np.abs(pp) ** 2
    ss_weight = np.abs(ss_amplitudes) ** 2
    return ModeData(
        distance_um=model.distance_um,
        energies_rel_mhz=energies,
        ss_amplitudes=ss_amplitudes,
        pp_weights=pp_weight,
        ss_weights=ss_weight,
        retained_ss_weight=float(np.sum(ss_weight)),
    )


def _build_mode_grid(
    distances_um: np.ndarray,
    config: BasisConfig,
    threshold: float,
    field_gauss: float = B_GAUSS,
    theta_deg: float = 0.0,
) -> list[ModeData]:
    modes = []
    for distance in distances_um:
        model = build_pair_model(
            field_gauss,
            config,
            distance_um=float(distance),
            theta_deg=theta_deg,
        )
        modes.append(_prepare_modes(model, threshold))
    return modes


def _target_returns(
    modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes: Lifetimes | None,
    amplitude_scale: float = 1.0,
    detuning_offset_mhz: float = 0.0,
    duration_scale: float = 1.0,
) -> tuple[complex, complex]:
    unblocked_state = np.array([1.0, 0.0], dtype=complex)
    blocked_state = np.zeros(len(modes.energies_rel_mhz) + 1, dtype=complex)
    blocked_state[0] = 1

    for segment in pulse:
        omega = amplitude_scale * segment.omega_mhz
        detuning = segment.detuning_mhz + detuning_offset_mhz

        h_unblocked = np.array(
            [[0.0, omega / 2], [omega / 2, -detuning]], dtype=complex
        )
        if lifetimes is not None:
            h_unblocked[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        unblocked_state = expm(
            -2j * np.pi * h_unblocked * segment.duration_us * duration_scale
        ) @ unblocked_state

        h_blocked = np.zeros(
            (len(modes.energies_rel_mhz) + 1,) * 2, dtype=complex
        )
        h_blocked[0, 1:] = omega * np.conj(modes.ss_amplitudes) / 2
        h_blocked[1:, 0] = omega * modes.ss_amplitudes / 2
        diagonal = modes.energies_rel_mhz - detuning
        if lifetimes is not None:
            h_blocked[0, 0] = -0.5j / (2 * np.pi * lifetimes.rb_56s_us)
            pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
            ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
            spectator_weight = np.maximum(
                0.0, 1 - modes.pp_weights - modes.ss_weights
            )
            rates = (
                modes.pp_weights * pp_rate
                + modes.ss_weights * ss_rate
                + spectator_weight * max(pp_rate, ss_rate)
            )
            diagonal = diagonal - 0.5j * rates / (2 * np.pi)
        h_blocked[1:, 1:] = np.diag(diagonal)
        blocked_state = expm(
            -2j * np.pi * h_blocked * segment.duration_us * duration_scale
        ) @ blocked_state

    return complex(unblocked_state[0]), complex(blocked_state[0])


def _control_pi(
    lifetime_us: float | None,
    amplitude_scale: float = 1.0,
    detuning_offset_mhz: float = 0.0,
) -> np.ndarray:
    h = np.array(
        [
            [0.0, amplitude_scale * OMEGA_RB_MHZ / 2],
            [
                amplitude_scale * OMEGA_RB_MHZ / 2,
                -detuning_offset_mhz,
            ],
        ],
        dtype=complex,
    )
    if lifetime_us is not None:
        h[1, 1] -= 0.5j / (2 * np.pi * lifetime_us)
    return expm(-2j * np.pi * h * RB_PI_DURATION_US)


def _kraus_diagonal(
    modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes: Lifetimes | None,
    target_amplitude_scale: float = 1.0,
    detuning_offset_mhz: float = 0.0,
    duration_scale: float = 1.0,
    control_amplitude_scale: float = 1.0,
    control_detuning_offset_mhz: float = 0.0,
) -> np.ndarray:
    unblocked, blocked = _target_returns(
        modes,
        pulse,
        lifetimes,
        amplitude_scale=target_amplitude_scale,
        detuning_offset_mhz=detuning_offset_mhz,
        duration_scale=duration_scale,
    )
    target_duration = duration_scale * sum(segment.duration_us for segment in pulse)
    control_lifetime = None if lifetimes is None else lifetimes.rb_56s_us
    control_pi = _control_pi(
        control_lifetime,
        control_amplitude_scale,
        control_detuning_offset_mhz,
    )
    control_idle = (
        1.0
        if control_lifetime is None
        else np.exp(-target_duration / (2 * control_lifetime))
    )
    control_only = complex(
        (control_pi @ np.diag([1.0, control_idle]) @ control_pi)[0, 0]
    )
    both = complex(
        (control_pi @ np.diag([unblocked, blocked]) @ control_pi)[0, 0]
    )
    return np.array([1.0, unblocked, control_only, both], dtype=complex)


def _fixed_correction_fidelity(
    kraus: np.ndarray,
    correction: tuple[float, float],
) -> float:
    alpha, beta = correction
    ideal = np.array([1, 1, 1, -1], dtype=complex)
    local_z = np.array(
        [1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))]
    )
    survival = float(np.vdot(kraus, kraus).real / 4)
    overlap = complex(np.vdot(ideal, local_z * kraus))
    return float((4 * survival + abs(overlap) ** 2) / 20)


def _nominal_correction(
    nominal_modes: ModeData,
    pulse: tuple[Segment, ...],
    lifetimes: Lifetimes,
) -> tuple[float, float]:
    metrics = _local_z_metrics(_kraus_diagonal(nominal_modes, pulse, lifetimes))
    return (
        float(metrics["optimal_local_z_alpha_rad"]),
        float(metrics["optimal_local_z_beta_rad"]),
    )


def _evaluate_grid(
    modes: list[ModeData],
    pulse: tuple[Segment, ...],
    lifetimes: Lifetimes | None,
    correction: tuple[float, float],
    **errors: float,
) -> np.ndarray:
    return np.array(
        [
            _fixed_correction_fidelity(
                _kraus_diagonal(mode, pulse, lifetimes, **errors), correction
            )
            for mode in modes
        ]
    )


def _optimize_pulse(
    seed: np.ndarray,
    modes: list[ModeData],
    lifetimes: Lifetimes,
) -> tuple[np.ndarray, dict[str, object]]:
    nominal_index = int(
        np.argmin([abs(mode.distance_um - R0_UM) for mode in modes])
    )
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        pulse = _pulse_from_parameters(parameters)
        correction = _nominal_correction(modes[nominal_index], pulse, lifetimes)
        nominal = _evaluate_grid(modes, pulse, lifetimes, correction)
        amplitude_minus = _evaluate_grid(
            modes,
            pulse,
            lifetimes,
            correction,
            target_amplitude_scale=0.99,
        )
        amplitude_plus = _evaluate_grid(
            modes,
            pulse,
            lifetimes,
            correction,
            target_amplitude_scale=1.01,
        )
        distances = np.array([mode.distance_um for mode in modes])
        weights = np.exp(-0.5 * ((distances - R0_UM) / SIGMA_R_UM) ** 2)
        weights /= np.sum(weights)
        ensemble_fidelities = np.array(
            [
                np.sum(weights * nominal),
                np.sum(weights * amplitude_minus),
                np.sum(weights * amplitude_plus),
                nominal[nominal_index],
            ]
        )
        # An eighth-power mean is a smooth approximation to the worst of the
        # nominal and +/-1% amplitude Gaussian-ensemble fidelities.
        infidelities = 1 - ensemble_fidelities
        return float(np.mean(infidelities**8) ** (1 / 8))

    result = minimize(
        objective,
        seed,
        method="Nelder-Mead",
        bounds=PARAMETER_BOUNDS,
        options={"maxiter": 500, "xatol": 2e-8, "fatol": 2e-11},
    )
    return np.asarray(result.x), {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": evaluations,
        "objective": float(result.fun),
    }


def _metrics_for_pulse(
    modes: list[ModeData],
    distances: np.ndarray,
    weights: np.ndarray,
    pulse: tuple[Segment, ...],
    lifetimes: Lifetimes,
    correction: tuple[float, float],
) -> tuple[dict[str, float], np.ndarray]:
    fidelities = _evaluate_grid(modes, pulse, lifetimes, correction)
    nominal_index = int(np.argmin(abs(distances - R0_UM)))
    kraus = _kraus_diagonal(modes[nominal_index], pulse, lifetimes)
    metrics = _local_z_metrics(kraus)
    return {
        "nominal_fidelity": float(fidelities[nominal_index]),
        "gaussian_ensemble_fidelity": float(np.sum(weights * fidelities)),
        "worst_fidelity": float(np.min(fidelities)),
        "mean_computational_survival": float(
            metrics["mean_computational_survival"]
        ),
        "conditional_phase_error_rad": float(
            metrics["conditional_phase_error_rad"]
        ),
        "total_gate_time_us": float(
            2 * RB_PI_DURATION_US
            + sum(segment.duration_us for segment in pulse)
        ),
    }, fidelities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="refine the stored pulse on a five-point scan-basis robust grid",
    )
    parser.add_argument(
        "--verify-quadrature",
        action="store_true",
        help="also evaluate a 17-point scan-basis Gauss-Hermite ensemble",
    )
    args = parser.parse_args()

    lt0 = query_lifetimes(0.0)
    lt300 = query_lifetimes(300.0)

    parameters = SELECTED_PARAMETERS.copy()
    optimization: dict[str, object] = {
        "performed": False,
        "note": "stored bounded-search pulse evaluated without refinement",
    }
    if args.optimize:
        optimization_distances = np.linspace(3.30, 3.50, 5)
        print("Building scan-basis optimization grid...", flush=True)
        optimization_modes = _build_mode_grid(
            optimization_distances, SCAN_BASIS, threshold=1e-6
        )
        print("Refining the robust composite pulse...", flush=True)
        parameters, optimization = _optimize_pulse(parameters, optimization_modes, lt0)
        optimization["performed"] = True
        print("optimized parameters:", parameters.tolist(), flush=True)

    pulse = _pulse_from_parameters(parameters)
    square_pulse = _square_pulse()

    gh_nodes, gh_weights = hermgauss(9)
    ensemble_distances = R0_UM + np.sqrt(2) * SIGMA_R_UM * gh_nodes
    ensemble_weights = gh_weights / np.sqrt(np.pi)
    print("Building primary-basis Gaussian ensemble...", flush=True)
    primary_modes = _build_mode_grid(
        ensemble_distances, PRIMARY_BASIS, threshold=1e-8
    )
    nominal_index = int(np.argmin(abs(ensemble_distances - R0_UM)))
    correction = _nominal_correction(primary_modes[nominal_index], pulse, lt0)
    square_correction = _nominal_correction(
        primary_modes[nominal_index], square_pulse, lt0
    )

    robust_0k, robust_curve = _metrics_for_pulse(
        primary_modes,
        ensemble_distances,
        ensemble_weights,
        pulse,
        lt0,
        correction,
    )
    square_0k, square_curve = _metrics_for_pulse(
        primary_modes,
        ensemble_distances,
        ensemble_weights,
        square_pulse,
        lt0,
        square_correction,
    )

    robustness = {}
    for label, errors in {
        "target_amplitude_minus_1pct": {"target_amplitude_scale": 0.99},
        "target_amplitude_plus_1pct": {"target_amplitude_scale": 1.01},
        "target_detuning_minus_50khz": {"detuning_offset_mhz": -0.05},
        "target_detuning_plus_50khz": {"detuning_offset_mhz": 0.05},
        "target_duration_minus_0p5pct": {"duration_scale": 0.995},
        "target_duration_plus_0p5pct": {"duration_scale": 1.005},
        "control_amplitude_minus_1pct": {"control_amplitude_scale": 0.99},
        "control_amplitude_plus_1pct": {"control_amplitude_scale": 1.01},
    }.items():
        curve = _evaluate_grid(primary_modes, pulse, lt0, correction, **errors)
        robustness[label] = {
            "nominal_fidelity": float(curve[nominal_index]),
            "gaussian_ensemble_fidelity": float(np.sum(ensemble_weights * curve)),
            "worst_fidelity": float(np.min(curve)),
        }

    field_robustness = {}
    for field in (B_GAUSS - 0.05, B_GAUSS + 0.05):
        field_mode = _build_mode_grid(
            np.array([R0_UM]),
            PRIMARY_BASIS,
            threshold=1e-8,
            field_gauss=field,
        )[0]
        field_fidelity = _evaluate_grid(
            [field_mode], pulse, lt0, correction
        )[0]
        field_robustness[f"{field:.2f}"] = float(field_fidelity)

    temperature_results = {}
    for label, lifetimes in (("300k", lt300), ("coherent", None)):
        curve = _evaluate_grid(primary_modes, pulse, lifetimes, correction)
        temperature_results[label] = {
            "nominal_fidelity": float(curve[nominal_index]),
            "gaussian_ensemble_fidelity": float(np.sum(ensemble_weights * curve)),
            "worst_fidelity": float(np.min(curve)),
        }

    threshold_convergence = {}
    nominal_model = build_pair_model(
        B_GAUSS, PRIMARY_BASIS, distance_um=R0_UM, theta_deg=0.0
    )
    for threshold in (1e-6, 1e-8, 1e-10):
        mode = _prepare_modes(nominal_model, threshold)
        kraus = _kraus_diagonal(mode, pulse, lt0)
        threshold_convergence[str(threshold)] = {
            "active_modes": len(mode.energies_rel_mhz),
            "retained_ss_weight": mode.retained_ss_weight,
            "fidelity": _fixed_correction_fidelity(kraus, correction),
        }

    quadrature_convergence: dict[str, object] = {
        "primary_basis_9point": {
            "robust_ensemble_fidelity": robust_0k["gaussian_ensemble_fidelity"],
            "square_ensemble_fidelity": square_0k["gaussian_ensemble_fidelity"],
        }
    }
    if args.verify_quadrature:
        nodes_17, weights_17 = hermgauss(17)
        distances_17 = R0_UM + np.sqrt(2) * SIGMA_R_UM * nodes_17
        weights_17 /= np.sqrt(np.pi)
        print("Building 17-point scan-basis convergence ensemble...", flush=True)
        modes_17 = _build_mode_grid(
            distances_17, SCAN_BASIS, threshold=1e-8
        )
        nominal_17 = int(np.argmin(abs(distances_17 - R0_UM)))
        correction_17 = _nominal_correction(modes_17[nominal_17], pulse, lt0)
        square_correction_17 = _nominal_correction(
            modes_17[nominal_17], square_pulse, lt0
        )
        robust_17 = _evaluate_grid(modes_17, pulse, lt0, correction_17)
        square_17 = _evaluate_grid(
            modes_17, square_pulse, lt0, square_correction_17
        )
        quadrature_convergence["scan_basis_17point"] = {
            "robust_nominal_fidelity": float(robust_17[nominal_17]),
            "robust_ensemble_fidelity": float(np.sum(weights_17 * robust_17)),
            "square_nominal_fidelity": float(square_17[nominal_17]),
            "square_ensemble_fidelity": float(np.sum(weights_17 * square_17)),
        }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "pairinteraction": getattr(pi, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "model": {
            "field_gauss": B_GAUSS,
            "mean_distance_um": R0_UM,
            "sigma_distance_um": SIGMA_R_UM,
            "primary_basis": asdict(PRIMARY_BASIS),
            "pulse_sequence": "Rb pi; palindromic five-segment Yb target; Rb pi",
            "fidelity": "loss-aware average CZ fidelity with one nominal local-Z correction held fixed",
            "decay": "non-Hermitian no-jump evolution; decay counted as orthogonal erasure",
            "projection": "complex PairInteraction bare-state amplitudes",
        },
        "optimization": optimization,
        "pulse_parameters": parameters.tolist(),
        "segments": [asdict(segment) for segment in pulse],
        "local_z_correction_rad": {
            "alpha": correction[0],
            "beta": correction[1],
        },
        "robust_pulse": {"0k": robust_0k, **temperature_results},
        "square_baseline_recomputed": square_0k,
        "robustness": robustness,
        "field_robustness": field_robustness,
        "threshold_convergence": threshold_convergence,
        "quadrature_convergence": quadrature_convergence,
        "ensemble": {
            "quadrature": "nine-point Gauss-Hermite",
            "distance_um": ensemble_distances.tolist(),
            "weights": ensemble_weights.tolist(),
            "robust_fidelity": robust_curve.tolist(),
            "square_fidelity": square_curve.tolist(),
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.35))
    ax = axes[0]
    edges = np.r_[0, np.cumsum([segment.duration_us for segment in pulse])]
    omega = np.array([segment.omega_mhz for segment in pulse])
    detuning = np.array([segment.detuning_mhz for segment in pulse])
    ax.stairs(omega, 1000 * edges, color="#0072B2", label=r"$\Omega/2\pi$")
    ax.set_xlabel("target-pulse time (ns)")
    ax.set_ylabel(r"$\Omega/2\pi$ (MHz)", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax_det = ax.twinx()
    ax_det.stairs(detuning, 1000 * edges, color="#D55E00", label=r"$\Delta/2\pi$")
    ax_det.set_ylabel(r"$\Delta/2\pi$ (MHz)", color="#D55E00")
    ax_det.tick_params(axis="y", labelcolor="#D55E00")
    ax.set_title("(a) five-segment target pulse")

    ax = axes[1]
    order = np.argsort(ensemble_distances)
    ax.plot(
        ensemble_distances[order],
        100 * (1 - robust_curve[order]),
        "o-",
        label="shaped pulse",
        color="#0072B2",
    )
    ax.plot(
        ensemble_distances[order],
        100 * (1 - square_curve[order]),
        "o--",
        label="square revival",
        color="#CC79A7",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ (µm)")
    ax.set_ylabel(r"fixed-correction $1-F_{\rm avg}$ (%)")
    ax.set_title("(b) distance response, 0 K")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    labels = [
        "nominal",
        r"$\Omega_t-1\%$",
        r"$\Omega_t+1\%$",
        r"$\Delta-50$ kHz",
        r"$\Delta+50$ kHz",
        r"$T_t-0.5\%$",
        r"$T_t+0.5\%$",
        r"$\Omega_c-1\%$",
        r"$\Omega_c+1\%$",
    ]
    fidelities = [robust_0k["gaussian_ensemble_fidelity"]] + [
        row["gaussian_ensemble_fidelity"] for row in robustness.values()
    ]
    values = 1e4 * (1 - np.array(fidelities))
    bars = ax.bar(np.arange(len(values)), values, color="#009E73")
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(np.arange(len(values)), labels, rotation=48, ha="right")
    ax.set_ylabel(r"ensemble infidelity $10^4(1-F_{\rm avg})$")
    ax.set_title("(c) fixed-pulse robustness")

    fig.tight_layout()
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print("wrote", OUT, flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".pdf"), flush=True)
    print("wrote", FIGURE_STEM.with_suffix(".png"), flush=True)
    print(
        f"robust pulse: nominal={robust_0k['nominal_fidelity']:.9f}, "
        f"Gaussian={robust_0k['gaussian_ensemble_fidelity']:.9f}; "
        f"square Gaussian={square_0k['gaussian_ensemble_fidelity']:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
