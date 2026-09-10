#!/usr/bin/env python3
"""Read stored curves; no propagation, fitting, file writes, or third-party imports.

Run from publication repo:
  python3 -B scripts/recompute_claim_diagnostics.py --simulations-dir \
    data/manuscript_claims_2026_09_10/simulations

The output is a NEW post-hoc reconstruction, not a recovered execution log.
All integrals use trapezoids on stored samples, with time in microseconds.
Full transformed-loss propagation and the optical spectator scan are NOT tested.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/manuscript_claims_2026_09_10/simulations"


def cumulative_trapezoid(time, values):
    """Integrate on the saved nonuniform microsecond grid, including zero at start."""
    out = [0.0]
    for i in range(len(time) - 1):
        out.append(out[-1] + (time[i + 1] - time[i]) * (values[i + 1] + values[i]) / 2)
    return out


def compute(baseline, curves):
    """Reconstruct estimates, without fitting them to manuscript-rounded values."""
    trajectory = curves["population_trajectory"]
    time = trajectory["time_us"]
    lifetimes = baseline["lifetimes_us"]
    keys = [
        "unblocked_yb_rydberg",
        "blocked_computational",
        "blocked_pp",
        "blocked_ss",
        "blocked_spectator",
        "blocked_loss",
    ]
    if (
        len(time) < 2
        or time[0] != 0
        or not all(math.isfinite(t) for t in time)
        or not all(b > a for a, b in zip(time, time[1:], strict=False))
    ):
        raise ValueError("population_trajectory.time_us must start at zero and strictly increase")
    for key in keys:
        if len(trajectory[key]) != len(time) or not all(math.isfinite(v) for v in trajectory[key]):
            raise ValueError(f"population_trajectory.{key} must be finite and match time_us")
    if not all(math.isfinite(v) and v > 0 for v in lifetimes.values()):
        raise ValueError("lifetimes_us must contain positive finite lifetimes")

    def cumulative(values):
        return cumulative_trapezoid(time, values)

    residence = {key: cumulative(trajectory[key])[-1] for key in keys if key != "blocked_loss"}
    # Historical labels yb_52p/yb_53s identify the target P/S states here.
    rb_s = 1 / lifetimes["rb_56s_us"]
    rb_p = 1 / lifetimes["rb_56p_us"]
    yb_s = 1 / lifetimes["yb_53s_us"]
    yb_p = 1 / lifetimes["yb_52p_us"]
    ss_rate, pp_rate = rb_s + yb_s, rb_p + yb_p
    # Same modal trajectory on BOTH sides: not a new trajectory under full D.
    bare_rate = [
        rb_s * c + ss_rate * s + pp_rate * p + max(ss_rate, pp_rate) * o
        for c, s, p, o in zip(
            trajectory["blocked_computational"],
            trajectory["blocked_ss"],
            trajectory["blocked_pp"],
            trajectory["blocked_spectator"],
            strict=False,
        )
    ]
    bare_cumulative = cumulative(bare_rate)
    differences = [
        modal - bare
        for modal, bare in zip(trajectory["blocked_loss"], bare_cumulative, strict=False)
    ]
    max_index = max(range(len(time)), key=lambda i: abs(differences[i]))

    # Ideal square pi pulse at Omega/(2*pi)=5 MHz: 100 ns, mean Rydberg P=1/2.
    pi_duration_us = 1 / (2 * 5.0)
    two_pi_residence_us = 2 * pi_duration_us / 2
    rb10_residence_us = time[-1] + two_pi_residence_us
    # First-order population-times-rate species estimate, equal input weights.
    # 00=0; 01=Yb only; 10=Rb only; 11=Rb pi pulses plus stored target window.
    # Omit spectator species assignment and opening/closing attenuation; no
    # exact full-sequence jump counting, nor infidelity attribution is claimed.
    species_by_input = {
        "00": {"Rb": 0.0, "Yb": 0.0},
        "01": {"Rb": 0.0, "Yb": residence["unblocked_yb_rydberg"] * yb_s},
        "10": {"Rb": rb10_residence_us * rb_s, "Yb": 0.0},
        "11": {
            "Rb": (
                two_pi_residence_us + residence["blocked_computational"] + residence["blocked_ss"]
            )
            * rb_s
            + residence["blocked_pp"] * rb_p,
            "Yb": residence["blocked_ss"] * yb_s + residence["blocked_pp"] * yb_p,
        },
    }
    # Constants deliberately explicit to reproduce the previous arithmetic.
    constants = dict(
        epsilon0_F_m=8.8541878128e-12,
        c_m_s=299792458,
        h_J_s=6.62607015e-34,
        e_C=1.602176634e-19,
        a0_m=5.29177210903e-11,
    )
    waist_m, dipole_ea0, frequency_hz = 12e-6, 0.00239, 11.778e6
    omega = 2 * math.pi * frequency_hz
    hbar = constants["h_J_s"] / (2 * math.pi)
    dipole = dipole_ea0 * constants["e_C"] * constants["a0_m"]
    power_mw = (
        math.pi
        * waist_m**2
        * constants["epsilon0_F_m"]
        * constants["c_m_s"]
        * (hbar * omega / dipole) ** 2
        / 4
        * 1000
    )
    output = {
        "scope": "Post-hoc finite-sample diagnostic and approximate budget; NOT full-D propagation",
        "input_keys": {
            "lifetimes": "baseline.lifetimes_us",
            "time": "curves.population_trajectory.time_us",
            "populations": ["curves.population_trajectory." + k for k in keys],
        },
        "lifetimes_us": lifetimes,
        "power": {
            "formula": "P=pi*w^2*epsilon0*c*(hbar*Omega/d)^2/4",
            "constants": constants,
            "waist_1e2_intensity_radius_m": waist_m,
            "assumed_dipole_ea0": dipole_ea0,
            "command_rabi_Hz": frequency_hz,
            "mW_before_optical_losses": power_mw,
            "selected_pulse_command_rabi_Hz": max(baseline["pulse_parameters"][:3]) * 1e6,
            "selected_pulse_mW_before_optical_losses": power_mw
            * (max(baseline["pulse_parameters"][:3]) * 1e6 / frequency_hz) ** 2,
        },
        "sample_count": len(time),
        "target_window_us": time[-1],
        "integrated_populations_ns": {k: 1000 * v for k, v in residence.items()},
        "Rb10_ideal_residence_ns": 1000 * rb10_residence_us,
        "Rb_pi_pair_ideal_residence_ns": 1000 * two_pi_residence_us,
        "diagnostic": {
            "Gamma_other_per_us": max(ss_rate, pp_rate),
            "bare_population_integral_endpoint": bare_cumulative[-1],
            "modal_norm_loss_endpoint": trajectory["blocked_loss"][-1],
            "endpoint_modal_minus_bare": differences[-1],
            "maximum_absolute_sampled_cumulative_difference": abs(differences[max_index]),
            "maximum_at_ns": 1000 * time[max_index],
            "warning": "Not a continuous-time bound or two-propagator comparison",
        },
        "approx_species_by_input": species_by_input,
        "approx_species_equal_input_mean": {
            atom: sum(row[atom] for row in species_by_input.values()) / 4 for atom in ["Rb", "Yb"]
        },
        "species_assumptions": [
            "Stored no-jump target-window populations; ideal pi-pulse residence",
            "Omit spectator species allocation and full-sequence attenuation corrections",
            "Equal 1/4 computational-input weights; first-order population/rate estimate",
            "Does not establish exact 6.70e-4/1.68e-4 historical budget or gate infidelity",
        ],
    }
    return output


def reproduce(simulations_dir=DATA):
    names = ["forster_control_decay_recheck.json", "forster_control_decay_driven_curves.json"]
    raw = [(simulations_dir / name).read_bytes() for name in names]
    baseline, curves = [json.loads(value) for value in raw]
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in zip(names, raw, strict=False)
    }
    provenance = curves["provenance"]
    if (
        provenance["baseline_sha256"] != hashes[names[0]]
        or provenance["baseline_signature"] != baseline["signature"]
    ):
        raise ValueError("Driven curves do not match the supplied corrected baseline")
    output = compute(baseline, curves)
    output["input_sha256"] = hashes
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.simulations_dir), indent=2))


if __name__ == "__main__":
    main()
