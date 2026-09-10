#!/usr/bin/env python3
"""S+S ('reverse-channel') Forster CZ variant with 0 K and 300 K lifetimes.

User-preferred scheme: both atoms are excited to S Rydberg states --
  * Yb: direct one-photon 302 nm pulse 6s6p 3P0 -> nu=48.369927 S (F=1/2),
  * Rb: effective two-photon pulse to 56S_1/2.
No microwave anywhere, so the control/target ordering is unconstrained;
both orderings are evaluated.  During the target pulse the optically bright
pair state is |SS> and the Forster-hybridized partner is |PP> (the reverse
of the PP-driven baseline in simulate_forster_gate.py / optimize_forster_gate.py).

Lifetimes: PairInteraction model values at 0 K (radiative only) and 300 K.

Output: simulations/ss_scheme_0k_results.json + console tables.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pairinteraction as pi
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations"))
from simulate_forster_gate import (  # noqa: E402
    PRIMARY_BASIS,
    SCAN_BASIS,
    Lifetimes,
    _kets,
    _local_z_metrics,
    build_pair_model,
    query_lifetimes,
)
import simulate_forster_gate as _sg  # noqa: E402
from rb_rydberg_hyperfine import target_stretched_shift_mhz  # noqa: E402


# --- State-selector note ---------------------------------------------------
# The documented selectors are correct under the Yb171_mqdt v1.4 database
# (P target: n=52, l=1, s=0 -> 50395.616439 cm^-1; S relay: n=53, l=0, s=1
# -> 50396.314307 cm^-1; SS-PP defect -0.763732 MHz at B=0).
# CAUTION: under the older v1.2 database the same P selector returned a state
# 13.3 GHz away; the correct v1.2 state needed float s~0.55.  The guard below
# refuses to run if the defect deviates from the documented value.
def _kets_repaired():
    rb_pp = pi.KetAtom("Rb", n=56, l=1, j=0.5, m=0.5)
    rb_ss = pi.KetAtom("Rb", n=56, l=0, j=0.5, m=0.5)
    yb_pp = pi.KetAtom("Yb171_mqdt", n=52, l=1, s=0, f=0.5, m=0.5)
    yb_ss = pi.KetAtom("Yb171_mqdt", n=53, l=0, s=1, f=0.5, m=0.5)
    return rb_pp, rb_ss, yb_pp, yb_ss


_sg._kets = _kets_repaired  # build_pair_model / query_lifetimes resolve _kets at call time

OUT = ROOT / "simulations" / "ss_scheme_0k_results.json"

FIELDS_COARSE = [0.0, 1.0, 2.0, 3.0, 3.95, 5.0]
SEARCH_THRESHOLD = 1e-6
FINAL_THRESHOLD = 1e-8
OMEGA_YB_OPTICAL_MHZ = 15.0   # direct 302 nm pi pulse (demonstrated class)
OMEGA_RB_TWOPhoton_MHZ = 5.0  # effective two-photon pi pulse
OMEGA_TARGET_MAX = {"yb_control": 5.0, "rb_control": 15.0}  # target pulse caps


def two_level_pi(omega_mhz: float, tau_us: float | None):
    """Resonant square pi pulse |g><->|r>, amplitude decay of |r>."""
    t_pi = 1.0 / (2.0 * omega_mhz)
    h = np.array([[0.0, omega_mhz / 2], [omega_mhz / 2, 0.0]], dtype=complex)
    if tau_us is not None:
        h[1, 1] -= 0.5j / (2 * np.pi * tau_us)
    return expm(-2j * np.pi * h * t_pi), t_pi


def ss_origin_offset_mhz(field_gauss: float) -> float:
    """Hyperfine-field-dressed [E(Rb56S)+E(YbS)]-[E(Rb56P)+E(YbP)]."""
    rb_pp, rb_ss, yb_pp, yb_ss = _kets_repaired()
    rb_basis = pi.BasisAtom.from_kets([rb_pp, rb_ss], delta_n=0, delta_l=0,
                                      delta_j=0, delta_m=0)
    yb_basis = pi.BasisAtom.from_kets([yb_pp, yb_ss], delta_n=0, delta_l=0,
                                      delta_f=0, delta_m=0)
    rb_sys = pi.SystemAtom(rb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    yb_sys = pi.SystemAtom(yb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    pp_e = rb_sys.get_corresponding_energy(rb_pp, unit="MHz") + yb_sys.get_corresponding_energy(yb_pp, unit="MHz")
    ss_e = rb_sys.get_corresponding_energy(rb_ss, unit="MHz") + yb_sys.get_corresponding_energy(yb_ss, unit="MHz")
    rb_hyperfine_difference = target_stretched_shift_mhz(
        56, 0, field_gauss
    ) - target_stretched_shift_mhz(56, 1, field_gauss)
    return float(ss_e - pp_e + rb_hyperfine_difference)


def prepare_ss_modes(model, lifetimes: Lifetimes, threshold: float):
    mask = np.abs(model.ss_overlap) ** 2 > threshold
    pp = model.pp_overlap[mask]
    ss = model.ss_overlap[mask]
    pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
    ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
    pp_w = np.abs(pp) ** 2
    ss_w = np.abs(ss) ** 2
    spec_w = np.maximum(0.0, 1 - pp_w - ss_w)
    rates = pp_w * pp_rate + ss_w * ss_rate + spec_w * max(pp_rate, ss_rate)
    return model.energies_mhz[mask], pp, ss, rates


def target_amplitudes(energies_rel, ss_overlap, rates, omega_mhz,
                      detuning_mhz, tau_target_us, duration_override=None):
    """Generalized 2pi pulse on the target atom; blocked branch driven via SS."""
    duration = (1.0 / np.sqrt(omega_mhz**2 + detuning_mhz**2)
                if duration_override is None else duration_override)
    h_un = np.array([[0.0, omega_mhz / 2],
                     [omega_mhz / 2, -detuning_mhz]], dtype=complex)
    if tau_target_us is not None:
        h_un[1, 1] -= 0.5j / (2 * np.pi * tau_target_us)
    unblocked = complex(expm(-2j * np.pi * h_un * duration)[0, 0])

    n = len(energies_rel)
    h_bl = np.zeros((n + 1, n + 1), dtype=complex)
    h_bl[0, 1:] = omega_mhz * np.conj(ss_overlap) / 2
    h_bl[1:, 0] = omega_mhz * ss_overlap / 2
    diag = energies_rel - detuning_mhz
    if rates is not None:
        diag = diag - 0.5j * rates / (2 * np.pi)
    h_bl[1:, 1:] = np.diag(diag)
    blocked = complex(expm(-2j * np.pi * h_bl * duration)[0, 0])
    return duration, unblocked, blocked


def gate_metrics(variant, modes, ss_offset, omega_t, det, lt: Lifetimes | None,
                 duration_override=None, control_omega_override=None):
    energies, pp, ss, rates = modes
    energies_rel = energies - ss_offset
    if variant == "yb_control":
        ctrl_omega, ctrl_tau = OMEGA_YB_OPTICAL_MHZ, (lt.yb_53s_us if lt else None)
        tgt_tau = lt.rb_56s_us if lt else None
    else:
        ctrl_omega, ctrl_tau = OMEGA_RB_TWOPhoton_MHZ, (lt.rb_56s_us if lt else None)
        tgt_tau = lt.yb_53s_us if lt else None
    if control_omega_override is not None:
        ctrl_omega = control_omega_override
    u_c, t_pi = two_level_pi(ctrl_omega, ctrl_tau)
    duration, unblocked, blocked = target_amplitudes(
        energies_rel, ss, rates if lt else None, omega_t, det, tgt_tau,
        duration_override=duration_override)
    idle = float(np.exp(-duration / (2 * ctrl_tau))) if lt else 1.0
    m_down = np.diag([1.0, idle])
    m_up = np.diag([unblocked, blocked])
    k_ctrl = complex((u_c @ m_down @ u_c)[0, 0])
    k_full = complex((u_c @ m_up @ u_c)[0, 0])
    if variant == "yb_control":
        kraus = np.array([1, k_ctrl, unblocked, k_full], dtype=complex)
    else:
        kraus = np.array([1, unblocked, k_ctrl, k_full], dtype=complex)
    metrics = _local_z_metrics(kraus)
    metrics.update({
        "omega_target_mhz": float(omega_t),
        "target_detuning_mhz": float(det),
        "target_duration_us": float(duration),
        "total_sequence_time_us": float(2 * t_pi + duration),
        "control_pi_time_us": float(t_pi),
    })
    return metrics


def optimize_target(variant, modes, ss_offset, lt, omega_max, starts=None):
    def objective(x):
        return -gate_metrics(variant, modes, ss_offset, float(x[0]), float(x[1]), lt)["average_gate_fidelity"]

    if starts is None:
        grid = np.linspace(0.4, omega_max, 117)
        vals = np.array([-objective(np.array([w, 0.0])) for w in grid])
        idx = [i for i in range(1, len(grid) - 1)
               if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]]
        starts = [(float(grid[i]), 0.0) for i in sorted(idx, key=lambda i: -vals[i])[:5]]
    best = None
    for s in starts:
        r = minimize(objective, np.asarray(s), method="Nelder-Mead",
                     bounds=[(0.3, omega_max), (-5.0, 5.0)],
                     options={"maxiter": 600, "xatol": 2e-9, "fatol": 1e-13})
        m = gate_metrics(variant, modes, ss_offset, float(r.x[0]), float(r.x[1]), lt)
        if best is None or m["average_gate_fidelity"] > best["average_gate_fidelity"]:
            best = m
    return best


def main():
    # --- guard: state identity must reproduce the documented defect
    offset0 = ss_origin_offset_mhz(0.0)
    print(
        f"hyperfine-dressed SS-PP offset at B=0: {offset0:.6f} MHz "
        "(expected -0.693745)",
        flush=True,
    )
    if abs(offset0 + 0.693745) > 0.01:
        raise RuntimeError("state-identity check failed: wrong Yb P/S selectors "
                           "(Yb171_mqdt database older than v1.4?)")
    lt0 = query_lifetimes(0.0)
    lt300 = query_lifetimes(300.0)
    print("0 K lifetimes (us):", asdict(lt0), flush=True)
    print("300 K lifetimes (us):", asdict(lt300), flush=True)

    results = {"lifetimes_0k_us": asdict(lt0), "lifetimes_300k_us": asdict(lt300),
               "assumptions": {
                   "yb_control_pi_mhz": OMEGA_YB_OPTICAL_MHZ,
                   "rb_control_pi_mhz": OMEGA_RB_TWOPhoton_MHZ,
                   "target_caps_mhz": OMEGA_TARGET_MAX,
                   "geometry": {"R_um": 3.4, "theta_deg": 0.0},
                   "ordering_note": "no microwave used; ordering unconstrained",
               },
               "coarse_scan": [], "final": {}}

    # Phase A: coarse field scan with scan basis, 0 K, both orderings
    best_by_variant = {}
    for B in FIELDS_COARSE:
        model = build_pair_model(B, SCAN_BASIS)
        offset = model.ss_asymptote_mhz
        modes = prepare_ss_modes(model, lt0, SEARCH_THRESHOLD)
        for variant in ("yb_control", "rb_control"):
            m = optimize_target(variant, modes, offset, lt0, OMEGA_TARGET_MAX[variant])
            row = {"field_gauss": B, "variant": variant, "ss_offset_mhz": offset,
                   "basis": asdict(SCAN_BASIS), **{k: m[k] for k in (
                       "average_gate_fidelity", "omega_target_mhz",
                       "target_detuning_mhz", "total_sequence_time_us",
                       "conditional_phase_error_rad")}}
            results["coarse_scan"].append(row)
            print(f"B={B:5.2f} {variant:11s} F_avg={m['average_gate_fidelity']:.7f} "
                  f"Om={m['omega_target_mhz']:.4f} det={m['target_detuning_mhz']:+.4f} "
                  f"t={m['total_sequence_time_us']:.4f}us", flush=True)
            if (variant not in best_by_variant
                    or m["average_gate_fidelity"] > best_by_variant[variant]["average_gate_fidelity"]):
                best_by_variant[variant] = {**row}

    # Phase B: primary basis at the winning field, 0 K / 300 K / coherent
    for variant, row in best_by_variant.items():
        B = row["field_gauss"]
        model = build_pair_model(B, PRIMARY_BASIS)
        offset = model.ss_asymptote_mhz
        start = [(row["omega_target_mhz"], row["target_detuning_mhz"]),
                 (row["omega_target_mhz"] * 1.02, row["target_detuning_mhz"])]
        for tag, lt in (("0k", lt0), ("300k", lt300), ("coherent", None)):
            lt_drive = lt if lt is not None else lt0
            modes_opt = prepare_ss_modes(model, lt_drive, SEARCH_THRESHOLD)
            m = optimize_target(variant, modes_opt, offset, lt_drive,
                                OMEGA_TARGET_MAX[variant], starts=start)
            modes_final = prepare_ss_modes(model, lt0 if lt is None else lt,
                                           FINAL_THRESHOLD)
            m = gate_metrics(variant, modes_final, offset,
                             m["omega_target_mhz"], m["target_detuning_mhz"], lt)
            key = f"{variant}@B{B:g}_{tag}"
            results["final"][key] = {"field_gauss": B, "variant": variant,
                                     "temperature": tag,
                                     "basis": asdict(PRIMARY_BASIS), **m}
            print(f"FINAL {key}: F_avg={m['average_gate_fidelity']:.7f} "
                  f"survival={m['mean_computational_survival']:.7f} "
                  f"phase_err={m['conditional_phase_error_rad']:+.5f} rad "
                  f"loss_by_input={['%.5f' % x for x in m['computational_loss_by_input']]}",
                  flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
