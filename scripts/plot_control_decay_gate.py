#!/usr/bin/env python3
"""Render Figs. 3 and 4 from corrected stored calculations, without rebuilding physics.

Preserves all pre-correction records. Run this instead of the historical
minimax --plot-only command when regenerating the manuscript gate figure.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/manuscript_claims_2026_09_10"
DATA = ARCHIVE / "simulations"
sys.path.insert(0, str(ROOT))
sys.path.append(str(ARCHIVE / "figures"))
from pulseviz import cz_phase_errors, first_order_response  # noqa: E402

from data.manuscript_claims_2026_09_10.figures.style import (  # noqa: E402
    DOUBLE_COL,
    RB_COLOR,
    SINGLE_COL,
    YB_COLOR,
    panel_label,
    use_paper_style,
)


def verify_archive(archive=ARCHIVE):
    """Check imported bytes, not obsolete per-run source hashes outside the snapshot."""
    manifest = json.loads((archive / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        raw = (archive / name).read_bytes()
        if (
            len(raw) != expected["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Corrected figure archive hash mismatch: {name}")


def load_results(data=DATA):
    verify_archive(data.parent)

    def load(suffix):
        return json.loads((data / f"forster_control_decay_{suffix}.json").read_text())

    baseline = load("recheck")
    signature = hashlib.sha256(
        json.dumps(baseline["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    if baseline["signature"] != signature:
        raise ValueError("Corrected baseline source signature mismatch")
    baseline_hash = hashlib.sha256(
        (data / "forster_control_decay_recheck.json").read_bytes()
    ).hexdigest()
    curves, p1, field = load("driven_curves"), load("p1_1"), load("field_scan")
    for record, key in (
        (curves, "provenance"),
        (p1, "control_decay_recheck_provenance"),
        (field, "control_decay_recheck_provenance"),
    ):
        if (
            record[key]["baseline_signature"] != signature
            or record[key]["baseline_sha256"] != baseline_hash
        ):
            raise ValueError(f"Corrected figure record baseline mismatch: {key}")
    grid = p1["response_grid"]
    values = np.asarray(grid["fidelity_grid"])
    iy = grid["yb_amplitude_scales"].index(1.0)
    ir = grid["rb_amplitude_scales"].index(1.0)
    axial = [(0.0, float(values[0, 0, iy, ir]))]
    for ri, radius in enumerate(grid["radii_um"][1:], start=1):
        for cosine in (-1.0, 1.0):
            ci = grid["direction_cosines"].index(cosine)
            axial.append((1000 * radius * cosine, float(values[ri, ci, iy, ir])))
    axial.sort()
    coarse = sorted(
        (
            r
            for r in field["fields"].values()
            if r["pair_window_ghz"] == 40
            and r["field_gauss"] in set(np.arange(0.0, 5.01, 0.5)) | {3.1}
        ),
        key=lambda r: r["field_gauss"],
    )
    if len(coarse) != 12:
        raise ValueError("Corrected magnetic figure requires all 12 coarse field points")
    nominal = baseline["summary"]["nominal_overlap"]
    np.testing.assert_allclose(
        [
            dict(axial)[0.0],
            curves["response_time_scan"][2]["recalibrated_nominal_fidelity"],
            field["fields"]["3.10000000/40"]["nominal"],
        ],
        nominal,
        rtol=0,
        atol=1e-10,
    )
    result = {
        "baseline": baseline,
        "constraints": curves["constraints"],
        "parameters": baseline["pulse_parameters"],
        "response_time_scan": curves["response_time_scan"],
        "target_amplitude_scan": curves["target_amplitude_scan"],
        "population_trajectory": curves["population_trajectory"],
        "axial_position_scan": {
            "delta_z_nm": [x for x, _ in axial],
            "nominal_amplitudes_fidelity": [y for _, y in axial],
        },
        "magnetic_field_scan": {
            "coarse_field_gauss": [r["field_gauss"] for r in coarse],
            "coarse_phase_recalibrated_fidelity": [r["nominal"] for r in coarse],
        },
    }
    return result


def endpoint_phase_errors(baseline):
    """Phase residuals relative to CZ, after the recorded local-Z corrections."""
    row = baseline["rows"]["0.000000/0.000000"]
    pairs = np.asarray(row["kraus_diagonal_re_im"])
    k = pairs[:, 0] + 1j * pairs[:, 1]
    errors, conditional_error = cz_phase_errors(k, row["fixed_local_z_rad"])
    np.testing.assert_allclose(
        conditional_error,
        row["phase_recalibrated_metrics"]["conditional_phase_error_rad"],
        atol=1e-12,
        rtol=0,
    )
    return 1000 * errors, 1000 * conditional_error


def save(fig, stem):
    path = ROOT / "figures" / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {path.name}.pdf and .png")


def plot_protocol(result):
    parameters = np.asarray(result["parameters"])
    # The stored symmetric five-segment command: a, b, c, b, a.
    order = [0, 1, 2, 1, 0]
    command_values = np.column_stack((parameters[:3][order], parameters[3:6][order]))
    duration = parameters[6]
    command_edges_us = np.r_[0.0, np.cumsum(np.full(5, duration))]
    rise_us = result["constraints"]["aom_rise_time_ns"] / 1000
    # Match the archived controller's seven-time-constant amplitude ring-down;
    # its detuning command stays at the last segment's value.
    all_edges_us = np.r_[command_edges_us, command_edges_us[-1] + 7 * rise_us / np.log(9)]
    all_values = np.vstack((command_values, [0.0, command_values[-1, 1]]))
    edges_us, filtered_values = first_order_response(
        all_edges_us,
        all_values,
        rise_time=rise_us,
        max_step=result["baseline"]["step_ns"] / 1000,
        initial=[0.0, command_values[0, 1]],
    )
    edges = 1000 * edges_us
    t = 1000 * np.asarray(result["population_trajectory"]["time_us"])
    np.testing.assert_allclose(edges, t, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        filtered_values[:, 0].max(),
        result["constraints"]["filtered_peak_omega_mhz"],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [filtered_values[:, 1].min(), filtered_values[:, 1].max()],
        result["constraints"]["filtered_detuning_range_mhz"],
        rtol=0,
        atol=1e-10,
    )
    offset = (1000 * result["constraints"]["total_gate_time_us"] - t[-1]) / 2
    end = offset + t[-1]
    total = end + offset
    np.testing.assert_allclose(total, 1000 * result["constraints"]["total_gate_time_us"], atol=1e-9)

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.65), layout="constrained")
    amplitude, unblocked, loss = axes[0]
    detuning, blocked, phase = axes[1]
    for ax, attribute, label in [
        (amplitude, 0, r"$\Omega_{\mathrm{Yb}}/2\pi$ (MHz)"),
        (detuning, 1, r"$\Delta_{\mathrm{Yb}}/2\pi$ (MHz)"),
    ]:
        ax.stairs(
            all_values[:, attribute],
            1000 * all_edges_us,
            baseline=None,
            color=RB_COLOR,
            linestyle="--",
            label="Command",
        )
        ax.stairs(
            filtered_values[:, attribute], edges, baseline=None, color=YB_COLOR, label="Filtered"
        )
        ax.set_ylabel(label)
    amplitude.set_ylim(0, 14)
    amplitude.set_yticks([0, 5, 10])
    amplitude.legend(loc="upper right", fontsize=7)
    detuning.set_ylim(-1.5, 3.1)
    detuning.set_yticks([-1, 0, 1, 2, 3])
    traj = result["population_trajectory"]
    unblocked.plot(t, traj["unblocked_yb_rydberg"], color=YB_COLOR)
    unblocked.text(
        0.95,
        0.9,
        r"$|01\rangle$: Yb Rydberg",
        transform=unblocked.transAxes,
        ha="right",
        fontsize=8,
    )
    for key, color, ls, label in [
        ("blocked_computational", YB_COLOR, "-", "Return channel"),
        ("blocked_pp", RB_COLOR, "-", r"$PP$"),
        ("blocked_ss", RB_COLOR, "--", r"$SS$"),
    ]:
        blocked.plot(t, traj[key], color=color, ls=ls, label=label)
    blocked.legend(loc="center", fontsize=7, handlelength=1.5)
    for ax in (unblocked, blocked):
        ax.set_ylim(-0.025, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Population")
    for ax in (amplitude, detuning, unblocked, blocked):
        ax.set_xlim(0, t[-1])
        ax.set_xticks([0, 40, 80, 120, 160])
    for ax in (amplitude, unblocked):
        ax.tick_params(labelbottom=False)
    for ax in (detuning, blocked):
        ax.set_xlabel("Target-window time (ns)")
    row = result["baseline"]["rows"]["0.000000/0.000000"]
    pairs = np.asarray(row["kraus_diagonal_re_im"])
    return_loss = 1 - np.sum(pairs**2, axis=1)
    np.testing.assert_allclose(
        return_loss,
        row["phase_recalibrated_metrics"]["computational_loss_by_input"],
        atol=1e-12,
        rtol=0,
    )
    loss.vlines(range(4), 0, 1000 * return_loss, color=YB_COLOR, lw=0.9)
    loss.plot(range(4), 1000 * return_loss, "o", color=YB_COLOR, mfc="white")
    loss.set_ylabel(r"Return loss ($10^{-3}$)")
    loss.set_ylim(bottom=-0.05)
    errors, conditional_error = endpoint_phase_errors(result["baseline"])
    phase.axhline(0, color="0.5", ls="--", lw=0.6)
    phase.vlines(range(4), 0, errors, color=RB_COLOR, lw=0.9)
    phase.plot(range(4), errors, "s", color=RB_COLOR, mfc="white")
    phase.set_ylim(-1.25, 1.25)
    phase.set_yticks([-1, 0, 1])
    phase.set_ylabel("Phase error (mrad)")
    phase.text(
        0.5,
        0.06,
        rf"$\Phi_{{\mathrm{{CZ}}}}-\pi={conditional_error:.3f}$ mrad",
        transform=phase.transAxes,
        ha="center",
        fontsize=7.5,
    )
    for ax in (loss, phase):
        ax.set_xticks(
            range(4), [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
        )
        ax.set_xlim(-0.4, 3.4)
    phase.set_xlabel("Computational input")
    for ax, title in zip(
        axes[0], ["Yb controls", "Conditional dynamics", "Full-gate return"], strict=False
    ):
        ax.set_title(title, fontsize=9, pad=7)
    for ax, letter in zip(axes.T.flat, "abcdef", strict=False):
        panel_label(ax, f"({letter})", x=-0.28)
    save(fig, "corrected_control_decay_gate")


def plot_robustness(result):
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL, 3.15), layout="constrained")
    response, position, amplitude, field = axes.flat
    rows = result["response_time_scan"]
    x = [r["rise_time_ns"] for r in rows]
    for key, ls, marker, color, label in [
        ("fixed_10ns_correction_nominal_fidelity", "-", "o", YB_COLOR, "Fixed local $Z$"),
        ("recalibrated_nominal_fidelity", "--", "s", RB_COLOR, "Recalibrated $Z$"),
    ]:
        response.plot(
            x,
            1 - np.array([r[key] for r in rows]),
            ls=ls,
            marker=marker,
            color=color,
            markerfacecolor="white",
            label=label,
        )
    response.set_xlabel(r"Rise time $t_{10-90}$ (ns)")
    response.legend(loc="upper left", fontsize=7, handlelength=1.3, borderpad=0.2)
    position.plot(
        result["axial_position_scan"]["delta_z_nm"],
        1 - np.array(result["axial_position_scan"]["nominal_amplitudes_fidelity"]),
        "o-",
        color=YB_COLOR,
        mfc="white",
    )
    position.set_xlabel(r"Axial offset $\delta z$ (nm)")
    scan = result["target_amplitude_scan"]
    for prefix, key, color, fmt, label in [
        ("yb", "nominal_position_rb_nominal_fidelity", YB_COLOR, "o-", "Yb"),
        ("rb", "nominal_position_yb_nominal_fidelity", RB_COLOR, "s--", "Rb"),
    ]:
        amplitude.plot(
            scan[prefix + "_rabi_error_pct"],
            1 - np.array(scan[key]),
            fmt,
            color=color,
            mfc="white",
            label=label,
        )
    amplitude.set_xlabel(r"Rabi error $\delta\Omega/\Omega$ (%)")
    amplitude.legend(loc="upper center", ncol=2, fontsize=7, handlelength=1.2, columnspacing=0.7)
    magnetic = result["magnetic_field_scan"]
    field.semilogy(
        magnetic["coarse_field_gauss"],
        1 - np.array(magnetic["coarse_phase_recalibrated_fidelity"]),
        "o-",
        color=YB_COLOR,
        mfc="white",
    )
    field.axvline(3.1, color="0.45", ls="--", lw=1)
    field.set_xlabel(r"Magnetic field $B$ (G)")
    field.text(
        0.97,
        0.93,
        "Carrier tracking\nRecalibrated $Z$",
        transform=field.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
    )
    for ax in (response, position, amplitude):
        ax.axhline(0.001, color="0.55", ls=":", lw=1)
        ax.set_ylim(0.00075, 0.0013)
        ax.set_yticks([0.0008, 0.001, 0.0012])
    field.set_yticks([0.001, 0.002, 0.005])
    field.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e-3:g}"))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$1-F_{\mathrm{avg}}$ ($10^{-3}$)")
    field.axhline(0.001, color="0.55", ls=":", lw=1)
    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"], strict=False):
        panel_label(ax, label)
    save(fig, "corrected_control_decay_robustness")


def main():
    use_paper_style()
    result = load_results()
    plot_protocol(result)
    plot_robustness(result)


if __name__ == "__main__":
    main()
