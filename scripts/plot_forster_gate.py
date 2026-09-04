#!/usr/bin/env python3
"""Regenerate the composite-gate figure from the committed gate record."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rbyb-publication")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from style import OKABE_ITO, use_paper_style  # noqa: E402


def main() -> None:
    use_paper_style()
    data = json.loads((ROOT / "data" / "forster_gate_results.json").read_text())
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    command = data["command_segments"]
    target_start_ns = 100.0
    target_edges_ns = (
        target_start_ns + 1000 * np.r_[0.0, np.cumsum([row["duration_us"] for row in command])]
    )
    target_end_ns = target_start_ns + 1000 * data["short_minimax"]["target_duration_us"]
    gate_end_ns = 1000 * data["short_minimax"]["total_gate_time_us"]

    command_spec = axes[0, 0].get_subplotspec()
    axes[0, 0].remove()
    command_grid = command_spec.subgridspec(2, 1, height_ratios=(0.7, 2), hspace=0.06)
    rb_axis = fig.add_subplot(command_grid[0])
    command_axis = fig.add_subplot(command_grid[1], sharex=rb_axis)
    rb_axis.barh(
        [0, 0],
        [100, 100],
        left=[0, target_end_ns],
        height=0.5,
        color=OKABE_ITO["vermillion"],
    )
    rb_axis.text(50, 0, r"$\pi$", ha="center", va="center", fontsize=9)
    rb_axis.text(target_end_ns + 50, 0, r"$\pi$ reverse", ha="center", va="center", fontsize=8)
    rb_axis.set_ylim(-0.48, 0.48)
    rb_axis.set_yticks([0], ["Rb control"])
    rb_axis.set_title("(a) ideal command-pulse sequence")
    rb_axis.spines[["left", "right", "top", "bottom"]].set_visible(False)
    rb_axis.tick_params(axis="both", length=0, labelbottom=False)

    edges = np.r_[0, target_edges_ns, target_end_ns, gate_end_ns]
    omega = np.r_[0, [row["omega_mhz"] for row in command], 0, 0]
    detuning = np.r_[0, [row["detuning_mhz"] for row in command], 0, 0]
    omega_plot = command_axis.stairs(
        omega, edges, color=OKABE_ITO["blue"], lw=1.8, label=r"$\Omega(t)$"
    )
    detuning_axis = command_axis.twinx()
    detuning_plot = detuning_axis.stairs(
        detuning,
        edges,
        color=OKABE_ITO["vermillion"],
        ls="--",
        lw=1.6,
        label=r"$\Delta(t)$",
    )
    command_axis.set_xlim(-5, gate_end_ns + 5)
    command_axis.set_ylabel(r"Yb $\Omega/2\pi$ (MHz)")
    detuning_axis.set_ylabel(r"Yb $\Delta/2\pi$ (MHz)")
    command_axis.set_xlabel("command-sequence time (ns)")
    command_axis.legend(
        [omega_plot, detuning_plot],
        [r"$\Omega(t)$", r"$\Delta(t)$"],
        frameon=False,
        fontsize=8,
        loc="upper left",
        ncol=2,
    )

    robustness_spec = axes[0, 1].get_subplotspec()
    axes[0, 1].remove()
    robustness_grid = robustness_spec.subgridspec(1, 3, wspace=0.18)
    response_axis = fig.add_subplot(robustness_grid[0])
    position_axis = fig.add_subplot(robustness_grid[1], sharey=response_axis)
    amplitude_axis = fig.add_subplot(robustness_grid[2], sharey=response_axis)

    response = data["response_time_scan"]
    response_axis.plot(
        [row["rise_time_ns"] for row in response],
        100 * np.array([row["recalibrated_nominal_fidelity"] for row in response]),
        "o-",
        color=OKABE_ITO["green"],
        ms=3.5,
    )
    response_axis.set_title("(b) response", fontsize=9)
    response_axis.set_xlabel(r"$t_{10-90}$ (ns)", fontsize=8)
    response_axis.set_ylabel(r"$F_{\rm avg}$ (%)", fontsize=8)

    position = data["axial_position_scan"]
    position_axis.plot(
        position["delta_z_nm"],
        100 * np.array(position["nominal_amplitudes_fidelity"]),
        "o-",
        color=OKABE_ITO["blue"],
        ms=3.5,
    )
    position_axis.set_title("position", fontsize=9)
    position_axis.set_xlabel(r"$\delta z$ (nm)", fontsize=8)
    position_axis.tick_params(axis="y", labelleft=False)

    amplitude = data["target_amplitude_scan"]
    amplitude_axis.plot(
        amplitude["yb_rabi_error_pct"],
        100 * np.array(amplitude["nominal_position_rb_nominal_fidelity"]),
        "o-",
        color=OKABE_ITO["purple"],
        ms=3.5,
        label="Yb",
    )
    amplitude_axis.plot(
        amplitude["rb_rabi_error_pct"],
        100 * np.array(amplitude["nominal_position_yb_nominal_fidelity"]),
        "s--",
        color=OKABE_ITO["orange"],
        ms=3.2,
        label="Rb",
    )
    amplitude_axis.set_title("amplitude", fontsize=9)
    amplitude_axis.set_xlabel(r"$\delta\Omega/\Omega$ (%)", fontsize=8)
    amplitude_axis.tick_params(axis="y", labelleft=False)
    amplitude_axis.legend(frameon=False, fontsize=7, loc="lower center")
    for axis in (response_axis, position_axis, amplitude_axis):
        axis.axhline(99.9, color="0.4", ls=":", lw=0.9)
        axis.set_ylim(99.88, 99.935)
        axis.tick_params(axis="both", labelsize=7)

    field = data["magnetic_field_scan"]
    field_axis = axes[1, 0]
    field_axis.plot(
        field["coarse_field_gauss"],
        100 * np.array(field["coarse_phase_recalibrated_fidelity"]),
        "o-",
        color=OKABE_ITO["vermillion"],
        ms=3.5,
        lw=1,
    )
    field_axis.axvspan(2.8, 3.4, color=OKABE_ITO["green"], alpha=0.1, lw=0)
    field_axis.axvline(3.1, color=OKABE_ITO["green"], ls="--", lw=0.9)
    field_axis.set_xlabel(r"$B$ (G)")
    field_axis.set_ylabel(r"$F_{\rm avg}$ (%)")
    field_axis.set_title("(c) gate fidelity vs magnetic field")

    trajectory = data["population_trajectory"]
    population_axis = axes[1, 1]
    time = np.array(trajectory["time_us"])
    for key, label, color, linestyle in (
        ("unblocked_yb_rydberg", r"$|01\rangle$: Yb Rydberg", "0.45", "-"),
        ("blocked_computational", r"$|11\rangle$: computational", OKABE_ITO["orange"], "-"),
        ("blocked_pp", r"$|11\rangle$: $|PP\rangle$", OKABE_ITO["vermillion"], "-"),
        ("blocked_ss", r"$|11\rangle$: $|SS\rangle$", OKABE_ITO["blue"], "--"),
    ):
        population_axis.plot(
            time, 100 * np.array(trajectory[key]), color=color, ls=linestyle, label=label
        )
    population_axis.set_xlabel(r"$t$ during filtered Yb target pulse ($\mu$s)")
    population_axis.set_ylabel("population (%)")
    population_axis.set_ylim(-3, 105)
    population_axis.set_title("(d) population transfer")
    population_axis.legend(frameon=False, fontsize=6.5, loc="center right")

    fig.tight_layout(h_pad=4)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(output / f"bounded_minimax_forster_gate.{suffix}", **options)
    plt.close(fig)
    print("wrote figures/bounded_minimax_forster_gate.{pdf,png}")


if __name__ == "__main__":
    main()
