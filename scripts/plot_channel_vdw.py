#!/usr/bin/env python3
"""Figure 4: static van der Waals branch for Rb 66S + Yb S(nu=62.6823).

Pair data are read from ``data/vdw_dense_data.json``.
This figure reports U(R) and bare-product weight only; it does not plot
a gate fidelity.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channel-vdw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import DOUBLE_COL, OKABE_ITO, use_paper_style  # noqa: E402

use_paper_style()
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)

ROOT = HERE.parent
OUT = ROOT / "figures"
DATA = json.loads((ROOT / "data" / "vdw_dense_data.json").read_text())
TRACK = DATA["distance_track_b25_pp"]
SECTORS = DATA["sector_track_b25"]
C6_GHZ_UM6 = DATA["configuration"]["zero_field_c6_ghz_um6"]
R_STAR = 3.3


def values(rows, key):
    return np.array([row[key] for row in rows], dtype=float)


def panel_label(ax, text):
    ax.text(
        -0.16,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.15), layout="constrained")
    axu, axw = axes

    r_track = values(TRACK, "distance_um")
    u_track = values(TRACK, "u_mhz")
    overlap_track = values(TRACK, "overlap")

    r_guide = np.linspace(3.0, 5.0, 200)
    u_guide = C6_GHZ_UM6 * 1e3 / r_guide**6

    for ax in axes:
        ax.axvline(R_STAR, color=OKABE_ITO["vermillion"], ls=":", lw=1.0)
        ax.set_xlim(2.95, 5.05)
        ax.set_xlabel(r"$R$ ($\mu\mathrm{m}$)")

    axu.plot(
        r_track,
        u_track,
        color=OKABE_ITO["blue"],
        marker="o",
        ms=3.5,
        lw=1.4,
        label="sampled eigenvalues, $B=25\\,\\mathrm{G}$",
    )
    axu.plot(
        r_guide,
        u_guide,
        ls="--",
        color="0.35",
        lw=1.1,
        label=r"$C_6/(h R^6)$ ($B=0$, $M=+1$)",
    )
    axu.scatter(
        np.full(len(SECTORS), R_STAR),
        values(SECTORS, "u_mhz"),
        s=22,
        color=OKABE_ITO["vermillion"],
        zorder=3,
        edgecolors="white",
        linewidths=0.4,
        label="magnetic sectors",
    )
    axu.set_ylim(0, 110)
    axu.set_ylabel(r"$U/h$ (MHz)")
    axu.legend(frameon=False, loc="upper right")
    panel_label(axu, r"$\mathrm{(a)}$")

    axw.plot(
        r_track,
        overlap_track,
        color=OKABE_ITO["purple"],
        marker="o",
        ms=3.5,
        lw=1.4,
        label="sampled eigenvalues, $B=25\\,\\mathrm{G}$",
    )
    axw.scatter(
        np.full(len(SECTORS), R_STAR),
        values(SECTORS, "overlap"),
        s=22,
        color=OKABE_ITO["vermillion"],
        zorder=3,
        edgecolors="white",
        linewidths=0.4,
    )
    axw.set_ylim(0.935, 1.001)
    axw.set_ylabel(r"bare-product weight $w$")
    panel_label(axw, r"$\mathrm{(b)}$")

    fig.savefig(OUT / "channel_vdw_ur.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "channel_vdw_ur.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("wrote channel_vdw_ur.pdf")


if __name__ == "__main__":
    main()
