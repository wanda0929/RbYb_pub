#!/usr/bin/env python3
"""Generate the selection-rule-allowed candidate excitation schematic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rbyb-publication")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from style import OKABE_ITO, RB_COLOR, SINGLE_COL, YB_COLOR, use_paper_style  # noqa: E402


def main() -> None:
    use_paper_style()
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(SINGLE_COL, 3.15))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.82])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.04, 0.91, "excitation protocol", fontsize=9, va="bottom", ha="left")

    ax.plot([0.49, 0.49], [0.03, 0.97], color="0.85", lw=0.6)
    ax.text(0.03, 0.97, "Yb (target)", fontsize=8, fontweight="bold", color=YB_COLOR, va="top")
    ax.plot([0.05, 0.44], [0.16, 0.16], color="0.15", lw=1.25)
    ax.text(0.05, 0.08, r"$^3P_0,\,F{=}1/2$", fontsize=6.2)
    ax.plot([0.05, 0.44], [0.82, 0.82], color=YB_COLOR, lw=1.25)
    ax.text(0.05, 0.86, r"$S,\,\nu{=}48.37$", fontsize=6.2, color=YB_COLOR)
    ax.annotate(
        "",
        xy=(0.24, 0.81),
        xytext=(0.24, 0.17),
        arrowprops={"arrowstyle": "-|>", "color": YB_COLOR, "lw": 1.3, "mutation_scale": 10},
    )
    ax.text(0.26, 0.48, "302.043 nm\none photon", fontsize=6.3, color=YB_COLOR, va="center")

    ax.text(0.53, 0.97, "Rb (control)", fontsize=8, fontweight="bold", color=RB_COLOR, va="top")
    y_ground, y_virtual, y_intermediate, y_rydberg = 0.16, 0.42, 0.56, 0.82
    x_left, x_right = 0.53, 0.97
    ax.plot([x_left, x_right], [y_ground, y_ground], color="0.15", lw=1.25)
    ax.text(x_left, 0.08, r"$5S_{1/2}$", fontsize=6.2)
    ax.plot([x_left, x_right], [y_rydberg, y_rydberg], color=RB_COLOR, lw=1.25)
    ax.text(x_left, 0.86, r"$56S_{1/2}$", fontsize=6.2, color=RB_COLOR)
    ax.plot([x_left, x_right], [y_intermediate, y_intermediate], color="0.15", lw=1.15)
    ax.text(
        x_right,
        y_intermediate + 0.018,
        r"$5P_{3/2}$",
        fontsize=6.2,
        ha="right",
        va="bottom",
        color="0.2",
    )
    ax.plot(
        [x_left, x_right],
        [y_virtual, y_virtual],
        color="0.4",
        lw=1,
        ls=(0, (2, 1.5)),
    )

    sky = OKABE_ITO["skyblue"]
    vermillion = OKABE_ITO["vermillion"]
    ax.annotate(
        "",
        xy=(0.66, y_virtual),
        xytext=(0.66, y_ground + 0.01),
        arrowprops={"arrowstyle": "-|>", "color": vermillion, "lw": 1.2, "mutation_scale": 9},
    )
    ax.annotate(
        "",
        xy=(0.84, y_rydberg - 0.01),
        xytext=(0.84, y_virtual),
        arrowprops={"arrowstyle": "-|>", "color": sky, "lw": 1.2, "mutation_scale": 9},
    )
    ax.text(0.64, 0.28, "780 nm", fontsize=6.2, color=vermillion, ha="right")
    ax.text(0.86, 0.64, "480 nm", fontsize=6.2, color=sky)
    ax.annotate(
        "",
        xy=(0.58, y_intermediate),
        xytext=(0.58, y_virtual),
        arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 0.8, "mutation_scale": 7},
    )
    ax.text(
        0.57,
        0.5 * (y_intermediate + y_virtual),
        r"$\Delta$",
        fontsize=7,
        ha="right",
        va="center",
        color="0.35",
    )

    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(
            output / f"channel_forster_ss_excitation.{suffix}",
            bbox_inches="tight",
            pad_inches=0.15,
            **options,
        )
    plt.close(fig)
    print("wrote figures/channel_forster_ss_excitation.{pdf,png}")


if __name__ == "__main__":
    main()
