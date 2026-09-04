#!/usr/bin/env python3
"""Generate the Förster-channel characterization figure.

The characterization panels read the machine-generated output of
``data/forster_characterization.json``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channel-forster")

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from style import DOUBLE_COL, OKABE_ITO, RB_COLOR, YB_COLOR, use_paper_style  # noqa: E402

use_paper_style()

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE = OKABE_ITO["blue"]
GREEN = OKABE_ITO["green"]
ORANGE = OKABE_ITO["orange"]
PURPLE = OKABE_ITO["purple"]
SKY = OKABE_ITO["skyblue"]
VERMILLION = OKABE_ITO["vermillion"]

DATA_PATH = ROOT / "data" / "forster_characterization.json"
with DATA_PATH.open() as handle:
    CHARACTERIZATION = json.load(handle)

DISTANCE_SCAN = CHARACTERIZATION["distance_scan"]
FIXED_M_FIELD_SCAN = CHARACTERIZATION["fixed_m_field_scan"]
ANGLE_SCAN = CHARACTERIZATION["fixed_m_angle_scan"]
OPERATING_POINT = CHARACTERIZATION["operating_point"]
ENERGY_EXCHANGE = CHARACTERIZATION["target_states"]["energy_exchange"]

R_UM = np.array([point["distance_um"] for point in DISTANCE_SCAN])
TRANSFER_PCT = 100 * np.array(
    [point["first_exchange_maximum"]["pp_population"] for point in DISTANCE_SCAN]
)
LEAKAGE_PCT = 100 * np.array(
    [point["first_exchange_maximum"]["spectator_population"] for point in DISTANCE_SCAN]
)
B_G = np.array([point["field_gauss"] for point in FIXED_M_FIELD_SCAN])
B_CONTRAST_PCT = 100 * np.array(
    [point["first_exchange_maximum"]["pp_population"] for point in FIXED_M_FIELD_SCAN]
)
ANGLE_DEG = np.array([point["theta_deg"] for point in ANGLE_SCAN])
ANGLE_CONTRAST_PCT = 100 * np.array(
    [point["first_exchange_maximum"]["pp_population"] for point in ANGLE_SCAN]
)

MW_GHZ = ENERGY_EXCHANGE["yb_released_ghz"]
RB_SP_GHZ = ENERGY_EXCHANGE["rb_absorbed_ghz"]
YB_SP_DIPOLE_EA0 = 287.563677
OPTICAL_S_NM = 302.04338880657144
DIRECT_P_NM = 302.04975559878864


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {OUT / f'{stem}.pdf'} and .png")


def characterize_channel() -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 5.25), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1])

    # (a) Species-resolved energy-level diagram for the exchange channel.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        -0.13,
        1.06,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.06,
        1.06,
        "Rb–Yb resonant energy exchange",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.text(0.24, 0.77, r"$^{87}$Rb", color=RB_COLOR, ha="center", fontsize=8.2, fontweight="bold")
    ax.text(0.76, 0.77, r"$^{171}$Yb", color=YB_COLOR, ha="center", fontsize=8.2, fontweight="bold")

    y_high, y_low = 0.60, 0.28
    ax.plot([0.06, 0.42], [y_high, y_high], color=RB_COLOR, lw=1.6)
    ax.plot([0.06, 0.42], [y_low, y_low], color=RB_COLOR, lw=1.6)
    ax.text(0.24, y_high + 0.035, r"$56P_{1/2}$", ha="center", va="bottom", fontsize=7.3)
    ax.text(0.24, y_low - 0.035, r"$56S_{1/2}$", ha="center", va="top", fontsize=7.3)
    ax.annotate(
        "",
        xy=(0.11, y_high - 0.02),
        xytext=(0.11, y_low + 0.02),
        arrowprops=dict(arrowstyle="-|>", color=RB_COLOR, lw=1.3),
    )
    ax.text(
        0.15,
        0.44,
        "absorbs\n" + rf"${RB_SP_GHZ:.6f}$ GHz",
        color=RB_COLOR,
        ha="left",
        va="center",
        fontsize=6.3,
    )

    ax.plot([0.58, 0.94], [y_high, y_high], color=YB_COLOR, lw=1.6)
    ax.plot([0.58, 0.94], [y_low, y_low], color=YB_COLOR, lw=1.6)
    ax.text(
        0.76,
        y_high + 0.035,
        r"$S$: $\nu{=}48.370$, $L{=}0$",
        ha="center",
        va="bottom",
        fontsize=6.7,
    )
    ax.text(
        0.76,
        y_low - 0.035,
        r"$P$: $\nu{=}48.014$, $L{=}1$",
        ha="center",
        va="top",
        fontsize=6.7,
    )
    ax.annotate(
        "",
        xy=(0.89, y_low + 0.02),
        xytext=(0.89, y_high - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=YB_COLOR, lw=1.3),
    )
    ax.text(
        0.85,
        0.44,
        "releases\n" + rf"${MW_GHZ:.6f}$ GHz",
        color=YB_COLOR,
        ha="right",
        va="center",
        fontsize=6.3,
    )

    ax.text(
        0.5,
        0.10,
        r"$|56P_{1/2},P_{\rm Yb}\rangle\ \leftrightarrow\ "
        r"|56S_{1/2},S_{\rm Yb}\rangle$",
        color=PURPLE,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
    )

    # (b) Bright eigenstate decomposition.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    bright_states = list(reversed(OPERATING_POINT["bright_states"]))
    labels = [r"$|\widetilde{+}\rangle$", r"$|\widetilde{-}\rangle$"]
    pp = 100 * np.array([state["pp_weight"] for state in bright_states])
    ss = 100 * np.array([state["ss_weight"] for state in bright_states])
    other = 100 * np.array([state["other_weight"] for state in bright_states])
    y = np.arange(2)
    ax.barh(y, pp, color=VERMILLION, label=r"$|PP\rangle$")
    ax.barh(y, ss, left=pp, color=BLUE, label=r"$|SS\rangle$")
    ax.barh(y, other, left=pp + ss, color="0.65", label="other")
    for i in range(2):
        ax.text(pp[i] / 2, i, f"{pp[i]:.1f}%", ha="center", va="center", color="white", fontsize=7)
        ax.text(
            pp[i] + ss[i] / 2,
            i,
            f"{ss[i]:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("bare-pair weight (%)")
    ax.set_title("Same Hamiltonian as the transfer curve", fontsize=8)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3)

    # (c) Distance scan.
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "(c)")
    ax.plot(R_UM, TRANSFER_PCT, "o-", color=BLUE, label="first-max transfer")
    operating_transfer = 100 * OPERATING_POINT["first_exchange_maximum"]["pp_population"]
    ax.scatter(
        [3.4],
        [operating_transfer],
        marker="*",
        s=75,
        color=GREEN,
        edgecolor="0.2",
        linewidth=0.4,
        zorder=5,
    )
    ax.set_xlabel(r"separation $R$ ($\mu$m)")
    ax.set_ylabel("transfer (%)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(94, 100)
    ax2 = ax.twinx()
    ax2.plot(R_UM, LEAKAGE_PCT, "s--", color=VERMILLION, label="out-of-subspace leakage")
    ax2.set_ylabel("coherent leakage (%)", color=VERMILLION)
    ax2.tick_params(axis="y", colors=VERMILLION)
    ax2.set_ylim(0, 0.75)
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="transfer"),
        Line2D([0], [0], color=VERMILLION, marker="s", ls="--", label="leakage"),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        loc="upper right",
        fontsize=7,
    )
    ax.set_title(r"Fixed-$m$ distance scan at $B=0$, $\theta=0$", fontsize=8)

    # (d) Field and geometry robustness, kept as separate x axes.
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.03)
    ax_b = fig.add_subplot(sub[0, 0])
    panel_label(ax_b, "(d)", x=-0.18)
    ax_b.plot(B_G, B_CONTRAST_PCT, "o-", color=PURPLE, lw=0.9, ms=2.2, label="fixed-electronic-$m$")
    # Hyperfine-resolved reconstruction from the same Hamiltonian and data
    # record as the composite-gate calculation.
    result_path = ROOT / "data" / "forster_gate_results.json"
    with open(result_path) as fh:
        field_scan = json.load(fh)["magnetic_field_scan"]
    allm_b = np.asarray(field_scan["coarse_field_gauss"])
    allm_swap = 100 * np.asarray(
        [point["maximum_ss_population"] for point in field_scan["coarse_static_transfer"]]
    )
    ax_b.plot(
        allm_b,
        allm_swap,
        "D--",
        color=VERMILLION,
        lw=0.8,
        ms=2.2,
        label="all-$(m_J,m_I)$",
        zorder=5,
    )
    # Adopted S+S gate operating point: B* = 3.10 G.
    b_star = 3.10
    ax_b.axvline(b_star, color=GREEN, lw=0.9, ls="--", zorder=1)
    ax_b.set_xlabel(r"$B$ (G)")
    ax_b.set_ylabel("swap contrast (%)")
    ax_b.set_ylim(82, 101)
    ax_b.set_title("field scan", fontsize=8)
    ax_b.legend(frameon=False, loc="lower right", fontsize=6.5)

    ax_t = fig.add_subplot(sub[0, 1], sharey=ax_b)
    ax_t.plot(ANGLE_DEG, ANGLE_CONTRAST_PCT, "o-", color=ORANGE, lw=0.9, ms=2.2)
    ax_t.axvspan(0, 30, color=GREEN, alpha=0.10, lw=0)
    ax_t.set_xlabel(r"angle $\theta$ (deg)")
    ax_t.set_title(r"fixed-$m$ angle scan", fontsize=8)
    ax_t.tick_params(labelleft=False)

    fig.suptitle(
        r"Rb–Yb Förster channel: finite-basis $SS\leftrightarrow PP$ characterization",
        fontsize=10,
    )
    save_figure(fig, "channel_forster_characterization")


def evolve_cycles_hamiltonian(h_mhz: np.ndarray, time_us: float, initial: np.ndarray) -> np.ndarray:
    """Evolve exp(-i 2π H t) for H in MHz and t in microseconds."""
    eigenvalues, eigenvectors = np.linalg.eigh(h_mhz)
    phases = np.exp(-2j * np.pi * eigenvalues * time_us)
    return eigenvectors @ (phases * (eigenvectors.conj().T @ initial))


def reduced_blockade_result(
    omega_mhz: float,
    v_mhz: float = 15.4,
    delta_mhz: float = 0.4,
) -> tuple[float, float, np.ndarray]:
    """Return population error, return phase, and final three-state vector."""
    h = np.array(
        [
            [0.0, omega_mhz / 2.0, 0.0],
            [omega_mhz / 2.0, 0.0, v_mhz],
            [0.0, v_mhz, -delta_mhz],
        ]
    )
    final = evolve_cycles_hamiltonian(h, 1.0 / omega_mhz, np.array([1.0, 0.0, 0.0]))
    return 1.0 - abs(final[0]) ** 2, np.angle(final[0]), final


def gate_model() -> None:
    omega_star = 15.4 / np.sqrt(8.0**2 - 0.25)
    v_star = 15.4
    delta = 0.4

    fig = plt.figure(figsize=(DOUBLE_COL, 5.35), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # (a) Sequential protocol and pair-state mechanism.
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "(a)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    edges = [0.05, 0.28, 0.73, 0.96]
    colors = [YB_COLOR, RB_COLOR, YB_COLOR]
    labels = [r"Yb $\pi$", r"Rb $2\pi$", r"Yb $\pi$"]
    for left, right, color, label in zip(edges[:-1], edges[1:], colors, labels, strict=True):
        ax.add_patch(
            Rectangle(
                (left, 0.68), right - left, 0.16, facecolor=color, alpha=0.20, edgecolor=color
            )
        )
        ax.text((left + right) / 2, 0.76, label, ha="center", va="center", fontsize=7.2)
    ax.annotate(
        "time", xy=(0.97, 0.60), xytext=(0.05, 0.60), arrowprops=dict(arrowstyle="->", lw=0.8)
    )
    ax.text(0.5, 0.50, "During the target pulse", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.08, 0.31, r"$|1_{\rm Rb},r_{\rm Yb}\rangle$", ha="left", va="center", fontsize=7)
    ax.text(0.50, 0.31, r"$|PP\rangle$", ha="center", va="center", fontsize=7)
    ax.text(0.90, 0.31, r"$|SS\rangle$", ha="right", va="center", fontsize=7)
    ax.annotate(
        "",
        xy=(0.43, 0.31),
        xytext=(0.25, 0.31),
        arrowprops=dict(arrowstyle="<->", color=RB_COLOR, lw=1.2),
    )
    ax.annotate(
        "",
        xy=(0.80, 0.31),
        xytext=(0.59, 0.31),
        arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.2),
    )
    ax.text(0.34, 0.36, r"$\Omega_{\rm Rb}/2$", color=RB_COLOR, ha="center", fontsize=6.8)
    ax.text(0.695, 0.36, r"$V/h=15.4$ MHz", color=PURPLE, ha="center", fontsize=6.8)
    ax.text(
        0.5,
        0.09,
        "Microwave off: the Förster splitting moves the target-excited branch.",
        ha="center",
        fontsize=6.5,
        color="0.3",
    )
    ax.set_title("Role-reversed sequential blockade-CZ candidate", fontsize=8)

    # (b) Explicit reduced-model dynamics for the target 2π pulse.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    t_gate = 1.0 / omega_star
    times = np.linspace(0, t_gate, 450)
    unblocked_excited = np.sin(np.pi * omega_star * times) ** 2
    blockaded = []
    h = np.array(
        [
            [0.0, omega_star / 2.0, 0.0],
            [omega_star / 2.0, 0.0, v_star],
            [0.0, v_star, -delta],
        ]
    )
    for time in times:
        state = evolve_cycles_hamiltonian(h, time, np.array([1.0, 0.0, 0.0]))
        blockaded.append(np.abs(state) ** 2)
    blockaded = np.asarray(blockaded)
    ax.plot(times, unblocked_excited, color=RB_COLOR, label="unblocked Rb Rydberg")
    ax.plot(times, blockaded[:, 0], color="0.15", label="blockaded return state")
    ax.plot(times, blockaded[:, 1], color=VERMILLION, ls="--", label=r"$|PP\rangle$")
    ax.plot(times, blockaded[:, 2], color=PURPLE, ls=":", label=r"$|SS\rangle$")
    ax.set_xlabel(r"target-pulse time ($\mu$s)")
    ax.set_ylabel("population")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title(rf"Reduced dynamics: $\Omega_{{\rm Rb}}/2\pi={omega_star:.3f}$ MHz", fontsize=8)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        edgecolor="none",
        ncol=1,
        loc="center right",
        fontsize=6.5,
    )

    # (c) Reduced return error and coherent return phase versus drive.
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "(c)")
    omegas = np.linspace(0.8, 4.0, 360)
    result = np.array([reduced_blockade_result(omega, v_star, delta)[:2] for omega in omegas])
    errors = result[:, 0]
    phases = np.unwrap(result[:, 1])
    ax.semilogy(omegas, errors, color=VERMILLION, label="return population error")
    star_error, star_phase, _ = reduced_blockade_result(omega_star, v_star, delta)
    ax.scatter(
        [omega_star],
        [star_error],
        marker="*",
        s=65,
        color=GREEN,
        edgecolor="0.2",
        linewidth=0.4,
        zorder=5,
    )
    ax.set_xlabel(r"effective Rb drive $\Omega_{\rm Rb}/2\pi$ (MHz)")
    ax.set_ylabel("reduced return error", color=VERMILLION)
    ax.tick_params(axis="y", colors=VERMILLION)
    ax.set_ylim(1e-5, 1)
    ax2 = ax.twinx()
    ax2.plot(omegas, phases, color=BLUE, alpha=0.8, label="return phase")
    ax2.scatter(
        [omega_star],
        [star_phase],
        marker="*",
        s=65,
        color=GREEN,
        edgecolor="0.2",
        linewidth=0.4,
        zorder=5,
    )
    ax2.set_ylabel("blockaded return phase (rad)", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(-0.9, 0.9)
    ax.text(
        0.04,
        0.08,
        rf"$k=8$: {100 * star_error:.3f}% error" + "\n" + rf"phase {star_phase:.4f} rad",
        transform=ax.transAxes,
        fontsize=6.8,
        color=GREEN,
    )
    ax.set_title(r"At $R=3.4\,\mu$m and $\delta/2\pi=0.4$ MHz", fontsize=8)

    # (d) Geometry/drive design map using V proportional to R^-3.
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "(d)")
    rs = np.linspace(3.0, 4.3, 145)
    omega_grid = np.linspace(0.8, 4.0, 170)
    err_map = np.empty((len(omega_grid), len(rs)))
    for j, radius in enumerate(rs):
        v = v_star * (3.4 / radius) ** 3
        for i, omega in enumerate(omega_grid):
            err_map[i, j] = max(reduced_blockade_result(omega, v, delta)[0], 1e-7)
    mesh = ax.pcolormesh(
        rs, omega_grid, err_map, shading="auto", cmap="magma_r", norm=LogNorm(1e-5, 1)
    )
    ax.scatter([3.4], [omega_star], marker="*", s=62, color=SKY, edgecolor="white", linewidth=0.6)
    ax.set_xlabel(r"separation $R$ ($\mu$m)")
    ax.set_ylabel(r"$\Omega_{\rm Rb}/2\pi$ (MHz)")
    ax.set_title(r"Reduced return error; $V\propto R^{-3}$", fontsize=8)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("return error")

    fig.suptitle(
        "Candidate π–2π–π gate: reduced coherent model (not a gate fidelity)",
        fontsize=10,
    )
    save_figure(fig, "channel_forster_gate_model")


def draw_level(
    ax: plt.Axes, y: float, label: str, color: str = "0.15", x0: float = 0.12, x1: float = 0.88
) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=1.5)
    ax.text((x0 + x1) / 2, y + 0.035, label, ha="center", va="bottom", fontsize=7.1, color=color)


def yb_excitation() -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 6.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])

    # (a) Selection-rule verdict and optical-microwave ladder.
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "(a)", x=-0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_g, y_p, y_s = 0.12, 0.62, 0.76
    draw_level(ax, y_g, r"$|g^o\rangle=6s6p\,{}^3P_0^o(F=1/2)$", YB_COLOR)
    ax.plot([0.12, 0.88], [y_p, y_p], color=VERMILLION, lw=1.5)
    ax.text(
        0.12,
        y_p - 0.035,
        r"$|r^o\rangle$: $\nu=48.014,L=1,F=1/2$",
        ha="left",
        va="top",
        fontsize=7.1,
        color=VERMILLION,
    )
    draw_level(ax, y_s, r"$|a^e\rangle$: $\nu=48.370,L=0,F=1/2$", BLUE)
    ax.annotate(
        "",
        xy=(0.36, y_s - 0.01),
        xytext=(0.36, y_g + 0.015),
        arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.5),
    )
    ax.text(
        0.31,
        0.45,
        "302.043 nm optical\nodd→even: E1 allowed",
        ha="right",
        va="center",
        color=PURPLE,
        fontsize=6.5,
    )
    ax.annotate(
        "",
        xy=(0.70, y_p + 0.005),
        xytext=(0.70, y_s - 0.005),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5),
    )
    ax.text(
        0.74,
        0.69,
        "~20.9 GHz MW\nemission; E1 allowed",
        ha="left",
        va="center",
        color=ORANGE,
        fontsize=6.3,
    )
    ax.annotate(
        "",
        xy=(0.92, y_p - 0.01),
        xytext=(0.92, y_g + 0.015),
        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.9, linestyle="--"),
    )
    ax.text(
        0.92, 0.38, "×", ha="center", va="center", color=VERMILLION, fontsize=15, fontweight="bold"
    )
    ax.text(
        0.86,
        0.34,
        "direct 302.050 nm\nodd→odd\nE1 forbidden",
        ha="right",
        va="center",
        color="0.35",
        fontsize=6.2,
    )
    ax.text(
        0.5,
        0.97,
        r"$\langle r|d_0|a\rangle=287.6\,ea_0$"
        "\n$\\Omega_{\\mu}/2\\pi=3.68$ MHz per V m$^{-1}$",
        ha="center",
        va="top",
        fontsize=6.4,
    )
    ax.text(0.98, 0.52, "energy not to scale", ha="right", fontsize=5.9, color="0.45")
    ax.set_title("Selection-rule verdict", fontsize=8)

    # (b) Detuned Raman construction.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)", x=-0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.08, 0.35], [0.20, 0.20], color=YB_COLOR, lw=1.5)
    ax.plot([0.65, 0.92], [0.20, 0.20], color=VERMILLION, lw=1.5)
    ax.plot([0.36, 0.64], [0.83, 0.83], color=BLUE, lw=1.5)
    ax.plot([0.36, 0.64], [0.72, 0.72], color="0.55", lw=0.8, ls="--")
    ax.text(0.215, 0.13, r"$|g\rangle$", ha="center", fontsize=7.2)
    ax.text(0.785, 0.13, r"$|r\rangle$", ha="center", fontsize=7.2)
    ax.text(0.50, 0.88, r"$|a\rangle$ (S relay)", ha="center", fontsize=7.2)
    ax.annotate(
        "",
        xy=(0.43, 0.71),
        xytext=(0.28, 0.22),
        arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.3),
    )
    ax.annotate(
        "",
        xy=(0.57, 0.71),
        xytext=(0.72, 0.22),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3),
    )
    ax.annotate(
        "",
        xy=(0.69, 0.22),
        xytext=(0.31, 0.22),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.4),
    )
    ax.annotate(
        "",
        xy=(0.67, 0.82),
        xytext=(0.67, 0.72),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.8),
    )
    ax.text(0.69, 0.77, r"$\Delta$", fontsize=7)
    ax.text(0.34, 0.49, r"$\Omega_o$", color=PURPLE, fontsize=7)
    ax.text(0.66, 0.49, r"$\Omega_\mu$", color=ORANGE, fontsize=7)
    ax.text(
        0.50,
        0.27,
        r"$\Omega_{\rm eff}=-\Omega_o\Omega_\mu/(2\Delta)$",
        ha="center",
        color=GREEN,
        fontsize=7.3,
    )
    ax.text(
        0.50,
        0.02,
        r"Tune $\delta+(|\Omega_o|^2-|\Omega_\mu|^2)/(4\Delta)=0$; "
        r"switch MW off before the Rb target pulse.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="0.25",
    )
    ax.set_title("Simultaneous difference-frequency Raman drive", fontsize=8)

    # (c) Exact three-level dynamics at the gate-simulation target.
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "(c)", x=-0.08, y=1.08)
    omega = 15.0
    delta = 57.15005715008573
    t_pi = 0.262466929133727
    hamiltonian = np.array(
        [[0, omega / 2, 0], [omega / 2, delta, omega / 2], [0, omega / 2, 0]],
        dtype=complex,
    )
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    times = np.linspace(0, t_pi, 500)
    coefficients = eigenvectors.conj().T @ np.array([1, 0, 0], dtype=complex)
    states = (
        eigenvectors @ (coefficients[:, None] * np.exp(-2j * np.pi * np.outer(energies, times)))
    ).T
    ax.plot(times, np.abs(states[:, 0]) ** 2, color=YB_COLOR, label=r"$|g\rangle$")
    ax.plot(times, np.abs(states[:, 2]) ** 2, color=VERMILLION, label=r"$|r\rangle$")
    ax.plot(times, np.abs(states[:, 1]) ** 2, color=PURPLE, label=r"$|a\rangle$")
    ax.set_xlabel(r"pulse time ($\mu$s)")
    ax.set_ylabel("population")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(frameon=False, ncol=3, loc="center")
    ax.text(
        0.98,
        0.96,
        r"$\Omega_o/2\pi=\Omega_\mu/2\pi=15$ MHz"
        "\n"
        r"$\Delta/2\pi=57.150$ MHz"
        "\n"
        r"$\max P_a=6.05\%$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.92, "pad": 1.5},
    )
    ax.set_title(r"Exact cyclic three-level $\pi$ pulse", fontsize=8, loc="right")

    # (d) All-optical fallbacks as a compact comparison panel.
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "(d)", x=-0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "All-optical fallback", fontsize=8, fontweight="bold", va="top")
    ax.text(0.02, 0.72, r"via $5d6s\,{}^3D_1$", color=BLUE, fontsize=7.2, fontweight="bold")
    ax.text(
        0.04, 0.54, "1388.763 nm + 386.003 nm\nτ = 332(11) ns; Γ/2π = 479(16) kHz", fontsize=6.8
    )
    ax.text(0.02, 0.33, r"via $6s7s\,{}^3S_1$", color=VERMILLION, fontsize=7.2, fontweight="bold")
    ax.text(
        0.04, 0.15, "649.087 nm + 564.942 nm\nτ = 13.8(17) ns; Γ/2π = 11.5(14) MHz", fontsize=6.8
    )
    ax.text(
        0.02,
        0.00,
        "Upper-leg Rydberg matrix elements remain unverified.",
        fontsize=6.2,
        color="0.35",
    )

    fig.suptitle(
        "Yb P excitation: forbidden direct E1 line and a candidate S-relay route", fontsize=10
    )
    save_figure(fig, "channel_forster_yb_excitation")


def main() -> None:
    characterize_channel()


if __name__ == "__main__":
    main()
