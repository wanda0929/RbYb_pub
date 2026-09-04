"""Shared matplotlib style for paper figures (PRA/PRX, REVTeX 4.2).

Rules from plan/paper_plan_v2.md:
- STIX/serif fonts everywhere, >= 7 pt at final size
- vector PDF, pdf.fonttype 42
- Okabe-Ito colorblind-safe palette
- fixed species code: Rb = orange #E69F00, Yb = blue #0072B2, Rb-Rb baseline = gray
- single column figsize=(3.375, h); double column figsize=(7.0, h)
"""

from matplotlib import rcParams

# Fixed species color code (all figures)
RB_COLOR = "#E69F00"  # orange
YB_COLOR = "#0072B2"  # blue
RBRB_COLOR = "#888888"  # gray baseline

# Okabe-Ito palette
OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Laser-arrow colors
LASER_780 = OKABE_ITO["vermillion"]  # 780 nm, red
LASER_480 = OKABE_ITO["skyblue"]  # 480 nm, blue
LASER_302 = OKABE_ITO["purple"]  # 302 nm, UV -> reddish purple
LASER_399 = "#7D5BA6"  # 399 nm, violet

SINGLE_COL = 3.375  # in (86 mm)
DOUBLE_COL = 7.0  # in (178 mm)


def use_paper_style():
    rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.6,
        }
    )
