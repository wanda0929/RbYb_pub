"""Shared matplotlib style for paper figures (PRA/PRX, REVTeX 4.2).

Current manuscript conventions:
- STIX/serif fonts everywhere, 8–9 pt at final size
- vector PDF, pdf.fonttype 42
- Huang 2018 blue/orange, thin boxed axes
- fixed species code: Rb = orange #D95319, Yb = blue #0072BD, Rb-Rb baseline = gray
- single column figsize=(3.375, h); double column figsize=(7.0, h)
"""

from matplotlib import rcParams
from pulseviz import paper_style, MODEL_COLORS

# Fixed species color code (all figures)
RB_COLOR = "#D95319"   # orange
YB_COLOR = "#0072BD"   # blue
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
LASER_780 = OKABE_ITO["vermillion"]   # 780 nm, red
LASER_480 = OKABE_ITO["skyblue"]      # 480 nm, blue
LASER_302 = OKABE_ITO["purple"]       # 302 nm, UV -> reddish purple
LASER_399 = "#7D5BA6"                 # 399 nm, violet

SINGLE_COL = 3.375  # in (86 mm)
DOUBLE_COL = 7.0    # in (178 mm)


def use_paper_style():
    paper_style(model="huang")
    rcParams.update({
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.minor.width": .5, "ytick.minor.width": .5,
    })


PAPER_COLORS = MODEL_COLORS['huang']


def panel_label(ax, label, x=-.16, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            ha='left', va='bottom')
