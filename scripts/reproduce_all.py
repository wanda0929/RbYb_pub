#!/usr/bin/env python3
"""Single entry point for checking or reproducing the publication archive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*arguments: str) -> None:
    command = list(arguments)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def figures() -> None:
    run(PYTHON, "scripts/plot_channel_forster.py", "--characterization-only")
    run(PYTHON, "scripts/plot_candidate_excitation.py")
    run(PYTHON, "scripts/reproduce_forster_gate.py", "--plot-only")
    run(PYTHON, "scripts/plot_channel_vdw.py")


def full(rebuild_figure1: bool = False) -> None:
    run(PYTHON, "scripts/verify_pairinteraction_databases.py")
    run(PYTHON, "scripts/optimize_forster_p0_4_reference.py")
    run(PYTHON, "scripts/evaluate_forster_p0_4.py")
    run(PYTHON, "scripts/evaluate_forster_p1_1.py")
    run(PYTHON, "scripts/evaluate_forster_p1_2.py")
    run(PYTHON, "scripts/evaluate_forster_p1_4.py")
    characterization = [PYTHON, "scripts/reproduce_forster_characterization.py"]
    if rebuild_figure1:
        characterization.append("--rebuild-fixed-m")
    run(*characterization)
    run(PYTHON, "scripts/scan_forster_p0_4_field.py")
    run(PYTHON, "scripts/scan_forster_p0_4_field.py", "--fields",
        "3.05", "3.075", "3.1", "3.125", "3.15", "--grid", "endpoints", "--workers", "4")
    run(PYTHON, "scripts/scan_forster_p0_4_field.py", "--fields",
        "3.085", "3.1025", "--grid", "full", "--workers", "4")
    run(PYTHON, "scripts/scan_forster_p0_4_field.py", "--fields",
        "3.1", "3.1025", "--grid", "endpoints", "--window", "60", "--workers", "4")
    run(PYTHON, "scripts/scan_forster_p0_4_field.py", "--assess-only")
    run(PYTHON, "scripts/reproduce_forster_gate.py")
    run(PYTHON, "scripts/reproduce_vdw_dense.py")
    run(PYTHON, "scripts/evaluate_vdw_p1_5.py")
    figures()
    run(PYTHON, "-m", "pytest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true",
        help="resume all numerical workflows, then regenerate figures and tests")
    mode.add_argument("--figures", action="store_true", help="render figures from committed data")
    parser.add_argument("--rebuild-figure1", action="store_true",
        help="with --full, also rebuild Figure 1's historical fixed-m calculations")
    args = parser.parse_args()

    if args.full:
        full(args.rebuild_figure1)
    elif args.figures:
        figures()
    else:
        run(PYTHON, "-m", "pytest")


if __name__ == "__main__":
    main()
