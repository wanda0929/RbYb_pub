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
    run(PYTHON, "scripts/check_control_decay_data.py")
    run(PYTHON, "scripts/plot_channel_forster.py", "--characterization-only")
    run(PYTHON, "scripts/plot_candidate_excitation.py")
    run(PYTHON, "scripts/plot_control_decay_gate.py")
    run(PYTHON, "scripts/plot_channel_vdw.py")


def historical_full(rebuild_figure1: bool = False) -> None:
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
    run(
        PYTHON,
        "scripts/scan_forster_p0_4_field.py",
        "--fields",
        "3.05",
        "3.075",
        "3.1",
        "3.125",
        "3.15",
        "--grid",
        "endpoints",
        "--workers",
        "4",
    )
    run(
        PYTHON,
        "scripts/scan_forster_p0_4_field.py",
        "--fields",
        "3.085",
        "3.1025",
        "--grid",
        "full",
        "--workers",
        "4",
    )
    run(
        PYTHON,
        "scripts/scan_forster_p0_4_field.py",
        "--fields",
        "3.1",
        "3.1025",
        "--grid",
        "endpoints",
        "--window",
        "60",
        "--workers",
        "4",
    )
    run(PYTHON, "scripts/scan_forster_p0_4_field.py", "--assess-only")
    run(PYTHON, "scripts/reproduce_forster_gate.py")
    run(PYTHON, "scripts/reproduce_vdw_dense.py")
    run(PYTHON, "scripts/evaluate_vdw_p1_5.py")
    run(PYTHON, "-m", "pytest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--historical-full",
        action="store_true",
        help="expensive PRE-CORRECTION replay; not the current manuscript gate workflow",
    )
    mode.add_argument(
        "--figures",
        action="store_true",
        help="check and render all five current figure subjects from stored data; no physics rerun",
    )
    mode.add_argument(
        "--quick-check",
        action="store_true",
        help="check archive and freshly rebuild/propagate one corrected nominal Hamiltonian",
    )
    parser.add_argument(
        "--rebuild-figure1",
        action="store_true",
        help="with --historical-full, also rebuild Figure 1's fixed-m calculations",
    )
    args = parser.parse_args()

    if args.rebuild_figure1 and not args.historical_full:
        parser.error("--rebuild-figure1 requires --historical-full")
    if args.historical_full:
        historical_full(args.rebuild_figure1)
    elif args.figures:
        figures()
    elif args.quick_check:
        run(PYTHON, "scripts/check_control_decay_data.py")
        run(PYTHON, "scripts/quick_check.py")
    else:
        run(PYTHON, "-m", "pytest")


if __name__ == "__main__":
    main()
