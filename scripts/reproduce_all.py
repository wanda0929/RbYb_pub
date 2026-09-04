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
    run(PYTHON, "scripts/plot_channel_forster.py")
    run(PYTHON, "scripts/plot_candidate_excitation.py")
    run(PYTHON, "scripts/plot_forster_gate.py")
    run(PYTHON, "scripts/plot_channel_vdw.py")


def full() -> None:
    run(PYTHON, "scripts/verify_pairinteraction_databases.py")
    run(PYTHON, "scripts/reproduce_forster_characterization.py")
    run(PYTHON, "scripts/reproduce_forster_gate.py")
    run(PYTHON, "scripts/evaluate_forster_p0_4.py")
    run(PYTHON, "scripts/evaluate_forster_p1_1.py")
    run(PYTHON, "scripts/evaluate_forster_p1_2.py")
    run(PYTHON, "scripts/evaluate_forster_p1_4.py")
    run(PYTHON, "scripts/reproduce_vdw_dense.py")
    run(PYTHON, "scripts/evaluate_vdw_p1_5.py")
    figures()
    run(PYTHON, "-m", "pytest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="recompute every record and figure")
    mode.add_argument("--figures", action="store_true", help="render figures from committed data")
    args = parser.parse_args()

    if args.full:
        full()
    elif args.figures:
        figures()
    else:
        run(PYTHON, "-m", "pytest")


if __name__ == "__main__":
    main()
