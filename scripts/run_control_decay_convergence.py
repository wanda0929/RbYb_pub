#!/usr/bin/env python3
"""Isolated, checkpointed local refinement; never promotes manuscript inputs.

Run against a frozen checkout. Each invocation requires a new output
directory. A recovered best point can be supplied with --seed-checkpoint; this
starts a new simplex, rather than claiming to resume the old optimizer state.
Use --recheck-only with a seed checkpoint to evaluate without optimizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optimize_forster_p0_4_reference as reference
import pairinteraction
import scipy
from scipy.optimize import minimize


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument(
        "--recheck-only",
        action="store_true",
        help="only reevaluate the seed candidate and baseline at fine steps",
    )
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.recheck_only and args.seed_checkpoint is None:
        parser.error("--recheck-only requires --seed-checkpoint")
    start = time.monotonic()
    seed = reference.minimax.SELECTED_PARAMETERS.copy()
    if args.seed_checkpoint:
        seed = np.array(json.loads(args.seed_checkpoint.read_text())["best_parameters"])
    bounds = np.asarray(reference.minimax.PARAMETER_BOUNDS)
    if (
        seed.shape != (7,)
        or not np.all(np.isfinite(seed))
        or np.any(seed < bounds[:, 0])
        or np.any(seed > bounds[:, 1])
    ):
        raise ValueError("seed must contain seven finite, in-bounds parameters")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "building_models",
        "decay_convention": "includes Rb56S control-excited target-window decay",
        "include_blocked_control_decay": True,
        "manuscript_selected": False,
        "full_grid_validated": False,
        "global_optimum_certified": False,
        "recheck_only": args.recheck_only,
        "seed_parameters": seed.tolist(),
        "baseline_parameters": reference.minimax.SELECTED_PARAMETERS.tolist(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pairinteraction": pairinteraction.__version__,
        },
        "basis": asdict(reference.p0_4.REFERENCE_BASIS),
        "training_specs": reference.OPTIMIZATION_SPECS,
        "training_configurations": 9,
        "optimization_step_ns": reference.OPTIMIZATION_STEP_NS,
        "options": {"maxiter": args.maxiter, "xatol": 2e-8, "fatol": 2e-11, "adaptive": True},
        "evaluations": 0,
    }
    checkpoint = args.output_dir / "checkpoint.json"
    save(checkpoint, state)
    reference._validate_database_assets()
    lifetimes = reference.query_lifetimes(0.0)
    geometries = []
    for spec in reference.OPTIMIZATION_SPECS:
        print(f"building geometry {spec}", flush=True)
        geometries.append(reference._prepare_geometry(spec))
    state["lifetimes_us"] = asdict(lifetimes)
    state["model_build_seconds"] = time.monotonic() - start
    state["status"] = "optimizing"
    print(f"models ready in {state['model_build_seconds']:.1f}s", flush=True)

    def objective(free):
        parameters = np.r_[free, seed[-1]]
        began = time.monotonic()
        metrics = reference._training_metrics(
            parameters,
            geometries,
            lifetimes,
            reference.OPTIMIZATION_STEP_NS,
            include_blocked_control_decay=True,
        )
        value = float(metrics["endpoint_objective"])
        if not np.isfinite(value):
            raise ValueError("nonfinite training objective")
        state["evaluations"] += 1
        row = {
            "evaluation": state["evaluations"],
            "parameters": parameters.tolist(),
            "metrics": metrics,
            "seconds": time.monotonic() - began,
        }
        with (args.output_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if "best_objective" not in state or value < state["best_objective"]:
            state.update(
                best_objective=value, best_parameters=parameters.tolist(), best_metrics=metrics
            )
        state["wall_seconds"] = time.monotonic() - start
        state["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        save(checkpoint, state)
        if state["evaluations"] <= 15 or state["evaluations"] % 25 == 0:
            print(
                f"eval {state['evaluations']}: J={value:.12g}, "
                f"best={state['best_objective']:.12g}, {row['seconds']:.2f}s, "
                f"RSS peak={state['peak_rss_mib']:.0f} MiB",
                flush=True,
            )
        return value

    if args.recheck_only:
        state["best_parameters"] = seed.tolist()
        state["optimizer"] = None
    else:
        result = minimize(
            objective, seed[:-1], method="Nelder-Mead", bounds=bounds[:-1], options=state["options"]
        )
        state["optimizer"] = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "final_simplex": [x.tolist() for x in result.final_simplex],
        }
    state["status"] = "fine_step_recheck"
    save(checkpoint, state)
    state["fine_step_recheck"] = {}
    for label, parameters in (
        ("baseline", reference.minimax.SELECTED_PARAMETERS),
        ("candidate", np.array(state["best_parameters"])),
    ):
        for step in (0.125, 0.0625):
            print(f"rechecking {label} at {step} ns", flush=True)
            state["fine_step_recheck"][f"{label}_{step}ns"] = reference._training_metrics(
                parameters, geometries, lifetimes, step, include_blocked_control_decay=True
            )
            save(checkpoint, state)
    state["status"] = "finished"
    state["wall_seconds"] = time.monotonic() - start
    state["scope"] = (
        "nine-point time-step recheck; optimization only if recheck_only is false; "
        "no global certificate or full-grid acceptance"
    )
    save(checkpoint, state)
    save(args.output_dir / "result.json", state)
    print(json.dumps(state["optimizer"]), flush=True)


if __name__ == "__main__":
    main()
