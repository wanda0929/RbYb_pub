#!/usr/bin/env python3
"""Checkpointed fixed-pulse reevaluation after adding control-excited decay.

Does not reoptimize or overwrite the pre-correction P0/P1 records. Start with
--grid endpoints, then extend the same output to --grid full. Independent
geometries can run with --workers on a CPU host. Set BLAS thread counts in the
launcher to avoid oversubscribing the machine.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pairinteraction
import scipy

import evaluate_forster_p0_4 as p0
import optimize_bounded_minimax_forster_gate as minimax
import optimize_hardware_aware_forster_gate as hardware
import simulate_robust_shaped_forster_gate as shaped
from simulate_forster_gate import query_lifetimes

HERE = Path(__file__).resolve().parent
OUT = HERE / "forster_control_decay_recheck.json"


def evaluate(spec, correction):
    started = time.monotonic()
    radius, cosine = spec
    print(f"building r={radius:g} um, cos={cosine:g}", flush=True)
    model = minimax._build_geometry(radius, cosine, p0.REFERENCE_BASIS).model
    modes = shaped._prepare_modes(model, p0.MODE_CUTOFF)
    lifetimes = query_lifetimes(0.0)
    pulse = p0._pulse()
    nominal_k = hardware._kraus(modes, pulse, lifetimes)
    local = shaped._local_z_metrics(nominal_k)
    if correction is None:
        correction = (local["optimal_local_z_alpha_rad"],
                      local["optimal_local_z_beta_rad"])
    grid = minimax._scenario_kraus([modes], pulse, lifetimes)[0]
    fidelities = minimax._fidelities_from_kraus(grid, correction)
    old = json.loads(p0.OUT.read_text())["fixed_inputs"]
    old_correction = (old["fixed_reference_local_z_alpha_rad"],
                      old["fixed_reference_local_z_beta_rad"])
    population = p0._population_summary(model, pulse, lifetimes)
    row = {
        "radius_um": radius, "cosine": cosine,
        "distance_um": model.distance_um, "theta_deg": model.theta_deg,
        "basis_size": model.symmetry_component_size,
        "active_modes": len(modes.energies_rel_mhz),
        "nominal_rabi_fixed_z_overlap": shaped._fixed_correction_fidelity(
            nominal_k, correction),
        "nominal_rabi_old_z_overlap": shaped._fixed_correction_fidelity(
            nominal_k, old_correction),
        "nominal_rabi_mean_survival": float(np.vdot(nominal_k, nominal_k).real / 4),
        "phase_recalibrated_metrics": local,
        "kraus_diagonal_re_im": [[float(k.real), float(k.imag)] for k in nominal_k],
        "amplitude_vertex_overlaps": fidelities.tolist(),
        "amplitude_vertex_kraus_re_im": np.stack([grid.real, grid.imag], axis=-1).tolist(),
        "population": population,
        "fixed_local_z_rad": list(correction),
        "wall_seconds": time.monotonic() - started,
    }
    print(f"done {spec}: nominal={row['nominal_rabi_fixed_z_overlap']:.10f}, "
          f"vertex min={np.min(fidelities):.10f}, {row['wall_seconds']:.1f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("endpoints", "full"), default="endpoints")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    sources = [Path(__file__), HERE / "optimize_hardware_aware_forster_gate.py",
               HERE / "simulate_robust_shaped_forster_gate.py",
               HERE / "optimize_bounded_minimax_forster_gate.py",
               HERE / "simulate_forster_gate.py", HERE / "rb_rydberg_hyperfine.py",
               HERE / "evaluate_forster_p0_4.py", p0.OUT]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = {
        "signature": signature, "source_sha256": hashes,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "decay_convention": "radiative no-jump including Rb56S control-excited target state",
        "pulse_reoptimized": False,
        "pulse_parameters": minimax.SELECTED_PARAMETERS.tolist(),
        "basis": asdict(p0.REFERENCE_BASIS),
        "mode_cutoff": p0.MODE_CUTOFF, "step_ns": hardware.FINAL_STEP_NS,
        "field_gauss": shaped.B_GAUSS, "temperature_k": 0.0,
        "lifetimes_us": asdict(query_lifetimes(0.0)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pairinteraction": pairinteraction.__version__},
        "rows": {},
    }
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result["signature"] != signature:
            raise ValueError("checkpoint sources changed; use a new --output")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    def key(spec):
        return f"{spec[0]:.6f}/{spec[1]:.6f}"

    if key((0., 0.)) not in result["rows"]:
        result["rows"][key((0., 0.))] = evaluate((0., 0.), None)
        save()
    correction = result["rows"][key((0., 0.))]["fixed_local_z_rad"]
    specs = ([(0., 0.), (.05, -1.), (.05, 1.)] if args.grid == "endpoints" else
             [(0., 0.)] + [(float(r), float(c)) for r in minimax.VALIDATION_RADII_UM[1:]
                           for c in minimax.VALIDATION_COSINES])
    pending = [s for s in specs if key(s) not in result["rows"]]
    if args.workers == 1:
        for spec in pending:
            result["rows"][key(spec)] = evaluate(spec, correction)
            save()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [(spec, pool.submit(evaluate, spec, correction)) for spec in pending]
            for spec, future in futures:
                result["rows"][key(spec)] = future.result()
                save()
    rows = [result["rows"][key(s)] for s in specs]
    worst_row = min(rows, key=lambda r: np.min(r["amplitude_vertex_overlaps"]))
    vertex = np.unravel_index(np.argmin(worst_row["amplitude_vertex_overlaps"]), (2, 2))
    result["summary"] = {
        "grid": args.grid, "geometries": len(rows),
        "nominal_overlap": rows[0]["nominal_rabi_fixed_z_overlap"],
        "position_only_minimum": min(r["nominal_rabi_fixed_z_overlap"] for r in rows),
        "joint_sampled_minimum": float(np.min(worst_row["amplitude_vertex_overlaps"])),
        "worst_radius_um": worst_row["radius_um"], "worst_cosine": worst_row["cosine"],
        "worst_yb_scale": minimax.AMPLITUDE_VERTICES[vertex[0]],
        "worst_rb_scale": minimax.AMPLITUDE_VERTICES[vertex[1]],
    }
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
