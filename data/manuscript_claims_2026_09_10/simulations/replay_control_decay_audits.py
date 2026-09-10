#!/usr/bin/env python3
"""Replay existing audits with corrected sources and distinct output files.

Run stages sequentially on the 8 GiB fallback Orb. No optimization is performed.
The original P0/P1 JSON records and old-decay field checkpoints are never written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_forster_p0_4 as p0
from simulate_forster_gate import query_lifetimes

HERE = Path(__file__).resolve().parent
P0 = HERE / "forster_control_decay_p0_4.json"
BASELINE = HERE / "forster_control_decay_recheck.json"


def provenance():
    sources = sorted(HERE.glob("*.py")) + [HERE / "forster_characterization.json"]
    return {
        "decay_convention": "radiative no-jump including Rb56S control-excited target state",
        "reoptimized_during_decay_recheck": False,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "baseline_signature": json.loads(BASELINE.read_text())["signature"],
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
    }


def annotate(path):
    data = json.loads(path.read_text())
    data["control_decay_recheck_provenance"] = provenance()
    if path != P0:
        data["control_decay_recheck_provenance"]["corrected_p0_sha256"] = hashlib.sha256(P0.read_bytes()).hexdigest()
    path.write_text(json.dumps(data, indent=2) + "\n")


def p0_audit():
    # Preserve completed numerical evaluations even if a larger basis exceeds
    # the Orb's memory. These wrappers only record returns; propagation is unchanged.
    progress = HERE / "forster_control_decay_p0_progress.jsonl"
    progress.write_text(json.dumps({"provenance": provenance()}) + "\n")
    original_metrics, original_row = p0._gate_metrics, p0._model_row

    def record(row):
        with progress.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def metrics(model, pulse, lifetimes, fixed_correction, mode_cutoff=p0.MODE_CUTOFF, **errors):
        result = original_metrics(model, pulse, lifetimes, fixed_correction, mode_cutoff, **errors)
        record({"kind": "gate", "basis": asdict(model.basis_config),
            "electric_field_v_cm": model.electric_field_v_cm,
            "defect_offset_mhz": model.defect_offset_mhz,
            "maximum_actual_step_ns": 1000 * max(s.duration_us for s in pulse),
            "cutoff": mode_cutoff, "errors": errors, "metrics": result})
        return result

    def model_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        record({"kind": "basis", "row": result})
        return result

    p0._gate_metrics, p0._model_row = metrics, model_row
    p0.OUT = P0
    try:
        p0.main()
        annotate(P0)
    finally:
        p0._gate_metrics, p0._model_row = original_metrics, original_row


def tight_steps():
    baseline = json.loads(BASELINE.read_text())
    correction = baseline["rows"]["0.000000/0.000000"]["fixed_local_z_rad"]
    model = p0._build(p0.REFERENCE_BASIS)
    lifetimes = query_lifetimes(0.0)
    rows = []
    for step in (0.125, 0.0625, 0.03125):
        row = {"step_ns": step, **p0._gate_metrics(model, p0._pulse(step), lifetimes, correction)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = HERE / "forster_control_decay_tight_steps.json"
    path.write_text(json.dumps({"rows": rows, "provenance": provenance()}, indent=2) + "\n")


def driven_curves():
    import optimize_bounded_minimax_forster_gate as minimax
    import optimize_hardware_aware_forster_gate as hardware
    import simulate_robust_shaped_forster_gate as shaped

    baseline = json.loads(BASELINE.read_text())
    correction = baseline["rows"]["0.000000/0.000000"]["fixed_local_z_rad"]
    model = p0._build(p0.REFERENCE_BASIS)
    modes = shaped._prepare_modes(model, p0.MODE_CUTOFF)
    lifetimes = query_lifetimes(0.)
    parameters = minimax.SELECTED_PARAMETERS
    pulse = p0._pulse()
    response = []
    for rise in (2., 5., 10., 15., 20.):
        filtered = hardware._filtered_pulse(parameters, rise, .125)
        kraus = hardware._kraus(modes, filtered, lifetimes)
        response.append({
            "rise_time_ns": rise,
            "target_duration_us": sum(s.duration_us for s in filtered),
            "recalibrated_nominal_fidelity": shaped._local_z_metrics(kraus)["average_gate_fidelity"],
            "fixed_10ns_correction_nominal_fidelity": shaped._fixed_correction_fidelity(kraus, correction),
        })
    errors = np.linspace(-1., 1., 9)
    scales = tuple(1 + errors / 100)
    yb = minimax._scenario_fidelities([modes], pulse, lifetimes, correction,
        target_scales=scales, control_scales=(1.,))[0, :, 0]
    rb = minimax._scenario_fidelities([modes], pulse, lifetimes, correction,
        target_scales=(1.,), control_scales=scales)[0, 0]
    trajectory = minimax._population_trajectory(modes,
        model.pp_overlap[np.abs(model.ss_overlap)**2 > p0.MODE_CUTOFF], pulse, lifetimes)
    np.testing.assert_allclose([response[2]["recalibrated_nominal_fidelity"], yb[4], rb[4]],
        baseline["summary"]["nominal_overlap"], rtol=0, atol=1e-10)
    assert all(lo <= value <= hi for value, (lo, hi) in zip(parameters, minimax.PARAMETER_BOUNDS))
    data = {
        "provenance": provenance(), "response_time_scan": response,
        "target_amplitude_scan": {"yb_rabi_error_pct": errors.tolist(), "rb_rabi_error_pct": errors.tolist(),
            "nominal_position_rb_nominal_fidelity": yb.tolist(), "nominal_position_yb_nominal_fidelity": rb.tolist()},
        "population_trajectory": trajectory,
        "population_summary": {key: {"maximum": max(values), "final": values[-1]}
            for key, values in trajectory.items() if key != "time_us"},
        "constraints": {"parameters_within_existing_bounds": True,
            "parameter_bounds": minimax.PARAMETER_BOUNDS,
            "aom_rise_time_ns": hardware.AOM_RISE_TIME_NS,
            "filtered_peak_omega_mhz": max(s.omega_mhz for s in pulse),
            "filtered_detuning_range_mhz": [min(s.detuning_mhz for s in pulse), max(s.detuning_mhz for s in pulse)],
            "target_duration_us": sum(s.duration_us for s in pulse),
            "total_gate_time_us": 2 * shaped.RB_PI_DURATION_US + sum(s.duration_us for s in pulse)},
    }
    path = HERE / "forster_control_decay_driven_curves.json"
    path.write_text(json.dumps(data, indent=2) + "\n")


def field_scan():
    import scan_forster_p0_4_field as field

    field.OUT = HERE / "forster_control_decay_field_scan.json"
    expected = provenance()
    if field.OUT.exists():
        old = json.loads(field.OUT.read_text())
        if old.get("control_decay_recheck_provenance") != expected:
            raise ValueError("Corrected field checkpoint provenance changed; do not resume")
    study = field.Study()
    study.data["control_decay_recheck_provenance"] = expected
    study.data["started_utc"] = datetime.now(timezone.utc).isoformat()
    study.save()
    study.grid(3.1, full=True)
    baseline = json.loads(BASELINE.read_text())["summary"]
    np.testing.assert_allclose(study.nominal(3.1)["nominal"], baseline["nominal_overlap"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(study.nominal(3.1)["full_grid"]["joint_minimum"], baseline["joint_sampled_minimum"], rtol=0, atol=1e-10)
    study.data["corrected_baseline_reproduced"] = True
    for value in sorted(set(np.arange(0, 5.01, .5).tolist() + [3.1])):
        study.nominal(value)
    for value in (3.05, 3.075, 3.1, 3.125, 3.15):
        study.grid(value)
    for value in (3.085, 3.1025):
        study.grid(value, full=True)
    for value in (3.1, 3.1025):
        study.grid(value, window=60.)
    study.data["assessment"] = field.field_comparison(study.data)
    study.data["completed_utc"] = datetime.now(timezone.utc).isoformat()
    study.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("p0", "p1", "projection", "field", "tight-steps", "curves"))
    args = parser.parse_args()
    if args.stage == "p0":
        p0_audit()
    elif args.stage in ("p1", "projection"):
        if args.stage == "p1":
            import evaluate_forster_p1_1 as audit
            audit.OUT = HERE / "forster_control_decay_p1_1.json"
        else:
            import evaluate_forster_p1_2 as audit
            audit.OUT = HERE / "forster_control_decay_p1_2.json"
        audit.P0_RESULT = P0
        audit.main()
        annotate(audit.OUT)
    elif args.stage == "field":
        field_scan()
    elif args.stage == "curves":
        driven_curves()
    else:
        tight_steps()


if __name__ == "__main__":
    main()
