#!/usr/bin/env python3
"""Check corrected manuscript Figs. 3/4 data, without rerunning physics.

Adapted from figures/check_manuscript_data.py in the manuscript repository;
unrelated historical channel/vdW checks remain in the publication test suite.
"""

import json

import numpy as np
from plot_control_decay_gate import DATA, endpoint_phase_errors, load_results


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def probability(values):
    a = np.asarray(values)
    assert np.all(np.isfinite(a)) and np.all((a >= -1e-12) & (a <= 1 + 1e-12))


def main():
    # The renderer verifies archive bytes and the corrected records' common baseline.
    result = load_results()
    curves = load("forster_control_decay_driven_curves")
    trajectory = curves["population_trajectory"]
    time = np.asarray(trajectory["time_us"])
    assert time[0] == 0 and np.all(np.diff(time) > 0)
    for key, values in trajectory.items():
        if key == "time_us":
            continue
        assert len(values) == len(time)
        probability(values)
        summary = curves["population_summary"][key]
        np.testing.assert_allclose(
            [max(values), values[-1]], [summary["maximum"], summary["final"]], atol=1e-12, rtol=0
        )
    # No-jump curves retain their norm loss; they are not normalized to survivors.
    total = sum(
        np.asarray(trajectory[k])
        for k in (
            "blocked_computational",
            "blocked_pp",
            "blocked_ss",
            "blocked_spectator",
            "blocked_loss",
        )
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-10, rtol=0)
    assert np.all(np.diff(trajectory["blocked_loss"]) >= -1e-12)
    np.testing.assert_allclose(
        time[-1], result["constraints"]["target_duration_us"], atol=1e-12, rtol=0
    )
    row = result["baseline"]["rows"]["0.000000/0.000000"]
    pairs = np.asarray(row["kraus_diagonal_re_im"])
    k = pairs[:, 0] + 1j * pairs[:, 1]
    loss = 1 - abs(k) ** 2
    probability(loss)
    np.testing.assert_allclose(
        loss, row["phase_recalibrated_metrics"]["computational_loss_by_input"], atol=1e-12, rtol=0
    )
    alpha, beta = row["fixed_local_z_rad"]
    corrected = k * np.exp(1j * np.array([0, beta, alpha, alpha + beta]))
    fidelity = (np.sum(abs(k) ** 2) + abs(np.dot([1, 1, 1, -1], corrected)) ** 2) / 20
    nominal = result["baseline"]["summary"]["nominal_overlap"]
    np.testing.assert_allclose(fidelity, nominal, atol=1e-12, rtol=0)
    errors, conditional = endpoint_phase_errors(result["baseline"])
    np.testing.assert_allclose(
        errors[3] - errors[2] - errors[1] + errors[0], conditional, atol=1e-10, rtol=0
    )
    print(f"Fig. 3: {len(time)} trajectory samples; return losses {loss.tolist()}")
    print(f"Fig. 3: conditional phase error {conditional:.9f} mrad; fidelity {fidelity:.12f}")
    for prefix, key in [
        ("yb", "nominal_position_rb_nominal_fidelity"),
        ("rb", "nominal_position_yb_nominal_fidelity"),
    ]:
        scan = result["target_amplitude_scan"]
        probability(scan[key])
        center = scan[prefix + "_rabi_error_pct"].index(0.0)
        np.testing.assert_allclose(scan[key][center], nominal, atol=1e-10, rtol=0)
    for row in result["response_time_scan"]:
        probability(
            [row["fixed_10ns_correction_nominal_fidelity"], row["recalibrated_nominal_fidelity"]]
        )
    field_fidelity = np.asarray(result["magnetic_field_scan"]["coarse_phase_recalibrated_fidelity"])
    probability(field_fidelity)
    assert np.all(1 - field_fidelity > 0)  # No invented floor on the logarithmic axis.
    axial = result["axial_position_scan"]
    probability(axial["nominal_amplitudes_fidelity"])
    assert np.all(np.diff(axial["delta_z_nm"]) > 0)
    print(
        f"Fig. 4: {len(result['response_time_scan'])} response, "
        f"{len(axial['delta_z_nm'])} axial, "
        f"{len(scan['yb_rabi_error_pct'])} per-species amplitude, "
        f"{len(field_fidelity)} field samples"
    )
    print("Passed corrected gate/robustness data checks; no physics rerun.")


if __name__ == "__main__":
    main()
