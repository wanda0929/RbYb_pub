"""Recovered evidence, post-hoc arithmetic, and corrected figure provenance."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "data/manuscript_support_2026_09_10"
sys.path.insert(0, str(ROOT / "scripts"))

import check_control_decay_data as checks  # noqa: E402
import plot_control_decay_gate as figures  # noqa: E402
import recompute_claim_diagnostics as diagnostics  # noqa: E402


def test_preserved_note_excerpts_and_reconstruction():
    manifest = json.loads((SUPPORT / "source_manifest.json").read_text())
    raw = (SUPPORT / "source_excerpts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["source_excerpts_sha256"]
    for note in json.loads(raw):
        assert note["git_status"].startswith("?? History_plans/")
        for excerpt in note["excerpts"]:
            assert len(excerpt["text"].splitlines()) == (
                excerpt["end_line"] - excerpt["start_line"] + 1
            )
    actual = diagnostics.reproduce()
    expected = json.loads((SUPPORT / "recomputed_diagnostics.json").read_text())
    assert actual == expected
    assert actual["power"]["mW_before_optical_losses"] == pytest.approx(44.5305551938)
    d = actual["diagnostic"]
    assert d["bare_population_integral_endpoint"] == pytest.approx(0.000853507991941344)
    assert d["endpoint_modal_minus_bare"] == pytest.approx(5.55440299565e-5)
    assert d["maximum_absolute_sampled_cumulative_difference"] == pytest.approx(5.57929583377e-5)
    assert "NOT full-D propagation" in actual["scope"]


def synthetic_inputs():
    baseline = {
        "lifetimes_us": {"rb_56s_us": 2.0, "rb_56p_us": 4.0, "yb_53s_us": 5.0, "yb_52p_us": 10.0},
        "pulse_parameters": [9.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.02],
    }
    curves = {
        "population_trajectory": {
            "time_us": [0.0, 0.1, 0.3],
            "unblocked_yb_rydberg": [0.6, 0.6, 0.6],
            "blocked_computational": [0.2, 0.2, 0.2],
            "blocked_ss": [0.1, 0.1, 0.1],
            "blocked_pp": [0.3, 0.3, 0.3],
            "blocked_spectator": [0.4, 0.4, 0.4],
            "blocked_loss": [0.0, 0.2, 0.2],
        }
    }
    return baseline, curves


def test_nonuniform_integrals_and_species_assignment_not_total_return_loss():
    # Integral of 2t+1 on [0,1,3] is [0,2,12]; a uniform-step rule fails.
    assert diagnostics.cumulative_trapezoid([0.0, 1.0, 3.0], [1.0, 3.0, 7.0]) == [0.0, 2.0, 12.0]
    result = diagnostics.compute(*synthetic_inputs())
    assert result["integrated_populations_ns"]["blocked_pp"] == pytest.approx(90.0)
    assert result["Rb10_ideal_residence_ns"] == pytest.approx(400.0)
    # Constant bare rate is 0.555/us. The largest discrepancy is at 0.1us,
    # not the endpoint. Spectators belong in this proxy, not the species split.
    assert result["diagnostic"]["bare_population_integral_endpoint"] == pytest.approx(0.1665)
    assert result["diagnostic"]["endpoint_modal_minus_bare"] == pytest.approx(0.0335)
    assert result["diagnostic"]["maximum_absolute_sampled_cumulative_difference"] == pytest.approx(
        0.1445
    )
    assert result["diagnostic"]["maximum_at_ns"] == pytest.approx(100.0)
    assert result["approx_species_by_input"]["10"]["Rb"] == pytest.approx(0.2)
    assert result["approx_species_by_input"]["11"]["Rb"] == pytest.approx(0.1175)
    assert result["approx_species_equal_input_mean"] == pytest.approx(
        {"Rb": 0.079375, "Yb": 0.01275}
    )
    assert result["power"]["selected_pulse_command_rabi_Hz"] == 9e6


@pytest.mark.parametrize("time", [[0.0, 0.1, 0.1], [0.0, float("nan"), 0.3], [0.1, 0.2, 0.3]])
def test_invalid_saved_time_is_rejected(time):
    baseline, curves = synthetic_inputs()
    curves["population_trajectory"]["time_us"] = time
    with pytest.raises(ValueError, match="time_us"):
        diagnostics.compute(baseline, curves)


def test_corrected_figures_match_records_and_reject_changed_bytes(tmp_path, capsys):
    checks.main()
    assert "fidelity 0.999125784769" in capsys.readouterr().out
    result = figures.load_results()
    phases, conditional = figures.endpoint_phase_errors(result["baseline"])
    assert conditional == pytest.approx(-1.64058190446)
    assert len(phases) == 4
    axial = result["axial_position_scan"]
    assert axial["delta_z_nm"][0] == -50.0
    assert axial["nominal_amplitudes_fidelity"][0] == pytest.approx(0.9990080680690021)
    assert len(result["magnetic_field_scan"]["coarse_field_gauss"]) == 12
    shutil.copytree(figures.ARCHIVE, tmp_path / "archive")
    data = tmp_path / "archive/simulations"
    path = data / "forster_control_decay_driven_curves.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        figures.load_results(data)
    # Even the standalone post-hoc reader rejects a different baseline.
    path = data / "forster_control_decay_recheck.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="do not match"):
        diagnostics.reproduce(data)


def test_phase_correction_uses_both_recorded_local_z_angles():
    baseline = figures.load_results()["baseline"]
    row = baseline["rows"]["0.000000/0.000000"]
    original, _ = figures.endpoint_phase_errors(baseline)
    row["fixed_local_z_rad"][0] += 0.02
    shifted, conditional = figures.endpoint_phase_errors(baseline)
    np.testing.assert_allclose(shifted - original, [0.0, 0.0, 20.0, 20.0], atol=1e-10)
    assert conditional == pytest.approx(-1.64058190446)
