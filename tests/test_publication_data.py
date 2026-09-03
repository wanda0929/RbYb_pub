from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text())


def test_forster_characterization_matches_figure_1() -> None:
    data = load_json("forster_characterization.json")
    assert data["initial_pair"] == "SS"
    assert data["final_pair"] == "PP"
    assert data["transitions"]["rb"]["process"] == "absorbs"
    assert data["transitions"]["yb"]["process"] == "releases"
    assert data["signed_defect_definition"] == "E(SS)-E(PP)"
    assert data["signed_defect_mhz"] == pytest.approx(-0.763732, abs=5e-7)

    distance_scan = data["fixed_m_distance_scan"]
    field_scan = data["fixed_m_field_scan"]
    angle_scan = data["fixed_m_angle_scan"]
    assert len(distance_scan) == 10
    assert len(field_scan) == 9
    assert len(angle_scan) == 7
    operating_point = next(row for row in distance_scan if row["distance_um"] == 3.4)
    assert operating_point["splitting_mhz"] == pytest.approx(30.799762725830078, abs=5e-7)
    assert operating_point["first_maximum_transfer"] == pytest.approx(0.9968927002410238, abs=5e-10)
    assert operating_point["coherent_spectator_leakage"] == pytest.approx(
        0.002871060035187445, abs=5e-10
    )
    assert operating_point["first_maximum_time_ns"] == 16.25

    reported = data["figure1b_manuscript_reported_weights"]
    assert reported["caption_distance_um"] == 3.4
    assert reported["reconstructed_source_distance_um"] == 3.0
    assert reported["values"] == [
        {"pp_weight": 0.544, "ss_weight": 0.452, "spectator_weight": 0.004},
        {"pp_weight": 0.454, "ss_weight": 0.543, "spectator_weight": 0.003},
    ]
    bright_3p0 = data["bright_eigenstates_at_3p0_um"]
    bright_3p4 = data["bright_eigenstates_at_3p4_um"]
    assert [row["pp_weight"] for row in bright_3p0] == pytest.approx(
        [0.5443903252428, 0.45396478797055373], abs=5e-10
    )
    assert [row["ss_weight"] for row in bright_3p0] == pytest.approx(
        [0.45233020616488084, 0.54280755358507], abs=5e-10
    )
    assert [row["pp_weight"] for row in bright_3p4] == pytest.approx(
        [0.5074042584263688, 0.49182086275955056], abs=5e-10
    )
    assert [row["ss_weight"] for row in bright_3p4] == pytest.approx(
        [0.4911250078533566, 0.5065693965582376], abs=5e-10
    )
    assert [row["first_maximum_transfer"] for row in field_scan] == pytest.approx(
        [
            0.9968927002410238,
            0.9952388084425496,
            0.9920220608346899,
            0.980774877662725,
            0.9641845375440418,
            0.9416698353745745,
            0.9153192322482656,
            0.8510429850201758,
            0.740478978207746,
        ],
        abs=5e-10,
    )
    assert [row["first_maximum_time_ns"] for row in field_scan] == pytest.approx(
        [16.25, 16.25, 16.25, 16.0, 16.0, 15.7, 15.7, 15.1, 14.2], abs=5e-12
    )

    provenance = data["database_provenance"]
    assert provenance["figure1_fixed_m_archived"].endswith(
        "pairinteraction_database_manifest_figure1_fixed_m.json"
    )
    assert provenance["transition_states_and_all_m"].endswith(
        "pairinteraction_database_manifest.json"
    )
    assert data["fixed_m_v1p4_recalculation_at_3p4_um"]["first_maximum_transfer"] == pytest.approx(
        0.97286139, abs=5e-8
    )

    all_m = data["all_m_field_scan"]
    assert [len(all_m[name]) for name in ("coarse", "fine", "extension")] == [
        13,
        25,
        8,
    ]
    for scan in (distance_scan, field_scan, angle_scan, *all_m.values()):
        assert np.isfinite(
            [value for row in scan for value in row.values() if isinstance(value, int | float)]
        ).all()


def test_forster_gate_matches_figure_3_and_table_i() -> None:
    data = load_json("forster_gate_results.json")
    assert data["database_manifest"] == "provenance/pairinteraction_database_manifest.json"
    assert data["parameters"] == pytest.approx(
        [
            4.410416083986085,
            9.418791104832165,
            11.734642228670836,
            -1.1443728579899664,
            -0.9263577893268974,
            1.744859450470417,
            0.025634115265641792,
        ],
        abs=1e-14,
    )
    point = data["operating_point"]
    assert [point[key] for key in ("distance_um", "theta_deg", "magnetic_field_gauss")] == [
        3.4,
        0.0,
        3.1,
    ]
    assert point["rb_effective_rabi_mhz"] == 5.0
    assert point["rb_pi_duration_ns"] == 100.0

    segments = data["command_segments"]
    assert len(segments) == 5
    assert [segment["omega_mhz"] for segment in segments] == pytest.approx(
        [
            4.410416083986085,
            9.418791104832165,
            11.734642228670836,
            9.418791104832165,
            4.410416083986085,
        ]
    )
    assert [segment["detuning_mhz"] for segment in segments] == pytest.approx(
        [
            -1.1443728579899664,
            -0.9263577893268974,
            1.744859450470417,
            -0.9263577893268974,
            -1.1443728579899664,
        ]
    )
    assert {segment["duration_us"] for segment in segments} == {0.025634115265641792}

    result = data["short_minimax"]
    assert result["target_duration_us"] == pytest.approx(0.1600289492601458, abs=1e-14)
    assert result["total_gate_time_us"] == pytest.approx(0.36002894926014584, abs=1e-14)
    assert result["nominal_fidelity"] == pytest.approx(0.9992936955793696, abs=5e-10)
    assert result["position_only_worst_fidelity"] == pytest.approx(0.999092875323293, abs=5e-10)
    assert result["amplitude_vertex_worst_fidelity"] == pytest.approx(0.9984383350388321, abs=5e-10)
    assert np.asarray(result["vertex_fidelity_grid"]).shape == (19, 2, 2)

    assumptions = data["assumptions"]
    assert assumptions["temperature_k"] == 0.0
    assert assumptions["maximum_relative_displacement_um"] == 0.05
    assert assumptions["amplitude_error_box"] == {
        "yb_effective_rabi_scale": [0.99, 1.01],
        "rb_effective_rabi_scale": [0.99, 1.01],
    }
    assert assumptions["aom_10_to_90_rise_time_ns"] == 10.0
    assert data["validation"]["propagation_step_ns"] == 0.125
    assert data["validation"]["number_of_geometries"] == 19
    assert [row["step_ns"] for row in data["propagation_convergence"]] == [
        1.0,
        0.5,
        0.25,
        0.125,
    ]


def test_vdw_dense_data_matches_figure_4() -> None:
    data = load_json("vdw_dense_data.json")
    assert "not a driven multichannel gate" in data["scope"]
    assert data["configuration"]["zero_field_c6_ghz_um6"] == pytest.approx(
        73.57944850063153, abs=5e-7
    )
    distance_track = data["distance_track_b25_pp"]
    sectors = data["sector_track_b25"]
    field_track = data["magnetic_field_track_r3p3_pp"]
    assert len(distance_track) == 12
    assert len(sectors) == 4
    assert len(field_track) == 6
    assert [row["distance_um"] for row in distance_track] == sorted(
        row["distance_um"] for row in distance_track
    )
    working_point = next(row for row in distance_track if row["distance_um"] == 3.3)
    assert working_point["u_mhz"] == pytest.approx(57.0720534324646, abs=2e-6)
    assert working_point["overlap"] == pytest.approx(0.9697184016029612, abs=5e-10)
    assert [(row["m_rb"], row["m_yb"]) for row in sectors] == [
        (-0.5, -0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (0.5, 0.5),
    ]


def test_vdw_feasibility_matches_appendix_a_and_table_iii() -> None:
    data = load_json("vdw_feasibility_results.json")
    assert "not a driven multichannel gate simulation" in data["scope"]
    assert data["model"]["effective_rabi_mhz"] == 2.0
    assert data["model"]["sequence_duration_us"] == 1.0
    assert data["model"]["rb_lifetime_0k_us"] == 323.60
    assert data["model"]["yb_lifetime_0k_us"] == 87.56

    expected = [
        (-0.5, -0.5, 56.793, 1764.3, 0.997709, 0.997852, 0.999579),
        (-0.5, 0.5, 57.127, 1801.0, 0.997709, 0.997852, 0.999579),
        (0.5, -0.5, 56.860, 1798.6, 0.997696, 0.997839, 0.999566),
        (0.5, 0.5, 57.072, 1770.6, 0.997699, 0.997841, 0.999568),
    ]
    sectors = data["four_magnetic_sectors"]
    assert len(sectors) == 4
    for row, values in zip(sectors, expected, strict=True):
        m_rb, m_yb, u_mhz, defect_mhz, fidelity, survival, lossless = values
        assert (row["m_rb"], row["m_yb"]) == (m_rb, m_yb)
        assert f"{row['u_mhz']:.3f}" == f"{u_mhz:.3f}"
        assert f"{row['delta_eff_mhz']:.1f}" == f"{defect_mhz:.1f}"
        assert f"{row['reduced_model_average_fidelity']:.6f}" == f"{fidelity:.6f}"
        assert f"{row['mean_computational_survival']:.6f}" == f"{survival:.6f}"
        assert f"{row['lossless_reduced_model_average_fidelity']:.6f}" == f"{lossless:.6f}"

    point = data["working_point"]
    assert point["conditional_phase_error_rad"] == pytest.approx(0.0534, abs=5e-5)
    assert point["minimum_basis_return"] == pytest.approx(0.996531, abs=5e-7)


def test_table_ii_csv_every_cell() -> None:
    with (DATA / "prior_work.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "work": "Zhang et al.",
            "initial_pair": "|rR> (symbolic)",
            "final_pair": "|r'R'> (symbolic)",
            "forster_defect": "not reported",
            "interaction_strength": "assumed V_Rr = 2pi x 200 MHz",
            "pair_potential": "not reported",
            "gate_result": "architecture numerics with assumed V",
            "citation_key": "Zhang2025Dual",
        },
        {
            "work": "Xu et al.",
            "initial_pair": "not reported",
            "final_pair": "not reported",
            "forster_defect": "not reported",
            "interaction_strength": "not reported",
            "pair_potential": "not reported",
            "gate_result": "not reported",
            "citation_key": "DAMOP2026_Xu",
        },
        {
            "work": "This work (Forster)",
            "initial_pair": "Rb 56S_1/2 + Yb S (nu=48.369927)",
            "final_pair": "Rb 56P_1/2 + Yb P (nu=48.014048)",
            "forster_defect": "-0.763732 MHz",
            "interaction_strength": "2|V|/h = 30.8 MHz at R=3.4 um",
            "pair_potential": "computed pair Hamiltonian",
            "gate_result": "modeled composite CZ: F_avg=99.93%; sampled minimum=99.84%",
            "citation_key": "this_work",
        },
        {
            "work": "This work (vdW)",
            "initial_pair": "Rb 66S_1/2 + Yb S (nu=62.6823)",
            "final_pair": "not a Forster pair",
            "forster_defect": "315 MHz (nearest partner)",
            "interaction_strength": "U/h = 57.1 MHz at R=3.3 um",
            "pair_potential": "C6/h = +73.58 GHz um^6",
            "gate_result": "not assigned",
            "citation_key": "this_work",
        },
    ]


def test_vdw_feasibility_script_reproduces_committed_data(tmp_path: Path) -> None:
    output = tmp_path / "vdw_feasibility_results.json"
    deviation = tmp_path / "vdw_r6_deviation.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduce_vdw_feasibility.py"),
            "--output",
            str(output),
            "--deviation-output",
            str(deviation),
        ],
        cwd=ROOT,
        check=True,
    )
    assert json.loads(output.read_text()) == load_json("vdw_feasibility_results.json")
    assert deviation.read_bytes() == (DATA / "vdw_r6_deviation.csv").read_bytes()


def test_database_manifests_are_distinct_and_complete() -> None:
    primary = json.loads(
        (ROOT / "provenance" / "pairinteraction_database_manifest.json").read_text()
    )
    fixed_m = json.loads(
        (ROOT / "provenance" / "pairinteraction_database_manifest_figure1_fixed_m.json").read_text()
    )
    assert primary["pairinteraction_version"] == fixed_m["pairinteraction_version"] == "2.5.0"
    assert primary["assets"]["Rb"]["version"] == fixed_m["assets"]["Rb"]["version"] == "v1.2"
    assert primary["assets"]["Yb171_mqdt"]["version"] == "v1.4"
    assert fixed_m["assets"]["Yb171_mqdt"]["version"] == "v1.2"
    for manifest, expected_count in ((primary, 12), (fixed_m, 11)):
        assert len(manifest["files"]) == expected_count
        assert len({entry["relative_path"] for entry in manifest["files"]}) == expected_count
        assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_manuscript_manifest_identifies_reviewed_pdf() -> None:
    manifest = json.loads((ROOT / "provenance" / "manuscript_manifest.json").read_text())
    assert manifest["source_pdf_sha256"] == (
        "f9a663c18db95812c3f7bb01c0bb01ead5ba5966da3fa73f34169b69416c32f1"
    )
    assert manifest["source_pdf_size_bytes"] == 547072
    assert manifest["source_repository_commit"] == "0e702172d365f5650ea8e9162d87657f7bbd0610"
