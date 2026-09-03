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


def assert_json_close(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_json_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_json_close(actual_item, expected_item)
    elif isinstance(expected, int | float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=1e-12, abs=5e-12)
    else:
        assert actual == expected


def test_forster_characterization_matches_figure_1() -> None:
    data = load_json("forster_characterization.json")
    exchange = data["target_states"]["energy_exchange"]
    assert exchange["direction"] == "SS to PP"
    assert exchange["rb_absorbed_ghz"] == pytest.approx(20.92229928433895, abs=5e-12)
    assert exchange["yb_released_ghz"] == pytest.approx(20.921535552740096, abs=5e-12)
    assert exchange["defect_ss_minus_pp_mhz"] == pytest.approx(-0.7637315988553439, abs=5e-9)

    fixed_model = data["fixed_m_model"]
    assert fixed_model["pair_basis_size"] == 2411
    assert fixed_model["target_projection"] == "complex PairInteraction get_amplitudes vectors"
    assert fixed_model["pair_energy_window_ghz"] == [-80.0, 80.0]

    point = data["operating_point"]
    projected = point["projected_two_state_model"]
    assert abs(complex(*projected["coupling_re_im_mhz"])) == pytest.approx(
        15.401772345881476, abs=5e-9
    )
    assert projected["generalized_splitting_mhz"] == pytest.approx(30.813011072442077, abs=5e-9)
    assert projected["maximum_transfer_probability"] == pytest.approx(0.9993856539028271, abs=5e-12)
    assert projected["first_maximum_time_ns"] == pytest.approx(16.226911379238103, abs=5e-9)

    assert point["full_bright_splitting_mhz"] == pytest.approx(31.109926469257402, abs=5e-9)
    bright = point["bright_states"]
    assert [row["pp_weight"] for row in bright] == pytest.approx(
        [0.4216754837049728, 0.5775496984412997], abs=5e-12
    )
    assert [row["ss_weight"] for row in bright] == pytest.approx(
        [0.5767393673319066, 0.4209874252977235], abs=5e-12
    )
    maximum = point["first_exchange_maximum"]
    assert maximum == pytest.approx(
        {
            "time_ns": 15.989779013311212,
            "pp_population": 0.9728638650531252,
            "residual_ss_population": 0.023770344054174445,
            "spectator_population": 0.0033657908927003898,
        },
        abs=5e-12,
    )
    assert point["unitarity_transfer_upper_bound"] == pytest.approx(0.9730961725133787, abs=5e-12)
    assert maximum["pp_population"] <= point["unitarity_transfer_upper_bound"]

    distance_scan = data["distance_scan"]
    field_scan = data["fixed_m_field_scan"]
    angle_scan = data["fixed_m_angle_scan"]
    all_m_scan = data["all_m_field_scan"]["points"]
    assert [len(scan) for scan in (distance_scan, field_scan, angle_scan, all_m_scan)] == [
        10,
        10,
        7,
        10,
    ]
    all_m_by_field = {row["field_gauss"]: row for row in all_m_scan}
    assert all_m_by_field[0.0]["first_exchange_maximum"]["pp_population"] == pytest.approx(
        0.9189690756718504, abs=5e-12
    )
    assert all_m_by_field[3.1]["first_exchange_maximum"]["pp_population"] == pytest.approx(
        0.982180111486594, abs=5e-12
    )
    assert all_m_by_field[5.0]["first_exchange_maximum"]["pp_population"] == pytest.approx(
        0.9954373079226287, abs=5e-12
    )
    assert (
        max(all_m_scan, key=lambda row: row["first_exchange_maximum"]["pp_population"])[
            "field_gauss"
        ]
        == 5.0
    )

    assert data["convergence"]["passed"] is True
    assert len(data["convergence"]["checks"]) == 3
    assert all(data["validation"].values())
    for scan in (distance_scan, field_scan, angle_scan, all_m_scan):
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
            "interaction_strength": "Delta nu = 31.1 MHz (finite basis) at R=3.4 um",
            "pair_potential": "finite-basis multichannel model",
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
    assert_json_close(json.loads(output.read_text()), load_json("vdw_feasibility_results.json"))
    assert deviation.read_bytes() == (DATA / "vdw_r6_deviation.csv").read_bytes()


def test_database_manifest_is_complete() -> None:
    primary = json.loads(
        (ROOT / "provenance" / "pairinteraction_database_manifest.json").read_text()
    )
    embedded = load_json("forster_characterization.json")["database"]
    assert primary["pairinteraction_version"] == "2.5.0"
    assert primary["assets"]["Rb"]["version"] == "v1.2"
    assert primary["assets"]["Yb171_mqdt"]["version"] == "v1.4"
    assert primary["assets"]["misc"]["version"] == "v1.4"
    assert len(primary["files"]) == 13
    assert len({entry["relative_path"] for entry in primary["files"]}) == 13
    assert all(len(entry["sha256"]) == 64 for entry in primary["files"])
    assert embedded["versions"] == {"Rb": "v1.2", "Yb171_mqdt": "v1.4", "misc": "v1.4"}
    assert {
        (entry["relative_path"], entry["size_bytes"], entry["sha256"])
        for entry in embedded["files"]
    } == {
        (entry["relative_path"], entry["size_bytes"], entry["sha256"]) for entry in primary["files"]
    }


def test_manuscript_manifest_identifies_reviewed_pdf() -> None:
    manifest = json.loads((ROOT / "provenance" / "manuscript_manifest.json").read_text())
    assert manifest["source_pdf_sha256"] == (
        "507d9130579157924c48d44ee3f196b657a1baeb482db6c7b38f54c26c420a4c"
    )
    assert manifest["source_pdf_size_bytes"] == 515155
    assert manifest["source_repository_commit"] == "9e9c0717ad517c3bef0af6a94fdaaccb1f728222"
