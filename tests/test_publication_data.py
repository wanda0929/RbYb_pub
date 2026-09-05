from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text())


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
    assert projected["maximum_transfer_probability"] == pytest.approx(0.9993856539028271)
    assert projected["first_maximum_time_ns"] == pytest.approx(16.226911379238103, abs=5e-9)

    assert point["full_bright_splitting_mhz"] == pytest.approx(31.109926469257402, abs=5e-9)
    bright = point["bright_states"]
    assert [row["pp_weight"] for row in bright] == pytest.approx(
        [0.4216754837049728, 0.5775496984412997]
    )
    assert [row["ss_weight"] for row in bright] == pytest.approx(
        [0.5767393673319066, 0.4209874252977235]
    )
    maximum = point["first_exchange_maximum"]
    assert maximum["pp_population"] == pytest.approx(0.9728638650531252)
    assert maximum["pp_population"] <= point["unitarity_transfer_upper_bound"]

    scans = (
        data["distance_scan"],
        data["fixed_m_field_scan"],
        data["fixed_m_angle_scan"],
        data["all_m_field_scan"]["points"],
    )
    assert [len(scan) for scan in scans] == [10, 10, 7, 10]
    assert data["convergence"]["passed"] is True
    assert all(data["validation"].values())


def test_forster_gate_matches_figure_3_search_record() -> None:
    data = load_json("forster_gate_results.json")
    assert data["software"]["pairinteraction"] == "2.5.0"
    assert data["parameters"] == pytest.approx(
        [
            4.480374546413655,
            9.229350329961736,
            11.778486107199452,
            -0.3647742336217171,
            -1.0235176743455907,
            2.4312627812906094,
            0.025634115265641792,
        ]
    )

    segments = data["command_segments"]
    assert len(segments) == 5
    assert [segment["omega_mhz"] for segment in segments] == pytest.approx(
        [4.480374546413655, 9.229350329961736, 11.778486107199452]
        + [9.229350329961736, 4.480374546413655]
    )
    assert [segment["detuning_mhz"] for segment in segments] == pytest.approx(
        [-0.3647742336217171, -1.0235176743455907, 2.4312627812906094]
        + [-1.0235176743455907, -0.3647742336217171]
    )

    result = data["short_minimax"]
    assert result["target_duration_us"] == pytest.approx(0.1600289492601458)
    assert result["total_gate_time_us"] == pytest.approx(0.36002894926014584)
    assert result["nominal_fidelity"] == pytest.approx(0.999318459478283)
    assert result["position_only_worst_fidelity"] == pytest.approx(0.9992023214963179)
    assert result["amplitude_vertex_worst_fidelity"] == pytest.approx(0.998693823160795)
    assert data["figure_model"].startswith("final P0-4")

    assumptions = data["assumptions"]
    assert "not CPTP process fidelity" in assumptions["metric"]
    assert data["fixed_inputs"]["field_gauss"] == 3.1
    assert data["optimization"]["performed"] is False


def test_p0_4_reference_and_bounded_values() -> None:
    data = load_json("forster_p0_4_uncertainty_convergence.json")
    bounded = data["reference_model_bounded_validation"]
    assert bounded["pulse_reoptimized"] is True
    assert "not converged/global" in bounded["pulse_reoptimization_provenance"]
    assert bounded["nominal_fidelity"] == pytest.approx(0.999318459478283)
    assert bounded["position_only_minimum_fidelity"] == pytest.approx(0.9992023214963179)
    assert bounded["position_and_amplitude_vertex_minimum_fidelity"] == pytest.approx(
        0.998693823160795
    )
    assert bounded["worst_case"] == pytest.approx(
        {
            "geometry_index": 10,
            "radial_displacement_nm": 50,
            "direction_cosine": -1,
            "delta_z_nm": -50,
            "transverse_nm": 0,
            "theta_deg": 0,
            "yb_rabi_scale": 0.99,
            "rb_rabi_scale": 0.99,
            "fidelity": 0.998693823160795,
        }
    )
    assert len(data["one_at_a_time_numerical_convergence"]) == 8
    assert [row["maximum_step_ns"] for row in data["propagation_step_convergence"]] == [
        1.0,
        0.5,
        0.25,
        0.125,
    ]
    assert len(data["spectroscopy_sensitivity_envelope"]["vertices"]) == 4


def test_p1_1_continuous_domain_and_thermal_record() -> None:
    data = load_json("forster_p1_1_robustness.json")
    domain = data["bounded_domain"]
    assert domain == {
        "max_displacement_um": 0.05,
        "yb_amplitude_bounds": [0.99, 1.01],
        "rb_amplitude_bounds": [0.99, 1.01],
    }
    sobol = data["nested_sobol"]["minimum_trace"]
    assert [row["sample_count"] for row in sobol] == [128, 256, 512, 1024]
    assert [row["minimum"] for row in sobol] == pytest.approx(
        [0.9988244811501916] * 4
    )
    local = data["local_adversarial_refinement"]
    assert len(local["surface_results"]) == 12
    assert len(local["full_hamiltonian_rechecks"]) == 3
    assert local["minimum_direct_recheck_fidelity"] == pytest.approx(0.9986938231607982)
    assert "not a global certificate" in local["claim_scope"]
    thermal = data["thermal_motion"]
    assert thermal["sample_count"] == 262144
    assert thermal["seed"] == 20260904
    assert "not a same-apparatus gate prediction" in thermal["interpretation"]


def test_p1_2_all_mode_projection_record() -> None:
    data = load_json("forster_p1_2_projection_audit.json")
    assert data["mode_cutoffs"] == [1e-6, 1e-8, "all modes"]
    assert data["maximum_absolute_all_mode_projection_fidelity_difference"] == pytest.approx(
        7.183481587347273e-10
    )
    assert [row["label"] for row in data["points"]] == [
        "nominal",
        "P0-4 limiting axial vertex",
    ]
    for point in data["points"]:
        assert [row["active_modes"] for row in point["cutoff_rows"]][-1] == 3684
        assert point["cutoff_rows"][-1]["bright_mode_cutoff"] is None
    assert "does not upgrade" in data["angular_scope"]["claim_limit"]


def test_p1_4_reference_spectrum_record() -> None:
    data = load_json("forster_p1_4_reference_spectrum.json")
    assert "finite P0-4 basis" in data["scope"]
    assert data["fixed_inputs"]["pair_basis_size"] == 3684
    assert data["fixed_inputs"]["connected_component_size"] == 3684
    spectrum = data["spectral_diagnostics"]
    target = spectrum["target_eigenstates"]
    assert [row["energy_relative_ss_asymptote_mhz"] for row in target] == pytest.approx(
        [-19.011564205670883, 11.635849585956572]
    )
    assert [row["ss_weight"] for row in target] == pytest.approx(
        [0.5297754958933687, 0.4663948004501662]
    )
    assert [row["pp_weight"] for row in target] == pytest.approx(
        [0.46626553146083427, 0.5290828896107129]
    )
    assert spectrum["nearest_spectator_gap_mhz"] == pytest.approx(48.64623245465875)
    largest = spectrum["largest_target_overlap_spectators"][0]
    assert largest["ss_weight"] + largest["pp_weight"] == pytest.approx(0.002284297331779965)


def test_vdw_dense_data_matches_figure_4() -> None:
    data = load_json("vdw_dense_data.json")
    assert "not a driven multichannel gate" in data["scope"]
    assert data["configuration"]["zero_field_c6_ghz_um6"] == pytest.approx(73.57944850063153)
    distance_track = data["distance_track_b25_pp"]
    sectors = data["sector_track_b25"]
    assert len(distance_track) == 12
    assert len(sectors) == 4
    working_point = next(row for row in distance_track if row["distance_um"] == 3.3)
    assert working_point["u_mhz"] == pytest.approx(57.0720534324646)
    assert working_point["overlap"] == pytest.approx(0.9697184016029612)


def test_p1_5_vdw_basis_convergence_record() -> None:
    data = load_json("vdw_p1_5_basis_convergence.json")
    assert data["status"] == "static vdW one-at-a-time basis convergence complete"
    assert data["software"]["database_hash_manifest"] == (
        "provenance/pairinteraction_database_manifest.json"
    )
    assert data["reference_reproduction"]["source"] == "data/vdw_dense_data.json"
    assert data["reference_reproduction"]["passed"] is True
    rows = data["one_at_a_time_rows"]
    assert len(rows) == 9
    reference = next(row for row in rows if row["comparison"] == "reference")
    assert reference["pair_basis_size"] == 16279
    assert reference["c6_ghz_um6"] == pytest.approx(73.57944835201508)
    assert reference["working_point_shift_mhz"] == pytest.approx(57.07205390930176)
    assert reference["bare_pair_weight"] == pytest.approx(0.969718401611007)
    expanded = [row for row in rows if row["comparison"] == "expanded"]
    assert [row["axis"] for row in expanded] == [
        "rb_radial_range",
        "l_max",
        "yb_radial_range",
        "pair_energy_window",
    ]
    maxima = data["convergence"]["maximum_changes_over_expanded_rows"]
    assert maxima["c6_ghz_um6"]["absolute_change"] == pytest.approx(0.01049242940258921)
    assert maxima["working_point_shift_mhz"]["absolute_change"] == pytest.approx(
        0.007853507995605469
    )
    assert maxima["bare_pair_weight"]["absolute_change"] == pytest.approx(1.3121926335069034e-05)
    assert "do not test cross-terms" in data["caveat"]


def test_table_ii_csv_matches_current_manuscript() -> None:
    with (DATA / "prior_work.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["publication_type"] for row in rows] == [
        "preprint",
        "conference abstract",
        "manuscript",
        "manuscript",
    ]
    assert rows[2]["forster_defect"] == "-0.764 MHz electronic; -0.694 MHz stretched HFS"
    assert rows[2]["gate_result"].endswith("F_avg=99.93%; sampled minimum=99.87%")
    assert rows[3]["work"] == "This work (vdW candidate)"
    assert rows[3]["gate_result"] == "not assigned"


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
    assert all(len(entry["sha256"]) == 64 for entry in primary["files"])
    assert embedded["versions"] == {"Rb": "v1.2", "Yb171_mqdt": "v1.4", "misc": "v1.4"}
    assert {
        (entry["relative_path"], entry["size_bytes"], entry["sha256"])
        for entry in embedded["files"]
    } == {
        (entry["relative_path"], entry["size_bytes"], entry["sha256"]) for entry in primary["files"]
    }


def test_manuscript_manifest_identifies_current_pdf() -> None:
    manifest = json.loads((ROOT / "provenance" / "manuscript_manifest.json").read_text())
    assert manifest["title"] == "87Rb-171Yb Rydberg interactions and a Förster CZ gate"
    assert manifest["source_pdf_sha256"] == (
        "e2ad80c3b39680cd18a922ec1dd89589d251dc9d9d38e414bf0603a4f62b3710"
    )
    assert manifest["source_pdf_size_bytes"] == 591570
    assert manifest["source_repository_commit"] == "a54e98ee5100296160b71871624d1bc83b816b9e"
    assert manifest["source_repository_commit_timestamp_utc"] == "2026-09-05T08:52:53Z"


def test_archive_map_and_reproduction_entry_point_are_complete() -> None:
    expected = [
        "data/forster_characterization.json",
        "data/forster_gate_results.json",
        "data/forster_p0_4_reoptimization.json",
        "data/forster_p0_4_field_scan.json",
        "data/forster_p0_4_uncertainty_convergence.json",
        "data/forster_p1_1_robustness.json",
        "data/forster_p1_2_projection_audit.json",
        "data/forster_p1_4_reference_spectrum.json",
        "data/vdw_dense_data.json",
        "data/vdw_p1_5_basis_convergence.json",
        "scripts/reproduce_all.py",
        "scripts/reproduce_forster_characterization.py",
        "scripts/reproduce_forster_gate.py",
        "scripts/optimize_forster_p0_4_reference.py",
        "scripts/scan_forster_p0_4_field.py",
        "scripts/evaluate_forster_p0_4.py",
        "scripts/evaluate_forster_p1_1.py",
        "scripts/evaluate_forster_p1_2.py",
        "scripts/evaluate_forster_p1_4.py",
        "scripts/reproduce_vdw_dense.py",
        "scripts/evaluate_vdw_p1_5.py",
        "scripts/plot_channel_forster.py",
        "scripts/plot_candidate_excitation.py",
        "scripts/plot_forster_gate.py",
        "scripts/plot_channel_vdw.py",
        "docs/INDEPENDENT_VERIFICATION_GUIDE.md",
        "docs/P1_1_ROBUSTNESS_METHOD.md",
        "provenance/runtime_environment.json",
        "tests/test_p1_projection.py",
        "tests/test_p1_robustness.py",
        "tests/test_publication_data.py",
        "tests/test_forster_characterization.py",
        "tests/test_forster_p0_4_field.py",
        "tests/test_forster_p0_4_reoptimization.py",
        "tests/test_rb_rydberg_hyperfine.py",
        "tests/test_vdw_p1_5.py",
    ]
    assert all((ROOT / relative).is_file() for relative in expected)
    readme = (ROOT / "README.md").read_text()
    assert "scripts/reproduce_all.py --full" in readme
    assert "not a gate design or process-fidelity result" in " ".join(readme.split())
    assert "33,180-state pair basis" in readme


def test_runtime_environment_records_backend_and_seeds() -> None:
    runtime = json.loads((ROOT / "provenance" / "runtime_environment.json").read_text())
    generated = runtime["record_generation_environment"]
    assert generated["pairinteraction"] == "2.5.0"
    assert generated["numpy"] == "2.4.6"
    assert "OpenBLAS" in generated["blas_lapack"]
    locked = runtime["locked_clean_environment_check"]
    assert locked["numpy"] == "2.3.5"
    assert locked["pytest"] == "9.1.1"
    assert runtime["randomness"]["thermal_benchmark_seed"] == 20260904
    resources = runtime["recorded_resource_observations"]["vdw_p1_5_basis_convergence"]
    assert resources["largest_pair_basis_size"] == 33180
    assert resources["peak_rss_kib"] == 10577180


def test_figure_3_input_hashes_match_archive_bytes() -> None:
    import hashlib

    gate = load_json("forster_gate_results.json")
    for record in gate["input_records"]:
        path = ROOT / record["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_archive_contains_no_private_paths_or_obsolete_loss_claims() -> None:
    private_paths = ("/Users/", "/home/user/", "QSTC_PROJECT", "RbYbLandscape")
    obsolete_claims = ("loss counted as erasure", "mapped to orthogonal erasure")
    checked = [
        ROOT / "README.md",
        *sorted((ROOT / "data").glob("*")),
        *sorted((ROOT / "docs").glob("*")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "provenance").glob("*")),
    ]
    for path in checked:
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        assert not any(value in text for value in private_paths), path
        assert not any(value in text.lower() for value in obsolete_claims), path
