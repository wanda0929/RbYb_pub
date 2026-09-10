"""Supplemental archive integrity and opt-in decay without changing legacy defaults."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_control_decay_convergence as runner  # noqa: E402
from simulate_forster_gate import Lifetimes  # noqa: E402

reference = runner.reference
hardware = reference.hardware
shaped = reference.shaped
ARCHIVE = ROOT / "data" / "forster_control_decay_convergence"


def mode():
    return shaped.ModeData(
        distance_um=3.4,
        energies_rel_mhz=np.array([1.9]),
        ss_amplitudes=np.array([1.0 + 0j]),
        pp_weights=np.array([0.0]),
        ss_weights=np.array([1.0]),
        retained_ss_weight=1.0,
    )


def test_archive_bytes_and_convergence_contract():
    hashes = {
        "result.json": "51a4514a90c114c717c8a3c87c1f4d77f0fcbde54c979df5861572ab3c4dd0cd",
        "evaluations.jsonl": "f4c423267542eb94c4cdcf6784463758592ceaeb9ff7003ec931d7adb04d6a69",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == expected
    result = json.loads((ARCHIVE / "result.json").read_text())
    rows = [json.loads(line) for line in (ARCHIVE / "evaluations.jsonl").read_text().splitlines()]
    assert result["status"] == "finished"
    assert result["optimizer"]["success"] is True
    assert result["optimizer"]["nit"] == 2091 < result["options"]["maxiter"]
    assert result["evaluations"] == result["optimizer"]["nfev"] == len(rows) == 3466
    assert result["training_configurations"] == 9
    assert result["training_specs"] == [[0.0, 0.0], [0.05, -1.0], [0.05, 1.0]]
    for number, row in enumerate(rows, 1):
        assert row["evaluation"] == number
        assert row["parameters"][-1] == result["seed_parameters"][-1]
        vertices = np.array(row["metrics"]["endpoint_vertex_fidelities"])
        assert vertices.shape == (2, 2, 2)
        assert vertices.min() == row["metrics"]["endpoint_vertex_minimum"]
    assert min(row["metrics"]["endpoint_objective"] for row in rows) == result["best_objective"]
    points, values = map(np.array, result["optimizer"]["final_simplex"])
    assert np.max(np.abs(points[1:] - points[0])) <= result["options"]["xatol"]
    assert np.max(np.abs(values[1:] - values[0])) <= result["options"]["fatol"]
    assert set(result["fine_step_recheck"]) == {
        f"{label}_{step}ns" for label in ("baseline", "candidate") for step in (0.125, 0.0625)
    }
    assert "no global certificate or full-grid acceptance" in result["scope"]
    np.testing.assert_array_equal(
        result["baseline_parameters"], reference.minimax.SELECTED_PARAMETERS
    )
    assert result["best_parameters"] != result["baseline_parameters"]


def test_zero_drive_control_decay_is_opt_in_and_no_loss_is_unchanged():
    pulse = (shaped.Segment(0.0, 0.7, 0.23),)
    lifetimes = Lifetimes(2.0, 0.7, 3.0, 1.3)
    legacy = hardware._target_returns(mode(), pulse, lifetimes)
    corrected = hardware._target_returns(
        mode(), pulse, lifetimes, include_blocked_control_decay=True
    )
    np.testing.assert_allclose(legacy, [1.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(corrected, [1.0, np.exp(-0.23 / (2 * 0.7))], atol=1e-14)
    np.testing.assert_allclose(
        hardware._target_returns(mode(), pulse, None, include_blocked_control_decay=True),
        [1.0, 1.0],
        atol=1e-14,
    )


def test_driven_decay_matches_independent_dense_generator_and_preserves_control_only():
    lifetimes = Lifetimes(2.0, 0.7, 3.0, 1.3)
    pulse = (shaped.Segment(2.3, -0.4, 0.13), shaped.Segment(1.1, 0.8, 0.17))
    state = np.array([1.0, 0.0], dtype=complex)
    for segment in pulse:
        coupling = -1j * np.pi * segment.omega_mhz * 0.99
        generator = np.array(
            [
                [-1 / (2 * 0.7), coupling],
                [
                    coupling,
                    -2j * np.pi * (1.9 - segment.detuning_mhz) - 1 / (2 * 0.7) - 1 / (2 * 1.3),
                ],
            ]
        )
        state = expm(generator * segment.duration_us) @ state
    actual = hardware._target_returns(
        mode(), pulse, lifetimes, amplitude_scale=0.99, include_blocked_control_decay=True
    )
    np.testing.assert_allclose(actual[1], state[0], atol=1e-13)
    legacy = hardware._kraus(mode(), pulse, lifetimes)
    corrected = hardware._kraus(mode(), pulse, lifetimes, include_blocked_control_decay=True)
    np.testing.assert_array_equal(legacy[:3], corrected[:3])
    assert abs(legacy[3] - corrected[3]) > 1e-3


def test_training_and_scenario_paths_forward_corrected_decay():
    lifetimes = Lifetimes(2.0, 0.7, 3.0, 1.3)
    parameters = reference.minimax.SELECTED_PARAMETERS
    geometries = [SimpleNamespace(modes=mode()) for _ in range(3)]
    pulse = reference._pulse(parameters, 1.0)
    nominal = hardware._kraus(mode(), pulse, lifetimes, include_blocked_control_decay=True)
    local = shaped._local_z_metrics(nominal)
    correction = (local["optimal_local_z_alpha_rad"], local["optimal_local_z_beta_rad"])
    actual = reference._training_metrics(
        parameters, geometries, lifetimes, 1.0, include_blocked_control_decay=True
    )
    assert actual["nominal_fidelity"] == pytest.approx(
        shaped._fixed_correction_fidelity(nominal, correction), abs=1e-13
    )
    for iy, sy in enumerate((0.99, 1.01)):
        for ir, sr in enumerate((0.99, 1.01)):
            kraus = hardware._kraus(
                mode(),
                pulse,
                lifetimes,
                target_amplitude_scale=sy,
                control_amplitude_scale=sr,
                include_blocked_control_decay=True,
            )
            assert actual["endpoint_vertex_fidelities"][0][iy][ir] == pytest.approx(
                shaped._fixed_correction_fidelity(kraus, correction), abs=1e-13
            )
    legacy = reference._training_metrics(parameters, geometries, lifetimes, 1.0)
    assert abs(legacy["nominal_fidelity"] - actual["nominal_fidelity"]) > 1e-3


@pytest.mark.parametrize("recheck_only", [False, True])
def test_runner_budget_rechecks_and_output_protection(tmp_path, monkeypatch, recheck_only):
    calls = []

    def metrics(parameters, geometries, lifetimes, step, *, include_blocked_control_decay):
        assert include_blocked_control_decay is True
        calls.append((parameters.copy(), step))
        return {"endpoint_objective": float(np.dot(parameters[:6], np.arange(1, 7)) * 1e-5)}

    monkeypatch.setattr(reference, "_training_metrics", metrics)
    monkeypatch.setattr(reference, "_validate_database_assets", lambda: None)
    monkeypatch.setattr(reference, "query_lifetimes", lambda _: Lifetimes(2.0, 0.7, 3.0, 1.3))
    monkeypatch.setattr(reference, "_prepare_geometry", lambda _: None)
    output = tmp_path / "new-run"
    argv = ["runner", "--output-dir", str(output), "--maxiter", "1"]
    if recheck_only:
        argv += ["--recheck-only", "--seed-checkpoint", str(ARCHIVE / "result.json")]
        monkeypatch.setattr(runner, "minimize", lambda *a, **k: pytest.fail("must not optimize"))
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "finished"
    assert result["manuscript_selected"] is False
    assert result["full_grid_validated"] is False
    assert result["global_optimum_certified"] is False
    if recheck_only:
        assert result["optimizer"] is None
        assert len(calls) == 4
    else:
        assert result["optimizer"]["success"] is False
        assert result["optimizer"]["status"] == 2
        assert result["evaluations"] == 7
    assert [step for _, step in calls[-4:]] == [0.125, 0.0625, 0.125, 0.0625]
    assert all(p[-1] == reference.minimax.SELECTED_PARAMETERS[-1] for p, _ in calls)
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.main()
    assert (output / "result.json").read_bytes() == before
