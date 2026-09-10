"""Displayed current-manuscript cells, separated from fresh physics validation."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data/manuscript_claims_2026_09_10/simulations"
sys.path.insert(0, str(ROOT / "scripts"))
import quick_check  # noqa: E402
import reproduce_all  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def test_table_iii_all_56_displayed_cells():
    # Independently transcribed from main.tex lines 600–607, hash in audit.
    displayed = np.array(
        [
            [2075, 30.72486, 99.54846, 98.32421, 0.07387, 99.89115, 99.90540],
            [3684, 30.64741, 99.54777, 98.81177, 0.07619, 99.91258, 99.91258],
            [5212, 30.65471, 99.54692, 98.79627, 0.07651, 99.91250, 99.91251],
            [4658, 30.64262, 99.54653, 98.79206, 0.07648, 99.91255, 99.91261],
            [6359, 30.64744, 99.54776, 98.81151, 0.07620, 99.91258, 99.91258],
            [4398, 30.64741, 99.54777, 98.81178, 0.07619, 99.91258, 99.91258],
            [1872, 30.78321, 99.74761, 98.84960, 0.01949, 99.87899, 99.90193],
            [3684, 30.66929, 99.54822, 98.81088, 0.07591, 99.91252, 99.91252],
        ]
    )
    rows = load(A / "forster_control_decay_p0_4.json")["one_at_a_time_numerical_convergence"]
    actual = np.array(
        [
            [
                r["connected_component_size"],
                r["target_bright_splitting_mhz"],
                100 * r["minimum_target_subspace_weight"],
                100 * r["static_transfer"]["maximum_ss_population"],
                100 * r["fixed_pulse_population"]["maximum_transient_spectator_population"],
                100 * r["gate"]["fixed_reference_local_z_fidelity"],
                100 * r["gate"]["phase_recalibrated_fidelity"],
            ]
            for r in rows
        ]
    )
    np.testing.assert_array_equal(actual[:, 0], displayed[:, 0])
    np.testing.assert_allclose(actual[:, 1:], displayed[:, 1:], atol=5.01e-6, rtol=0)


def test_table_v_all_20_displayed_cells_use_full_basis():
    displayed = np.array(
        [
            [16279, 73.57945, 57.07205, 0.969718],
            [19317, 73.57948, 57.07751, 0.969709],
            [19522, 73.57873, 57.07173, 0.969719],
            [33180, 73.57945, 57.06934, 0.969732],
            [21681, 73.58994, 57.07991, 0.969718],
        ]
    )
    rows = load(ROOT / "data/vdw_p1_5_basis_convergence.json")["one_at_a_time_rows"]
    by_axis = {r["axis"]: r for r in rows if r["comparison"] in ("reference", "expanded")}
    rows = [
        by_axis[axis]
        for axis in (
            "reference",
            "rb_radial_range",
            "yb_radial_range",
            "l_max",
            "pair_energy_window",
        )
    ]
    actual = np.array(
        [
            [
                r[k]
                for k in (
                    "pair_basis_size",
                    "c6_ghz_um6",
                    "working_point_shift_mhz",
                    "bare_pair_weight",
                )
            ]
            for r in rows
        ]
    )
    np.testing.assert_array_equal(actual[:, 0], displayed[:, 0])
    np.testing.assert_allclose(actual[:, 1:3], displayed[:, 1:3], atol=5.01e-6, rtol=0)
    np.testing.assert_allclose(actual[:, 3], displayed[:, 3], atol=5.01e-7, rtol=0)


def test_table_iv_derived_budget_and_table_ii_qualified_inputs():
    record = load(A / "forster_control_decay_recheck.json")
    row = record["rows"]["0.000000/0.000000"]
    k = np.asarray(row["kraus_diagonal_re_im"])
    loss = 1 - np.sum(k * k, axis=1)
    np.testing.assert_allclose(loss, [0.0, 6.2489e-4, 1.3550e-3, 1.5161e-3], atol=5.01e-8, rtol=0)
    np.testing.assert_allclose(
        loss / 4, [0.0, 1.5622e-4, 3.3875e-4, 3.7904e-4], atol=5.01e-9, rtol=0
    )
    fidelity = row["nominal_rabi_fixed_z_overlap"]
    assert loss.mean() == pytest.approx(8.7401e-4, abs=5.01e-9, rel=0)
    assert 1 - loss.mean() - fidelity == pytest.approx(2.08e-7, abs=5.01e-10, rel=0)
    assert 1 - fidelity == pytest.approx(8.7422e-4, abs=5.01e-9, rel=0)
    states = load(ROOT / "data/working_state_inputs.json")
    assert len(states["rows"]) == 5
    assert "quoted/unverified" in states["scope"]
    for state in states["rows"]:
        if "lifetime_record_key" in state:
            assert state["quoted_lifetime_us"] == pytest.approx(
                record["lifetimes_us"][state["lifetime_record_key"]], abs=0.005, rel=0
            )
        if "lifetime_record_keys" in state:
            np.testing.assert_allclose(
                state["quoted_lifetimes_us"],
                [record["lifetimes_us"][k] for k in state["lifetime_record_keys"]],
                atol=0.005,
                rtol=0,
            )


@pytest.mark.parametrize(
    "key",
    [
        "basis_size",
        "nominal_rabi_fixed_z_overlap",
        "kraus_diagonal_re_im",
        "amplitude_vertex_overlaps",
    ],
)
def test_quick_check_rejects_bad_physics_outputs(key):
    expected = load(A / "forster_control_decay_recheck.json")["rows"]["0.000000/0.000000"]
    actual = copy.deepcopy(expected)
    actual[key] = np.asarray(actual[key]) + 0.001
    with pytest.raises((ValueError, AssertionError)):
        quick_check.compare(actual, expected)


def test_reviewer_entrypoints_do_not_use_old_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(reproduce_all, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["reproduce_all.py", "--figures"])
    reproduce_all.main()
    assert [c[1] for c in calls] == [
        "scripts/check_control_decay_data.py",
        "scripts/plot_channel_forster.py",
        "scripts/plot_candidate_excitation.py",
        "scripts/plot_control_decay_gate.py",
        "scripts/plot_channel_vdw.py",
    ]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["reproduce_all.py", "--quick-check"])
    reproduce_all.main()
    assert [c[1] for c in calls] == [
        "scripts/check_control_decay_data.py",
        "scripts/quick_check.py",
    ]
