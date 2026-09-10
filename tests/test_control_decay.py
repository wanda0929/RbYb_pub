"""Control-decay regression tests retained without the unrelated optimizer archive."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimize_forster_p0_4_reference as reference  # noqa: E402
from simulate_forster_gate import Lifetimes  # noqa: E402

hardware = reference.hardware
shaped = reference.shaped


def mode():
    return shaped.ModeData(
        distance_um=3.4,
        energies_rel_mhz=np.array([1.9]),
        ss_amplitudes=np.array([1.0 + 0j]),
        pp_weights=np.array([0.0]),
        ss_weights=np.array([1.0]),
        retained_ss_weight=1.0,
    )


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
