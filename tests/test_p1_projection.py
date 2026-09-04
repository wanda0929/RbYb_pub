#!/usr/bin/env python3
"""Fast checks for the P1-2 sparse all-mode propagator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimize_hardware_aware_forster_gate as hardware  # noqa: E402
import simulate_robust_shaped_forster_gate as shaped  # noqa: E402
from evaluate_forster_p1_2 import _sparse_target_returns  # noqa: E402
from simulate_forster_gate import Lifetimes  # noqa: E402


class SparseProjectionTest(unittest.TestCase):
    def test_sparse_target_propagation_matches_existing_small_model(self) -> None:
        energies = np.array([-12.0, 8.0, 31.0])
        ss_amplitudes = np.array([0.7, -0.6, 0.02], dtype=complex)
        pp_amplitudes = np.array([0.5, 0.7, 0.01j], dtype=complex)
        model = SimpleNamespace(
            energies_mhz=energies,
            ss_asymptote_mhz=0.0,
            ss_overlap=ss_amplitudes,
            pp_overlap=pp_amplitudes,
        )
        pulse = (
            shaped.Segment(4.0, -1.0, 0.006),
            shaped.Segment(8.0, 0.5, 0.008),
        )
        lifetimes = Lifetimes(400.0, 190.0, 340.0, 80.0)
        threshold = 1e-3
        mask = np.abs(ss_amplitudes) ** 2 > threshold
        modes = shaped.ModeData(
            distance_um=3.4,
            energies_rel_mhz=energies[mask],
            ss_amplitudes=ss_amplitudes[mask],
            pp_weights=np.abs(pp_amplitudes[mask]) ** 2,
            ss_weights=np.abs(ss_amplitudes[mask]) ** 2,
            retained_ss_weight=float(np.sum(np.abs(ss_amplitudes[mask]) ** 2)),
        )
        expected = hardware._target_returns(modes, pulse, lifetimes, amplitude_scale=1.01)
        actual = _sparse_target_returns(
            model,
            pulse,
            lifetimes,
            threshold,
            target_amplitude_scale=1.01,
        )
        np.testing.assert_allclose(
            [actual["unblocked_return"], actual["blocked_return"]],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_none_cutoff_keeps_every_mode(self) -> None:
        model = SimpleNamespace(
            energies_mhz=np.array([-1.0, 2.0]),
            ss_asymptote_mhz=0.0,
            ss_overlap=np.array([1.0, 0.0], dtype=complex),
            pp_overlap=np.array([0.0, 1.0], dtype=complex),
        )
        actual = _sparse_target_returns(
            model,
            (shaped.Segment(0.0, 0.0, 0.001),),
            Lifetimes(400.0, 190.0, 340.0, 80.0),
            None,
            target_amplitude_scale=1.0,
        )
        self.assertEqual(actual["active_modes"], 2)
        self.assertAlmostEqual(actual["retained_ss_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
