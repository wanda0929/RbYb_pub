#!/usr/bin/env python3
"""Fast checks for the final-model P0-4 reoptimization contract."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import optimize_forster_p0_4_reference as reoptimization


class ReoptimizationConfigurationTest(unittest.TestCase):
    def test_final_reference_basis_and_training_set_are_fixed(self) -> None:
        basis = reoptimization.p0_4.REFERENCE_BASIS
        self.assertEqual(
            (
                basis.delta_n,
                basis.atomic_window_ghz,
                basis.pair_window_ghz,
                basis.delta_l,
                basis.interaction_order,
            ),
            (3, 80.0, 40.0, 2, 4),
        )
        self.assertEqual(
            reoptimization.OPTIMIZATION_SPECS,
            ((0.0, 0.0), (0.05, -1.0), (0.05, 1.0)),
        )
        self.assertEqual(reoptimization.MODE_CUTOFF, 1e-6)
        self.assertEqual(reoptimization.OPTIMIZATION_STEP_NS, 1.0)
        self.assertEqual(reoptimization.FINAL_STEP_NS, 0.125)
        self.assertEqual(len(reoptimization.validation_specs()), 19)
        self.assertEqual(
            reoptimization.DATABASE_ASSETS,
            {
                "Rb": "Rb_v1.2",
                "Yb171_mqdt": "Yb171_mqdt_v1.4",
                "misc": "misc_v1.4",
            },
        )

    def test_reoptimization_seed_is_independent_of_downstream_selection(self) -> None:
        expected_seed = np.array(
            [
                4.410416083986085,
                9.418791104832165,
                11.734642228670836,
                -1.1443728579899664,
                -0.9263577893268974,
                1.744859450470417,
                0.025634115265641792,
            ]
        )
        expected_selected = np.array(
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
        np.testing.assert_array_equal(
            reoptimization.REOPTIMIZATION_SEED_PARAMETERS, expected_seed
        )
        np.testing.assert_array_equal(
            reoptimization.minimax.SELECTED_PARAMETERS, expected_selected
        )
        self.assertFalse(
            np.shares_memory(
                reoptimization.REOPTIMIZATION_SEED_PARAMETERS,
                reoptimization.minimax.SELECTED_PARAMETERS,
            )
        )

    def test_objective_uses_eight_vertices_plus_nominal(self) -> None:
        endpoint = np.array([0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998])
        nominal = 0.999
        infidelities = 1 - np.r_[endpoint, nominal]
        expected = np.max(infidelities) + 0.02 * np.mean(infidelities)
        self.assertAlmostEqual(
            reoptimization.robust_objective(endpoint.reshape(2, 2, 2), nominal),
            expected,
        )
        with self.assertRaises(ValueError):
            reoptimization.robust_objective(endpoint[:-1], nominal)


class ReoptimizationAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = {
            "nominal_fidelity": 0.9989,
            "endpoint_objective": 0.0021,
            "endpoint_vertex_minimum": 0.9980,
            "full_grid_vertex_minimum": 0.9980,
        }

    def test_accepts_only_crossing_nonworsening_candidate(self) -> None:
        candidate = {
            "nominal_fidelity": 0.9991,
            "endpoint_objective": 0.0020,
            "endpoint_vertex_minimum": 0.9981,
            "full_grid_vertex_minimum": 0.9981,
        }
        checks = reoptimization.acceptance_checks(
            self.baseline, candidate, include_full_grid=True
        )
        self.assertTrue(checks["accepted"])
        self.assertTrue(all(checks.values()))

    def test_rejects_nominal_miss_or_hidden_robust_regression(self) -> None:
        nominal_miss = {
            **self.baseline,
            "nominal_fidelity": 0.99899,
            "endpoint_objective": 0.0020,
            "endpoint_vertex_minimum": 0.9981,
            "full_grid_vertex_minimum": 0.9981,
        }
        self.assertFalse(
            reoptimization.acceptance_checks(
                self.baseline, nominal_miss, include_full_grid=True
            )["accepted"]
        )

        robust_regression = {
            "nominal_fidelity": 0.9991,
            "endpoint_objective": 0.0020,
            "endpoint_vertex_minimum": 0.9979,
            "full_grid_vertex_minimum": 0.9979,
        }
        checks = reoptimization.acceptance_checks(
            self.baseline, robust_regression, include_full_grid=True
        )
        self.assertFalse(checks["endpoint_vertex_minimum_no_worse_than_baseline"])
        self.assertFalse(checks["full_grid_vertex_minimum_no_worse_than_baseline"])
        self.assertFalse(checks["accepted"])


if __name__ == "__main__":
    unittest.main()
