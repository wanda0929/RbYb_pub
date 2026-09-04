#!/usr/bin/env python3
"""Fast self-checks for the physics-independent P1-1 methodology."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, atomic_mass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p1_robustness import (  # noqa: E402
    HarmonicThermalState,
    PhaseSpaceSample,
    RobustnessDomain,
    RobustnessScenario,
    TwoSpeciesPhaseSpace,
    ballistic_relative_trajectory,
    doppler_detuning_mhz,
    harmonic_phase_space_widths,
    local_adversarial_search,
    nested_minimum_trace,
    rotate_phase_space,
    sample_two_species_phase_space,
    sobol_scenarios,
    worst_sample_seeds,
)


class SobolRobustnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = RobustnessDomain()

    def test_sobol_design_is_deterministic_and_nested(self) -> None:
        small = sobol_scenarios(3, self.domain)
        large = sobol_scenarios(4, self.domain)
        self.assertEqual(len(small), 8)
        for left, right in zip(small, large, strict=False):
            np.testing.assert_array_equal(left.as_array(), right.as_array())

    def test_sobol_design_respects_bounds_and_uniform_ball_moment(self) -> None:
        scenarios = sobol_scenarios(12, self.domain)
        points = np.array([scenario.as_array() for scenario in scenarios])
        radii = np.linalg.norm(points[:, :3], axis=1)
        self.assertLessEqual(float(np.max(radii)), self.domain.max_displacement_um)
        self.assertTrue(all(self.domain.contains(scenario) for scenario in scenarios))
        np.testing.assert_allclose(np.mean(points[:, :3], axis=0), 0.0, atol=4e-5)
        self.assertAlmostEqual(
            float(np.mean(radii**2)),
            3 * self.domain.max_displacement_um**2 / 5,
            delta=2e-6,
        )

    def test_nested_minimum_trace_and_seed_selection(self) -> None:
        scenarios = sobol_scenarios(3, self.domain)
        values = np.array([5.0, 4.0, 7.0, 3.0, 8.0, 9.0, 1.0, 2.0])
        trace = nested_minimum_trace(values, powers=(1, 2, 3))
        self.assertEqual(
            [(point.sample_count, point.minimum, point.index) for point in trace],
            [(2, 4.0, 1), (4, 3.0, 3), (8, 1.0, 6)],
        )
        self.assertEqual(worst_sample_seeds(scenarios, values, 2), (scenarios[6], scenarios[7]))

    def test_local_adversarial_search_obeys_ball_and_amplitude_bounds(self) -> None:
        seed = RobustnessScenario(0.0, 0.0, 0.0, 1.0, 1.0)

        def objective(scenario: RobustnessScenario) -> float:
            return scenario.delta_x_um + scenario.yb_amplitude_scale - scenario.rb_amplitude_scale

        result = local_adversarial_search(objective, (seed,), self.domain)[0]
        self.assertTrue(result.success, result.message)
        self.assertTrue(self.domain.contains(result.scenario, atol=1e-9))
        self.assertAlmostEqual(
            result.scenario.delta_x_um, -self.domain.max_displacement_um, places=8
        )
        self.assertAlmostEqual(
            result.scenario.yb_amplitude_scale,
            self.domain.yb_amplitude_bounds[0],
            places=8,
        )
        self.assertAlmostEqual(
            result.scenario.rb_amplitude_scale,
            self.domain.rb_amplitude_bounds[1],
            places=8,
        )


class ThermalMotionTest(unittest.TestCase):
    def test_classical_harmonic_widths(self) -> None:
        state = HarmonicThermalState(
            mass_u=87.0,
            temperature_uk=10.0,
            trap_frequencies_khz=(100.0, 120.0, 20.0),
        )
        position_um, velocity_um_per_us = harmonic_phase_space_widths(state, quantum=False)
        mass_kg = state.mass_u * atomic_mass
        expected_velocity = np.sqrt(Boltzmann * 10e-6 / mass_kg)
        expected_position_um = (
            1e6 * expected_velocity / (2 * np.pi * 1e3 * np.array(state.trap_frequencies_khz))
        )
        np.testing.assert_allclose(position_um, expected_position_um)
        np.testing.assert_allclose(velocity_um_per_us, expected_velocity)

    def test_quantum_widths_include_zero_point_motion(self) -> None:
        state = HarmonicThermalState(
            mass_u=171.0,
            temperature_uk=0.0,
            trap_frequencies_khz=(60.0, 60.0, 10.0),
        )
        position_um, velocity_um_per_us = harmonic_phase_space_widths(state)
        omega = 2 * np.pi * 1e3 * np.array(state.trap_frequencies_khz)
        np.testing.assert_allclose(
            position_um * velocity_um_per_us,
            1e6 * 1.054571817e-34 / (2 * state.mass_u * atomic_mass),
            rtol=1e-8,
        )
        np.testing.assert_allclose(velocity_um_per_us / (position_um * 1e-6), omega, rtol=1e-12)

    def test_two_species_sampling_is_seeded(self) -> None:
        yb = HarmonicThermalState(171.0, 2.9, (60.0, 60.0, 10.0))
        rb = HarmonicThermalState(87.0, 13.0, (154.0, 150.0, 30.0))
        first = sample_two_species_phase_space(yb, rb, 10, seed=17)
        second = sample_two_species_phase_space(yb, rb, 10, seed=17)
        np.testing.assert_array_equal(first.yb.position_um, second.yb.position_um)
        np.testing.assert_array_equal(first.rb.velocity_um_per_us, second.rb.velocity_um_per_us)

    def test_phase_space_rotation(self) -> None:
        sample = PhaseSpaceSample(
            position_um=np.array([[1.0, 2.0, 3.0]]),
            velocity_um_per_us=np.array([[4.0, 5.0, 6.0]]),
        )
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = rotate_phase_space(sample, rotation)
        np.testing.assert_allclose(rotated.position_um, [[-2.0, 1.0, 3.0]])
        np.testing.assert_allclose(rotated.velocity_um_per_us, [[-5.0, 4.0, 6.0]])

    def test_ballistic_relative_trajectory(self) -> None:
        samples = TwoSpeciesPhaseSpace(
            yb=PhaseSpaceSample(
                position_um=np.array([[0.1, 0.2, 0.3]]),
                velocity_um_per_us=np.array([[0.01, 0.02, 0.03]]),
            ),
            rb=PhaseSpaceSample(
                position_um=np.array([[0.01, 0.02, 0.03]]),
                velocity_um_per_us=np.array([[0.001, 0.002, 0.003]]),
            ),
        )
        trajectory = ballistic_relative_trajectory((0.0, 0.0, 3.4), samples, (0.0, 2.0))
        np.testing.assert_allclose(trajectory[0, 0], [0.09, 0.18, 3.67])
        np.testing.assert_allclose(trajectory[0, 1], [0.108, 0.216, 3.724])

    def test_doppler_detuning_units_and_sign(self) -> None:
        shift = doppler_detuning_mhz(np.array([[1.0, 0.0, 0.0]]), (2 * np.pi, 0.0, 0.0))
        np.testing.assert_allclose(shift, [-1.0])


if __name__ == "__main__":
    unittest.main()
