"""Fast static-exchange contracts; no PairInteraction models are constructed."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_forster_characterization as characterization


class StaticExchangeTests(unittest.TestCase):
    def test_detuned_two_level_first_exchange_and_resolution(self):
        coupling, defect = 15.4, -.76
        energies, vectors = np.linalg.eigh([[0., coupling], [coupling, defect]])
        pp, ss = vectors[0].conj(), vectors[1].conj()
        splitting = np.hypot(defect, 2*coupling)
        for samples in (801, 1601):
            result = characterization.analyze_spectrum(energies, pp, ss, False, samples)
            peak = result['first_exchange_maximum']
            self.assertAlmostEqual(peak['time_ns'], 1000/(2*splitting), places=6)
            self.assertAlmostEqual(peak['pp_population'], (2*coupling/splitting)**2, places=12)
            self.assertAlmostEqual(peak['spectator_population'], 0., places=12)
            self.assertAlmostEqual(sum(peak[k] for k in ('pp_population',
                'residual_ss_population', 'spectator_population')), 1., places=12)

    def test_ss_initial_population_matches_direct_unitary_not_pp_initial(self):
        h = np.array([[0., 10., 2.], [10., 1., 5.], [2., 5., 30.]])
        energies, vectors = np.linalg.eigh(h)
        pp, ss = vectors[0].conj(), vectors[1].conj()
        t = .016
        direct = expm(-2j*np.pi*h*t)
        actual = characterization.transfer_populations(t, energies, pp, ss)
        np.testing.assert_allclose(actual, np.abs(direct[:, 1])**2, atol=1e-12)
        self.assertGreater(abs(actual[2]-abs(direct[2, 0])**2), 1e-3)
        gauge = np.exp(1j*np.arange(3))
        np.testing.assert_allclose(actual, characterization.transfer_populations(
            t, energies+1234., pp*gauge, ss*gauge), atol=1e-12)

    def test_final_hfs_builder_receives_explicit_field_and_reference_basis(self):
        import evaluate_forster_p0_4 as p0
        with patch.object(characterization.gate_model, 'build_pair_model',
                side_effect=RuntimeError('sentinel')) as build:
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                characterization.final_hfs_field_point(3.1)
        self.assertEqual(build.call_args.args, (3.1, p0.REFERENCE_BASIS))
        self.assertEqual(build.call_args.kwargs, {'distance_um': 3.4, 'theta_deg': 0.})


if __name__ == '__main__':
    unittest.main()
