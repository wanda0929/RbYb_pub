"""Fast checks independent of downloaded atomic tables."""
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

import evaluate_forster_off_axis as audit


class OffAxisTests(unittest.TestCase):
    def test_spectrum_keeps_axial_dark_modes_and_q_population(self):
        h = csr_matrix(np.diag([-2., 1., 4., 10., 20., 30.]))
        ss = np.array([.6, 0, .8, 0, 0, 0])
        pp = np.array([.8, 0, -.6, 0, 0, 0])
        total_m = np.array([2.5, 1.5, 2.5, 3.5, 2.5, 2.5])
        e, cs, cp, gram, info = audit.spectrum(h, ss, pp, total_m, 3)
        np.testing.assert_allclose(e, [-2, 1, 4], atol=1e-12)
        np.testing.assert_allclose(cs**2, [.36, 0, .64], atol=1e-12)
        np.testing.assert_allclose(cp**2, [.64, 0, .36], atol=1e-12)
        np.testing.assert_allclose(gram, np.diag([0, 1, 0]), atol=1e-12)
        self.assertLess(info['omitted_ss_weight'], 1e-12)
        factor = audit.splu((h - .1234567*audit.eye(6)).tocsc())
        inverse = audit.LinearOperator(h.shape, matvec=factor.solve, dtype=h.dtype)
        cached = audit.spectrum(h, ss, pp, total_m, 3, inverse)
        np.testing.assert_allclose(cached[0], e, atol=1e-12)
        np.testing.assert_allclose(cached[1]**2, cs**2, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'between 1 and 5'):
            audit.spectrum(h, ss, pp, total_m, 6)

    def test_nuclear_lift_preserves_yb_and_crosses_electronic_m(self):
        # Deliberately permuted products: HFS can change nuclear and electronic
        # indices together, but must never change the Yb index.
        atomic = np.diag([2., 3., 5., 7.])
        atomic[1, 2] = atomic[2, 1] = 0.4
        result = audit.lift_hyperfine(csr_matrix(atomic),
                                     np.array([0, 1, 0, 1]),
                                     np.array([8, 8, 9, 9]),
                                     np.array([1, 0, 1, 0]), 2).toarray()
        np.testing.assert_allclose(result, [[5, .4, 0, 0], [.4, 3, 0, 0],
                                           [0, 0, 5, .4], [0, 0, .4, 3]])

    def test_propagation_matches_independent_dense_sequence(self):
        life = audit.base.Lifetimes(410., 190., 340., 80.)
        pulse = (audit.shaped.Segment(7., -1.2, .023),
                 audit.shaped.Segment(3., 2.1, .037))
        energies = np.array([-14., 19.])
        ss = np.array([.6, .7])
        pp = np.array([-.5, .2])
        gram = np.array([[.3, .1], [.1, .2]])
        scale, correction = .99, (.3, -.4)
        result = audit.propagate(energies, ss, pp, gram, pulse, scale, correction, life)
        u, b = np.array([1, 0], complex), np.array([1, 0, 0], complex)
        # Independent bare-weight sums: spectator remainder decays at SS rate.
        rates = np.array([.25*(1/410+1/340)+.75*(1/190+1/80),
                          .04*(1/410+1/340)+.96*(1/190+1/80)])
        for segment in pulse:
            om, det, t = scale*segment.omega_mhz, segment.detuning_mhz, segment.duration_us
            hu = np.array([[0, om/2], [om/2, -det-1j/(4*np.pi*80)]])
            hb = np.diag(np.r_[-1j/(4*np.pi*190), energies-det-1j*rates/(4*np.pi)])
            hb[0, 1:] = hb[1:, 0] = om*ss/2
            u = expm(-2j*np.pi*t*hu) @ u
            b = expm(-2j*np.pi*t*hb) @ b
        hc = np.array([[0, 5*scale/2], [5*scale/2, -1j/(4*np.pi*190)]])
        control = expm(-2j*np.pi*.1*hc)
        k = np.array([1, u[0], (control @ np.diag([1, np.exp(-.06/380)]) @ control)[0, 0],
                      (control @ np.diag([u[0], b[0]]) @ control)[0, 0]])
        actual = np.array([complex(*z) for z in result['kraus_re_im']])
        np.testing.assert_allclose(actual, k, atol=1e-12)
        cz = np.exp(1j*np.array([0, correction[1], correction[0], sum(correction)]))
        expected = (np.vdot(k, k).real + abs(np.sum(k*cz*np.array([1, 1, 1, -1])))**2)/20
        self.assertAlmostEqual(result['fidelity'], expected, places=12)
        self.assertAlmostEqual(result['final_outside_m_population'], np.vdot(b[1:], gram @ b[1:]).real, places=12)


if __name__ == '__main__':
    unittest.main()
