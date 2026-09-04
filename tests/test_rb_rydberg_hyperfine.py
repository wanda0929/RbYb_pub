#!/usr/bin/env python3
"""Self-checks for the 87Rb Rydberg hyperfine Hamiltonian."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rb_rydberg_hyperfine import (  # noqa: E402
    RB87_M_I,
    hyperfine_constants_mhz,
    isolated_hyperfine_hamiltonian_mhz,
    target_stretched_shift_mhz,
)


class RbRydbergHyperfineTest(unittest.TestCase):
    def test_56s_interval(self) -> None:
        a_mhz, b_mhz = hyperfine_constants_mhz(56, 0, 0.5)
        self.assertAlmostEqual(a_mhz, 0.1255297445, places=9)
        self.assertEqual(b_mhz, 0.0)
        eigenvalues = np.linalg.eigvalsh(isolated_hyperfine_hamiltonian_mhz(56, 0, 0.5))
        self.assertAlmostEqual(eigenvalues[-1] - eigenvalues[0], 2 * a_mhz)
        np.testing.assert_allclose(eigenvalues[:3], -1.25 * a_mhz, atol=1e-14)
        np.testing.assert_allclose(eigenvalues[3:], 0.75 * a_mhz, atol=1e-14)

    def test_56p_one_half_interval(self) -> None:
        a_mhz, b_mhz = hyperfine_constants_mhz(56, 1, 0.5)
        self.assertAlmostEqual(a_mhz, 0.0322127186, places=9)
        self.assertEqual(b_mhz, 0.0)
        eigenvalues = np.linalg.eigvalsh(isolated_hyperfine_hamiltonian_mhz(56, 1, 0.5))
        self.assertAlmostEqual(eigenvalues[-1] - eigenvalues[0], 2 * a_mhz)

    def test_p_three_half_quadrupole_spectrum(self) -> None:
        j = 1.5
        nuclear_spin = 1.5
        a_mhz, b_mhz = hyperfine_constants_mhz(56, 1, j)
        self.assertNotEqual(b_mhz, 0.0)
        expected = []
        for total_f in range(4):
            x = 0.5 * (total_f * (total_f + 1) - nuclear_spin * (nuclear_spin + 1) - j * (j + 1))
            quadrupole = (
                b_mhz
                * (3 * x**2 + 1.5 * x - nuclear_spin * (nuclear_spin + 1) * j * (j + 1))
                / (2 * nuclear_spin * (2 * nuclear_spin - 1) * j * (2 * j - 1))
            )
            expected.extend([a_mhz * x + quadrupole] * (2 * total_f + 1))
        eigenvalues = np.linalg.eigvalsh(isolated_hyperfine_hamiltonian_mhz(56, 1, j))
        np.testing.assert_allclose(eigenvalues, np.sort(expected), atol=1e-14)

    def test_stretched_state_is_an_eigenstate(self) -> None:
        field_gauss = 3.10
        hamiltonian = isolated_hyperfine_hamiltonian_mhz(56, 0, 0.5, field_gauss)
        # Ordering is m_I-major and m_J=(-1/2,+1/2).
        stretched_index = int(np.flatnonzero(RB87_M_I == 1.5)[0]) * 2 + 1
        stretched = np.zeros(8, dtype=complex)
        stretched[stretched_index] = 1
        expected = target_stretched_shift_mhz(56, 0, field_gauss)
        np.testing.assert_allclose(hamiltonian @ stretched, expected * stretched, atol=1e-14)

    def test_hamiltonian_is_hermitian(self) -> None:
        for orbital_l, j in ((0, 0.5), (1, 0.5), (1, 1.5), (2, 1.5), (2, 2.5)):
            hamiltonian = isolated_hyperfine_hamiltonian_mhz(56, orbital_l, j, 3.10)
            np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
