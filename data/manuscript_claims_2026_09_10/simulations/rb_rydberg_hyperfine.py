#!/usr/bin/env python3
"""Hyperfine--Zeeman Hamiltonian for the 87Rb Rydberg basis.

PairInteraction 2.5 supplies the electronic Rb Hamiltonian but does not carry
the 87Rb nuclear spin.  This module adds I=3/2 explicitly in the uncoupled
|n,l,j,m_J;m_I> basis and transforms the result to PairInteraction's
field-dressed electronic eigenbasis.  Frequencies are cyclic frequencies in
MHz, consistent with the simulation modules in this directory.

The directly measured nS_1/2 scaling is from Tauschinsky et al., PRA 87,
042522 (2013).  The nP_1/2 normalization is the Cardman--Raithel 85Rb
measurement, PRA 106, 052810 (2022), scaled by the isotope nuclear
g-factor ratio.  The much smaller P_3/2 and D_J constants are n*^-3 fits to
the low-n 87Rb data compiled by Arimondo et al., RMP 49, 31 (1977).  No F_J
normalization is available in that compilation, so F-state hyperfine terms
are set to zero rather than assigned an unsupported value.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron


RB87_NUCLEAR_SPIN = 1.5
RB87_M_I = np.array([-1.5, -0.5, 0.5, 1.5])
RB87_STRETCHED_M_I = 1.5

# Atomic-Hamiltonian convention H_Z/h = g_I (mu_B/h) B I_z.  The minus sign
# converts the positive tabulated nuclear magnetic moment to the electronic
# angular-momentum convention used in alkali-atom Hamiltonians.
RB87_G_I_BOHR = -0.0009951414
MU_B_OVER_H_MHZ_PER_G = 1.39962449361

# Modified Rydberg--Ritz coefficients delta(n)=delta_0+delta_2/(n-delta_0)^2
# are sufficient at n~56 at substantially better precision than the HFS
# normalizations.  Values are those used for the 87Rb series in ARC and are
# consistent with the Rb spectroscopy sources used by PairInteraction.
_QUANTUM_DEFECT = {
    (0, 0.5): (3.1311804, 0.1784),
    (1, 0.5): (2.6548849, 0.2900),
    (1, 1.5): (2.6416737, 0.2950),
    (2, 1.5): (1.348091, -0.60286),
    (2, 2.5): (1.3464657, -0.59600),
    (3, 2.5): (0.0165192, -0.085),
    (3, 3.5): (0.0165437, -0.086),
}

# A(n)/h = A_prefactor/(n*)^3 and B(n)/h = B_prefactor/(n*)^3.
# Prefactors are in MHz.  The P_3/2 and D values are fits (rounded more
# coarsely than the direct S and isotope-scaled P_1/2 determinations).
_HFS_PREFACTORS_MHZ = {
    (0, 0.5): (18_550.0, 0.0),
    (1, 0.5): (4_890.0, 0.0),
    (1, 1.5): (1_033.0, 143.0),
    (2, 1.5): (828.0, 50.0),
    (2, 2.5): (-357.0, 0.0),
}


def effective_principal_quantum_number(n: int, l: int, j: float) -> float:
    """Return n* for the Rb series used by the hyperfine scaling."""

    delta_0, delta_2 = _QUANTUM_DEFECT[(int(l), float(j))]
    delta = delta_0 + delta_2 / (n - delta_0) ** 2
    return float(n - delta)


def hyperfine_constants_mhz(
    n: int,
    l: int,
    j: float,
    f_state_prefactors_mhz: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Return magnetic-dipole A/h and electric-quadrupole B/h in MHz."""

    prefactors = (
        f_state_prefactors_mhz
        if int(l) == 3
        else _HFS_PREFACTORS_MHZ.get((int(l), float(j)))
    )
    if prefactors is None:
        return 0.0, 0.0
    n_eff = effective_principal_quantum_number(n, l, j)
    return prefactors[0] / n_eff**3, prefactors[1] / n_eff**3


def _angular_momentum_matrices(
    angular_momentum: float, magnetic_numbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return J_z, J_+, J_- in the supplied magnetic-number ordering."""

    magnetic_numbers = np.asarray(magnetic_numbers, dtype=float)
    dimension = len(magnetic_numbers)
    z = np.diag(magnetic_numbers)
    plus = np.zeros((dimension, dimension), dtype=complex)
    index_by_m = {round(float(m), 8): index for index, m in enumerate(magnetic_numbers)}
    for column, m in enumerate(magnetic_numbers):
        row = index_by_m.get(round(float(m + 1), 8))
        if row is not None:
            plus[row, column] = np.sqrt(
                angular_momentum * (angular_momentum + 1) - m * (m + 1)
            )
    return z, plus, plus.conj().T


def isolated_hyperfine_hamiltonian_mhz(
    n: int,
    l: int,
    j: float,
    field_gauss: float = 0.0,
    f_state_prefactors_mhz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return the isolated-manifold H_hfs+H_nZ matrix in |m_I,m_J>."""

    m_j = np.arange(-j, j + 1, 1.0)
    i_z, i_plus, i_minus = _angular_momentum_matrices(RB87_NUCLEAR_SPIN, RB87_M_I)
    j_z, j_plus, j_minus = _angular_momentum_matrices(j, m_j)
    x = (
        np.kron(i_z, j_z)
        + 0.5 * np.kron(i_plus, j_minus)
        + 0.5 * np.kron(i_minus, j_plus)
    )
    a_mhz, b_mhz = hyperfine_constants_mhz(
        n, l, j, f_state_prefactors_mhz=f_state_prefactors_mhz
    )
    hamiltonian = a_mhz * x
    if b_mhz and j >= 1:
        identity = np.eye(x.shape[0])
        numerator = (
            3 * (x @ x)
            + 1.5 * x
            - RB87_NUCLEAR_SPIN
            * (RB87_NUCLEAR_SPIN + 1)
            * j
            * (j + 1)
            * identity
        )
        denominator = (
            2
            * RB87_NUCLEAR_SPIN
            * (2 * RB87_NUCLEAR_SPIN - 1)
            * j
            * (2 * j - 1)
        )
        hamiltonian = hamiltonian + b_mhz * numerator / denominator
    nuclear_zeeman = (
        RB87_G_I_BOHR
        * MU_B_OVER_H_MHZ_PER_G
        * field_gauss
        * np.kron(i_z, np.eye(len(m_j)))
    )
    return np.asarray(hamiltonian + nuclear_zeeman, dtype=complex)


def target_stretched_shift_mhz(n: int, l: int, field_gauss: float) -> float:
    """HFS+nuclear-Zeeman shift of |j=1/2,m_J=1/2;m_I=3/2>."""

    a_mhz, _ = hyperfine_constants_mhz(n, l, 0.5)
    return float(
        a_mhz * RB87_STRETCHED_M_I * 0.5
        + RB87_G_I_BOHR
        * MU_B_OVER_H_MHZ_PER_G
        * field_gauss
        * RB87_STRETCHED_M_I
    )


def dressed_hyperfine_hamiltonian_mhz(
    rb_basis,
    rb_system,
    field_gauss: float,
    f_state_prefactors_mhz: tuple[float, float] = (0.0, 0.0),
) -> csr_matrix:
    """Build H_hfs+H_nZ in |m_I> times the dressed electronic Rb basis."""

    number_electronic = rb_basis.number_of_states
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    blocks: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, ket in enumerate(rb_basis.kets):
        blocks[(int(ket.n), int(ket.l), int(round(2 * ket.j)))].append(index)

    for (n, l, twice_j), electronic_indices in blocks.items():
        j = twice_j / 2
        electronic_indices.sort(key=lambda index: rb_basis.kets[index].m)
        first_ket = rb_basis.kets[electronic_indices[0]]
        local = isolated_hyperfine_hamiltonian_mhz(
            n,
            l,
            j,
            field_gauss=0.0,
            f_state_prefactors_mhz=f_state_prefactors_mhz,
        )
        local_electronic_size = len(electronic_indices)
        if local_electronic_size != int(2 * j + 1):
            raise ValueError(f"Incomplete magnetic manifold for {first_ket.get_label()}")
        local_rows, local_columns = np.nonzero(np.abs(local) > 0)
        for local_row, local_column in zip(local_rows, local_columns):
            nuclear_row, electronic_row = divmod(local_row, local_electronic_size)
            nuclear_column, electronic_column = divmod(
                local_column, local_electronic_size
            )
            rows.append(nuclear_row * number_electronic + electronic_indices[electronic_row])
            columns.append(
                nuclear_column * number_electronic
                + electronic_indices[electronic_column]
            )
            values.append(local[local_row, local_column])

    nuclear_diagonal = np.repeat(
        RB87_G_I_BOHR
        * MU_B_OVER_H_MHZ_PER_G
        * field_gauss
        * RB87_M_I,
        number_electronic,
    )
    diagonal_indices = np.arange(4 * number_electronic)
    rows.extend(diagonal_indices.tolist())
    columns.extend(diagonal_indices.tolist())
    values.extend(nuclear_diagonal.astype(complex).tolist())

    bare_hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(4 * number_electronic, 4 * number_electronic),
        dtype=complex,
    ).tocsr()
    coefficients = rb_system.get_eigenbasis().get_coefficients().tocsr().astype(complex)
    transformation = kron(eye(4, format="csr"), coefficients, format="csr")
    dressed = transformation.conj().T @ bare_hamiltonian @ transformation
    return dressed.tocsr()
