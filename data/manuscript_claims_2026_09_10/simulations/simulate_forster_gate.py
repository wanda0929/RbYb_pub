#!/usr/bin/env python3
"""Reproduce the driven Rb--Yb Förster-gate simulation.

The pair Hamiltonian includes every magnetic sublevel in a projected finite
PairInteraction basis.  Square preparation, target, and depreparation pulses
are then propagated explicitly.  Rydberg depopulation is included with a
non-Hermitian no-jump Hamiltonian; the destination of lost norm is not modeled.

Outputs:
  simulations/channel_forster_gate_results.json
  figures/channel_forster_gate_simulation.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pairinteraction as pi
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SIMULATION_DIR = ROOT / "simulations"
RESULT_PATH = ROOT / "simulations" / "channel_forster_gate_results.json"
FIGURE_STEM = FIGURE_DIR / "channel_forster_gate_simulation"

sys.path.insert(0, str(FIGURE_DIR))
sys.path.insert(0, str(SIMULATION_DIR))
from style import OKABE_ITO, use_paper_style  # noqa: E402
from rb_rydberg_hyperfine import (  # noqa: E402
    RB87_M_I,
    RB87_STRETCHED_M_I,
    dressed_hyperfine_hamiltonian_mhz,
    target_stretched_shift_mhz,
)

COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["purple"],
]


@dataclass(frozen=True)
class BasisConfig:
    delta_n: int
    atomic_window_ghz: float
    pair_window_ghz: float
    delta_l: int = 2
    interaction_order: int = 3


@dataclass
class PairModel:
    field_gauss: float
    electric_field_v_cm: float
    distance_um: float
    theta_deg: float
    defect_offset_mhz: float
    basis_config: BasisConfig
    energies_mhz: np.ndarray
    pp_overlap: np.ndarray
    ss_overlap: np.ndarray
    pair_basis_size: int
    symmetry_component_size: int
    pp_component_norm: float
    ss_component_norm: float
    hermiticity_error_mhz: float
    pp_asymptote_mhz: float
    ss_asymptote_mhz: float
    forster_defect_mhz: float
    rb_ss_carrier_shift_mhz: float
    yb_ss_carrier_shift_mhz: float
    rb_hyperfine_included: bool
    rb_f_hyperfine_prefactors_mhz: tuple[float, float]
    angular_delta_m_couplings_omitted: bool
    static_transfer: dict[str, float]
    spectral_diagnostics: dict[str, object]

    def mode_mask(self, threshold: float) -> np.ndarray:
        return np.abs(self.pp_overlap) ** 2 > threshold


@dataclass(frozen=True)
class Lifetimes:
    rb_56p_us: float
    rb_56s_us: float
    yb_52p_us: float
    yb_53s_us: float


@dataclass(frozen=True)
class GateParameters:
    distance_um: float = 3.4
    theta_deg: float = 0.0
    omega_yb_optical_mhz: float = 15.0
    omega_yb_microwave_mhz: float = 15.0
    raman_delta_mhz: float = 57.15005715008573
    raman_pi_time_us: float = 0.262466929133727


PARAMS = GateParameters()
PRIMARY_BASIS = BasisConfig(delta_n=4, atomic_window_ghz=160.0, pair_window_ghz=20.0)
SCAN_BASIS = BasisConfig(delta_n=3, atomic_window_ghz=80.0, pair_window_ghz=20.0)
PAIR_WINDOW_CHECK_BASIS = BasisConfig(
    delta_n=3, atomic_window_ghz=80.0, pair_window_ghz=10.0
)
TARGET_TOTAL_M = 2.5
ELECTRONIC_M_RANGE = (1, 4)


def _kets() -> tuple[pi.KetAtom, pi.KetAtom, pi.KetAtom, pi.KetAtom]:
    rb_pp = pi.KetAtom("Rb", n=56, l=1, j=0.5, m=0.5)
    rb_ss = pi.KetAtom("Rb", n=56, l=0, j=0.5, m=0.5)
    yb_pp = pi.KetAtom("Yb171_mqdt", n=52, l=1, s=0, f=0.5, m=0.5)
    yb_ss = pi.KetAtom("Yb171_mqdt", n=53, l=0, s=1, f=0.5, m=0.5)
    return rb_pp, rb_ss, yb_pp, yb_ss


def _dense_overlap(basis: pi.BasisPair, pair: tuple[pi.KetAtom, pi.KetAtom]) -> np.ndarray:
    overlap = basis.get_amplitudes(pair)
    if hasattr(overlap, "toarray"):
        overlap = overlap.toarray()
    return np.asarray(overlap, dtype=complex).reshape(-1)


def _dressed_index_by_bare_index(system: pi.SystemAtom) -> dict[int, int]:
    coefficients = system.get_eigenbasis().get_coefficients().toarray()
    dominant_bare_indices = np.argmax(np.abs(coefficients), axis=0)
    if len(set(dominant_bare_indices.tolist())) != len(dominant_bare_indices):
        raise ValueError("Atomic dressed states do not have unique dominant bare kets")
    return {
        int(bare_index): int(dressed_index)
        for dressed_index, bare_index in enumerate(dominant_bare_indices)
    }


def _pair_state_indices(
    pair_basis: pi.BasisPair,
    rb_system: pi.SystemAtom,
    yb_system: pi.SystemAtom,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify dressed-atom and magnetic indices of every pair-basis state."""

    rb_index_by_bare = _dressed_index_by_bare_index(rb_system)
    yb_index_by_bare = _dressed_index_by_bare_index(yb_system)
    rb_indices = []
    yb_indices = []
    electronic_m = []
    nuclear_m = []
    allowed_m_i = {round(float(value), 8) for value in RB87_M_I}
    for ket in pair_basis.kets:
        rb_state, yb_state = ket.state_atoms
        rb_indices.append(
            rb_index_by_bare[rb_state.get_corresponding_ket_index()]
        )
        yb_indices.append(
            yb_index_by_bare[yb_state.get_corresponding_ket_index()]
        )
        m_value = float(
            rb_state.get_corresponding_ket().m
            + yb_state.get_corresponding_ket().m
        )
        m_i_value = TARGET_TOTAL_M - m_value
        if round(m_i_value, 8) not in allowed_m_i:
            raise ValueError(f"Electronic M={m_value} is outside the target M_tot block")
        electronic_m.append(m_value)
        nuclear_m.append(m_i_value)
    return (
        np.asarray(rb_indices, dtype=int),
        np.asarray(yb_indices, dtype=int),
        np.asarray(electronic_m),
        np.asarray(nuclear_m),
    )


def _lift_rb_hyperfine_to_pair_basis(
    pair_basis: pi.BasisPair,
    rb_basis: pi.BasisAtom,
    rb_system: pi.SystemAtom,
    yb_system: pi.SystemAtom,
    field_gauss: float,
    rb_indices: np.ndarray,
    yb_indices: np.ndarray,
    nuclear_m: np.ndarray,
    f_state_prefactors_mhz: tuple[float, float],
):
    """Lift the single-Rb HFS Hamiltonian while leaving Yb unchanged."""

    atomic_hyperfine = dressed_hyperfine_hamiltonian_mhz(
        rb_basis,
        rb_system,
        field_gauss,
        f_state_prefactors_mhz=f_state_prefactors_mhz,
    )
    m_i_index = {
        round(float(value), 8): index for index, value in enumerate(RB87_M_I)
    }
    atomic_indices = np.asarray(
        [
            m_i_index[round(float(m_i), 8)] * rb_basis.number_of_states + rb_index
            for m_i, rb_index in zip(nuclear_m, rb_indices)
        ],
        dtype=int,
    )
    rows = []
    columns = []
    values = []
    for yb_index in np.unique(yb_indices):
        pair_indices = np.flatnonzero(yb_indices == yb_index)
        submatrix = atomic_hyperfine[atomic_indices[pair_indices]][
            :, atomic_indices[pair_indices]
        ].tocoo()
        rows.extend(pair_indices[submatrix.row].tolist())
        columns.extend(pair_indices[submatrix.col].tolist())
        values.extend(submatrix.data.tolist())
    return coo_matrix(
        (values, (rows, columns)),
        shape=(pair_basis.number_of_states, pair_basis.number_of_states),
        dtype=complex,
    ).tocsr()


def _pair_state_label(pair_basis: pi.BasisPair, index: int, m_i: float) -> str:
    rb_state, yb_state = pair_basis.kets[index].state_atoms
    return (
        f"{rb_state.get_corresponding_ket().get_label()},mI={m_i:+.1f} + "
        f"{yb_state.get_corresponding_ket().get_label()}"
    )


def build_pair_model(
    field_gauss: float,
    config: BasisConfig,
    distance_um: float | None = None,
    theta_deg: float | None = None,
    electric_field_v_cm: float = 0.0,
    defect_offset_mhz: float = 0.0,
    include_rb_hyperfine: bool = True,
    rb_f_hyperfine_prefactors_mhz: tuple[float, float] = (0.0, 0.0),
) -> PairModel:
    """Construct the electronic-magnetic and 87Rb-nuclear pair Hamiltonian.

    ``defect_offset_mhz`` changes E_SS-E_PP while holding the optically
    addressed SS asymptote fixed.  It is an effective target-pair sensitivity
    parameter, not a refit of the underlying single-atom structure.
    """

    if distance_um is None:
        distance_um = PARAMS.distance_um
    if theta_deg is None:
        theta_deg = PARAMS.theta_deg
    rb_pp, rb_ss, yb_pp, yb_ss = _kets()
    rb_basis = pi.BasisAtom.from_kets(
        [rb_pp, rb_ss],
        delta_n=config.delta_n,
        delta_l=config.delta_l,
        delta_j=2,
        delta_m=None,
        delta_energy=config.atomic_window_ghz,
        delta_energy_unit="GHz",
    )
    yb_basis = pi.BasisAtom.from_kets(
        [yb_pp, yb_ss],
        delta_n=config.delta_n,
        delta_l=config.delta_l,
        delta_f=3,
        delta_m=None,
        delta_energy=config.atomic_window_ghz,
        delta_energy_unit="GHz",
    )

    rb_system = pi.SystemAtom(rb_basis).set_magnetic_field(
        [0, 0, field_gauss], unit="G"
    )
    yb_system = pi.SystemAtom(yb_basis).set_magnetic_field(
        [0, 0, field_gauss], unit="G"
    )
    rb_system.diagonalize()
    yb_system.diagonalize()
    rb_ss_reference_mhz = float(
        rb_system.get_corresponding_energy(rb_ss, unit="MHz")
    )
    yb_ss_reference_mhz = float(
        yb_system.get_corresponding_energy(yb_ss, unit="MHz")
    )
    if electric_field_v_cm != 0.0:
        rb_system.set_electric_field([0, 0, electric_field_v_cm], unit="V/cm")
        yb_system.set_electric_field([0, 0, electric_field_v_cm], unit="V/cm")
        rb_system.diagonalize()
        yb_system.diagonalize()
    rb_ss_carrier_shift_mhz = float(
        rb_system.get_corresponding_energy(rb_ss, unit="MHz")
        - rb_ss_reference_mhz
    )
    yb_ss_carrier_shift_mhz = float(
        yb_system.get_corresponding_energy(yb_ss, unit="MHz")
        - yb_ss_reference_mhz
    )
    energy_zero_mhz = rb_system.get_corresponding_energy(
        rb_pp, unit="MHz"
    ) + yb_system.get_corresponding_energy(yb_pp, unit="MHz")
    pair_window_mhz = 1000 * config.pair_window_ghz
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        m=ELECTRONIC_M_RANGE,
        energy=(energy_zero_mhz - pair_window_mhz, energy_zero_mhz + pair_window_mhz),
        energy_unit="MHz",
    )
    pair_system = pi.SystemPair(pair_basis)
    pair_system.set_distance(
        distance_um, angle_degree=theta_deg, unit="micrometer"
    )
    pair_system.set_interaction_order(config.interaction_order)
    hamiltonian = pair_system.get_hamiltonian(unit="MHz").tocsr().astype(complex)
    hamiltonian -= energy_zero_mhz * identity(
        pair_basis.number_of_states, format="csr", dtype=complex
    )
    rb_indices, yb_indices, electronic_m, nuclear_m = _pair_state_indices(
        pair_basis, rb_system, yb_system
    )

    # The M_tot=5/2 block is exact for theta=0.  For the sub-degree transverse
    # geometries in the bounded-position scan we retain the angle-dependent
    # Delta-M=0 tensor term but omit the much smaller Delta-M=+/-1,+/-2 terms;
    # that approximation is explicitly reported with the model.
    matrix = hamiltonian.tocoo()
    same_electronic_m = np.isclose(
        electronic_m[matrix.row], electronic_m[matrix.col], atol=1e-10
    )
    hamiltonian = coo_matrix(
        (
            matrix.data[same_electronic_m],
            (matrix.row[same_electronic_m], matrix.col[same_electronic_m]),
        ),
        shape=matrix.shape,
        dtype=complex,
    ).tocsr()
    if include_rb_hyperfine:
        hamiltonian += _lift_rb_hyperfine_to_pair_basis(
            pair_basis,
            rb_basis,
            rb_system,
            yb_system,
            field_gauss,
            rb_indices,
            yb_indices,
            nuclear_m,
            rb_f_hyperfine_prefactors_mhz,
        )

    stretched_nuclear_mask = np.isclose(nuclear_m, RB87_STRETCHED_M_I)
    pp_full = _dense_overlap(pair_basis, (rb_pp, yb_pp)) * stretched_nuclear_mask
    ss_full = _dense_overlap(pair_basis, (rb_ss, yb_ss)) * stretched_nuclear_mask

    # Keep the connected block containing the target.  At theta=0 this is the
    # exact M_tot=5/2 sector containing all four 87Rb nuclear projections.
    adjacency = abs(hamiltonian)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data = np.ones(adjacency.nnz, dtype=adjacency.dtype)
    _, labels = connected_components(adjacency, directed=False)
    target_labels = np.unique(labels[np.flatnonzero(np.abs(pp_full) > 1e-12)])
    component = np.flatnonzero(np.isin(labels, target_labels))

    h_component = hamiltonian[component][:, component].toarray()
    scale = max(float(np.max(np.abs(h_component))), 1.0)
    hermiticity_error = float(np.max(np.abs(h_component - h_component.conj().T)))
    assert hermiticity_error / scale < 1e-11, "Pair Hamiltonian is not Hermitian"

    pp_component = pp_full[component]
    ss_component = ss_full[component]
    pp_norm = float(np.vdot(pp_component, pp_component).real)
    ss_norm = float(np.vdot(ss_component, ss_component).real)
    assert pp_norm > 0.999, f"PP target representation is incomplete: {pp_norm}"
    assert ss_norm > 0.99, f"SS target representation is incomplete: {ss_norm}"
    pp_component /= np.sqrt(pp_norm)
    ss_component /= np.sqrt(ss_norm)

    # Holding E_SS and the target-laser reference fixed, lowering E_PP by
    # defect_offset_mhz increases the target-pair defect E_SS-E_PP by the same
    # amount.  This projector scan isolates sensitivity to that pair defect.
    h_component -= defect_offset_mhz * np.outer(
        pp_component, pp_component.conj()
    )

    energies, eigenvectors = np.linalg.eigh(h_component)
    pp_overlap = eigenvectors.conj().T @ pp_component
    ss_overlap = eigenvectors.conj().T @ ss_component

    target_weight = np.abs(pp_overlap) ** 2 + np.abs(ss_overlap) ** 2
    target_modes = np.argsort(target_weight)[-2:]

    def eigenstate_summary(mode_index: int) -> dict[str, object]:
        pair_weights = np.abs(eigenvectors[:, mode_index]) ** 2
        top_components = np.argsort(pair_weights)[-6:][::-1]
        return {
            "mode_index": int(mode_index),
            "energy_mhz": float(energies[mode_index]),
            "ss_weight": float(abs(ss_overlap[mode_index]) ** 2),
            "pp_weight": float(abs(pp_overlap[mode_index]) ** 2),
            "residual_weight": float(max(0.0, 1 - target_weight[mode_index])),
            "top_pair_components": [
                {
                    "state": _pair_state_label(
                        pair_basis,
                        int(component[pair_component]),
                        float(nuclear_m[component[pair_component]]),
                    ),
                    "weight": float(pair_weights[pair_component]),
                }
                for pair_component in top_components
            ],
        }

    spectator_modes = np.setdiff1d(
        np.arange(len(energies)), target_modes, assume_unique=True
    )
    distance_to_target = np.min(
        np.abs(energies[spectator_modes, None] - energies[target_modes]), axis=1
    )
    nearest_spectator = int(spectator_modes[np.argmin(distance_to_target)])
    strongest_spectators = spectator_modes[
        np.argsort(target_weight[spectator_modes])[-5:][::-1]
    ]
    spectral_diagnostics = {
        "target_eigenstates": [
            eigenstate_summary(int(index))
            for index in target_modes[np.argsort(energies[target_modes])]
        ],
        "nearest_spectator_gap_mhz": float(np.min(distance_to_target)),
        "nearest_spectator_eigenstate": eigenstate_summary(nearest_spectator),
        "largest_target_overlap_spectators": [
            eigenstate_summary(int(index)) for index in strongest_spectators
        ],
    }

    times = np.linspace(0, 0.080, 4001)
    pair_phase = np.exp(-2j * np.pi * np.outer(times, energies))
    ss_amplitude = pair_phase @ (np.conj(ss_overlap) * pp_overlap)
    pp_amplitude = pair_phase @ (np.abs(pp_overlap) ** 2)
    ss_population = np.abs(ss_amplitude) ** 2
    maximum_index = int(np.argmax(ss_population))
    static_transfer = {
        "maximum_ss_population": float(ss_population[maximum_index]),
        "time_us": float(times[maximum_index]),
        "pp_population_at_maximum": float(np.abs(pp_amplitude[maximum_index]) ** 2),
        "spectator_population_at_maximum": float(
            max(
                0.0,
                1
                - ss_population[maximum_index]
                - np.abs(pp_amplitude[maximum_index]) ** 2,
            )
        ),
    }

    pp_asymptote_mhz = float(
        target_stretched_shift_mhz(56, 1, field_gauss)
        if include_rb_hyperfine
        else 0.0
    ) - defect_offset_mhz
    ss_asymptote_mhz = float(
        rb_system.get_corresponding_energy(rb_ss, unit="MHz")
        + yb_system.get_corresponding_energy(yb_ss, unit="MHz")
        - energy_zero_mhz
        + (
            target_stretched_shift_mhz(56, 0, field_gauss)
            if include_rb_hyperfine
            else 0.0
        )
    )
    forster_defect_mhz = ss_asymptote_mhz - pp_asymptote_mhz

    return PairModel(
        field_gauss=field_gauss,
        electric_field_v_cm=electric_field_v_cm,
        distance_um=distance_um,
        theta_deg=theta_deg,
        defect_offset_mhz=defect_offset_mhz,
        basis_config=config,
        energies_mhz=energies,
        pp_overlap=pp_overlap,
        ss_overlap=ss_overlap,
        pair_basis_size=hamiltonian.shape[0],
        symmetry_component_size=len(component),
        pp_component_norm=pp_norm,
        ss_component_norm=ss_norm,
        hermiticity_error_mhz=hermiticity_error,
        pp_asymptote_mhz=pp_asymptote_mhz,
        ss_asymptote_mhz=ss_asymptote_mhz,
        forster_defect_mhz=forster_defect_mhz,
        rb_ss_carrier_shift_mhz=rb_ss_carrier_shift_mhz,
        yb_ss_carrier_shift_mhz=yb_ss_carrier_shift_mhz,
        rb_hyperfine_included=include_rb_hyperfine,
        rb_f_hyperfine_prefactors_mhz=rb_f_hyperfine_prefactors_mhz,
        angular_delta_m_couplings_omitted=abs(theta_deg) > 1e-12,
        static_transfer=static_transfer,
        spectral_diagnostics=spectral_diagnostics,
    )


def query_lifetimes(temperature_k: float) -> Lifetimes:
    rb_pp, rb_ss, yb_pp, yb_ss = _kets()
    return Lifetimes(
        rb_56p_us=float(
            rb_pp.get_lifetime(temperature=temperature_k, temperature_unit="K", unit="us")
        ),
        rb_56s_us=float(
            rb_ss.get_lifetime(temperature=temperature_k, temperature_unit="K", unit="us")
        ),
        yb_52p_us=float(
            yb_pp.get_lifetime(temperature=temperature_k, temperature_unit="K", unit="us")
        ),
        yb_53s_us=float(
            yb_ss.get_lifetime(temperature=temperature_k, temperature_unit="K", unit="us")
        ),
    )


def raman_propagator(lifetimes: Lifetimes | None) -> np.ndarray:
    # Basis: |g>=6s6p 3P0, |a>=53S triplet relay, |r>=52P singlet target.
    # PairInteraction reports cyclic frequencies, so all Hamiltonians below
    # are converted to angular units by 2*pi before exponentiation.
    h = np.array(
        [
            [0, PARAMS.omega_yb_optical_mhz / 2, 0],
            [PARAMS.omega_yb_optical_mhz / 2, PARAMS.raman_delta_mhz, PARAMS.omega_yb_microwave_mhz / 2],
            [0, PARAMS.omega_yb_microwave_mhz / 2, 0],
        ],
        dtype=complex,
    )
    if lifetimes is not None:
        h[1, 1] -= 0.5j / (2 * np.pi * lifetimes.yb_53s_us)
        h[2, 2] -= 0.5j / (2 * np.pi * lifetimes.yb_52p_us)
    return expm(-2j * np.pi * h * PARAMS.raman_pi_time_us)


def unblocked_rb_return(omega_mhz: float, lifetime_us: float | None) -> complex:
    h = np.array([[0, omega_mhz / 2], [omega_mhz / 2, 0]], dtype=complex)
    if lifetime_us is not None:
        h[1, 1] -= 0.5j / (2 * np.pi * lifetime_us)
    return complex(expm(-2j * np.pi * h / omega_mhz)[0, 0])


def blockaded_return(
    model: PairModel,
    omega_mhz: float,
    lifetimes: Lifetimes | None,
    mode_threshold: float,
    return_generator: bool = False,
) -> complex | tuple[complex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = model.mode_mask(mode_threshold)
    energies = model.energies_mhz[mask]
    pp = model.pp_overlap[mask]
    ss = model.ss_overlap[mask]
    n_modes = len(energies)
    assert n_modes > 0

    h = np.zeros((n_modes + 1, n_modes + 1), dtype=complex)
    h[0, 1:] = omega_mhz * np.conj(pp) / 2
    h[1:, 0] = omega_mhz * pp / 2
    h[1:, 1:] = np.diag(energies)
    if lifetimes is not None:
        pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
        ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
        pp_weight = np.abs(pp) ** 2
        ss_weight = np.abs(ss) ** 2
        spectator_weight = np.maximum(0.0, 1 - pp_weight - ss_weight)
        mode_rates = (
            pp_weight * pp_rate
            + ss_weight * ss_rate
            + spectator_weight * max(pp_rate, ss_rate)
        )
        h[1:, 1:] -= 0.5j * np.diag(mode_rates) / (2 * np.pi)

    generator = -2j * np.pi * h
    duration = 1 / omega_mhz
    amplitude = complex(expm(generator * duration)[0, 0])
    if return_generator:
        return amplitude, generator, pp, ss, energies
    return amplitude


def full_mode_blockaded_return_sparse(
    model: PairModel, omega_mhz: float, lifetimes: Lifetimes
) -> complex:
    """Propagate every eigenmode once as an independent projection check."""

    pp = model.pp_overlap
    ss = model.ss_overlap
    pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
    ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
    pp_weight = np.abs(pp) ** 2
    ss_weight = np.abs(ss) ** 2
    spectator_weight = np.maximum(0.0, 1 - pp_weight - ss_weight)
    mode_rates = (
        pp_weight * pp_rate
        + ss_weight * ss_rate
        + spectator_weight * max(pp_rate, ss_rate)
    )

    n_modes = len(model.energies_mhz)
    h = diags(np.r_[0, model.energies_mhz], dtype=complex, format="lil")
    h[0, 1:] = omega_mhz * np.conj(pp) / 2
    h[1:, 0] = omega_mhz * pp[:, None] / 2
    generator = (-2j * np.pi * h).tocsr()
    generator.setdiag(generator.diagonal() - 0.5 * np.r_[0, mode_rates])
    initial = np.zeros(n_modes + 1, dtype=complex)
    initial[0] = 1
    final = expm_multiply(generator / omega_mhz, initial)
    return complex(final[0])


def _local_z_metrics(kraus_diagonal: np.ndarray) -> dict[str, float | list[float]]:
    ideal = np.array([1, 1, 1, -1], dtype=complex)

    def negative_overlap(alpha: float) -> float:
        first = kraus_diagonal[0] + np.exp(1j * alpha) * kraus_diagonal[2]
        second = kraus_diagonal[1] - np.exp(1j * alpha) * kraus_diagonal[3]
        return -(abs(first) + abs(second))

    alpha_grid = np.linspace(-np.pi, np.pi, 2049)
    alpha0 = float(alpha_grid[np.argmin([negative_overlap(a) for a in alpha_grid])])
    step = float(alpha_grid[1] - alpha_grid[0])
    alpha = float(
        minimize_scalar(
            negative_overlap,
            bounds=(alpha0 - 2 * step, alpha0 + 2 * step),
            method="bounded",
            options={"xatol": 1e-13},
        ).x
    )
    first = kraus_diagonal[0] + np.exp(1j * alpha) * kraus_diagonal[2]
    second = kraus_diagonal[1] - np.exp(1j * alpha) * kraus_diagonal[3]
    beta = float(np.angle(first) - np.angle(second))
    correction = np.array(
        [1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))]
    )
    overlap = complex(np.vdot(ideal, correction * kraus_diagonal))
    survival = float(np.vdot(kraus_diagonal, kraus_diagonal).real / 4)
    average_fidelity = float(
        (4 * survival + abs(overlap) ** 2) / (4 * (4 + 1))
    )
    conditional_phase = float(
        np.angle(
            kraus_diagonal[3]
            * kraus_diagonal[0]
            / (kraus_diagonal[2] * kraus_diagonal[1])
        )
    )
    phase_error = float(np.angle(np.exp(1j * (conditional_phase - np.pi))))
    return {
        "average_gate_fidelity": average_fidelity,
        "entanglement_fidelity": float(abs(overlap) ** 2 / 16),
        "mean_computational_survival": survival,
        "conditional_phase_rad": conditional_phase,
        "conditional_phase_error_rad": phase_error,
        "optimal_local_z_alpha_rad": alpha,
        "optimal_local_z_beta_rad": beta,
        "computational_loss_by_input": [
            float(max(0, 1 - abs(amplitude) ** 2)) for amplitude in kraus_diagonal
        ],
    }


def simulate_gate(
    model: PairModel,
    omega_rb_mhz: float,
    lifetimes: Lifetimes | None,
    mode_threshold: float,
) -> dict[str, object]:
    uy = raman_propagator(lifetimes)
    coherent = lifetimes is None
    rb_pp_lifetime = None if coherent else lifetimes.rb_56p_us
    yb_pp_lifetime = None if coherent else lifetimes.yb_52p_us
    yb_ss_lifetime = None if coherent else lifetimes.yb_53s_us
    unblocked = unblocked_rb_return(omega_rb_mhz, rb_pp_lifetime)
    blocked = blockaded_return(
        model, omega_rb_mhz, lifetimes, mode_threshold=mode_threshold
    )

    middle_01 = np.diag(
        [
            1,
            np.exp(0 if coherent else -0.5 / yb_ss_lifetime / omega_rb_mhz),
            np.exp(0 if coherent else -0.5 / yb_pp_lifetime / omega_rb_mhz),
        ]
    )
    middle_11 = np.diag(
        [
            unblocked,
            unblocked
            * np.exp(0 if coherent else -0.5 / yb_ss_lifetime / omega_rb_mhz),
            blocked,
        ]
    )
    kraus = np.array(
        [
            1,
            (uy @ middle_01 @ uy)[0, 0],
            unblocked,
            (uy @ middle_11 @ uy)[0, 0],
        ],
        dtype=complex,
    )
    metrics = _local_z_metrics(kraus)
    metrics.update(
        {
            "omega_rb_mhz": float(omega_rb_mhz),
            "rb_2pi_time_us": float(1 / omega_rb_mhz),
            "total_sequence_time_us": float(
                2 * PARAMS.raman_pi_time_us + 1 / omega_rb_mhz
            ),
            "blocked_middle_return": [float(blocked.real), float(blocked.imag)],
            "unblocked_middle_return": [float(unblocked.real), float(unblocked.imag)],
            "kraus_diagonal_re_im": [
                [float(value.real), float(value.imag)] for value in kraus
            ],
            "mode_threshold": float(mode_threshold),
            "retained_mode_count": int(np.sum(model.mode_mask(mode_threshold))),
            "retained_pp_weight": float(
                np.sum(np.abs(model.pp_overlap[model.mode_mask(mode_threshold)]) ** 2)
            ),
        }
    )
    return metrics


def optimize_omega(
    model: PairModel, lifetimes: Lifetimes, mode_threshold: float = 1e-6
) -> tuple[float, dict[str, object]]:
    omega_grid = np.linspace(0.5, 6.0, 276)
    fidelities = np.array(
        [
            simulate_gate(model, omega, lifetimes, mode_threshold)[
                "average_gate_fidelity"
            ]
            for omega in omega_grid
        ]
    )
    candidates = [0, len(omega_grid) - 1]
    candidates.extend(
        index
        for index in range(1, len(omega_grid) - 1)
        if fidelities[index] >= fidelities[index - 1]
        and fidelities[index] >= fidelities[index + 1]
    )
    candidates = sorted(candidates, key=lambda index: fidelities[index], reverse=True)[:12]
    optimum = None
    for index in candidates:
        lower = omega_grid[max(0, index - 1)]
        upper = omega_grid[min(len(omega_grid) - 1, index + 1)]
        if lower == upper:
            continue
        refined = minimize_scalar(
            lambda omega: -simulate_gate(model, omega, lifetimes, mode_threshold)[
                "average_gate_fidelity"
            ],
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 2e-10},
        )
        if optimum is None or refined.fun < optimum.fun:
            optimum = refined
    assert optimum is not None
    omega = float(optimum.x)
    return omega, simulate_gate(model, omega, lifetimes, mode_threshold)


def _verify_average_fidelity_formula(metrics: dict[str, object]) -> float:
    kraus = np.array(
        [complex(real, imag) for real, imag in metrics["kraus_diagonal_re_im"]]
    )
    alpha = float(metrics["optimal_local_z_alpha_rad"])
    beta = float(metrics["optimal_local_z_beta_rad"])
    correction = np.array(
        [1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))]
    )
    ideal = np.array([1, 1, 1, -1], dtype=complex)
    effective = np.conj(ideal) * correction * kraus
    rng = np.random.default_rng(20260830)
    samples = rng.normal(size=(200_000, 4)) + 1j * rng.normal(size=(200_000, 4))
    samples /= np.linalg.norm(samples, axis=1)[:, None]
    probabilities = np.abs(samples) ** 2
    monte_carlo = float(np.mean(np.abs(probabilities @ effective) ** 2))
    analytic = float(metrics["average_gate_fidelity"])
    assert abs(monte_carlo - analytic) < 1.5e-4
    return monte_carlo


def _target_trace(
    model: PairModel, omega: float, lifetimes: Lifetimes, threshold: float
) -> dict[str, np.ndarray]:
    _, generator, pp, ss, _ = blockaded_return(
        model,
        omega,
        lifetimes,
        mode_threshold=threshold,
        return_generator=True,
    )
    duration = 1 / omega
    times = np.linspace(0, duration, 401)
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    initial_coefficients = np.linalg.solve(
        eigenvectors, np.r_[1, np.zeros(generator.shape[0] - 1)]
    )
    states = (
        eigenvectors
        @ (
            initial_coefficients[:, None]
            * np.exp(np.outer(eigenvalues, times))
        )
    ).T
    ground = np.abs(states[:, 0]) ** 2
    pair_states = states[:, 1:]
    pp_population = np.abs(pair_states @ np.conj(pp)) ** 2
    ss_population = np.abs(pair_states @ np.conj(ss)) ** 2
    pair_population = np.sum(np.abs(pair_states) ** 2, axis=1)
    spectator = np.maximum(0, pair_population - pp_population - ss_population)
    loss = np.maximum(0, 1 - ground - pair_population)
    return {
        "time_us": times,
        "ground": ground,
        "pp": pp_population,
        "ss": ss_population,
        "spectator": spectator,
        "loss": loss,
    }


def _plot_figure(
    result: dict[str, object],
    primary: PairModel,
    lifetimes: Lifetimes,
) -> None:
    use_paper_style()
    optimized = result["optimized_gate"]
    omega = optimized["omega_rb_mhz"]
    curve_omega = np.linspace(1.5, 5.2, 220)
    curve_decay = [
        simulate_gate(primary, value, lifetimes, 1e-6)["average_gate_fidelity"]
        for value in curve_omega
    ]
    curve_coherent = [
        simulate_gate(primary, value, None, 1e-6)["average_gate_fidelity"]
        for value in curve_omega
    ]
    trace = _target_trace(primary, omega, lifetimes, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)

    ax = axes[0, 0]
    t_yb = PARAMS.raman_pi_time_us
    t_rb = 1 / omega
    ax.broken_barh([(0, t_yb), (t_yb + t_rb, t_yb)], (20, 8), facecolors=COLORS[1])
    ax.broken_barh([(t_yb, t_rb)], (8, 8), facecolors=COLORS[0])
    ax.text(t_yb / 2, 24, r"Yb Raman $\pi$", ha="center", va="center", fontsize=8)
    ax.text(t_yb + t_rb / 2, 12, r"Rb target $2\pi$", ha="center", va="center", fontsize=8)
    ax.text(t_yb + t_rb + t_yb / 2, 24, r"Yb Raman $\pi$", ha="center", va="center", fontsize=8)
    ax.annotate(
        r"$|01\rangle\leftrightarrow|0r\rangle$",
        xy=(t_yb / 2, 20),
        xytext=(t_yb / 2, 33),
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "0.35"},
        fontsize=8,
    )
    ax.annotate(
        r"$|11\rangle\leftrightarrow|1r\rangle\leftrightarrow|PP\rangle\leftrightarrow|SS\rangle$",
        xy=(t_yb + t_rb / 2, 16),
        xytext=(t_yb + t_rb / 2, 34),
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "0.35"},
        fontsize=7.5,
    )
    ax.set_xlim(0, 2 * t_yb + t_rb)
    ax.set_ylim(4, 39)
    ax.set_yticks([])
    ax.set_xlabel(r"sequence time ($\mu$s)")
    ax.set_title("(a) Simulated three-pulse gate")
    ax.spines[["left", "right", "top"]].set_visible(False)

    ax = axes[0, 1]
    ax.plot(trace["time_us"], trace["ground"], label=r"$|1_{\rm Rb},r_{\rm Yb}\rangle$", color=COLORS[0])
    ax.plot(trace["time_us"], trace["pp"], label=r"$|PP\rangle$", color=COLORS[1])
    ax.plot(trace["time_us"], trace["ss"], label=r"$|SS\rangle$", color=COLORS[2])
    ax.plot(trace["time_us"], trace["spectator"], label="spectators", color=COLORS[3])
    ax.plot(trace["time_us"], trace["loss"], label="decay loss", color="0.4", linestyle="--")
    ax.set_xlabel(r"Rb target-pulse time ($\mu$s)")
    ax.set_ylabel("no-jump population")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(fontsize=6.7, ncol=2, loc="center")
    ax.set_title("(b) All-$m$ pair-basis dynamics")

    ax = axes[1, 0]
    ax.plot(curve_omega, curve_coherent, label="coherent", color=COLORS[2])
    ax.plot(curve_omega, curve_decay, label="300 K lifetimes", color=COLORS[0])
    ax.scatter([omega], [optimized["average_gate_fidelity"]], color=COLORS[1], zorder=4)
    ax.axvline(omega, color="0.5", lw=0.7, linestyle="--")
    ax.set_xlabel(r"Rb target Rabi frequency $\Omega_{\rm Rb}/2\pi$ (MHz)")
    ax.set_ylabel("average CZ fidelity")
    ax.set_ylim(min(curve_decay) - 0.001, 1.0002)
    ax.legend(fontsize=7)
    ax.set_title("(c) Global target-pulse optimization")

    ax = axes[1, 1]
    labels = ["00", "01", "10", "11"]
    losses = 100 * np.asarray(optimized["computational_loss_by_input"])
    x = np.arange(4)
    ax.bar(x, losses, 0.45, color=COLORS[0])
    ax.set_xticks(x, labels)
    ax.set_xlabel("computational input")
    ax.set_ylabel("population outside ideal output (%)")
    ax.set_title("(d) Input-resolved loss")

    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _model_summary(model: PairModel, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "field_gauss": model.field_gauss,
        "electric_field_v_cm": model.electric_field_v_cm,
        "distance_um": model.distance_um,
        "theta_deg": model.theta_deg,
        "defect_offset_mhz": model.defect_offset_mhz,
        "basis": asdict(model.basis_config),
        "pair_basis_size": model.pair_basis_size,
        "symmetry_component_size": model.symmetry_component_size,
        "pp_component_norm": model.pp_component_norm,
        "ss_component_norm": model.ss_component_norm,
        "hermiticity_error_mhz": model.hermiticity_error_mhz,
        "pp_asymptote_mhz": model.pp_asymptote_mhz,
        "ss_asymptote_mhz": model.ss_asymptote_mhz,
        "forster_defect_mhz": model.forster_defect_mhz,
        "rb_ss_carrier_shift_mhz": model.rb_ss_carrier_shift_mhz,
        "yb_ss_carrier_shift_mhz": model.yb_ss_carrier_shift_mhz,
        "rb_hyperfine_included": model.rb_hyperfine_included,
        "rb_f_hyperfine_prefactors_mhz": model.rb_f_hyperfine_prefactors_mhz,
        "angular_delta_m_couplings_omitted": model.angular_delta_m_couplings_omitted,
        "static_transfer": model.static_transfer,
        "spectral_diagnostics": model.spectral_diagnostics,
        "optimized_omega_rb_mhz": metrics["omega_rb_mhz"],
        "optimized_average_gate_fidelity": metrics["average_gate_fidelity"],
    }


def run_simulation() -> dict[str, object]:
    print("Querying PairInteraction lifetimes ...")
    lifetimes_300 = query_lifetimes(300)
    lifetimes_0 = query_lifetimes(0)

    coherent_uy = raman_propagator(None)
    coherent_raman_transfer = float(abs(coherent_uy[2, 0]) ** 2)
    assert coherent_raman_transfer > 1 - 1e-12
    raman_times = np.linspace(0, PARAMS.raman_pi_time_us, 2001)
    h_raman = np.array(
        [
            [0, PARAMS.omega_yb_optical_mhz / 2, 0],
            [PARAMS.omega_yb_optical_mhz / 2, PARAMS.raman_delta_mhz, PARAMS.omega_yb_microwave_mhz / 2],
            [0, PARAMS.omega_yb_microwave_mhz / 2, 0],
        ],
        dtype=complex,
    )
    evals, evecs = np.linalg.eigh(h_raman)
    raman_states = (
        evecs
        @ (
            (evecs.conj().T @ np.array([1, 0, 0], dtype=complex))[:, None]
            * np.exp(-2j * np.pi * np.outer(evals, raman_times))
        )
    ).T
    peak_relay_population = float(np.max(np.abs(raman_states[:, 1]) ** 2))
    raman_300 = raman_propagator(lifetimes_300)
    raman_300_transfer = float(abs(raman_300[2, 0]) ** 2)

    field_values = [0.0, 1.0, 2.0, 3.0, 3.2, 3.4, 3.6, 4.0, 5.0, 6.0]
    field_models: dict[float, PairModel] = {}
    field_scan = []
    print("Scanning magnetic field with all magnetic sublevels ...")
    for field in field_values:
        model = build_pair_model(field, SCAN_BASIS)
        field_models[field] = model
        omega, metrics = optimize_omega(model, lifetimes_300)
        field_scan.append(
            {
                "field_gauss": field,
                "optimized_omega_rb_mhz": omega,
                "average_gate_fidelity": metrics["average_gate_fidelity"],
                "static_maximum_ss_population": model.static_transfer[
                    "maximum_ss_population"
                ],
            }
        )
        print(
            f"  B={field:3.1f} G: Favg={metrics['average_gate_fidelity']:.9f}, "
            f"Omega={omega:.4f} MHz"
        )

    selected_field = float(
        max(field_scan, key=lambda item: item["average_gate_fidelity"])["field_gauss"]
    )
    scan_model = field_models[selected_field]
    scan_omega, scan_metrics = optimize_omega(scan_model, lifetimes_300)

    print(f"Building primary converged basis at B={selected_field:.1f} G ...")
    primary = build_pair_model(selected_field, PRIMARY_BASIS)
    preliminary_omega, _ = optimize_omega(primary, lifetimes_300, mode_threshold=1e-6)
    final_optimization = minimize_scalar(
        lambda omega: -simulate_gate(primary, omega, lifetimes_300, 1e-8)[
            "average_gate_fidelity"
        ],
        bounds=(preliminary_omega - 0.03, preliminary_omega + 0.03),
        method="bounded",
        options={"xatol": 2e-11},
    )
    optimized_omega = float(final_optimization.x)
    optimized = simulate_gate(primary, optimized_omega, lifetimes_300, 1e-8)
    coherent = simulate_gate(primary, optimized_omega, None, 1e-8)
    mode_check = simulate_gate(primary, optimized_omega, lifetimes_300, 1e-6)
    full_mode_blocked = full_mode_blockaded_return_sparse(
        primary, optimized_omega, lifetimes_300
    )
    projected_blocked = complex(*optimized["blocked_middle_return"])
    full_mode_amplitude_difference = abs(full_mode_blocked - projected_blocked)

    pair_window_check = build_pair_model(selected_field, PAIR_WINDOW_CHECK_BASIS)
    _, pair_window_metrics = optimize_omega(pair_window_check, lifetimes_300)

    basis_difference = abs(
        optimized["average_gate_fidelity"] - scan_metrics["average_gate_fidelity"]
    )
    mode_difference = abs(
        optimized["average_gate_fidelity"] - mode_check["average_gate_fidelity"]
    )
    assert basis_difference < 5e-5, f"Atomic-basis convergence failed: {basis_difference}"
    assert mode_difference < 2e-5, f"Bright-mode convergence failed: {mode_difference}"
    assert full_mode_amplitude_difference < 2e-6, (
        f"Full-mode projection failed: {full_mode_amplitude_difference}"
    )

    monte_carlo = _verify_average_fidelity_formula(optimized)
    sensitivity = []
    for fraction in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        metrics = simulate_gate(
            primary, optimized_omega * (1 + fraction), lifetimes_300, 1e-8
        )
        sensitivity.append(
            {
                "fractional_omega_error": fraction,
                "average_gate_fidelity": metrics["average_gate_fidelity"],
            }
        )

    result: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pairinteraction": "2.5.0",
        },
        "model_scope": {
            "pair_hamiltonian": (
                "PairInteraction electronic Hamiltonian plus explicit 87Rb I=3/2 "
                "hyperfine and nuclear Zeeman terms in the M_tot=5/2 block at theta=0"
            ),
            "pulses": "square pulses propagated explicitly",
            "decay": "non-Hermitian no-jump evolution; every 300 K depopulation event is mapped to orthogonal erasure",
            "not_included": [
                "atomic motion and position distribution",
                "laser amplitude, phase, and frequency noise",
                "finite pulse rise and fall times",
                "state-resolved spontaneous-decay branching back into the computational subspace",
                "full optical intermediate manifolds for the assumed effective Rabi frequencies",
            ],
        },
        "parameters": {
            **asdict(PARAMS),
            "field_gauss": selected_field,
            "omega_rb_mhz": optimized_omega,
        },
        "pairinteraction_lifetimes_300k_us": asdict(lifetimes_300),
        "pairinteraction_lifetimes_radiative_0k_us": asdict(lifetimes_0),
        "yb_raman": {
            "coherent_transfer_probability": coherent_raman_transfer,
            "transfer_probability_with_300k_lifetimes": raman_300_transfer,
            "peak_relay_population": peak_relay_population,
        },
        "field_scan": field_scan,
        "selected_pair_model": _model_summary(primary, optimized),
        "optimized_gate": optimized,
        "coherent_gate": coherent,
        "omega_sensitivity": sensitivity,
        "convergence": {
            "atomic_basis_check": _model_summary(scan_model, scan_metrics),
            "pair_window_check": _model_summary(pair_window_check, pair_window_metrics),
            "primary_minus_atomic_check_fidelity": float(basis_difference),
            "bright_mode_threshold_1e-6_fidelity": mode_check[
                "average_gate_fidelity"
            ],
            "bright_mode_threshold_difference": float(mode_difference),
            "full_mode_sparse_blocked_return_re_im": [
                float(full_mode_blocked.real),
                float(full_mode_blocked.imag),
            ],
            "full_mode_minus_projected_amplitude": float(
                full_mode_amplitude_difference
            ),
        },
        "validation": {
            "raman_exact_cyclic_condition_passed": True,
            "pair_hamiltonian_hermiticity_passed": True,
            "target_state_representation_passed": True,
            "atomic_basis_convergence_passed": True,
            "bright_mode_convergence_passed": True,
            "full_mode_sparse_projection_passed": True,
            "haar_monte_carlo_average_fidelity": monte_carlo,
            "analytic_minus_monte_carlo": float(
                optimized["average_gate_fidelity"] - monte_carlo
            ),
        },
    }

    _plot_figure(result, primary, lifetimes_300)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT_PATH.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_STEM.relative_to(ROOT)}.pdf and .png")
    print(
        f"Final: B={selected_field:.1f} G, Omega_Rb/2pi={optimized_omega:.6f} MHz, "
        f"Favg={optimized['average_gate_fidelity']:.9f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_simulation()


if __name__ == "__main__":
    main()
