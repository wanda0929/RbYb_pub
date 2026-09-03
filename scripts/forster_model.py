"""Shared all-magnetic-sublevel PairInteraction model for the Förster channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pairinteraction as pi
from scipy.optimize import minimize_scalar
from scipy.sparse import identity
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class BasisConfig:
    delta_n: int
    atomic_window_ghz: float
    pair_window_ghz: float


@dataclass
class PairModel:
    field_gauss: float
    distance_um: float
    theta_deg: float
    basis_config: BasisConfig
    energies_mhz: np.ndarray
    pp_overlap: np.ndarray
    ss_overlap: np.ndarray
    pair_basis_size: int
    symmetry_component_size: int
    pp_component_norm: float
    ss_component_norm: float
    hermiticity_error_mhz: float
    static_transfer: dict[str, float]


@dataclass(frozen=True)
class Lifetimes:
    rb_56p_us: float
    rb_56s_us: float
    yb_52p_us: float
    yb_53s_us: float


PRIMARY_BASIS = BasisConfig(delta_n=4, atomic_window_ghz=160.0, pair_window_ghz=20.0)
SCAN_BASIS = BasisConfig(delta_n=3, atomic_window_ghz=80.0, pair_window_ghz=20.0)


def initialize_database() -> None:
    if pi.Database.get_global_database() is None:
        pi.Database.initialize_global_database(download_missing=False)


def forster_kets() -> tuple[pi.KetAtom, pi.KetAtom, pi.KetAtom, pi.KetAtom]:
    """Return Rb PP/SS and Yb PP/SS kets, respectively."""
    rb_pp = pi.KetAtom("Rb", n=56, l=1, j=0.5, m=0.5)
    rb_ss = pi.KetAtom("Rb", n=56, l=0, j=0.5, m=0.5)
    yb_pp = pi.KetAtom("Yb171_mqdt", n=52, l=1, s=0, f=0.5, m=0.5)
    yb_ss = pi.KetAtom("Yb171_mqdt", n=53, l=0, s=1, f=0.5, m=0.5)
    return rb_pp, rb_ss, yb_pp, yb_ss


def dense_amplitudes(basis: pi.BasisPair, pair: tuple[pi.KetAtom, pi.KetAtom]) -> np.ndarray:
    amplitudes = basis.get_amplitudes(pair)
    if hasattr(amplitudes, "toarray"):
        amplitudes = amplitudes.toarray()
    return np.asarray(amplitudes, dtype=complex).reshape(-1)


def build_pair_model(
    field_gauss: float,
    config: BasisConfig,
    distance_um: float = 3.4,
    theta_deg: float = 0.0,
) -> PairModel:
    """Construct and diagonalize the connected all-m Förster Hamiltonian."""
    initialize_database()
    rb_pp, rb_ss, yb_pp, yb_ss = forster_kets()
    rb_basis = pi.BasisAtom.from_kets(
        [rb_pp, rb_ss],
        delta_n=config.delta_n,
        delta_l=2,
        delta_j=2,
        delta_m=None,
        delta_energy=config.atomic_window_ghz,
        delta_energy_unit="GHz",
    )
    yb_basis = pi.BasisAtom.from_kets(
        [yb_pp, yb_ss],
        delta_n=config.delta_n,
        delta_l=2,
        delta_f=3,
        delta_m=None,
        delta_energy=config.atomic_window_ghz,
        delta_energy_unit="GHz",
    )
    rb_system = (
        pi.SystemAtom(rb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    )
    yb_system = (
        pi.SystemAtom(yb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    )
    energy_zero_mhz = float(
        rb_system.get_corresponding_energy(rb_pp, unit="MHz")
        + yb_system.get_corresponding_energy(yb_pp, unit="MHz")
    )
    pair_window_mhz = 1000 * config.pair_window_ghz
    pair_basis = pi.BasisPair(
        [rb_system, yb_system],
        energy=(energy_zero_mhz - pair_window_mhz, energy_zero_mhz + pair_window_mhz),
        energy_unit="MHz",
    )
    hamiltonian = (
        pi.SystemPair(pair_basis)
        .set_distance(distance_um, angle_degree=theta_deg, unit="micrometer")
        .set_interaction_order(3)
        .get_hamiltonian(unit="MHz")
        .tocsr()
        .astype(complex)
    )
    hamiltonian -= energy_zero_mhz * identity(
        pair_basis.number_of_states, format="csr", dtype=complex
    )
    pp_full = dense_amplitudes(pair_basis, (rb_pp, yb_pp))
    ss_full = dense_amplitudes(pair_basis, (rb_ss, yb_ss))

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
    if hermiticity_error / scale >= 1e-11:
        raise RuntimeError("Pair Hamiltonian is not Hermitian")

    pp_component = pp_full[component]
    ss_component = ss_full[component]
    pp_norm = float(np.vdot(pp_component, pp_component).real)
    ss_norm = float(np.vdot(ss_component, ss_component).real)
    if pp_norm <= 0.999 or ss_norm <= 0.99:
        raise RuntimeError(f"incomplete target representation: PP={pp_norm:.9f}, SS={ss_norm:.9f}")
    pp_component /= np.sqrt(pp_norm)
    ss_component /= np.sqrt(ss_norm)

    energies, eigenvectors = np.linalg.eigh(h_component)
    pp_overlap = eigenvectors.conj().T @ pp_component
    ss_overlap = eigenvectors.conj().T @ ss_component

    times = np.linspace(0, 0.080, 4001)
    pair_phase = np.exp(-2j * np.pi * np.outer(times, energies))
    ss_amplitude = pair_phase @ (np.conj(ss_overlap) * pp_overlap)
    pp_amplitude = pair_phase @ (np.abs(pp_overlap) ** 2)
    ss_population = np.abs(ss_amplitude) ** 2
    maximum_index = int(np.argmax(ss_population))
    pp_population = float(np.abs(pp_amplitude[maximum_index]) ** 2)
    static_transfer = {
        "maximum_ss_population": float(ss_population[maximum_index]),
        "time_us": float(times[maximum_index]),
        "pp_population_at_maximum": pp_population,
        "spectator_population_at_maximum": float(
            max(0.0, 1 - ss_population[maximum_index] - pp_population)
        ),
    }
    return PairModel(
        field_gauss=field_gauss,
        distance_um=distance_um,
        theta_deg=theta_deg,
        basis_config=config,
        energies_mhz=energies,
        pp_overlap=pp_overlap,
        ss_overlap=ss_overlap,
        pair_basis_size=hamiltonian.shape[0],
        symmetry_component_size=len(component),
        pp_component_norm=pp_norm,
        ss_component_norm=ss_norm,
        hermiticity_error_mhz=hermiticity_error,
        static_transfer=static_transfer,
    )


def query_lifetimes(temperature_k: float) -> Lifetimes:
    initialize_database()
    rb_pp, rb_ss, yb_pp, yb_ss = forster_kets()
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


def ss_origin_offset_mhz(field_gauss: float) -> float:
    """Return dressed E(SS)-E(PP) in cyclic MHz at the specified field."""
    initialize_database()
    rb_pp, rb_ss, yb_pp, yb_ss = forster_kets()
    rb_basis = pi.BasisAtom.from_kets([rb_pp, rb_ss], delta_n=0, delta_l=0, delta_j=0, delta_m=0)
    yb_basis = pi.BasisAtom.from_kets([yb_pp, yb_ss], delta_n=0, delta_l=0, delta_f=0, delta_m=0)
    rb_system = (
        pi.SystemAtom(rb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    )
    yb_system = (
        pi.SystemAtom(yb_basis).set_magnetic_field([0, 0, field_gauss], unit="G").diagonalize()
    )
    pp_energy = rb_system.get_corresponding_energy(
        rb_pp, unit="MHz"
    ) + yb_system.get_corresponding_energy(yb_pp, unit="MHz")
    ss_energy = rb_system.get_corresponding_energy(
        rb_ss, unit="MHz"
    ) + yb_system.get_corresponding_energy(yb_ss, unit="MHz")
    return float(ss_energy - pp_energy)


def local_z_metrics(kraus: np.ndarray) -> dict[str, float | list[float]]:
    """Loss-aware average CZ fidelity after optimal virtual local-Z gates."""
    ideal = np.array([1, 1, 1, -1], dtype=complex)

    def negative_overlap(alpha: float) -> float:
        first = kraus[0] + np.exp(1j * alpha) * kraus[2]
        second = kraus[1] - np.exp(1j * alpha) * kraus[3]
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
    first = kraus[0] + np.exp(1j * alpha) * kraus[2]
    second = kraus[1] - np.exp(1j * alpha) * kraus[3]
    beta = float(np.angle(first) - np.angle(second))
    correction = np.array([1, np.exp(1j * beta), np.exp(1j * alpha), np.exp(1j * (alpha + beta))])
    overlap = complex(np.vdot(ideal, correction * kraus))
    survival = float(np.vdot(kraus, kraus).real / 4)
    conditional_phase = float(np.angle(kraus[3] * kraus[0] / (kraus[2] * kraus[1])))
    return {
        "average_gate_fidelity": float((4 * survival + abs(overlap) ** 2) / 20),
        "mean_computational_survival": survival,
        "conditional_phase_rad": conditional_phase,
        "conditional_phase_error_rad": float(np.angle(np.exp(1j * (conditional_phase - np.pi)))),
        "optimal_local_z_alpha_rad": alpha,
        "optimal_local_z_beta_rad": beta,
        "computational_loss_by_input": [
            float(max(0, 1 - abs(amplitude) ** 2)) for amplitude in kraus
        ],
    }
