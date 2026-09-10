#!/usr/bin/env python3
"""Fixed-pulse cross-M audit, explicit nuclear product basis and sparse spectra.

No axial-brightness selection is applied when building the Hilbert space.
The only additional truncations are an explicit total-M band and the number
of exact eigenpairs nearest the addressed SS asymptote. Both must be checked.
Decay uses the published secular pair-eigenmode prescription, unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pairinteraction as pi
import scipy
from scipy.linalg import expm
from scipy.sparse import block_diag, coo_matrix, diags, eye
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply, splu

import evaluate_forster_p0_4 as p0
import simulate_forster_gate as base
import simulate_robust_shaped_forster_gate as shaped
from rb_rydberg_hyperfine import RB87_M_I, dressed_hyperfine_hamiltonian_mhz
from reproduce_forster_characterization import database_manifest

HERE = Path(__file__).resolve().parent


def lift_hyperfine(atomic_hfs, rb_indices, yb_indices, nuclear_indices, n_rb):
    """Restrict H_Rb tensor identity_Yb to explicit (pair,mI) products."""
    atomic_indices = nuclear_indices * n_rb + rb_indices
    rows, cols, data = [], [], []
    for yb in np.unique(yb_indices):
        idx = np.flatnonzero(yb_indices == yb)
        block = atomic_hfs[atomic_indices[idx]][:, atomic_indices[idx]].tocoo()
        rows.extend(idx[block.row])
        cols.extend(idx[block.col])
        data.extend(block.data)
    return coo_matrix((data, (rows, cols)), shape=(len(rb_indices),) * 2).tocsr()


def build_sparse(distance, theta, sector_radius, config=p0.REFERENCE_BASIS):
    rb_pp, rb_ss, yb_pp, yb_ss = base._kets()
    common = dict(delta_n=config.delta_n, delta_l=config.delta_l,
                  delta_m=None, delta_energy=config.atomic_window_ghz,
                  delta_energy_unit="GHz")
    rb_basis = pi.BasisAtom.from_kets([rb_pp, rb_ss], delta_j=2, **common)
    yb_basis = pi.BasisAtom.from_kets([yb_pp, yb_ss], delta_f=3, **common)
    rb = pi.SystemAtom(rb_basis).set_magnetic_field([0, 0, 3.10], unit="G")
    yb = pi.SystemAtom(yb_basis).set_magnetic_field([0, 0, 3.10], unit="G")
    rb.diagonalize()
    yb.diagonalize()
    zero = rb.get_corresponding_energy(rb_pp, unit="MHz") + yb.get_corresponding_energy(yb_pp, unit="MHz")
    ss_zero = (rb.get_corresponding_energy(rb_ss, unit="MHz")
               + yb.get_corresponding_energy(yb_ss, unit="MHz") - zero
               + base.target_stretched_shift_mhz(56, 0, 3.10))
    pairs = pi.BasisPair([rb, yb], energy=(zero - config.pair_window_ghz * 1000,
                                          zero + config.pair_window_ghz * 1000),
                         energy_unit="MHz")
    system = pi.SystemPair(pairs)
    system.set_distance(distance, angle_degree=theta, unit="micrometer")
    system.set_interaction_order(config.interaction_order)
    electronic = system.get_hamiltonian(unit="MHz").tocsr()
    electronic = electronic - zero * eye(pairs.number_of_states, format="csr")
    rb_map = base._dressed_index_by_bare_index(rb)
    yb_map = base._dressed_index_by_bare_index(yb)
    rb_idx, yb_idx, electronic_m = [], [], []
    for ket in pairs.kets:
        r, y = ket.state_atoms
        rb_idx.append(rb_map[r.get_corresponding_ket_index()])
        yb_idx.append(yb_map[y.get_corresponding_ket_index()])
        electronic_m.append(r.get_corresponding_ket().m + y.get_corresponding_ket().m)
    n = pairs.number_of_states
    nuclear_idx = np.repeat(np.arange(4), n)
    total_m = np.tile(electronic_m, 4) + RB87_M_I[nuclear_idx]
    keep = np.flatnonzero(np.abs(total_m - base.TARGET_TOTAL_M) <= sector_radius + 1e-9)
    # Slice before copying nuclear blocks, avoiding the full 4N sparse product.
    blocks = []
    for nuclear in range(4):
        idx = keep[nuclear_idx[keep] == nuclear] % n
        blocks.append(electronic[idx][:, idx])
    h = block_diag(blocks, format="csr")
    hfs = dressed_hyperfine_hamiltonian_mhz(rb_basis, rb, 3.10)
    h += lift_hyperfine(hfs, np.tile(rb_idx, 4)[keep], np.tile(yb_idx, 4)[keep],
                       nuclear_idx[keep], rb_basis.number_of_states)
    nuclear_target = np.isclose(RB87_M_I, 1.5).astype(float)
    pp = np.kron(nuclear_target, base._dense_overlap(pairs, (rb_pp, yb_pp)))[keep]
    ss = np.kron(nuclear_target, base._dense_overlap(pairs, (rb_ss, yb_ss)))[keep]
    assert abs(np.vdot(ss, ss).real - 1) < 1e-9
    assert abs(np.vdot(pp, pp).real - 1) < 1e-9
    assert np.max(np.abs(h.data.imag)) < 1e-9, "xz-plane Hamiltonian must be real"
    h = h.real.tocsr()
    herm = h - h.T
    assert max(abs(herm.data), default=0) < 1e-8
    h -= ss_zero * eye(len(keep), format="csr")
    h.eliminate_zeros()
    return h, ss.real, pp.real, total_m[keep], {
        "electronic_pair_size": n, "nuclear_product_size": 4 * n,
        "retained_size": len(keep), "nnz": h.nnz,
        "ss_asymptote_relative_electronic_pp_mhz": float(ss_zero),
        "sector_sizes": {str(m): int(np.sum(total_m[keep] == m)) for m in np.unique(total_m[keep])},
    }


def spectrum(h, ss, pp, total_m, count, inverse=None):
    if not 0 < count < h.shape[0]:
        raise ValueError(f"--modes must be between 1 and {h.shape[0]-1}; got {count}")
    # Random seed rather than SS: the solver must also resolve axial-dark modes.
    energies, vectors = eigsh(h, k=count, sigma=0.1234567, which="LM",
                             v0=np.random.default_rng(20260910).normal(size=h.shape[0]),
                             tol=1e-11, OPinv=inverse)
    order = np.argsort(energies)
    energies, vectors = energies[order], vectors[:, order]
    residual = np.linalg.norm(h @ vectors - vectors * energies, axis=0)
    orthogonality = np.max(abs(vectors.T @ vectors - np.eye(count)))
    assert residual.max() < 1e-5, residual.max()
    assert orthogonality < 1e-7, orthogonality
    c_ss, c_pp = vectors.T @ ss, vectors.T @ pp
    outside = ~np.isclose(total_m, base.TARGET_TOTAL_M)
    q_vectors = vectors[outside]
    gram = q_vectors.T @ q_vectors
    info = {
        "mode_count": count, "energy_range_mhz": [float(energies.min()), float(energies.max())],
        "omitted_ss_weight": float(max(0, 1 - c_ss @ c_ss)),
        "omitted_pp_weight": float(max(0, 1 - c_pp @ c_pp)),
        "max_eigen_residual_mhz": float(residual.max()),
        "orthogonality_error": float(orthogonality),
    }
    return energies, c_ss, c_pp, gram, info


def propagate(energies, c_ss, c_pp, gram, pulse, scale, correction, lifetimes):
    pp_rate = 1 / lifetimes.rb_56p_us + 1 / lifetimes.yb_52p_us
    ss_rate = 1 / lifetimes.rb_56s_us + 1 / lifetimes.yb_53s_us
    rates = c_pp**2 * pp_rate + c_ss**2 * ss_rate + np.maximum(0, 1-c_pp**2-c_ss**2) * max(pp_rate, ss_rate)
    count = len(energies)
    idx = np.arange(1, count + 1)
    drive = coo_matrix((np.r_[c_ss, c_ss] / 2,
                        (np.r_[idx, np.zeros(count, dtype=int)],
                         np.r_[np.zeros(count, dtype=int), idx])), shape=(count+1,) * 2).tocsr()
    state = np.zeros(count + 1, complex)
    state[0] = 1
    unblocked = np.array([1, 0], complex)
    peak_q = 0.0
    peak_pair = 0.0
    for segment in pulse:
        omega, detuning, dt = scale * segment.omega_mhz, segment.detuning_mhz, segment.duration_us
        diagonal = np.r_[-1j/(4*np.pi*lifetimes.rb_56s_us),
                          energies-detuning-1j*rates/(4*np.pi)]
        generator = -2j*np.pi*dt*(diags(diagonal) + omega*drive)
        state = expm_multiply(generator, state, traceA=-2j*np.pi*dt*diagonal.sum())
        h01 = np.array([[0, omega/2], [omega/2, -detuning-1j/(4*np.pi*lifetimes.yb_53s_us)]])
        unblocked = expm(-2j*np.pi*dt*h01) @ unblocked
        q_pop = float(np.vdot(state[1:], gram @ state[1:]).real)
        peak_q = max(peak_q, q_pop)
        peak_pair = max(peak_pair, float(np.vdot(state[1:], state[1:]).real))
    control = shaped._control_pi(lifetimes.rb_56s_us, amplitude_scale=scale)
    idle = np.exp(-sum(s.duration_us for s in pulse)/(2*lifetimes.rb_56s_us))
    k = np.array([1, unblocked[0], (control @ np.diag([1, idle]) @ control)[0, 0],
                  (control @ np.diag([unblocked[0], state[0]]) @ control)[0, 0]])
    metrics = shaped._local_z_metrics(k)
    return {
        "scale_rb_and_yb": scale,
        "fidelity": shaped._fixed_correction_fidelity(k, correction),
        "mean_survival": float(np.vdot(k, k).real/4),
        "phase_error_rad": metrics["conditional_phase_error_rad"],
        "kraus_re_im": [[float(z.real), float(z.imag)] for z in k],
        "peak_outside_m_population": peak_q,
        "final_outside_m_population": q_pop,
        "peak_pair_population": peak_pair,
        "target_window_return_probability": float(abs(state[0])**2),
        "target_window_final_pair_population": float(np.vdot(state[1:], state[1:]).real),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0, help="micrometres")
    parser.add_argument("--dz", type=float, default=0, help="micrometres, pair axis")
    parser.add_argument("--sector-radius", type=int, default=0)
    parser.add_argument("--modes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--step-ns", type=float, default=0.125)
    parser.add_argument("--check-half-step", action="store_true",
                        help="repeat propagation at half the step for the largest mode count")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sector_radius < 0 or min(args.modes) < 1:
        parser.error("--sector-radius must be nonnegative and --modes positive")
    if not np.isfinite([args.dx, args.dz, args.step_ns]).all() or args.step_ns <= 0:
        parser.error("offsets must be finite and --step-ns finite and positive")
    if np.hypot(3.4 + args.dz, args.dx) == 0:
        parser.error("the interatomic distance must be positive")
    if args.output.exists():
        parser.error(f"{args.output} already exists (possibly partial). Preserve it and use a new --output and --modes for the remaining counts; automatic resume is not supported.")
    pi.Database.initialize_global_database(download_missing=False)
    started = time.monotonic()
    distance = float(np.hypot(3.4 + args.dz, args.dx))
    theta = float(np.degrees(np.arctan2(args.dx, 3.4 + args.dz)))
    reference = json.loads((HERE / "forster_control_decay_recheck.json").read_text())
    correction = reference["rows"]["0.000000/0.000000"]["fixed_local_z_rad"]
    # Reuse the reference's pinned zero-temperature lifetimes. Re-querying the
    # atomic decay database while holding the LU factors has a large memory peak.
    lifetimes = base.Lifetimes(**reference["lifetimes_us"])
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": {"dx_um": args.dx, "dz_um": args.dz, "distance_um": distance, "theta_deg": theta},
        "sector_radius": args.sector_radius, "basis": asdict(p0.REFERENCE_BASIS),
        "fixed_local_z_rad": correction, "step_ns": args.step_ns,
        "lifetimes_us": asdict(lifetimes),
        "scope": "frozen xz-plane geometry; fixed published pulse; secular eigenmode absorbing decay; no thermal average",
        "population_convention": "target window initialized in control-excited state; excludes opening/closing pulse attenuation; peaks sampled at integration boundaries",
        "software": {"pairinteraction": pi.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                          [Path(__file__).name, "simulate_forster_gate.py", "rb_rydberg_hyperfine.py",
                           "evaluate_forster_p0_4.py", "optimize_hardware_aware_forster_gate.py",
                           "simulate_robust_shaped_forster_gate.py", "optimize_bounded_minimax_forster_gate.py",
                           "forster_control_decay_recheck.json"]},
        "database": database_manifest(), "rows": [],
    }
    print(f"Building R={distance}, theta={theta}, M band radius={args.sector_radius}", flush=True)
    h, ss, pp, total_m, result["construction"] = build_sparse(distance, theta, args.sector_radius)
    print(result["construction"], flush=True)
    pulse = p0._pulse(args.step_ns)
    if max(args.modes) >= h.shape[0]:
        parser.error(f"--modes must be smaller than retained dimension {h.shape[0]}")
    # Construction releases large temporary sparse blocks. Return free glibc
    # arenas before factorization rather than retaining them under the orb cap.
    gc.collect()
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
    print("Factorizing once for all requested eigenmode counts", flush=True)
    factor = splu((h - 0.1234567 * eye(h.shape[0])).tocsc(), permc_spec="MMD_AT_PLUS_A")
    result["factorization_ordering"] = "MMD_AT_PLUS_A"
    inverse = LinearOperator(h.shape, matvec=factor.solve, dtype=h.dtype)
    for count in args.modes:
        print(f"Solving {count} eigenmodes", flush=True)
        energies, c_ss, c_pp, gram, info = spectrum(h, ss, pp, total_m, count, inverse)
        info["scenarios"] = [propagate(energies, c_ss, c_pp, gram, pulse, scale, correction, lifetimes)
                             for scale in (1.0, 0.99)]
        if args.check_half_step and count == max(args.modes):
            info["half_step_check"] = {
                "step_ns": args.step_ns / 2,
                "scenarios": [propagate(energies, c_ss, c_pp, gram,
                                        p0._pulse(args.step_ns / 2), scale, correction, lifetimes)
                              for scale in (1.0, 0.99)],
            }
        result["rows"].append(info)
        result["wall_seconds"] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(info), flush=True)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
