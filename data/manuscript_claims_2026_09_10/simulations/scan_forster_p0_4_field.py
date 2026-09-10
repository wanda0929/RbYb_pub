#!/usr/bin/env python3
"""Exploratory fixed-pulse field study; never replaces published P0/P1 data.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Checkpoints are resumable;
only small prepared modes are retained during a run, not PairInteraction models.
Both phase conventions track local atomic carrier transitions at every field.
The retained angular block omits nonzero Delta-M couplings for tilted pairs.

Recorded replay sequence (arguments to this script, in order):
  [no arguments: baseline full grid, then coarse 0--5 G nominal scan]
  --fields 3.05 3.075 3.1 3.125 3.15 --grid endpoints --workers 4
  --fields 3.085 3.1025 --grid full --workers 4
  --fields 3.1 3.1025 --grid endpoints --window 60 --workers 4
  --assess-only
  --plot-only

Existing checkpoints skip completed propagations. Move the result aside before
an independent clean replay; the script never reoptimizes the selected pulse.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pairinteraction as pi
import scipy

import evaluate_forster_p0_4 as p0
import optimize_bounded_minimax_forster_gate as minimax
import optimize_forster_p0_4_reference as reference
import optimize_hardware_aware_forster_gate as hardware
import simulate_robust_shaped_forster_gate as shaped
from simulate_forster_gate import build_pair_model, query_lifetimes

OUT = Path(__file__).with_name('forster_p0_4_field_scan.json')
PARAMETERS = np.array([4.480374546413655, 9.229350329961736,
    11.778486107199452, -0.3647742336217171, -1.0235176743455907,
    2.4312627812906094, 0.025634115265641792])


def coordinates(radius, cosine):
    axial = shaped.R0_UM + radius * cosine
    transverse = radius * np.sqrt(max(0., 1 - cosine**2))
    return float(np.hypot(axial, transverse)), float(np.degrees(np.arctan2(transverse, axial)))


def summarize(rows, nominal):
    endpoints = [r for r in rows if r['radius_um'] == .05 and abs(r['cosine']) == 1]
    worst = min(rows, key=lambda r: np.min(r['vertices']))
    iy, ir = np.unravel_index(np.argmin(worst['vertices']), (2, 2))
    return {'nominal': nominal, 'position_minimum': min(r['position'] for r in rows),
        'joint_minimum': float(np.min(worst['vertices'])),
        'endpoint_objective': reference.robust_objective(np.array([r['vertices'] for r in endpoints]), nominal),
        'worst': {k: worst[k] for k in ('radius_um', 'cosine', 'distance_um', 'theta_deg')}
            | {'yb_scale': minimax.AMPLITUDE_VERTICES[iy], 'rb_scale': minimax.AMPLITUDE_VERTICES[ir]}}


class Study:
    def __init__(self):
        self.start = time.monotonic()
        self.data = json.loads(OUT.read_text()) if OUT.exists() else {
            'pulse_parameters': PARAMETERS.tolist(), 'basis': asdict(p0.REFERENCE_BASIS),
            'cutoff': 1e-6, 'step_ns': .125, 'temperature_k': 0,
            'scope': 'Fixed pulse, carrier-tracked local atomic transitions. Zero-T no-jump loss-aware overlap, not CPTP process fidelity. Sampled finite retained angular block; no continuous/global optimum or fixed-laser magnetic-noise prediction.',
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'pairinteraction': pi.__version__},
            'fields': {}, 'runs': []}
        assert self.data['pulse_parameters'] == PARAMETERS.tolist()
        assert np.array_equal(PARAMETERS, minimax.SELECTED_PARAMETERS)
        self.cache = {}
        self.pulse = reference._pulse(PARAMETERS, .125)
        self.lifetimes = query_lifetimes(0.)
        self.data['lifetimes'] = asdict(self.lifetimes)
        self.data['database_assets'] = reference.DATABASE_ASSETS
        self.data['runs'].append({'wall_seconds': 0, 'command': sys.argv})
        self.workers = 1

    def save(self):
        self.data['runs'][-1].update(wall_seconds=time.monotonic()-self.start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maximum_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        temporary = OUT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.data, indent=2)+'\n')
        temporary.replace(OUT)

    def modes(self, field, radius=0., cosine=0., window=40.):
        key = (field, radius, cosine, window)
        if key not in self.cache:
            distance, theta = coordinates(radius, cosine)
            start = time.monotonic()
            model = build_pair_model(field, replace(p0.REFERENCE_BASIS, pair_window_ghz=window),
                distance_um=distance, theta_deg=theta)
            modes = shaped._prepare_modes(model, 1e-6)
            metadata = {'radius_um': radius, 'cosine': cosine, 'distance_um': distance,
                'theta_deg': theta, 'active_modes': len(modes.energies_rel_mhz),
                'retained_ss_weight': modes.retained_ss_weight,
                'connected_component_size': model.symmetry_component_size,
                'pair_basis_size': model.pair_basis_size,
                'model_seconds': time.monotonic()-start}
            self.cache[key] = (modes, metadata)
            del model
            gc.collect()
        return self.cache[key]

    def prefetch(self, keys):
        keys = [key for key in keys if key not in self.cache]
        if self.workers > 1 and keys:
            with ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                for key, prepared in zip(keys, pool.map(prepare_modes, keys)):
                    self.cache[key] = prepared

    def nominal(self, field, window=40.):
        key = f'{field:.8f}/{window:g}'
        if key not in self.data['fields']:
            modes, metadata = self.modes(field, window=window)
            kraus = hardware._kraus(modes, self.pulse, self.lifetimes)
            metrics = shaped._local_z_metrics(kraus)
            correction = [metrics['optimal_local_z_alpha_rad'], metrics['optimal_local_z_beta_rad']]
            fixed = correction if field == 3.1 and window == 40 else self.nominal(3.1)['correction']
            self.data['fields'][key] = {'field_gauss': field, 'pair_window_ghz': window,
                'correction': correction, 'nominal': metrics['average_gate_fidelity'],
                'survival': metrics['mean_computational_survival'],
                'fixed_3_10G_nominal': shaped._fixed_correction_fidelity(kraus, fixed),
                'nominal_model': metadata, 'rows': {}}
            if window != 40:
                self.data['fields'][key]['fixed_same_field_40GHz_nominal'] = (
                    shaped._fixed_correction_fidelity(kraus, self.nominal(field)['correction']))
            self.save()
            print('nominal', key, self.data['fields'][key]['nominal'], flush=True)
        return self.data['fields'][key]

    def grid(self, field, full=False, window=40.):
        row = self.nominal(field, window)
        specs = reference.validation_specs() if full else reference.OPTIMIZATION_SPECS
        self.prefetch([(field, r, c, window) for r, c in specs
            if f'{r:g}/{c:g}' not in row['rows']])
        for radius, cosine in specs:
            key = f'{radius:g}/{cosine:g}'
            if key not in row['rows']:
                modes, metadata = self.modes(field, radius, cosine, window)
                kraus = hardware._kraus(modes, self.pulse, self.lifetimes)
                vertices = minimax._scenario_kraus([modes], self.pulse, self.lifetimes)
                row['rows'][key] = metadata | {
                    'position': shaped._fixed_correction_fidelity(kraus, row['correction']),
                    'vertices': minimax._fidelities_from_kraus(vertices, row['correction'])[0].tolist()}
                if window != 40:
                    fixed = self.nominal(field)['correction']
                    row['rows'][key].update(
                        fixed_same_field_40GHz_position=shaped._fixed_correction_fidelity(kraus, fixed),
                        fixed_same_field_40GHz_vertices=minimax._fidelities_from_kraus(vertices, fixed)[0].tolist())
                self.save()
                print('grid', field, key, flush=True)
        selected = [row['rows'][f'{r:g}/{c:g}'] for r, c in specs]
        row['full_grid' if full else 'endpoints'] = summarize(selected, row['nominal'])
        self.save()


def prepare_modes(key):
    # Workers build explicit models and return only small mode arrays.
    study = object.__new__(Study)
    study.cache = {}
    return study.modes(*key)


def field_comparison(data):
    """Compare the two fully validated alternatives and matched 60-GHz checks."""
    fields = data['fields']
    baseline = fields['3.10000000/40']['full_grid']
    comparisons = []
    for b in (3.085, 3.1025):
        candidate = fields[f'{b:.8f}/40']['full_grid']
        comparisons.append({'field_gauss': b, **candidate,
            'nominal_change': candidate['nominal']-baseline['nominal'],
            'joint_minimum_change': candidate['joint_minimum']-baseline['joint_minimum'],
            'endpoint_objective_change': candidate['endpoint_objective']-baseline['endpoint_objective']})
    convergence = []
    for b in (3.1, 3.1025):
        r = fields[f'{b:.8f}/60']
        endpoint_rows = [r['rows'][key] for key in ('0.05/-1', '0.05/1')]
        fixed_vertices = np.array([row['fixed_same_field_40GHz_vertices'] for row in endpoint_rows])
        convergence.append({'field_gauss': b,
            'nominal_fixed_40GHz_z': r['fixed_same_field_40GHz_nominal'],
            'joint_minimum_fixed_40GHz_z': float(np.min(fixed_vertices)),
            'nominal_recalibrated_60GHz_z': r['nominal'],
            'joint_minimum_recalibrated_60GHz_z': r['endpoints']['joint_minimum'],
            'nominal_window_change_fixed_z': r['fixed_same_field_40GHz_nominal']-fields[f'{b:.8f}/40']['nominal'],
            'joint_minimum_window_change_fixed_z': float(np.min(fixed_vertices))-fields[f'{b:.8f}/40']['full_grid']['joint_minimum']})
    gain60 = convergence[1]['joint_minimum_fixed_40GHz_z']-convergence[0]['joint_minimum_fixed_40GHz_z']
    return {'baseline': baseline, 'alternatives': comparisons, 'pair_window_checks': convergence,
        'robust_candidate_joint_change_at_60GHz_fixed_z': gain60,
        'recommendation': 'retain 3.10 G' if gain60 <= 0 else 'review materiality before any operating-point change',
        'interpretation': 'The 40-to-60 GHz differences are deterministic basis sensitivities, not statistical error bars. No pulse optimization was performed; this is not a joint pulse/field optimum.',
        'adoption_dependencies': ['P0-4 basis/sensitivity/convergence and fixed-Z calibration',
            'P1-1 grid/Sobol/local checks', 'P1-2 projection audit',
            'P1-4 static spectrum at changed field', 'all Fig. 3 curves and associated manuscript values']}


def plot_checkpoint():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(OUT.read_text())
    rows = sorted((r for r in data['fields'].values() if r['pair_window_ghz'] == 40),
        key=lambda r: r['field_gauss'])
    baseline = data['fields']['3.10000000/40']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, label in [('nominal', 'Nominal Z recalibrated at each B'),
                        ('fixed_3_10G_nominal', 'Z fixed at 3.10 G')]:
        axes[0].plot([r['field_gauss'] for r in rows], [100*r[name] for r in rows],
            '.-', label=label)
        zoom = [r for r in rows if 3.0 <= r['field_gauss'] <= 3.2]
        axes[1].plot([r['field_gauss'] for r in zoom],
            [1e6*(r[name]-baseline['nominal']) for r in zoom], '.-', label=label)
    robust = [r for r in rows if 'endpoints' in r or 'full_grid' in r]
    axes[2].plot([r['field_gauss'] for r in robust],
        [1e6*(r.get('endpoints', r.get('full_grid'))['joint_minimum']-
            baseline['full_grid']['joint_minimum']) for r in robust], '.-', label='Axial vertex minimum')
    full = [r for r in rows if 'full_grid' in r]
    axes[2].scatter([r['field_gauss'] for r in full],
        [1e6*(r['full_grid']['joint_minimum']-baseline['full_grid']['joint_minimum']) for r in full],
        marker='x', s=60, color='black', label='Full 19-geometry check')
    for ax in axes:
        ax.axvline(3.1, ls=':', color='gray')
        ax.set_xlabel('Magnetic field (G)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.axhline(0, color='gray', lw=.7)
    axes[0].set_ylabel('Nominal loss-aware overlap (%)')
    axes[1].set_ylabel('Nominal change from 3.10 G (×10⁻⁶)')
    axes[2].set_ylabel('Minimum change from 3.10 G (×10⁻⁶)')
    fig.suptitle('Fixed accepted pulse · final P0-4 model · 0.125 ns · carrier tracking\n'
        'Exploratory sampled scan; zero-T no-jump overlap, not CPTP fidelity', fontsize=11)
    destination = OUT.parents[1] / '.amp/in/artifacts/forster_p0_4_field_scan.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=float, nargs='+')
    parser.add_argument('--grid', choices=['none', 'endpoints', 'full'], default='none')
    parser.add_argument('--window', type=float, choices=[40., 60.], default=40.)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--assess-only', action='store_true')
    args = parser.parse_args()
    if args.assess_only:
        data = json.loads(OUT.read_text())
        data['assessment'] = field_comparison(data)
        OUT.write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(data['assessment'], indent=2))
        return
    if args.plot_only:
        plot_checkpoint()
        return
    study = Study()
    study.workers = args.workers
    study.grid(3.1, full=True)
    baseline = study.nominal(3.1)
    assert abs(baseline['nominal'] - .9993184594782827) < 1e-10
    assert abs(baseline['full_grid']['joint_minimum'] - .9986938231607946) < 1e-10
    study.data['baseline_reproduced'] = True
    fields = args.fields or sorted(set(np.arange(0, 5.01, .5).tolist()+[3.1]))
    study.prefetch([(field, 0., 0., args.window) for field in fields
        if f'{field:.8f}/{args.window:g}' not in study.data['fields']])
    for field in fields:
        study.nominal(field, args.window)
        if args.grid != 'none':
            study.grid(field, args.grid == 'full', args.window)
    study.save()


if __name__ == '__main__':
    main()
