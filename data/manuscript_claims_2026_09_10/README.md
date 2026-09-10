# Code and data supporting the revised manuscript's gate claims

This is an archival import of **existing calculations**, not a new calculation
or pulse optimization. All 41 imported files are byte-identical to the source
[manuscript snapshot](https://github.com/wanda0929/draft-RbYb/tree/fd728db62804157051467b32ed4ef4a9184e29da).
[manifest.json](manifest.json) identifies that snapshot, its LaTeX/PDF hashes,
and the size and SHA-256 of each imported file. The code and numerical data
needed for the workflows below are included here; access to the manuscript
repository is not needed to read this archive.

## Which records support which claims?

Paths below are relative to this directory. The seven corrected-decay records
keep the selected pulse fixed and include Rb 56S decay in the control-excited
state during the target window. They are distinct from the older pre-correction
publication records. The non-selected local-optimization experiment is excluded
from this publication package; no imported file in the 41-file manifest changed.

| Claim or diagnostic | Existing record under `simulations/` | Fields / source |
| --- | --- | --- |
| Nominal gate fidelity 99.91%, sampled joint minimum 99.85% | [forster_control_decay_recheck.json](simulations/forster_control_decay_recheck.json) | `summary`, 19 geometry `rows`; `evaluate_forster_control_decay.py` |
| Basis, multipole, spectroscopy and axial electric-field sensitivity | [forster_control_decay_p0_4.json](simulations/forster_control_decay_p0_4.json) | `one_at_a_time_numerical_convergence`, `spectroscopy_sensitivity_envelope`, `pure_target_pair_defect_scan`, `axial_dc_electric_field_scan` |
| Dense-grid, Sobol and local adversarial robustness checks | [forster_control_decay_p1_1.json](simulations/forster_control_decay_p1_1.json) | `response_grid`, `nested_sobol`, `local_adversarial_refinement`; not a continuous-domain certificate |
| Axial bright-mode projection check | [forster_control_decay_p1_2.json](simulations/forster_control_decay_p1_2.json) | `points`, `maximum_absolute_all_mode_projection_fidelity_difference`; not an all-M off-axis test |
| Magnetic-field sensitivity | [forster_control_decay_field_scan.json](simulations/forster_control_decay_field_scan.json) | `fields`, `runs`, `assessment` |
| Finer integration steps | [forster_control_decay_tight_steps.json](simulations/forster_control_decay_tight_steps.json) | `rows` at 0.125, 0.0625 and 0.03125 ns |
| Finite control response, amplitude scans and population trajectories | [forster_control_decay_driven_curves.json](simulations/forster_control_decay_driven_curves.json) | `response_time_scan`, `target_amplitude_scan`, `population_trajectory`, `population_summary`, `constraints` |
| Finite-sector off-axis corrections | [forster_off_axis/](simulations/forster_off_axis/) | Eleven JSON records; [report](simulations/forster_off_axis/REPORT.md) and [method](simulations/forster_off_axis/README.md) |

The six corrected audit stages after the baseline are implemented in
[replay_control_decay_audits.py](simulations/replay_control_decay_audits.py).
Its imported model and propagation modules are included unchanged. The off-axis
generator is [evaluate_forster_off_axis.py](simulations/evaluate_forster_off_axis.py).
Names containing `optimize` in the dependency closure do not mean an optimization
was performed for this import: those modules also hold the fixed pulse and helpers.

The corrected baseline's exact nominal fidelity is **0.999125784769183**;
its position-only minimum is **0.9990080680690021** and joint sampled minimum is
**0.9984994300039455**. Its nominal fixed local-Z angles are
[-3.1407724432579953, 2.941141606972484] rad. Do not substitute the optimized
non-selected candidate's phases or the historical archive's 0.999318459478283 nominal value.

## Off-axis records and limits

- `axial.json`, `axial_inward.json`: axial reference and inward 50 nm point.
- `ball_max_angle_projected.json`, `ball_max_angle_m1.json`,
  `ball_max_angle_m2.json`, `ball_max_angle_m2_512.json`: matched 50 nm
  maximum-angle point in one, three and five total-M sectors. The `m2` file
  contains a completed 256-mode checkpoint from an interrupted run; the
  separate `m2_512` file is the completed 512-mode follow-up. Both are retained.
- `ball_mixed_projected.json`, `ball_mixed_m1.json`: transverse 30 nm and
  axial -40 nm, in one and three sectors.
- `thermal_projected.json`, `thermal_m1.json`, `thermal_m2.json`: transverse
  266 nm frozen displacement, about 4.47344 degrees, in one, three and five sectors.

The nominal five-sector results are 0.9991222471242031 at the maximum-angle
ball point and 0.9990343729665522 at the thermal-scale point. Differences from
their respective projections are about -3.54 × 10⁻⁶ and -8.81 × 10⁻⁵.
The five-sector thermal-scale calculation is not an all-M convergence result
or a three-dimensional thermal average. The report separates spectral,
time-step and sector-count checks and documents the population convention.

**Numerical precision and scope:** the maximum-angle correction magnitude is
3.5402568 × 10⁻⁶ at nominal amplitude, or 3.5676584 × 10⁻⁶ with both amplitudes
at 0.99. Thus about 3.6 × 10⁻⁶ describes these tested points. These discrete
diagnostics are not a strict bound on the entire ball or a uniform
coupling/energy-gap perturbative bound.

## Preservation, dependencies and verification

The original `simulations/` layout is deliberate. It preserves imports, relative
data paths and source bytes without mixing in the adapted historical `scripts/`
implementations. The imported `figures/style.py` and `figures/pulseviz/__init__.py`
complete the plotting-helper imports used by those original simulation modules.
The two additional input records are
`forster_characterization.json` (database provenance) and
`forster_p0_4_uncertainty_convergence.json` (historical calibration comparison).
They are inputs, not substitutes for the corrected-decay outputs.

Original per-run source hashes and timestamps have not been rewritten. Some runs
predate the final source snapshot, notably off-axis CLI/memory refinements;
the final snapshot is not claimed byte-identical to every earlier source hash.
The corrected-audit provenance also lists scripts outside this bounded import
closure; those lists are historical records, not a declaration that every listed
script is needed or included here. The import manifest separately checks the
exact files actually archived.

The original off-axis README's commands assume this directory as the working
root. They are optional, expensive replay instructions, **not required to use
or validate the published results**. Some archived replay stages write to their
original result paths: use a disposable copy for any future recalculation.
The original version/database requirements remain documented in the records
and README. No database installation, eigensolve, propagation or optimization
is performed by the archive-integrity check:

```bash
uv run python -m pytest tests/test_manuscript_claims_archive.py
```

Run this command from the publication repository root. It checks hashes,
recorded values, completion status, source syntax and local import closure without
importing or executing the archived numerical code. Source-to-archive byte equality
was checked at import; future offline tests check archive-to-manifest consistency.
Existing publication figures and
historical replay defaults are intentionally unchanged.
