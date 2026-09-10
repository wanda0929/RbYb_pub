# Code-availability audit: core results covered, some claims still unsupported

Audited on 2026-09-10 against the local modified `draft-RbYb/main.tex`, SHA-256
`b620bdebe7c2db4a268e17ed5c7d1fd85a73d6774be0cf4e0ddff09a4e391658`.
This is not a submitted-version certificate: the final manuscript must be
frozen and its PDF/source hashes recorded before release.

There are **five figures and five tables**: main-text Figs. 1–4 and Table I;
appendix Fig. 5 and Tables II–V. There is no literature-comparison table in this
revision. The prior Table-II CSV has been removed rather than mislabeled as
the current state dictionary.

## Figure and table data map

Below, `A/` means `data/manuscript_claims_2026_09_10/simulations/` and ordinary
data paths begin at the repository root. Figure numbers identify scientific
subjects, not a claim that every schematic's graphical layout is identical to
the latest privately edited PDF. Drawing coordinates are not physical data.

| Item (main.tex lines) | Final serialized values and exact fields | Generator / status |
| --- | --- | --- |
| Fig. 1 (100–115), channel characterization | `data/forster_characterization.json`: `target_states.energy_exchange`, `operating_point.bright_states`, `distance_scan`, `fixed_m_field_scan`, `all_m_field_scan.points`, `fixed_m_angle_scan` | `scripts/reproduce_forster_characterization.py`; `scripts/plot_channel_forster.py --characterization-only`. Use final HFS scan, not `historical_all_m_field_scan`. |
| Fig. 2 (202–211), excitation paths | Schematic; states/wavelengths in `scripts/plot_candidate_excitation.py` and `data/working_state_inputs.json` | No physical curve required. Does not validate optical selectivity. |
| Fig. 3 (232–242), corrected gate | `A/forster_control_decay_recheck.json`: `pulse_parameters`, nominal `rows` Kraus amplitudes and `fixed_local_z_rad`; `A/forster_control_decay_driven_curves.json`: `constraints`, `population_trajectory` (1286 points) | `A/evaluate_forster_control_decay.py`, `A/replay_control_decay_audits.py curves`; `scripts/plot_control_decay_gate.py`. |
| Fig. 4 (376–387), robustness | `A/forster_control_decay_driven_curves.json`: `response_time_scan`, `target_amplitude_scan`; `A/forster_control_decay_p1_1.json.response_grid`; `A/forster_control_decay_field_scan.json.fields` (40 GHz rows) | `A/replay_control_decay_audits.py curves/p1/field`; corrected renderer. |
| Fig. 5 (763–774), static vdW | `data/vdw_dense_data.json`: `distance_track_b25_pp`, `sector_track_b25`, `configuration.zero_field_c6_ghz_um6` | `scripts/reproduce_vdw_dense.py`, `scripts/plot_channel_vdw.py`. No redundant `vdw_gate_distance_sweep.json` or driven vdW gate required. |
| Table I (319–349), pulse and performance | Corrected `recheck`: `pulse_parameters`, `basis`, `field_gauss`, `summary`, nominal row; `driven_curves.constraints`; Rb 5 MHz, 100 ns, 50 nm and ±1% are prescribed settings | Same corrected gate generators. Conditional overlap = fidelity / mean computational survival. |
| Table II (488–519), state dictionary | `data/working_state_inputs.json` transcribes all five roles with field-level distinction between quoted inputs and archived lifetime results; four Förster lifetimes in `A/forster_control_decay_recheck.json.lifetimes_us` | **Partial**: selectors and four lifetimes have model support; g factors/dipoles/vdW lifetimes lack a complete independent query record. Transcription is not validation. |
| Table III (587–610), 56 convergence cells | `A/forster_control_decay_p0_4.json.one_at_a_time_numerical_convergence`: connected block size, `target_bright_splitting_mhz`, `minimum_target_subspace_weight`, `static_transfer.maximum_ss_population`, `fixed_pulse_population.maximum_transient_spectator_population`, `gate` fixed/recalibrated fidelities | `A/replay_control_decay_audits.py p0`, not the historical root P0 default. Probabilities multiplied by 100 for percent columns. |
| Table IV (639–658), return-loss budget | Nominal corrected `recheck.rows` Kraus amplitudes; `driven_curves.population_trajectory` for 49.4 ns residence | For each input, L = 1 − abs(k)² and contribution L/4; mean loss = 1 − mean(abs(k)²); conditional defect = mean(abs(k)²) − F; infidelity = 1 − F. This exact return budget is not an exact radiative species budget. |
| Table V (806–825), 20 vdW convergence cells | `data/vdw_p1_5_basis_convergence.json.one_at_a_time_rows`: reference + four expanded rows, `pair_basis_size`, `c6_ghz_um6`, `working_point_shift_mhz`, `bare_pair_weight` | `scripts/evaluate_vdw_p1_5.py`. N is the full pair basis, not the 8399-state target component. Contracted rows also retained. |

`tests/test_code_availability.py` checks the displayed numerical convergence
cells and derived return budget against the stored full-precision records.
Existing tests check channel results, pulses, input integrity and method
regressions. Tests of data consistency are not independent physics replication.

## Supporting results that must not be pruned

- Appendix A exchange bounds, angular scans and convergence:
  `data/forster_characterization.json`; hyperfine constants in
  `scripts/rb_rydberg_hyperfine.py`.
- Appendix B spectrum: `data/forster_p1_4_reference_spectrum.json` and
  `scripts/evaluate_forster_p1_4.py`.
- Corrected P0 spectroscopy, pair defect and axial electric-field arrays;
  corrected P1 Sobol/adversarial/thermal arrays; field and tight-step records.
- All eleven off-axis JSON files and their generator/report. Three-/five-sector
  and 256-/512-mode comparisons are convergence evidence, not duplicate runs.
  The interrupted 256-mode checkpoint is explicitly distinguished from the
  completed 512-mode result. These are frozen points, not a thermal average.
- Zero-field C6 matrices and nearest-partner detunings inside
  `data/vdw_dense_data.json.configuration.zero_field_c6`, not just plotted U(R).
- The frozen archive's source/import closure. Files named `optimize` also
  define the selected pulse and helpers. The older root implementations remain
  dependencies of characterization, reference-spectrum and vdW audits/tests.

## Remaining evidence gaps: do not claim all findings are fully archived

| Claim | Evidence boundary / missing artifact |
| --- | --- |
| Appendix A 3×3 survey and 24 candidates at n=45–70 (449–450) | No traced candidate table and replayable selection manifest established in this audit. |
| Yb P purity 78.6% / 53% at 0.5 / 1 V/cm (456) | Missing single-atom Stark scan and producer. |
| Omitted F-state stress <2×10⁻¹⁴ (484) | Historical SCAN evidence only in `data/forster_gate_results.json.historical_scan_basis_diagnostics.record.rb_hyperfine_validation.omitted_f_state_stress_test`; not a corrected rerun. |
| Table II atomic g/dipole/vdW lifetime/term-energy claims, 301.8699 nm and 70.06/14.04 MHz estimates | Quoted inputs retained, but independent state-resolved atomic query output/producer incomplete. |
| Förster ±100 GHz spectator excitation <1.3×10⁻⁵ | Missing line list, pulse definition and single-atom scan. Power conversion does not validate dipoles or selectivity. |
| Full transformed-loss comparison <6×10⁻⁵ (628) and all spectator lifetimes bounded by proxy (629) | Only same-trajectory post-hoc diagnostic; no full-D propagation or per-state lifetime table. |
| Exact radiative/species and residual-excitation decomposition (662–663) | Available reconstruction is approximate. Table IV's exact computational-return loss is separately supported. |
| Thermal benchmark geometry (715–722) | Stored thermal-motion input uses its recorded trap-axis convention; changing the weak-axis orientation in prose is not a new propagated result. |
| Three-dimensional electric-field robustness (733–736) | Archived electric-field scan is axial only. |
| 0.1/0.2 MHz penalties (755), multi-MHz detuning/field recovery (757) | Curvature estimate and proposed recalibration, not direct archived propagation/recovery studies. |
| Appendix C ±100 GHz optical S/D scan (837–838): 2.36 GHz nearest spectator, 1.1×10⁻⁵ excitation, 1.52×10⁻³ opposite-clock excitation, −0.20 rad phase | Missing field-dressed optical scan/producer. Static vdW pair results cannot substitute for it. |

These gaps require either actual calculations and their records or changes to
the manuscript claims. Removing files cannot close them. See
[support notes](MANUSCRIPT_SUPPORT_NOTES.md) for the recovered diagnostic formulas.

## Reproduction and environment

- `uv sync --frozen` uses `pyproject.toml`/`uv.lock`; original per-run versions
  remain in the records. NumPy 2.3.5 is the locked compatibility environment,
  not the original 2.4.6 execution environment. Runtime/backend and seeds are
  recorded in `provenance/runtime_environment.json`.
- `provenance/pairinteraction_database_manifest.json` identifies PairInteraction
  2.5.0, Rb v1.2, Yb171_mqdt v1.4 and misc v1.4, with release URLs, archive
  SHA-256 and all 13 installed-file hashes. `verify_pairinteraction_databases.py`
  verifies the installed tables before the quick physics check.
- `reproduce_all.py --quick-check` checks corrected data, then freshly
  rebuilds the nominal 3684-state P0-4 Hamiltonian using the archived implementation
  and propagates nominal plus
  amplitude vertices and the population trajectory. It uses fixed published
  local-Z phases and compares complex amplitudes, populations and fidelities
  at absolute tolerance 10⁻⁸; it does not optimize or overwrite archives.
  The initial orb run took about 43 seconds (two BLAS threads), giving
  0.9991257847691977 versus stored 0.999125784769183. Runtime is not guaranteed
  on other hardware, and this one-point test is not the full convergence audit.
- `reproduce_all.py --figures` checks and renders all five figure subjects
  from their stored final data; no physics is rerun. Individual scripts and
  expensive corrected replay commands are documented in the frozen archive.
  Always use a disposable copy for replay stages that write their output paths.
- `--historical-full` explicitly names the older pre-correction replay. It is
  not the revised gate reproducer. The ambiguous old `--full` and redundant
  `--corrected-figures` options have been removed.

## Pruned material and release boundary

Removed the separate, non-manuscript-selected local-convergence experiment
(3466 optimizer evaluations, candidate, runner and experiment-only tests),
the unused plotting compatibility wrapper, and the obsolete literature table.
Retained the three control-decay method regressions and all claim-supporting
convergence data. Removed tracked files remain recoverable from Git history.

At audit time the GitHub repository was private. Code availability for readers
still requires a public release or a journal-approved reviewer-access mechanism,
a frozen manuscript version and preferably a persistent archival identifier.
No visibility change, release, DOI, or push is performed by these scripts.
