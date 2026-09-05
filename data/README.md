# Data dictionary and traceability

All numerical records use repository-relative provenance paths. JSON numbers
are stored at full precision; values displayed in the manuscript are rounded.
`../provenance/manuscript_manifest.json` identifies the exact manuscript PDF
and source commit against which this map was prepared.

## `forster_characterization.json`

- `target_states.energy_exchange`: Figure 1(a) transition direction and signed
  SS-minus-PP electronic-centroid defect; Rb absorbs and Yb releases energy.
- `fixed_m_model`: explicit fixed-electronic-m basis, pair-energy window,
  target norms, and complex-amplitude projection convention.
- `operating_point.projected_two_state_model`: direct PP/SS projection,
  including coupling, generalized splitting, transfer bound, and first maximum.
- `operating_point.bright_states`: Figure 1(b) finite-basis bright-state
  energies and PP/SS/other weights at 3.4 µm.
- `operating_point.first_exchange_maximum`, `trajectory`, and
  `unitarity_transfer_upper_bound`: full finite-basis propagation quantities.
- `distance_scan`, `fixed_m_field_scan`, `fixed_m_angle_scan`, and
  `all_m_field_scan`: Figure 1(c,d) records. The final P0-4 axial HFS scan has
  transfer 0.9881256840811098 at 3.10 G and a sampled maximum
  0.992047614356116 at 4 G; `historical_all_m_field_scan` preserves the old
  search-basis scan.
- `convergence`: pair-window, Δn, and ℓmax checks at the operating point.
- `database`: database versions and SHA-256 hashes.

## `forster_gate_results.json`

This is the current Figure 3 record. All driven panels use the final P0-4
numerical-reference model; historical SCAN-basis diagnostics are nested and
explicitly excluded from the current plot.

- `parameters` and `command_segments`: selected symmetric five-segment pulse.
- `short_minimax`: final-reference nominal overlap 0.999318459478283 and
  sampled joint minimum 0.998693823160795.
- `response_time_scan`, `axial_position_scan`, `target_amplitude_scan`,
  `magnetic_field_scan`, and `population_trajectory`: Figure 3 diagnostics.
- `assumptions`: response, decay, geometry, polarization, and effective-drive
  model boundaries.

## `forster_p0_4_reoptimization.json`

Deterministic final-reference local refinement and exact validation. The
accepted candidate exhausted 400 iterations / 625 evaluations and is not a
converged or global optimum.

## `forster_p0_4_field_scan.json`

Checkpointed fixed-pulse field study, fine peak scans, full-grid comparisons,
and the pair-window check supporting retention of the 3.10 G operating field.

## `forster_p0_4_uncertainty_convergence.json`

Post-optimization fixed-pulse audit in the enlarged numerical-reference model:

- `reference_model_bounded_validation`: nominal overlap 0.999318459478283,
  finite-set position minimum 0.999202321496318, and joint position/amplitude
  minimum 0.998693823160795.
- `one_at_a_time_numerical_convergence`: pair window, Δn, atomic window, ℓmax,
  and interaction-order variants with spectrum, static-transfer, gate, and
  transient-spectator diagnostics.
- `bright_mode_cutoff_convergence` and `propagation_step_convergence`:
  projection and time-step audits.
- `spectroscopy_sensitivity_envelope`, `pure_target_pair_defect_scan`, and
  `axial_dc_electric_field_scan`: deterministic sensitivity calculations, not
  probabilistic confidence intervals.
- `interaction_order_scope`: the exact meaning of PairInteraction's orders 3,
  4, and partial 5.

## `forster_p1_1_robustness.json`

- `bounded_domain`: the 50 nm position ball and independent continuous ±1%
  effective-Rabi bounds.
- `nested_sobol`: deterministic 128, 256, 512, and 1024-point nested
  five-dimensional Sobol prefixes.
- `local_adversarial_refinement`: 12 response-surface starts and direct P0-4
  retained-block numerical-reference reevaluation of the leading candidates.
- `thermal_motion`: literature-based cross-platform scale benchmark. It is not
  a same-apparatus thermal gate prediction.
- `status`: explicit claim boundary; no global minimum is certified.

The direct local-adversarial recheck reaches 0.998693823160798, consistent
with the finite-set axial boundary value.

## `forster_p1_2_projection_audit.json`

- `points`: nominal and limiting axial cases at bright-mode cutoffs 10⁻⁶,
  10⁻⁸, and with all connected modes retained.
- `maximum_absolute_all_mode_projection_fidelity_difference`: the largest
  all-mode-minus-10⁻⁶ fidelity difference, 7.183481587347273 × 10⁻¹⁰.
- `angular_scope`: distinguishes this axial hyperfine-resolved audit from the
  separate electronic-only full-angular control.

## `forster_p1_4_reference_spectrum.json`

- `fixed_inputs`: the final P0-4 basis, operating point, 3684-state pair block,
  field-dressed asymptotes, and Förster defect.
- `spectral_diagnostics.target_eigenstates`: energies relative to the stretched
  SS asymptote, SS/PP/residual weights, and leading bare-pair components of the
  two target modes.
- `spectral_diagnostics.nearest_spectator_eigenstate`: the closest eigenstate
  to the target doublet and its composition.
- `spectral_diagnostics.largest_target_overlap_spectators`: the five spectator
  eigenstates with the largest SS+PP weight.

This is a direct diagonalization of the stated finite axial P0-4 block, not an
untruncated atomic Hilbert space or transverse full-angular calculation.

## `vdw_dense_data.json`

- `configuration.zero_field_c6_ghz_um6`: perturbative stretched-sector value
  used for the Figure 4 C6/R6 guide.
- `configuration.zero_field_c6`: zero-field M = −1, 0, +1 C6 matrices.
- `distance_track_b25_pp`: Figure 4 shift and bare-product weight.
- `sector_track_b25`: four magnetic-sector markers at 3.3 µm.
- `magnetic_field_track_r3p3_pp`: supporting 15–40 G static field scan.

This record is a basis-truncated static pair-Hamiltonian calculation, not a
vdW gate calculation.

## `vdw_p1_5_basis_convergence.json`

- `one_at_a_time_rows`: reference, contracted, and expanded Rb radial, Yb
  radial, ℓmax, and pair-window calculations. Every non-reference row changes
  exactly one numerical basis axis.
- `reference_reproduction`: comparison with `vdw_dense_data.json` at the
  published reference setting, including the numerical tolerance.
- `convergence.maximum_changes_over_expanded_rows`: maximum absolute and
  relative changes in C6, the 3.3 µm finite-field pair shift, and bare-pair
  weight over the four expanded rows.
- `resource_usage`: measured wall time and peak resident memory of the recorded
  serial run.

These checks establish one-axis finite-basis stability only. They do not test
simultaneous-expansion cross-terms, higher multipoles, driven vdW dynamics, or
process fidelity.

## `prior_work.csv`

Every cell of manuscript Table II, including publication type, explicit
`not reported` entries, and citation keys. It is curated literature data, not
simulation output.

## Database provenance

`../provenance/pairinteraction_database_manifest.json` gives upstream release
URLs, archive hashes, and installed-file hashes and sizes for the Rb v1.2,
Yb171_mqdt v1.4, and `misc` v1.4 profile. Generated records embed or reference
the same profile. `../provenance/runtime_environment.json` records the reference
operating system, numerical backend, locked validation environment, and seed
conventions.
