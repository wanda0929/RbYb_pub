# Data dictionary and traceability

All numerical records use repository-relative provenance paths. JSON numbers
are stored at full precision; values displayed in the manuscript are rounded.
`../provenance/manuscript_manifest.json` identifies the exact manuscript PDF
and source commit against which this map was prepared.

## `forster_characterization.json`

- `states`, `transitions`, `signed_defect_mhz`: Figure 1(a). The transition
  direction is from `SS` to `PP`: Rb absorbs and Yb releases energy.
- `figure1b_manuscript_reported_weights`: the rounded bare-pair weights printed
  in Figure 1(b), together with the captioned and reconstructed source
  distances. The values came from the 3.0 µm calculation even though the
  manuscript caption says 3.4 µm; the original raw eigensystem was not kept.
- `bright_eigenstates_at_3p0_um`, `bright_eigenstates_at_3p4_um`: independent,
  full-precision v1.2 reconstructions that expose that distance mismatch.
- `fixed_m_distance_scan`: Figure 1(c) splitting, first-maximum transfer,
  spectator leakage, and transfer time.
- `fixed_m_field_scan`, `fixed_m_angle_scan`, `all_m_field_scan`: Figure 1(d).
- `fixed_m_v1p4_recalculation_at_3p4_um`: a transparent recalculation of the
  archived fixed-\(m\) operating point with the current Yb v1.4 database. The
  plotted fixed-\(m\) scans themselves are the exactly reproduced v1.2 record.
- `database_provenance`: the separate manifests used for the archived fixed-
  \(m\) scans and for the v1.4 transition/all-\(m\) calculations.

## `forster_gate_results.json`

- `operating_point`, `parameters`, `command_segments`: Figure 3(a) and Table I.
- `assumptions`: response, decay, position, and effective-Rabi model boundaries.
- `short_minimax`: nominal and sampled worst-case Table I results, plus the
  complete amplitude-vertex fidelity grid.
- `propagation_convergence`: the 1, 0.5, 0.25, and 0.125 ns boundary-aligned
  convergence checks supporting the reported final propagation step.
- `response_time_scan`, `axial_position_scan`, `target_amplitude_scan`:
  Figure 3(b).
- `magnetic_field_scan`: phase-recalibrated Figure 3(c) data.
- `population_trajectory`: nominal no-jump Figure 3(d) data.
- `validation`: basis and geometry sampling metadata.

## `vdw_dense_data.json`

- `configuration.zero_field_c6_ghz_um6`: perturbative stretched-sector
  coefficient used for the Figure 4(a) \(C_6/R^6\) guide.
- `configuration.zero_field_c6`: full zero-field \(M=-1,0,+1\) C6 matrices.
- `distance_track_b25_pp`: Figure 4(a,b) static shift and bare-product weight.
- `sector_track_b25`: the four Figure 4(a) magnetic-sector markers at 3.3 µm.
- `magnetic_field_track_r3p3_pp`: additional 15–40 G support for the prose
  stability statement; it is not plotted in Figure 4.

This file is a static pair-Hamiltonian result, not a vdW gate calculation.

## `vdw_feasibility_results.json`

Appendix B and Table III's reduced two-level screening estimate. The
`four_magnetic_sectors` array is in manuscript order and reports `u_mhz`,
`delta_eff_mhz`, `reduced_model_average_fidelity`,
`mean_computational_survival`, and
`lossless_reduced_model_average_fidelity`. These are surrogate-model estimates,
not a driven multichannel process fidelity or a second gate design.

`pair_track` adds the inexpensive reduced estimate to each exact static vdW
point. `distance_sweep` is PCHIP-interpolated between exact points and is a
guide only.

## CSV files

- `vdw_r6_deviation.csv`: pointwise deviation of the dense finite-field branch
  from the zero-field \(C_6/R^6\) guide.
- `prior_work.csv`: every cell of manuscript Table II, including explicit
  `not reported` entries and citation keys. It is curated data, not simulation
  output.

## Database provenance

`../provenance/pairinteraction_database_manifest.json` gives upstream release
URLs, archive hashes, and installed-file hashes and sizes for the primary Rb
v1.2 plus Yb171_mqdt v1.4 profile.
`../provenance/pairinteraction_database_manifest_figure1_fixed_m.json` records
the separate Rb v1.2 plus Yb171_mqdt v1.2 profile that exactly reproduces the
archived fixed-\(m\) Figure 1 scans. The characterization JSON discloses the
resulting version drift and includes a v1.4 recalculation at 3.4 µm.
