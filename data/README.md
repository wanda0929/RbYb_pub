# Data dictionary and traceability

All numerical records use repository-relative provenance paths. JSON numbers
are stored at full precision; values displayed in the manuscript are rounded.
`../provenance/manuscript_manifest.json` identifies the exact manuscript PDF
and source commit against which this map was prepared.

## `forster_characterization.json`

- `target_states.energy_exchange`: Figure 1(a) transition direction and signed
  `SS`-minus-`PP` defect; Rb absorbs and Yb releases energy.
- `fixed_m_model`: the explicit fixed-\(m\) basis, pair-energy window, target
  norms, and complex-amplitude projection convention.
- `operating_point.projected_two_state_model`: direct \(PP/SS\) projection,
  including \(V/h\), generalized detuned splitting, two-state transfer bound,
  and first-maximum time.
- `operating_point.bright_states`: Figure 1(b) full finite-basis bright-state
  energies and \(PP/SS/\)other weights at the captioned 3.4 µm distance.
- `operating_point.first_exchange_maximum`, `trajectory`, and
  `unitarity_transfer_upper_bound`: the full-Hamiltonian transfer quantities
  used to check that the weights and propagated population are consistent.
- `distance_scan`: Figure 1(c) full bright splitting and simultaneous
  first-maximum \(PP\), residual \(SS\), and spectator populations.
- `fixed_m_field_scan`, `fixed_m_angle_scan`, `all_m_field_scan`: Figure 1(d).
  `all_m_field_scan` is a static first-exchange diagnostic; its sampled 5 G
  maximum is not the objective used to choose the driven composite-gate field.
- `convergence`: pair-window, \(\Delta n\), and \(\ell_{\max}\) checks at the
  operating point, with explicit thresholds and changes from the primary
  calculation.
- `database`: versions and SHA-256 hashes of every PairInteraction table used.
- `validation`: machine-readable consistency conditions. All must be `true`.

The projected model, bright-state weights and splitting, full propagation,
and fixed-\(m\) scans all use the same Yb171_mqdt v1.4 Hamiltonian and explicit
complex \(PP\) and \(SS\) target amplitudes.

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
URLs, archive hashes, and installed-file hashes and sizes for the Rb v1.2,
Yb171_mqdt v1.4, and `misc` v1.4 profile used by all current calculations. The
characterization JSON embeds the same per-file hashes with each generated
record.
