# Rb–Yb publication data and simulations

Numerical records and simulation code supporting **“87Rb–171Yb Rydberg
interactions and a Förster CZ gate.”** This repository contains the reported
numerical outputs, the scripts that produce or plot them, the settings needed
for independent verification, and exact atomic-database provenance.

The manuscript snapshot covered by this archive is identified by PDF SHA-256
and source commit in `provenance/manuscript_manifest.json`. The manuscript PDF
and LaTeX sources are maintained in the separate manuscript repository.

## Scientific scope

- **Förster channel:** fixed-electronic-m and hyperfine-resolved all-mode
  characterization of the specified SS ↔ PP pair.
- **Förster gate:** a zero-temperature, non-Hermitian no-jump,
  hardware-response-aware composite-revival CZ candidate. The committed
  records include the finite validation set, post-optimization numerical and
  atomic-structure sensitivity audit, continuous-domain search, thermal-scale
  benchmark, all-mode projection audit, and the detailed spectrum of the final
  axial numerical-reference block.
- **Van der Waals channel:** a basis-truncated static pair shift and bare-pair
  weight with one-at-a-time Rb radial, Yb radial, angular-momentum, and
  pair-window convergence checks. It is not a gate design or process-fidelity
  result.
- **Excitation routes:** selection-rule-allowed candidate routes. They do not
  constitute complete multilevel laser-error budgets.

The pulse-search record (`data/forster_gate_results.json`) uses the less
expensive hyperfine-resolved search model documented in the manuscript. The
post-optimization reference values are in
`data/forster_p0_4_uncertainty_convergence.json`; these use the enlarged
numerical-reference basis and are the values used for the manuscript's
headline nominal and bounded results. These two records answer different
questions and should not be interchanged.

## Paper-to-archive map

| Paper item | Data or settings | Reproducing or plotting script |
| --- | --- | --- |
| Figure 1 | `data/forster_characterization.json` | `scripts/reproduce_forster_characterization.py`, `scripts/plot_channel_forster.py` |
| Figure 2 | candidate-route settings in the manuscript and verification guide | `scripts/plot_candidate_excitation.py` |
| Figure 3 and Table I pulse-search diagnostics | `data/forster_gate_results.json` | `scripts/reproduce_forster_gate.py`, `scripts/plot_forster_gate.py` |
| Post-optimization convergence and sensitivity audit | `data/forster_p0_4_uncertainty_convergence.json` | `scripts/evaluate_forster_p0_4.py` |
| Continuous-domain and thermal-scale audit | `data/forster_p1_1_robustness.json` | `scripts/evaluate_forster_p1_1.py` |
| All-mode projection audit | `data/forster_p1_2_projection_audit.json` | `scripts/evaluate_forster_p1_2.py` |
| Final axial reference-block spectrum | `data/forster_p1_4_reference_spectrum.json` | `scripts/evaluate_forster_p1_4.py` |
| Figure 4 | `data/vdw_dense_data.json` | `scripts/reproduce_vdw_dense.py`, `scripts/plot_channel_vdw.py` |
| vdW one-at-a-time basis convergence | `data/vdw_p1_5_basis_convergence.json` | `scripts/evaluate_vdw_p1_5.py` |
| Table II | `data/prior_work.csv` | curated literature table |
| Independent replication settings | `docs/INDEPENDENT_VERIFICATION_GUIDE.md` | verifier chooses an independent implementation |

See `data/README.md` for field-level traceability and
`docs/P1_1_ROBUSTNESS_METHOD.md` for the robustness protocol.

## Environment

Python dependencies are locked with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

The calculations require PairInteraction 2.5.0 and the Rb v1.2, Yb171_mqdt
v1.4, and `misc` v1.4 database releases. Release URLs and exact SHA-256 hashes
are recorded in `provenance/pairinteraction_database_manifest.json`. The
reference operating system, Python version, BLAS/LAPACK backend, and all random
or quasi-random seed conventions are recorded in
`provenance/runtime_environment.json`. Install the archived releases and verify
every table before recalculation:

```bash
uv run pairinteraction database download \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.2/Rb_v1.2.zip \
  https://github.com/pairinteraction/database-mqdt/releases/download/v1.4/Yb171_mqdt_v1.4.zip \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.4/misc_v1.4.zip
uv run python scripts/verify_pairinteraction_databases.py
```

The JSON files retain the NumPy version used to generate each record. The
lockfile uses NumPy 2.3.5 because it is compatible with PairInteraction 2.5.0
on all supported platforms; this patch-version difference does not alter the
stored provenance.

## One-command workflows

Run the fast numerical-method, publication-data, and provenance tests:

```bash
uv run python scripts/reproduce_all.py
```

Regenerate the four manuscript figures from the committed numerical records:

```bash
uv run python scripts/reproduce_all.py --figures
```

Recompute every numerical record, regenerate the figures, and run the tests:

```bash
uv run python scripts/reproduce_all.py --full
```

The full workflow is expensive. It performs repeated dense diagonalizations
for the Förster calculations and vdW sparse shift-invert eigensolves up to a
33,180-state pair basis. The recorded vdW convergence run took about 12.5
minutes and reached approximately 10.1 GiB peak RSS on a 16-logical-CPU Linux
host. Runtime and peak memory depend strongly on the BLAS implementation and
hardware. The default command deliberately tests the committed archive without
overwriting it.

`scripts/reproduce_forster_gate.py --optimize` optionally repeats local
adaptive Nelder–Mead refinement from the committed pulse. The default gate
command deterministically reevaluates the selected pulse. The stored search
used a 1 ns step; the committed validation uses a command-boundary-aligned
0.125 ns maximum step.

## Units and interpretation

Energies and frequencies are cyclic MHz or GHz (divided by h); distances are
micrometres; magnetic fields are gauss; and C6/h is in GHz·µm⁶. JSON records
state their assumptions and model boundaries explicitly.

The Förster result is a modeled no-jump overlap, not a trace-preserving
laboratory process fidelity. Lost norm is not assigned to state-resolved jump
branches or assumed to be erasure. The vdW result is a static pair-Hamiltonian
calculation only.

## Versioned release and archival DOI

`RELEASE_CHECKLIST.md` lists the maintainer actions needed to publish a stable
version, create a GitHub release, and optionally mint a Zenodo DOI. Until a
release and DOI exist, cite the repository URL together with the commit hash
used for the analysis; do not cite an unissued DOI.
