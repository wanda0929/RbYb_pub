# Rb–Yb publication data and simulations

Code and numerical records supporting **“87Rb–171Yb Rydberg interactions and a
Förster CZ gate.”** The corrected gate has nominal no-jump overlap
0.999125784769183 and sampled joint minimum 0.9984994300039455. These are not
CPTP laboratory process fidelities.

**Availability status: core results covered, not every manuscript claim.**
The [code-availability audit](docs/CODE_AVAILABILITY.md) maps all five figures
and five tables in the inspected manuscript to their data and generators,
and explicitly lists missing optical/atomic/dissipation validation artifacts.
The current Table II is a state dictionary, not a literature-comparison table.
Do not describe this package as supporting *all findings* until those gaps
and the final submitted manuscript version are resolved.

## Install the locked environment and physical inputs

```bash
uv sync --frozen
uv run pairinteraction database download \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.2/Rb_v1.2.zip \
  https://github.com/pairinteraction/database-mqdt/releases/download/v1.4/Yb171_mqdt_v1.4.zip \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.4/misc_v1.4.zip
uv run python scripts/verify_pairinteraction_databases.py
```

The downloader may ask for confirmation for each installed release. Use a
disposable database cache if you need to preserve a different profile. The
required profile is PairInteraction 2.5.0, Rb v1.2, Yb171_mqdt v1.4 and misc
v1.4. [Database provenance](provenance/pairinteraction_database_manifest.json)
records release URLs, archive hashes and all 13 installed-file SHA-256 hashes.

`pyproject.toml` and `uv.lock` pin dependencies. The records retain their
original NumPy 2.4.6 provenance; the tested lock uses compatible NumPy 2.3.5.
[Runtime provenance](provenance/runtime_environment.json) records Python,
BLAS/LAPACK, platform and seeds. A compatible lock is not a claim that the
original and present execution environments are byte-identical.

## Reviewer commands

```bash
# Archive, numerical-method and displayed-table tests (no large physics audit)
uv run python scripts/reproduce_all.py

# Actual nominal Hamiltonian reconstruction and corrected pulse propagation
uv run python scripts/reproduce_all.py --quick-check

# Render all five current figure subjects from stored final data
uv run python scripts/reproduce_all.py --figures

# Print recovered optical-power/post-hoc loss estimates as JSON
uv run python scripts/recompute_claim_diagnostics.py
```

The quick check rebuilds the nominal 3684-state axial hyperfine Hamiltonian,
retains 52 bright modes, and compares complex return amplitudes, populations
and fidelities with the corrected reference at absolute tolerance 10⁻⁸.
The initial orb run took about 43 seconds with two BLAS threads. It does not
optimize, reuse a model checkpoint, or write numerical archives. This is a
nominal physics check, not a global robustness or full convergence certificate.

The figure command uses corrected gate/robustness data, not the old Figure-3
record. Figures are generated under `figures/`; PDF/PNG outputs are not
versioned. Schematic drawing coordinates do not constitute simulation data.

## Data and reproduction map

- [All figures/tables and remaining gaps](docs/CODE_AVAILABILITY.md).
- [Corrected gate, convergence, sensitivity and off-axis archive](data/manuscript_claims_2026_09_10/README.md): byte-preserved data and minimal import closure, with explicit costly replay commands. Run writing workflows in a disposable copy.
- [Data dictionary](data/README.md): channel characterization, reference spectrum, static vdW matrices/scans/convergence and exact record fields.
- [Recovered support notes](docs/MANUSCRIPT_SUPPORT_NOTES.md): power conversion, post-hoc loss diagnostic and approximate species budget, with source excerpts and hashes. None is a full non-diagonal loss propagation or optical spectator scan.
- [Independent verification settings](docs/INDEPENDENT_VERIFICATION_GUIDE.md) and [robustness method](docs/P1_1_ROBUSTNESS_METHOD.md). Historical pre-correction numbers in the guide are not current gate claims; use the corrected archive map.

The Förster model is a zero-temperature, finite-basis, hardware-response-aware
no-jump calculation. The vdW channel is a static pair-Hamiltonian calculation,
not a gate design or process-fidelity result. Excitation schematics are not
complete multilevel laser-error budgets. Frequencies are cyclic MHz/GHz,
distances μm, magnetic fields G, and C6/h is in GHz·μm⁶.

## Historical dependencies and expensive replay

The old root P0/P1/gate records and adapted modules remain because they carry
selected-pulse provenance, historical controls, and dependencies of the channel,
spectrum and vdW workflows. The corrected snapshot preserves its own import
closure; similarly named modules must not be interchanged. The separate
non-manuscript-selected optimization experiment and obsolete literature table
have been removed; claim-supporting convergence records and method tests remain.

`uv run python scripts/reproduce_all.py --historical-full` explicitly runs the
older pre-correction replay. It is **not** a current gate reconstruction; use
`--quick-check` or the corrected archive's replay instructions instead. Add
`--rebuild-figure1` only with `--historical-full` to rebuild the old fixed-m scan.
Full audits can require a 33,180-state pair basis and roughly 10.1 GiB RSS;
the recorded vdW convergence run took about 12.5 minutes. Do not confuse
replotting committed data with rerunning these calculations.

## Publication release

The original PDF/source snapshot is identified by
`provenance/manuscript_manifest.json`; the corrected archive has its own
manifest. Neither automatically identifies the latest locally edited draft.
Follow [the release checklist](RELEASE_CHECKLIST.md) to freeze the submitted
version, resolve coverage gaps and arrange public or reviewer access. At the
last audit the repository was private. No release or DOI is assumed to exist;
cite an actual archived version/commit, not an unissued DOI.
