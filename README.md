# Rb–Yb publication data and simulations

Numerical data and simulation code supporting **“Optically accessible
87Rb–171Yb Rydberg interactions and a Förster CZ gate.”** This repository is
the publication-focused record: it contains the reported numerical outputs,
the code that generates them, and exact atomic-database provenance. Plotting
code, rendered figures, exploratory searches, and superseded calculations are
intentionally omitted.

The exact manuscript snapshot used for this artifact map is identified by PDF
SHA-256 and source commit in `provenance/manuscript_manifest.json`; the PDF and
LaTeX sources are not redistributed here.

## Scientific scope

- **Förster channel:** full state assignment, fixed-\(m\) and all-\(m\)
  characterization, and the hardware-aware composite-revival CZ construction.
- **van der Waals channel:** a static, repulsive, blockade-compatible
  interaction resource and its all-\(m\) pair track.
- **Appendix B only:** a reduced two-level feasibility estimate derived from
  the static vdW branch. It is **not** a driven multichannel gate simulation,
  a vdW gate design, or a process-fidelity result.

## Paper-to-data map

| Paper item | Data | Reproducing script |
| --- | --- | --- |
| Figure 1 | `data/forster_characterization.json` | `scripts/reproduce_forster_characterization.py` |
| Figure 2 | schematic; no numerical data | not applicable |
| Figure 3 and Table I | `data/forster_gate_results.json` | `scripts/reproduce_forster_gate.py` |
| Figure 4 | `data/vdw_dense_data.json` | `scripts/reproduce_vdw_dense.py` |
| Table II | `data/prior_work.csv` | curated literature table |
| Appendix B and Table III | `data/vdw_feasibility_results.json`, `data/vdw_r6_deviation.csv` | `scripts/reproduce_vdw_feasibility.py` |

See [`data/README.md`](data/README.md) for field-level traceability.

## Environment

Python dependencies are locked with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

The calculations require PairInteraction 2.5.0 and one database profile: Rb
v1.2, Yb171_mqdt v1.4, and the common `misc` v1.4 Wigner table. This profile is
used consistently for the transition identities, fixed- and all-\(m\) Förster
characterization, the Förster gate, and all vdW results.

The database files are not redistributed here. Download and verify them using
the release URLs and SHA-256 values in
`provenance/pairinteraction_database_manifest.json`. PairInteraction's active
database directory can be printed with:

```bash
uv run python -c \
  'import pairinteraction as p; print(p.Database(download_missing=False).database_dir)'
```

Install the three archived releases with PairInteraction's database command:

```bash
uv run pairinteraction database download \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.2/Rb_v1.2.zip \
  https://github.com/pairinteraction/database-mqdt/releases/download/v1.4/Yb171_mqdt_v1.4.zip \
  https://github.com/pairinteraction/database-sqdt/releases/download/v1.4/misc_v1.4.zip
```

Then verify every required database table before calculating:

```bash
uv run python scripts/verify_pairinteraction_databases.py
```

The lockfile uses NumPy 2.3.5 because it is compatible with PairInteraction
2.5.0 on all supported platforms. The NumPy patch/minor version recorded in
each JSON is retained as provenance.

### Figure 1 consistency provenance

`forster_characterization.json` now derives the projected two-state model,
finite-basis bright-state weights and splitting, propagated populations,
distance/field/angle scans, and convergence checks from explicit complex
\(PP\) and \(SS\) amplitudes in the same Yb171_mqdt v1.4 Hamiltonian. At
\(R=3.4\,\mu\mathrm{m}\), the direct projection gives
\(|V|/h=15.401772\,\mathrm{MHz}\) and a generalized splitting of
30.813011 MHz. The 2411-state calculation gives a 31.109926 MHz bright-state
splitting and 97.286387% first-maximum transfer at 15.989779 ns, below its
97.309617% phase-independent unitarity bound. The record also contains the
three basis-convergence checks used to set the manuscript's reporting
precision.

The static all-\(m\) exchange scan and the driven-gate scan answer different
questions. The sampled static transfer is largest at 5 G (99.5437%), whereas
the fixed composite pulse was optimized at 3.10 G and its phase-recalibrated
gate scan peaks at 3.15 G (99.92953%); the same pulse gives 99.69967% at 5 G.
The latter scan reoptimizes local phases, not pulse parameters, so it does not
claim a global joint field-and-pulse optimum.

## Reproduction

Quick checks validate database hashes, state identities, basis construction,
and one operating point without overwriting committed data:

```bash
uv run python scripts/reproduce_forster_characterization.py --quick-check
uv run python scripts/reproduce_forster_gate.py --quick-check
uv run python scripts/reproduce_vdw_dense.py --quick-check
```

Regenerate all committed numerical records:

```bash
uv run python scripts/reproduce_forster_characterization.py
uv run python scripts/reproduce_forster_gate.py
uv run python scripts/reproduce_vdw_dense.py
uv run python scripts/reproduce_vdw_feasibility.py
uv run pytest
```

The feasibility calculation and tests complete in seconds once the dense data
exist. The full PairInteraction calculations are CPU- and memory-intensive:
the Förster characterization performs dozens of dense diagonalizations, while
the vdW calculation follows a 16,000–17,600-state branch with sparse
shift-invert eigensolves. Runtime depends strongly on BLAS and hardware.

`reproduce_forster_gate.py --optimize` optionally repeats local adaptive
Nelder–Mead refinement from the committed selected pulse. The default command
re-evaluates that pulse over the complete validation set and is the appropriate
deterministic data-regeneration command. The stored optimization used a 1 ns
step; the reported validation and committed record use a command-boundary-
aligned 0.125 ns maximum step and include the propagation-convergence series.

## Units and interpretation

Energies and frequencies are cyclic MHz or GHz (that is, divided by \(h\));
distances are micrometres; magnetic fields are gauss; and \(C_6/h\) is in
GHz·µm⁶. JSON records state their assumptions and model boundaries explicitly.

The Förster result is a modeled, zero-temperature no-jump construction, not a
laboratory process fidelity. The vdW result is a static interaction
calculation. Appendix B's reduced surrogate does not establish optically
driven multichannel dynamics.
