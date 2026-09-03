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

The calculations require PairInteraction 2.5.0 and two database profiles:

- **Primary profile:** Rb v1.2 and Yb171_mqdt v1.4 for transition identities,
  all-\(m\) Förster characterization, the Förster gate, and all vdW results.
- **Archived Figure 1 fixed-\(m\) profile:** Rb v1.2 and Yb171_mqdt v1.2 in a
  separate database root.

The database files are not redistributed here. Download and verify them using
the release URLs and SHA-256 values in the two database manifests under
`provenance/`.
Extract the primary profile under PairInteraction's active `tables` directory,
which can be printed with:

```bash
uv run python -c \
  'import pairinteraction as p; print(p.Database(download_missing=False).database_dir)'
```

Create an isolated directory for the archived profile. Each downloaded ZIP
contains its versioned directory, so extraction is:

```bash
FIXED_M_DB=/path/to/figure1-v1p2
mkdir -p "$FIXED_M_DB/tables"
unzip Rb_v1.2.zip -d "$FIXED_M_DB/tables"
unzip Yb171_mqdt_v1.2.zip -d "$FIXED_M_DB/tables"
```

After extraction, verify every required database table before calculating:

```bash
uv run python scripts/verify_pairinteraction_databases.py
uv run python scripts/verify_pairinteraction_databases.py \
  --manifest provenance/pairinteraction_database_manifest_figure1_fixed_m.json \
  --database-dir "$FIXED_M_DB"
```

The lockfile uses NumPy 2.3.5 because it is compatible with PairInteraction
2.5.0 on all supported platforms. The NumPy patch/minor version recorded in
each JSON is retained as provenance.

### Figure 1 database provenance

A provenance audit found that the archived fixed-\(m\) distance, field, and
angle scans in Figure 1 are reproduced exactly with Yb171_mqdt v1.2, rather
than v1.4. For example, v1.2 gives the reported 30.799763 MHz splitting and
99.689270% transfer at 3.4 µm; the otherwise identical v1.4 recalculation gives
31.109927 MHz and 97.286139%. `forster_characterization.json` preserves the
reported v1.2 scans, includes the current-v1.4 operating-point recalculation,
and identifies both manifests. The transition identities, all-\(m\) data,
Förster gate, and vdW calculations use v1.4.

The Figure 1(b) percentages require a second distinction. The displayed
54.4/45.2 and 45.4/54.3 percent PP/SS weights reproduce the archived v1.2
calculation at 3.0 µm, although the manuscript caption assigns them to 3.4 µm.
At the captioned distance the reconstructed weights are approximately
50.74/49.11 and 49.18/50.66 percent. The JSON retains the displayed values and
both full-precision reconstructions, and flags the distance mismatch rather
than presenting the displayed values as a 3.4 µm recalculation.

## Reproduction

Quick checks validate database hashes, state identities, basis construction,
and one operating point without overwriting committed data:

```bash
uv run python scripts/reproduce_forster_characterization.py \
  --fixed-m-database-dir "$FIXED_M_DB" --quick-check
uv run python scripts/reproduce_forster_gate.py --quick-check
uv run python scripts/reproduce_vdw_dense.py --quick-check
```

Regenerate all committed numerical records:

```bash
uv run python scripts/reproduce_forster_characterization.py \
  --fixed-m-database-dir "$FIXED_M_DB"
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
