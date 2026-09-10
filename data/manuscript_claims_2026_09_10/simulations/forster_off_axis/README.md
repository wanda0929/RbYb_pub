# Fixed-pulse off-axis hyperfine audit

This audit extends the published pair Hamiltonian to explicit products of
electronic pair states and all four Rb nuclear projections. It changes neither
the pulse nor the corrected nominal local-Z calibration. It is a frozen-position
angular diagnostic, not a thermal average, pulse reoptimization, or certificate
over a continuous uncertainty region.

See [the Chinese computational report](REPORT.md) for results, convergence
comparisons, and the distinction between tested points and thermal averaging.

## Model and numerical checks

- PairInteraction 2.5.0; Rb v1.2, Yb171_mqdt v1.4, misc v1.4. JSON records carry
  database asset hashes and PairInteraction/NumPy/SciPy versions.
- The reference atomic and pair windows, radial/angular cutoffs and interaction
  terms through R^-4 are unchanged. The full electronic pair basis has 9,324
  states; explicit nuclear products have 37,296 states before sector selection.
- `--sector-radius j` retains all products with |M_tot - 5/2| <= j. Radius 0 is
  the original projection; radii 1 and 2 contain three and five sectors. No
  axial-brightness cutoff is used to select basis states. Finite sector bands
  must not be described as the full all-M Hilbert space.
- Sparse shift-invert eigenpairs are selected by energy near the SS asymptote,
  including axial-dark modes. Increasing `--modes` tests this additional spectral
  truncation. The omitted SS weight is reported without renormalizing it; its
  magnitude alone is not a gate-error estimate because omitted eigenmodes are
  far off resonance. Eigenpair residuals and orthogonality are checked.
- The drive addresses the same SS product as in the published model. The
  secular eigenmode absorbing-decay prescription, Rb control-window decay,
  and two effective Rb pi pulses are retained. This does not add a resolved
  optical excitation network for every spectator magnetic state or Lindblad
  return channels.
- All scenarios use the corrected phases in
  `forster_control_decay_recheck.json`, not the older pre-correction P0 record.
  Both Rabi scales are tested at 1.00 and 0.99; optical carriers and local Z are
  held fixed. The fidelity includes both Rb pulses.
  Current runs also read the same pinned zero-temperature lifetimes from that
  reference, avoiding a repeated atomic-decay query while LU factors are resident.
- Cross-sector populations start from unit control-excited population at the
  target-window entrance. They exclude opening/closing pulse attenuation.
  Reported peaks are maxima on integration boundaries, not certified continuous
  maxima. Final pair population is not identical to cross-sector population.

The 50 nm ball's maximum-angle geometry is dx = 0.049994593 um,
dz = -0.000735294 um (theta = 0.842615 degrees). A thermal-scale diagnostic uses
dx = 0.266 um, dz = 0: both its distance and angle change consistently. It is
outside the original ball and is not a sample-averaged thermal prediction.

## Reproduce

Install the numerical environment in an isolated venv:

```bash
uv venv .venv
uv pip install --python .venv/bin/python pairinteraction==2.5.0 numpy==2.4.6 scipy==1.17.1 matplotlib==3.11.1
```

Download the pinned tables (the plain `database download Rb` command may fetch
a newer version). Do not mix higher Rb table versions into this cache; the
library chooses the highest cached minor version. Inspect the cache before
accepting any replacement prompt, especially on a shared machine.

```bash
.venv/bin/pairinteraction database download https://github.com/pairinteraction/database-sqdt/releases/download/v1.2/Rb_v1.2.zip https://github.com/pairinteraction/database-sqdt/releases/download/v1.4/misc_v1.4.zip https://github.com/pairinteraction/database-mqdt/releases/download/v1.4/Yb171_mqdt_v1.4.zip
```

Run one physics worker at a time on an 8 GiB machine. For example:

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python simulations/evaluate_forster_off_axis.py --dx 0.049994593 --dz -0.000735294 --sector-radius 2 --modes 256 512 --output /tmp/ball_m2.json
.venv/bin/python simulations/evaluate_forster_off_axis.py --dx 0.266 --sector-radius 2 --modes 256 512 --check-half-step --output /tmp/thermal_m2.json
.venv/bin/python -m unittest discover -s simulations -p test_forster_off_axis.py
```

`--check-half-step` repeats propagation at half the requested time step for
the largest mode count, reusing the spectrum rather than rebuilding the matrix.
On this orb the workload soft memory limit is about 5.6 GiB, below the nominal
host RAM. Five-sector sparse LU can take more than ten minutes. Initial runs
hit memory throttling; the current implementation slices nuclear blocks early,
reuses the LU factorization with a symmetric-pattern ordering, releases temporary
allocator arenas, and avoids querying atomic lifetimes during propagation.

Each completed mode count is written to JSON. `completed_utc` means all requested
counts finished. Existing output files are never overwritten: after interruption,
preserve the partial file and use a new output path for remaining mode counts.
Source hashes identify the implementation used for each record; initial axial
and radius-1 runs preceded CLI/provenance and memory-allocation refinements.
The physical matrix and propagation conventions were unchanged by those refinements.
