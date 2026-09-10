# Supplemental control-decay local convergence experiment

This is a later, separate experiment, not a replacement for the selected pulse,
manuscript-linked figures, or historical validation records.

## Model and objective

The supplemental runner explicitly enables `include_blocked_control_decay=True`.
The blocked control-excited state during the target window carries the Rb 56S
amplitude decay rate 1/(2 τRb). Historical code omitted this term in that state;
pair-mode decay and the control-only branch already included their own losses.
The opt-in correction affects only the hardware-response training/recheck path;
it does not change historical defaults or claim to update the separate shaped,
sparse, or trajectory propagators. Both baseline and candidate below use the
corrected convention. They must not be compared directly to old loss-convention
metrics as though the difference were solely pulse improvement.

The training set contains **nine configurations**: nominal plus the two axial
displacement endpoints (±50 nm), each crossed with all four independent
(sRb, sYb) amplitude vertices in {0.99, 1.01}². Six amplitude/detuning parameters
vary; segment duration stays fixed at 0.025634115265641792 µs. The regularized
objective is the maximum nine-point infidelity plus 0.02 times their mean
infidelity. The training propagation step is 1 ns.

## Outcome and claim boundaries

Adaptive Nelder–Mead returned `success=true` after **2091 iterations and 3466
evaluations**, before the 3000-iteration budget. The final best-relative simplex
parameter spread was 1.8930605000244682e-8 (xatol 2e-8); objective spread was
2.4026136982713275e-14 (fatol 2e-11). Recorded effective runtime including model
construction and rechecks was approximately 37018 seconds.

At the finer 0.0625 ns evaluation step:

| Corrected-model metric | Selected baseline | Supplemental candidate |
| --- | ---: | ---: |
| Nominal no-jump overlap | 0.9991257839998615 | 0.9991397589660671 |
| Training-vertex minimum overlap | 0.9984994230238966 | 0.9985074824055115 |
| Regularized objective | 0.0015287779146971453 | 0.0015205409958427055 |

This establishes the specified optimizer stopping tolerances for the **1 ns
discrete training objective** only. The 0.125 and 0.0625 ns checks reevaluate
fixed pulses; they do not optimize at those steps. No multistart study,
continuous-domain/global minimax certificate, or new full-grid acceptance is
provided. These are zero-temperature no-jump overlaps, not CPTP process
fidelities. The new pulse is not manuscript-selected. The older 400-iteration
budget-limited run remains budget-limited; this experiment does not rewrite it.

## Original records and provenance

- `result.json`: unmodified completed result, including seed/baseline/best
  parameters, optimizer options, final simplex, source hashes, software versions,
  and all four fine-step rechecks.
- `evaluations.jsonl`: all 3466 objective evaluations, in order, with parameters,
  nine-point metrics, and timing. This is scientific evidence, not debug output.

These files are copied byte-for-byte from the manuscript repository's
[archived experiment](https://github.com/wanda0929/draft-RbYb/tree/6588139151e607f61480da4df322ce95a33a8595/simulations/forster_control_decay_convergence).
The original `source_sha256` names refer to that experiment's frozen
manuscript-layout simulation sources, **not** to the adapted `scripts/` files
in this repository. The raw provenance has deliberately not been rewritten.
Access to the original source repository may require permission; the adapted
runner and dependencies needed for numerical replay are included here.

| Original file | SHA-256 |
| --- | --- |
| `result.json` | `51a4514a90c114c717c8a3c87c1f4d77f0fcbde54c979df5861572ab3c4dd0cd` |
| `evaluations.jsonl` | `f4c423267542eb94c4cdcf6784463758592ceaeb9ff7003ec931d7adb04d6a69` |

The original run used Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1, and
PairInteraction 2.5.0. This repository locks NumPy 2.3.5; fresh runs record their
actual versions and adapted source hashes rather than claiming byte-identical
optimizer trajectories across environments. The original experiment verified
all 13 database table hashes; this runner validates the repository's database
manifest before building models. Use Rb v1.2, Yb171_mqdt v1.4, and misc v1.4,
not the latest Rb tables (see the root README for installation).

## Replay without optimization

From the repository root, install the locked environment with `uv sync --frozen`
and verify the database tables as described in the root README. Choose a **new,
nonexistent** output directory; never use the archived data directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/run_control_decay_convergence.py \
  --recheck-only \
  --seed-checkpoint data/forster_control_decay_convergence/result.json \
  --output-dir /tmp/rbyb-control-decay-recheck
```

This builds three geometries and reevaluates baseline and archived candidate at
both fine steps. It does not call the optimizer. The new result has
`optimizer=null`, `evaluations=0`, and `recheck_only=true`. Compare its
`fine_step_recheck` metrics with the original result, allowing numerical roundoff.

## Optional new optimization (expensive)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/run_control_decay_convergence.py \
  --output-dir /tmp/rbyb-control-decay-new-run --maxiter 3000
```

This starts from the selected baseline, writes a checkpoint after each
evaluation, and performs four fine-step rechecks after optimization returns.
`status=finished` means the workflow completed; inspect `optimizer.success` and
`optimizer.message` to distinguish convergence from budget exhaustion. Supplying
`--seed-checkpoint` starts a **new simplex** from the saved best point; it does
not resume optimizer state. Neither command overwrites manuscript inputs.

The default `scripts/reproduce_all.py` tests the archive and opt-in propagation
without rerunning the expensive experiment; `--full` does not include this
separate supplement. Unit tests also verify raw-file hashes and simplex bounds,
compare corrected propagation against an independent dense generator, and check
that a budget-limited run is not reported as optimizer success.
