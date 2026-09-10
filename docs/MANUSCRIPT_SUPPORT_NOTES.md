# Additional manuscript support recovered from local notes

This supplement separates recovered evidence from calculations that remain
unavailable. It does not change the selected pulse or any archived simulation.
The original 41-file September 10 snapshot and its manifest remain unchanged.

## Evidence and limits

| Manuscript claim | Available support | Limit |
| --- | --- | --- |
| 44.5 mW peak-command Yb power | Gaussian-beam conversion using the stated dipole, waist and Rabi frequency | The dipole is an input, not independently recomputed here. |
| Loss difference below 6 × 10⁻⁵ | Bare-population decay integral compared with stored modal norm loss on the saved trajectory | No propagation with the full non-diagonal loss matrix; no continuous-time bound. |
| Rb/Yb radiative contributions near 6.70/1.68 × 10⁻⁴ | Approximate species decomposition of stored populations and ideal control-pulse residence times | Not an exact decomposition of the complete damped gate; unresolved spectator loss is not assigned to a species. |
| Corrected manuscript gate and robustness figures | Corrected baseline, driven curves, robustness and field-scan JSON records | Rendering and data consistency checks do not rerun physics. |

The local `History_plans` notes were untracked when inspected. Their excerpts
and source-file hashes are preserved in
[source_excerpts.json](../data/manuscript_support_2026_09_10/source_excerpts.json); they must not be
represented as files from a published manuscript commit. Only task-relevant
excerpts are included, not the entire local planning directory.

## Reproduce the arithmetic and figures

From the repository root, after `uv sync`:

```bash
uv run python scripts/recompute_claim_diagnostics.py
uv run python scripts/reproduce_all.py --figures
uv run python -m pytest tests/test_manuscript_support.py
```

The arithmetic command prints JSON without writing files or importing the
physics code. Its default inputs are the corrected archive's `recheck` and
`driven_curves` records; `--simulations-dir` can select a disposable replay
directory with the same schema. Baseline hash/signature mismatches are rejected.
[recomputed_diagnostics.json](../data/manuscript_support_2026_09_10/recomputed_diagnostics.json)
is the new reconstruction, with input and script hashes, not a recovered old
execution log. Regenerate it explicitly with shell redirection only when
intending to update this supplement.

The figure workflow verifies the original 41-file archive manifest, shared
baseline provenance, populations, computational returns, phase and fidelity.
It produces `figures/corrected_control_decay_gate.{pdf,png}` (current Fig. 3)
and `figures/corrected_control_decay_robustness.{pdf,png}` (current Fig. 4).
These names do not overwrite historical gate figures. The `--figures` workflow
also renders the channel, excitation and static vdW subjects. The older
pre-correction replay is explicitly named `--historical-full`.

[source_manifest.json](../data/manuscript_support_2026_09_10/source_manifest.json)
identifies the original plotting/checking scripts and already archived helper
hashes. The adapted checker covers corrected Figs. 3/4 only; it does not require
the manuscript-only `vdw_gate_distance_sweep.json`. Per-run hashes of historical
scripts outside the frozen import are retained as provenance, not checked
against a different final source revision. No full-D propagation or optical
spectator scan is performed by these commands.

## Power conversion

For a Gaussian beam with 1/e² intensity radius \(w\),
\(I_0=2P/(\pi w^2)\), \(I_0=\epsilon_0 c E_0^2/2\), and
\(\Omega=dE_0/\hbar\). Thus

\[
P=\frac{\pi w^2\epsilon_0 c}{4}
  \left(\frac{\hbar\Omega}{d}\right)^2.
\]

The historical note uses a rounded cyclic Rabi frequency of 11.778 MHz,
\(d=0.00239\,ea_0\), and \(w=12\,\mu\mathrm m\), giving approximately
44.5306 mW. The script also reports the value using the full-precision selected
pulse amplitude. This is peak command power before optical losses, not the
filtered pulse's time-averaged power.

The older `channel_forster.md:363–382` excerpt quotes a different optical
route: π polarization (Δm = 0), initial and final mF = +1/2, with
\(|d_0|=0.00169\,ea_0\). The current manuscript instead assumes σ⁺
addressing (Δm = +1), from mF = −1/2 to +1/2, and
\(|d_{+1}|=0.00239\,ea_0\). These are different spherical components and
initial states, not two interchangeable dipole estimates for the same
addressed transition. Using the old component at the same waist and Rabi
frequency would require about 89 mW. The old route therefore does not
independently verify the current σ⁺ matrix element: 44.53 mW remains
conditional on the stated \(d_{+1}\).

## Post-hoc loss diagnostic, not a full-loss-matrix validation

Let \(P_c,P_{SS},P_{PP},P_o\) denote the saved blocked-branch populations,
and let \(\gamma_i=1/\tau_i\). On this *already propagated* trajectory,
integrate

\[
L_{\rm bare}(t)=\int_0^t [\gamma_{\rm Rb,S}P_c
+(\gamma_{\rm Rb,S}+\gamma_{\rm Yb,S})P_{SS}
+(\gamma_{\rm Rb,P}+\gamma_{\rm Yb,P})P_{PP}
+\gamma_oP_o]\,dt',
\]

using the trapezoidal rule at the saved, potentially nonuniform time points.
Here \(\gamma_o\) is the larger of the SS and PP total rates: a proxy,
not a rigorous upper bound on all other states' decay rates.

The endpoint integral is about 8.53508 × 10⁻⁴, versus stored modal loss
9.09052 × 10⁻⁴. The endpoint difference is about 5.55440 × 10⁻⁵; the
largest absolute difference at saved samples is about 5.57930 × 10⁻⁵.
These compare two loss prescriptions on the same saved population trajectory.
They do **not** demonstrate that repropagating with off-diagonal damping changes
branch loss by less than 6 × 10⁻⁵. The recovered referee report explicitly
recommended that latter calculation as further work.

## Approximate species budget

The unblocked Yb population integrates to approximately 49.3990 ns. The ideal
Rb-only branch spends the entire 160.02895 ns target window in Rb 56S;
two ideal 100 ns π pulses contribute 50 ns each, giving 260.02895 ns.
The control-pulse residence estimate is not itself a damped-trajectory integral.

The approximation averages equally over four computational inputs. Only the
two Rb-excited inputs receive the ideal control-pulse residence contribution;
the blocked input uses its saved computational, SS and PP residence integrals.
It ignores attenuation inherited from preceding pulses, changes in the final
control pulse due to incomplete return, and the unknown species composition
of spectator population. This gives approximately 6.70333 × 10⁻⁴ (Rb) and
1.67437 × 10⁻⁴ (Yb). Preserve these computed values rather than forcing the
latter to match the manuscript's 1.68 × 10⁻⁴.

The residual spectator proxy and the bare/modal discrepancy are separate from
these two estimates. Their sum is not asserted to equal the manuscript's
radiative error budget. Computational-return loss also includes residual
excitation and must not be identified with spontaneous emission alone.

## Still missing

- Independent generation of the 302.043 nm target dipole and a state-resolved
  ±100 GHz optical spectator scan supporting the <1.3 × 10⁻⁵ claim, including
  pulse, polarization, initial state, detunings and transition matrix elements.
- Propagation with the full transformed non-diagonal loss operator and a
  matched diagonal/full comparison.
- A species-resolved, complete-gate decay integral with the same propagation
  and loss convention as the reported gate fidelity.

These missing validations are not supplied by plotting the existing data or
by recomputing the post-hoc estimates above.
