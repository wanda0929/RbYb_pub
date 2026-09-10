# Generated manuscript figures

Run `uv run python scripts/reproduce_all.py --figures` to regenerate the four
numerical figure subjects and the excitation schematic (five figures total)
from committed data. Generated PDF/PNG files are intentionally not versioned.

The command checks the corrected data and writes
`corrected_control_decay_gate.{pdf,png}` (Fig. 3) and
`corrected_control_decay_robustness.{pdf,png}` (Fig. 4), alongside the channel,
excitation and static vdW plots. It does not regenerate the pre-correction gate
plot. The corrected renderer reuses the frozen style/pulseviz helpers.

See [the availability map](../docs/CODE_AVAILABILITY.md) for data fields,
generators and remaining evidence gaps. Schematic geometry is not physical data;
the command does not assert pixel-identical layout to a later edited manuscript.
