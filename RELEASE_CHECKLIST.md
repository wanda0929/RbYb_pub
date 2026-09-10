# Release checklist

These are maintainer actions, not steps performed by the numerical workflow.

1. Resolve or qualify every gap in `docs/CODE_AVAILABILITY.md`; do not claim all
   findings are archived while unsupported optical/atomic/loss claims remain.
   Run `uv sync --frozen` and `uv run python scripts/reproduce_all.py` on a clean clone.
2. Run `uv run python scripts/reproduce_all.py --figures` and compare the
   five generated figure subjects and all five tables with the submitted manuscript.
3. Install and verify the pinned databases, then run
   `uv run python scripts/reproduce_all.py --quick-check`. If resources permit,
   replay the corrected audit stages in a disposable copy as documented in the
   claim archive. `--historical-full` is pre-correction, not current gate validation.
4. Freeze and record the submitted PDF/source hashes and commit. The original
   `provenance/manuscript_manifest.json` and corrected archive manifest are
   historical snapshots, not automatic identifiers of the latest local draft.
5. Confirm the repository is public and all README links work without login.
6. Create an immutable version tag and GitHub release for the submitted
   artifact; record the tag in the manuscript or submission metadata.
7. To obtain an archival DOI, connect the public GitHub repository to Zenodo,
   archive the release, and add the issued DOI badge and citation metadata.
8. Do not advertise or cite a DOI until Zenodo has actually issued it.

Changing repository visibility, publishing a release, and minting a DOI are
external maintainer operations and are intentionally separate from the scripts.
