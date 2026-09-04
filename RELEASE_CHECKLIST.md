# Release checklist

These are maintainer actions, not steps performed by the numerical workflow.

1. Run `uv sync` and `uv run python scripts/reproduce_all.py` on a clean clone.
2. Run `uv run python scripts/reproduce_all.py --figures` and compare the
   generated figures with the manuscript.
3. If resources permit, run `uv run python scripts/reproduce_all.py --full` and
   inspect all changed JSON records before accepting them. The recorded vdW
   convergence run reached about 10.1 GiB peak RSS.
4. Confirm `provenance/manuscript_manifest.json` identifies the submitted PDF
   and manuscript source commit.
5. Confirm the repository is public and all README links work without login.
6. Create an immutable version tag and GitHub release for the submitted
   artifact; record the tag in the manuscript or submission metadata.
7. To obtain an archival DOI, connect the public GitHub repository to Zenodo,
   archive the release, and add the issued DOI badge and citation metadata.
8. Do not advertise or cite a DOI until Zenodo has actually issued it.

Changing repository visibility, publishing a release, and minting a DOI are
external maintainer operations and are intentionally separate from the scripts.
