#!/usr/bin/env python3
"""Record detailed spectrum diagnostics in the final P0-4 reference block.

Output:
  data/forster_p1_4_reference_spectrum.json
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_forster_p0_4 as p0  # noqa: E402
from verify_pairinteraction_databases import verify_database_manifest  # noqa: E402

OUT = ROOT / "data" / "forster_p1_4_reference_spectrum.json"


def main() -> None:
    verify_database_manifest()
    print("building final P0-4 axial reference spectrum", flush=True)
    model = p0._build(p0.REFERENCE_BASIS)
    diagnostics = model.spectral_diagnostics
    for key in ("target_eigenstates", "largest_target_overlap_spectators"):
        for state in diagnostics[key]:
            state["energy_relative_ss_asymptote_mhz"] = float(
                state["energy_mhz"] - model.ss_asymptote_mhz
            )
    diagnostics["nearest_spectator_eigenstate"]["energy_relative_ss_asymptote_mhz"] = float(
        diagnostics["nearest_spectator_eigenstate"]["energy_mhz"] - model.ss_asymptote_mhz
    )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "software": {
            "pairinteraction": importlib.metadata.version("pairinteraction"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "status": "final P0-4 axial reference-spectrum audit complete",
        "scope": (
            "hyperfine-resolved exact M_tot=5/2 block at theta=0; direct "
            "diagonalization within the stated finite P0-4 basis, not the full "
            "untruncated atomic Hilbert space or a transverse full-angular model"
        ),
        "fixed_inputs": {
            "field_gauss": model.field_gauss,
            "distance_um": model.distance_um,
            "theta_deg": model.theta_deg,
            "basis": asdict(model.basis_config),
            "pair_basis_size": model.pair_basis_size,
            "connected_component_size": model.symmetry_component_size,
            "pp_asymptote_mhz": model.pp_asymptote_mhz,
            "ss_asymptote_mhz": model.ss_asymptote_mhz,
            "forster_defect_mhz": model.forster_defect_mhz,
        },
        "spectral_diagnostics": diagnostics,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
