"""Read-only archive checks: never import or execute the archived physics code."""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "manuscript_claims_2026_09_10"
SIMULATIONS = ARCHIVE / "simulations"


def load(name):
    return json.loads((SIMULATIONS / name).read_text())


def test_import_manifest_preserves_all_files():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    assert manifest["source_commit"] == "fd728db62804157051467b32ed4ef4a9184e29da"
    assert len(manifest["files"]) == 41
    actual = {
        str(p.relative_to(ARCHIVE))
        for directory in [SIMULATIONS, ARCHIVE / "figures"]
        for p in directory.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert actual == set(manifest["files"])
    for name, expected in manifest["files"].items():
        data = (ARCHIVE / name).read_bytes()
        assert len(data) == expected["size_bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name


def test_corrected_baseline_is_not_historical_or_reoptimized():
    baseline = load("forster_control_decay_recheck.json")
    assert baseline["pulse_reoptimized"] is False
    assert baseline["completed_utc"]
    assert len(baseline["rows"]) == baseline["summary"]["geometries"] == 19
    assert baseline["summary"]["nominal_overlap"] == pytest.approx(
        0.999125784769183, abs=1e-14, rel=0
    )
    assert baseline["summary"]["joint_sampled_minimum"] == pytest.approx(
        0.9984994300039455, abs=1e-14, rel=0
    )
    assert baseline["rows"]["0.000000/0.000000"]["fixed_local_z_rad"] == [
        -3.1407724432579953,
        2.941141606972484,
    ]
    assert len(list(SIMULATIONS.glob("forster_control_decay_*.json"))) == 7


def test_off_axis_records_keep_partial_checkpoint_and_matched_comparisons():
    paths = sorted((SIMULATIONS / "forster_off_axis").glob("*.json"))
    assert len(paths) == 11
    reference = load("forster_control_decay_recheck.json")
    phases = reference["rows"]["0.000000/0.000000"]["fixed_local_z_rad"]
    for path in paths:
        record = json.loads(path.read_text())
        assert record["fixed_local_z_rad"] == phases
        assert bool(record.get("completed_utc")) == (path.name != "ball_max_angle_m2.json")
        assert record["database"] == load("forster_characterization.json")["database"]

    for name, expected, bounds in [
        ("ball_max_angle_m2_512", 0.9991222471242031, (3.54e-6, 3.55e-6)),
        ("thermal_m2", 0.9990343729665522, (8.80e-5, 8.81e-5)),
    ]:
        prefix = "ball_max_angle" if name.startswith("ball") else "thermal"
        expanded = load(f"forster_off_axis/{name}.json")
        projected = load(f"forster_off_axis/{prefix}_projected.json")
        assert expanded["geometry"] == projected["geometry"]
        assert expanded["sector_radius"] == 2
        assert expanded["rows"][-1]["mode_count"] == 512
        fidelity = expanded["rows"][-1]["scenarios"][0]["fidelity"]
        assert fidelity == pytest.approx(expected, rel=0, abs=1e-14)
        difference = projected["rows"][-1]["scenarios"][0]["fidelity"] - fidelity
        assert bounds[0] < difference < bounds[1]


def test_snapshot_syntax_and_document_links_without_importing_code():
    scripts = sorted(ARCHIVE.rglob("*.py"))
    assert len(scripts) == 19
    external = {"numpy", "scipy", "matplotlib", "pairinteraction", "cycler"}
    for path in scripts:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in set(modules) - sys.stdlib_module_names - external:
                assert any(
                    (directory / f"{module}.py").exists()
                    or (directory / module / "__init__.py").exists()
                    for directory in [SIMULATIONS, ARCHIVE / "figures"]
                ), (path, module)
    for name in [
        "forster_characterization.json",
        "forster_p0_4_uncertainty_convergence.json",
        "forster_control_decay_recheck.json",
        "forster_control_decay_p0_4.json",
    ]:
        assert (SIMULATIONS / name).is_file()
    for path in [ARCHIVE / "README.md", *SIMULATIONS.glob("forster_off_axis/*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target:
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
