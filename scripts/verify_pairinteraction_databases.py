#!/usr/bin/env python3
"""Verify the exact PairInteraction database assets used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pairinteraction as pi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "provenance" / "pairinteraction_database_manifest.json"


def database_tables_dir(database_dir: Path | None = None) -> Path:
    """Return PairInteraction's active database table directory."""
    if database_dir is None:
        database_dir = Path(pi.Database(download_missing=False).database_dir)
    return database_dir / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_database_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    database_dir: Path | None = None,
) -> Path:
    """Raise on a missing or changed database file and return the table path."""
    manifest = json.loads(manifest_path.read_text())
    installed_version = getattr(pi, "__version__", "unknown")
    expected_version = manifest["pairinteraction_version"]
    if installed_version != expected_version:
        raise RuntimeError(
            f"PairInteraction {installed_version} is installed; {expected_version} is required"
        )

    tables = database_tables_dir(database_dir)
    problems: list[str] = []
    for entry in manifest["files"]:
        path = tables / entry["relative_path"]
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            problems.append(f"size mismatch: {path} ({actual_size} != {entry['size_bytes']})")
            continue
        actual_hash = sha256(path)
        if actual_hash != entry["sha256"]:
            problems.append(f"SHA-256 mismatch: {path}")
    if problems:
        raise RuntimeError("Database verification failed:\n" + "\n".join(problems))
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--database-dir",
        type=Path,
        help="PairInteraction database root containing the tables directory",
    )
    args = parser.parse_args()
    tables = verify_database_manifest(args.manifest, args.database_dir)
    print(f"verified PairInteraction databases in {tables}")


if __name__ == "__main__":
    main()
