from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce_vdw_dense as vdw  # noqa: E402
from evaluate_vdw_p1_5 import (  # noqa: E402
    CONFIGURATIONS,
    REFERENCE_LABEL,
    convergence_summary,
)


def as_tuple(config: vdw.BasisConfig) -> tuple[int, float, int, float]:
    return (
        config.rb_n_half_width,
        config.yb_nu_half_width,
        config.l_max,
        config.pair_window_ghz,
    )


def row(
    label: str,
    comparison: str,
    c6: float,
    shift: float,
    weight: float,
    size: int,
) -> dict[str, object]:
    return {
        "label": label,
        "comparison": comparison,
        "c6_ghz_um6": c6,
        "working_point_shift_mhz": shift,
        "bare_pair_weight": weight,
        "pair_basis_size": size,
    }


def test_reference_config_preserves_published_bounds() -> None:
    config = vdw.REFERENCE_BASIS
    assert config.rb_n_half_width == 3
    assert config.yb_nu_half_width == 3.3
    assert config.l_max == 3
    assert config.pair_window_ghz == 80.0


def test_each_variant_changes_exactly_one_reference_axis() -> None:
    reference = vdw.REFERENCE_BASIS
    for label, _, comparison, config in CONFIGURATIONS:
        changed = sum(
            left != right for left, right in zip(as_tuple(config), as_tuple(reference), strict=True)
        )
        expected = 0 if comparison == "reference" else 1
        assert changed == expected, label


def test_summary_uses_only_expanded_rows() -> None:
    rows = [
        row(REFERENCE_LABEL, "reference", 10.0, 5.0, 0.9, 100),
        row("contracted", "contracted", 100.0, 50.0, 0.1, 50),
        row("radial expanded", "expanded", 10.2, 5.1, 0.91, 120),
        row("l expanded", "expanded", 9.9, 4.8, 0.89, 150),
    ]
    summary = convergence_summary(rows)
    changes = summary["maximum_changes_over_expanded_rows"]
    assert changes["c6_ghz_um6"]["absolute_change"] == pytest.approx(0.2)
    assert changes["working_point_shift_mhz"]["absolute_change"] == pytest.approx(0.2)
    assert changes["bare_pair_weight"]["absolute_change"] == pytest.approx(0.01)
    assert summary["maximum_expanded_pair_basis"]["pair_basis_size"] == 150
