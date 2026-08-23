"""Mechanical non-overlap check for every reserved seed range in the
project — see model/seed_registry.py for why this exists as code and
not just "chosen far apart by eye"."""

from __future__ import annotations

from model.seed_registry import RESERVED_RANGES, assert_no_overlap


def test_no_reserved_range_overlaps():
    assert_no_overlap()


def test_at_least_the_known_splits_are_registered():
    expected = {
        "TRAIN (real seeds 0-2)", "VAL", "GEOM (§3.1.5 / §3.3)",
        "PHASE2_LEAKAGE_A", "PHASE2_LEAKAGE_B", "PHASE2_COUNTERFACTUAL",
        "PHASE2_MARGIN", "PHASE2_MARGIN_BARM",
    }
    assert expected <= set(RESERVED_RANGES.keys())
