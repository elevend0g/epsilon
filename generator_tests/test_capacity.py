"""§1.3: both capacity bounds must be computed integers, asserted here —
not an unverified paragraph."""

from __future__ import annotations

from generator.capacity import (
    PROPOSED,
    CapacityConfig,
    cache_bits,
    check_lower_bound,
    check_upper_bound,
    lower_bound_ratio,
    state_bits,
    upper_bound_ratio,
)


def test_proposed_constants_satisfy_both_bounds():
    assert check_upper_bound(PROPOSED)
    assert check_lower_bound(PROPOSED)


def test_proposed_constants_match_worked_arithmetic():
    assert cache_bits(PROPOSED) == 1024 * 6 * 6  # log2(64) == 6
    assert state_bits(PROPOSED) == 64 * 8 * 16
    assert upper_bound_ratio(PROPOSED) == (1024 * 6 * 6) / (64 * 8 * 16)


def test_doc_counterexample_fails_upper_bound():
    """§1.3's own worked counterexample: d_model=256, d_state=16 makes the
    state larger than the cache — the asymmetry does not exist there."""
    bad = CapacityConfig(n_entries=1024, alphabet_size=64, d_model=256, d_state=16)
    assert state_bits(bad) > cache_bits(bad)
    assert not check_upper_bound(bad)


def test_d_model_128_is_roughly_a_coin_flip():
    coin_flip = CapacityConfig(n_entries=1024, alphabet_size=64, d_model=128, d_state=16)
    assert not check_upper_bound(coin_flip)  # ratio ~1.1, well under the >=4 bar


def test_lower_bound_rejects_a_pointer_only_state():
    """A state barely large enough to hold the pointer key alone should
    fail the lower bound — it has no room for trajectory features."""
    pointer_only = CapacityConfig(
        n_entries=1024, alphabet_size=64, d_model=2, d_state=1, n_trajectory_features=4
    )
    assert not check_lower_bound(pointer_only)
