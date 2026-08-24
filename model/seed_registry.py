"""Central registry of every reserved seed range used across the
project, and a mechanical non-overlap assertion.

Every reserved range was chosen "by eye" through this project (hundred-
million-spaced bases, picked to look far apart) and never mechanically
verified until now — the fourth thing in the project whose correctness
depended on construction rather than a check, after the gate-hardening
bug, the causal/query-PR conflation, and the buffer-thrashing bug
(RESULTS.md findings 1, 2, 6). Every new reserved split — Phase 3's
probe corpus included — must add a row here, computed from the actual
constant and consumption formula the consuming code uses (not a
separately-maintained guess), and pass assert_no_overlap()."""

from __future__ import annotations

from model.pilot_train import CHAIN_LENGTHS, GEOM_SEED, TRAIN_SEED_BASE, VAL_SEED_BASE
from model.phase2_leakage import LEAKAGE_SEED_A, LEAKAGE_SEED_B
from model.phase2_counterfactual import COUNTERFACTUAL_SEED
from model.phase2_margin import MARGIN_SEED
from model.phase2_margin_barm import BARM_SEED

MAX_L = max(CHAIN_LENGTHS)

# name -> (low, high) inclusive: every concrete seed value the named
# split can actually produce.
RESERVED_RANGES: dict[str, tuple[int, int]] = {
    # model/pilot_train.py run_pilot(): TRAIN_SEED_BASE + pilot_idx*10_000_000 + step,
    # pilot_idx in {0,1,2}, step in [0, STEP_CAP=102_400).
    "TRAIN (real seeds 0-2)": (TRAIN_SEED_BASE, TRAIN_SEED_BASE + 2 * 10_000_000 + 102_400),
    # model/pilot_train.py build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, ...), L in CHAIN_LENGTHS.
    "VAL": (VAL_SEED_BASE + 1, VAL_SEED_BASE + MAX_L),
    # model/pilot_train.py build_fixed_batch(3, N_GEOM, GEOM_SEED, ...) -- single point.
    "GEOM (§3.1.5 / §3.3)": (GEOM_SEED, GEOM_SEED),
    # model/phase2_leakage.py: LEAKAGE_SEED_A/B + L, L in CHAIN_LENGTHS.
    "PHASE2_LEAKAGE_A": (LEAKAGE_SEED_A + 1, LEAKAGE_SEED_A + MAX_L),
    "PHASE2_LEAKAGE_B": (LEAKAGE_SEED_B + 1, LEAKAGE_SEED_B + MAX_L),
    # model/phase2_counterfactual.py: single point (L fixed at 2 in that script).
    "PHASE2_COUNTERFACTUAL": (COUNTERFACTUAL_SEED, COUNTERFACTUAL_SEED),
    # model/phase2_margin.py: MARGIN_SEED + L, L in CHAIN_LENGTHS.
    "PHASE2_MARGIN": (MARGIN_SEED + 1, MARGIN_SEED + MAX_L),
    # model/phase2_margin_barm.py: BARM_SEED + L*10 + ARM_SEED_OFFSET[arm], L in CHAIN_LENGTHS, offset in [0,4].
    "PHASE2_MARGIN_BARM": (BARM_SEED + 1 * 10 + 0, BARM_SEED + MAX_L * 10 + 4),
    # model/phase3_probe_corpus.py: PROBE_SEED_BASE + band*100 + L*10 + arm_offset,
    # band in {0,1}, L in CHAIN_LENGTHS, arm_offset in [0,4].
    "PHASE3_PROBE_CORPUS (§5.1-§5.2)": (810_000_000 + 0 * 100 + 1 * 10 + 0, 810_000_000 + 1 * 100 + 3 * 10 + 4),
    # model/phase3_gap_diagnostic.py: GAP_SEED_BASE + n_hard_index, n_hard_index in [0,5]
    # (n_hard in {1,2,4,8,16,32}). Post-hoc diagnostic, §8/§3.1.6 precedent -- not a
    # preregistered criterion, but seed isolation discipline still applies.
    "PHASE3_GAP_DIAGNOSTIC (§8 post-hoc)": (820_000_000, 820_000_005),
}


def assert_no_overlap(ranges: dict[str, tuple[int, int]] | None = None) -> None:
    ranges = ranges if ranges is not None else RESERVED_RANGES
    items = list(ranges.items())
    for i in range(len(items)):
        name_i, (lo_i, hi_i) = items[i]
        assert lo_i <= hi_i, f"{name_i}: malformed range ({lo_i} > {hi_i})"
        for j in range(i + 1, len(items)):
            name_j, (lo_j, hi_j) = items[j]
            if lo_i <= hi_j and lo_j <= hi_i:
                raise AssertionError(
                    f"seed range overlap: {name_i} [{lo_i:,},{hi_i:,}] vs "
                    f"{name_j} [{lo_j:,},{hi_j:,}]"
                )


if __name__ == "__main__":
    assert_no_overlap()
    print(f"OK: {len(RESERVED_RANGES)} reserved ranges, no overlaps")
    for name, (lo, hi) in sorted(RESERVED_RANGES.items(), key=lambda kv: kv[1][0]):
        print(f"  {lo:>15,} - {hi:>15,}  {name}")
