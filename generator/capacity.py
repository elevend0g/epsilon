"""§1.3 capacity arithmetic — computed integers, not an asserted paragraph.

Two bounds, both meant to be written into PREREG.md as computed integers
once real architecture constants are chosen:

  upper: cache_bits / state_bits >= 4   (the table must not fit in state)
  lower: state_bits / trajectory_bits >= 16  (state must have ample room
         for more than a pointer)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

UPPER_BOUND_MIN_RATIO = 4
LOWER_BOUND_MIN_RATIO = 16


@dataclass(frozen=True)
class CapacityConfig:
    n_entries: int              # cache size
    alphabet_size: int          # |S|
    d_model: int
    d_state: int
    bits_per_element: int = 16  # fp16, a conservative (over-)estimate of true capacity
    n_trajectory_features: int = 4  # margin, displacement, gate state, integration count


def cache_bits(cfg: CapacityConfig) -> float:
    return cfg.n_entries * 6 * math.log2(cfg.alphabet_size)


def state_bits(cfg: CapacityConfig) -> int:
    return cfg.d_model * cfg.d_state * cfg.bits_per_element


def trajectory_bits(cfg: CapacityConfig) -> float:
    key_bits = 3 * math.log2(cfg.alphabet_size)
    feature_bits = cfg.n_trajectory_features * cfg.bits_per_element
    return key_bits + feature_bits


def upper_bound_ratio(cfg: CapacityConfig) -> float:
    return cache_bits(cfg) / state_bits(cfg)


def lower_bound_ratio(cfg: CapacityConfig) -> float:
    return state_bits(cfg) / trajectory_bits(cfg)


def check_upper_bound(cfg: CapacityConfig) -> bool:
    return upper_bound_ratio(cfg) >= UPPER_BOUND_MIN_RATIO


def check_lower_bound(cfg: CapacityConfig) -> bool:
    return lower_bound_ratio(cfg) >= LOWER_BOUND_MIN_RATIO


# Proposed architecture constants (§1.3): satisfy both bounds simultaneously.
# n_entries=1024, |S|=64 -> cache ~36.9 kbit; d_model=64, d_state=8, fp16
# -> state ~8.2 kbit (~4.5x below cache) and ~100x the trajectory floor.
PROPOSED = CapacityConfig(n_entries=1024, alphabet_size=64, d_model=64, d_state=8)
