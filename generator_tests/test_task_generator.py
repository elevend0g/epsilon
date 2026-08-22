"""Unit tests for generator/task_generator.py against docs/phase1.md v2 §1.

Covers the invariants named in §7 deliverable 2: axis orthogonality, twin
identity, follow-path uniqueness, no duplicate keys, no ordinal signal,
per-hop hard negatives, B1 decoy legality, B2 terminal distribution — plus
the mechanically-derived `rho` (§2.5) that falls out of the walk itself
rather than being asserted per arm.
"""

from __future__ import annotations

import pytest

from generator import (
    ABSTAIN,
    END,
    MIN_END_DECOYS,
    GeneratorConfig,
    TaskGenerator,
    verify_follow_path,
    verify_no_duplicate_keys,
)


def hamming(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def make_generator(**overrides) -> TaskGenerator:
    defaults = dict(alphabet_size=16, chain_length=3, n_distractors=24, n_hard=2, seed=0)
    defaults.update(overrides)
    return TaskGenerator(GeneratorConfig(**defaults))


ALL_ARMS = ("A", "B1", "B2", "C", "D")


def items(family):
    return (family.A, family.B1, family.B2, family.C, family.D)


# -- config validation --------------------------------------------------------


def test_config_rejects_n_hard_over_budget():
    with pytest.raises(ValueError):
        GeneratorConfig(alphabet_size=16, chain_length=3, n_distractors=4, n_hard=2, seed=0)


def test_config_rejects_small_alphabet():
    with pytest.raises(ValueError):
        GeneratorConfig(alphabet_size=3, chain_length=1, n_distractors=4, n_hard=1, seed=0)


# -- chain and arm correctness ------------------------------------------------


def test_chain_ends_in_end():
    gen = make_generator()
    for seed in range(20):
        gen.rng.seed(seed)
        family = gen.generate_family()
        assert family.A.chain_values[-1] == END
        for i in range(len(family.A.chain_keys) - 1):
            assert family.A.chain_values[i] == family.A.chain_keys[i + 1]


def test_arm_ground_truths_and_paths():
    gen = make_generator()
    for seed in range(50):
        gen.rng.seed(seed)
        family = gen.generate_family()

        assert family.A.answer == family.A.chain_keys[-1]
        assert family.D.answer == family.D.chain_keys[-1]
        assert family.B1.answer == ABSTAIN
        assert family.B2.answer == ABSTAIN
        assert family.C.answer == ABSTAIN

        for item in items(family):
            assert verify_follow_path(item), item.arm


def test_arm_c_start_key_genuinely_absent():
    gen = make_generator()
    for seed in range(20):
        gen.rng.seed(seed)
        family = gen.generate_family()
        cache_keys = {e.key for e in family.C.memory}
        assert family.C.start_key not in cache_keys


# -- requirement 5: no duplicate keys with differing values -------------------


def test_no_duplicate_keys_across_arms():
    gen = make_generator()
    for seed in range(50):
        gen.rng.seed(seed)
        family = gen.generate_family()
        for item in items(family):
            assert verify_no_duplicate_keys(item.memory)


# -- arm B1 (missing key, deletion-free): decoy legality ----------------------


def test_b1_matches_a_token_count():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        assert len(family.B1.memory) == len(family.A.memory)


def test_b1_hop_key_unfindable():
    """req 3: the true hop-j key is gone; the decoy occupying its cache
    position has an unrelated key that collides with no chain key."""
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        touched_key = family.A.chain_keys[family.B1.hop_j - 1]
        b1_keys = {e.key for e in family.B1.memory}
        assert touched_key not in b1_keys
        assert len(b1_keys) == len(family.B1.memory)  # still no duplicate keys


def test_b1_decoy_value_does_not_continue_chain():
    gen = make_generator(chain_length=3, n_distractors=18, n_hard=3)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        real_chain_values = set(family.A.chain_values) - {END}
        non_chain_entries = [e for e in family.B1.memory if e.key not in family.A.chain_keys]
        for e in non_chain_entries:
            assert e.value not in real_chain_values


# -- arm B2 (dead end, thesis arm): terminal legality -------------------------


def test_b2_matches_a_token_count():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        assert len(family.B2.memory) == len(family.A.memory)


def test_b2_terminal_value_not_a_key_anywhere():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        cache_keys = {e.key for e in family.B2.memory}
        assert family.B2.terminal_value not in cache_keys


def test_b2_hop_key_retrieves_terminal_not_true_value():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        touched_key = family.A.chain_keys[family.B2.hop_j - 1]
        b2_table = {e.key: e.value for e in family.B2.memory}
        assert b2_table[touched_key] == family.B2.terminal_value
        assert b2_table[touched_key] != family.A.chain_values[family.B2.hop_j - 1]


# -- rho (§2.5): falls out of the walk, checked per arm ------------------------


def test_rho_matches_walk_semantics():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        L = len(family.A.chain_keys)
        j = family.B1.hop_j

        assert family.A.rho == L - 1
        assert family.D.rho == L - 1
        assert family.C.rho == 0
        # B1: fails to retrieve at hop j -> j-1 successful hops
        assert family.B1.rho == j - 1
        # B2: hop j itself succeeds (retrieves the terminal); j+1 fails -> j successful hops
        assert family.B2.rho == j


def test_rho_not_conflated_between_b1_and_b2_at_same_hop():
    """The two arms share the same hop_j but must not share the same rho —
    this is the distinction the doc's 'prefix length for B1 and B2' phrasing
    does not spell out on its own."""
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        if family.B1.hop_j > 1:  # trivial when hop_j==1: both would be 0 vs 1, still distinct
            assert family.B1.rho != family.B2.rho
        assert family.B2.rho == family.B1.rho + 1


# -- twin identity (§1.4): B1/B2 same count as A, only C is shorter ----------


def test_arm_d_token_count_matches_c_only():
    gen = make_generator(chain_length=3, n_distractors=24, n_hard=2)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        assert len(family.D.memory) == len(family.C.memory)
        assert len(family.A.memory) == len(family.C.memory) + 1
        assert len(family.B1.memory) == len(family.A.memory)
        assert len(family.B2.memory) == len(family.A.memory)


# -- requirement 1: per-hop hard negatives -------------------------------------


def test_per_hop_hard_negatives_present():
    n_hard = 3
    gen = make_generator(chain_length=3, n_distractors=18, n_hard=n_hard)
    for seed in range(20):
        gen.rng.seed(seed)
        family = gen.generate_family()
        cache_keys = [e.key for e in family.A.memory]
        for hop_key in family.A.chain_keys:
            n_at_distance_1 = sum(1 for k in cache_keys if hamming(k, hop_key) == 1)
            assert n_at_distance_1 >= n_hard, (
                "hop hard negatives must not be clustered only around start_key"
            )


# -- requirement 2: hard negatives don't continue the active chain ------------


def test_distractors_never_continue_the_chain():
    """No distractor may equal a real chain value — but END is the
    documented exception (§3.1.6's decoys), checked separately above."""
    gen = make_generator(chain_length=3, n_distractors=18, n_hard=3)
    for seed in range(30):
        gen.rng.seed(seed)
        family = gen.generate_family()
        real_chain_values = set(family.A.chain_values) - {END}
        distractor_values = {
            e.value for e in family.A.memory if e.key not in family.A.chain_keys
        }
        assert distractor_values.isdisjoint(real_chain_values)


# -- requirement 6: shuffle carries no ordinal-position signal ----------------


def test_no_ordinal_position_signal():
    gen = make_generator(chain_length=2, n_distractors=24, n_hard=2)
    n_samples = 400
    relative_positions = []
    for seed in range(n_samples):
        gen.rng.seed(seed)
        family = gen.generate_family()
        item = family.A
        target_key = item.chain_keys[0]  # start_key's own entry
        idx = next(i for i, e in enumerate(item.memory) if e.key == target_key)
        relative_positions.append(idx / (len(item.memory) - 1))

    mean_pos = sum(relative_positions) / n_samples
    assert 0.4 < mean_pos < 0.6
    assert min(relative_positions) < 0.15
    assert max(relative_positions) > 0.85


# -- requirement 7: follow-path uniqueness, tolerant of stray END values -----


def test_follow_path_unique_even_with_extra_end_elsewhere():
    """Multiple END-valued entries are expected and fine — only the
    reachable path from start_key must be unique. Simulated here by forcing
    a distractor's value to END directly (outside the normal generator
    path, which never emits END on distractors) and confirming the walk
    still resolves correctly."""
    gen = make_generator(chain_length=2, n_distractors=24, n_hard=1)
    gen.rng.seed(0)
    family = gen.generate_family()
    from generator.task_generator import CacheEntry

    mutated = list(family.A.memory)
    for i, e in enumerate(mutated):
        if e.key not in family.A.chain_keys:
            mutated[i] = CacheEntry(e.key, END)
            break

    from generator import walk

    table = {e.key: e.value for e in mutated}
    visited, terminal = walk(table, family.A.start_key, len(mutated) + 1)
    assert terminal == "END"
    assert visited == family.A.chain_keys


# -- §3.1.6: the true terminus must not be the only findable END entry ------


def test_at_least_min_end_decoys_present():
    """A model could otherwise answer every item by finding "the" END
    entry and emitting its key, in one lookup, without ever following the
    chain — regardless of chain length. Checked on the arms where genuine
    terminus-finding matters (A, B1, B2, D); C's start_key is unreachable
    regardless of how many END entries exist, so it doesn't need this."""
    from generator.task_generator import END, MIN_END_DECOYS

    gen = make_generator(chain_length=3, n_distractors=40, n_hard=2)
    for seed in range(20):
        gen.rng.seed(seed)
        family = gen.generate_family()
        # A/D always keep the true terminus alongside the decoys — their
        # chain entries are never touched. B1 and B2 can legitimately drop
        # to exactly MIN_END_DECOYS: if hop_j == L, the entry being touched
        # *is* the true (kL, END) entry — B1 removes it (missing key at the
        # broken hop), B2 overwrites its value with a terminal (dead end at
        # the broken hop) — either way the true terminus is gone, by design.
        for item in (family.A, family.D):
            n_end = sum(1 for e in item.memory if e.value == END)
            assert n_end >= 1 + MIN_END_DECOYS, item.arm
        for item in (family.B1, family.B2):
            n_end = sum(1 for e in item.memory if e.value == END)
            assert n_end >= MIN_END_DECOYS, item.arm


def test_end_decoy_keys_unreachable_from_start_key():
    """Decoys must not be reachable via the true walk, or they'd change
    the answer rather than just adding noise."""
    from generator.task_generator import END

    gen = make_generator(chain_length=3, n_distractors=40, n_hard=2)
    for seed in range(20):
        gen.rng.seed(seed)
        family = gen.generate_family()
        item = family.A
        decoy_keys = {e.key for e in item.memory if e.value == END} - {item.chain_keys[-1]}
        assert decoy_keys.isdisjoint(item.chain_keys)


def test_config_rejects_too_few_end_decoys():
    with pytest.raises(ValueError):
        GeneratorConfig(
            alphabet_size=16, chain_length=3, n_distractors=40, n_hard=2, n_end_decoys=7,
        )


def test_config_rejects_end_decoys_over_budget():
    with pytest.raises(ValueError):
        GeneratorConfig(
            alphabet_size=16, chain_length=1, n_distractors=7, n_hard=0, n_end_decoys=8,
        )


# -- §1.5: difficulty axes vary independently ----------------------------------


def test_axis_orthogonality_cache_size_tracks_only_length_and_n_distractors():
    sizes_across_n_hard = set()
    for n_hard in (1, 2, 3):
        gen = make_generator(chain_length=3, n_distractors=24, n_hard=n_hard, seed=1)
        family = gen.generate_family()
        sizes_across_n_hard.add(len(family.A.memory))
    assert len(sizes_across_n_hard) == 1

    base = make_generator(chain_length=3, n_distractors=24, n_hard=2, seed=1).generate_family()
    bigger = make_generator(chain_length=3, n_distractors=32, n_hard=2, seed=1).generate_family()
    assert len(bigger.A.memory) - len(base.A.memory) == 32 - 24

    len1 = make_generator(chain_length=1, n_distractors=24, n_hard=2, seed=1).generate_family()
    len3 = make_generator(chain_length=3, n_distractors=24, n_hard=2, seed=1).generate_family()
    assert len(len3.A.memory) - len(len1.A.memory) == 3 - 1


def test_axis_orthogonality_full_grid_generates():
    for chain_length in (1, 2, 3):
        for n_distractors in (16, 24, 40):
            for n_hard in (0, 1, 2):
                if n_hard * chain_length + MIN_END_DECOYS > n_distractors:
                    continue
                gen = make_generator(
                    alphabet_size=32, chain_length=chain_length, n_distractors=n_distractors,
                    n_hard=n_hard, seed=chain_length * 100 + n_distractors + n_hard,
                )
                family = gen.generate_family()
                assert len(family.A.memory) == chain_length + n_distractors


# -- per-item graph resampling (§1.2): never a fixed global graph -------------


def test_successive_items_resample_the_graph():
    gen = make_generator()
    chains = set()
    for _ in range(20):
        family = gen.generate_family()
        chains.add(tuple(family.A.chain_keys))
    assert len(chains) > 1
