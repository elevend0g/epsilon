"""Task generator for the follow-until-END multi-hop lookup task.

Implements docs/phase1.md v2 §1: per-item graph resampling, END-terminated
chains, per-hop hard negatives, and the five condition arms (A/B1/B2/C/D).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

Symbol = int
Key = tuple[Symbol, Symbol, Symbol]
END = "END"
Value = Key | str  # a Key tuple, or the reserved sentinel END

ABSTAIN = "ABSTAIN"

Arm = Literal["A", "B1", "B2", "C", "D"]
Terminal = Literal["END", "DEAD", "TRUNCATED"]


@dataclass(frozen=True)
class CacheEntry:
    key: Key
    value: Value


MIN_END_DECOYS = 8  # §1.6 req 7 / §3.1.6: the true terminus must not be the only END-valued entry


@dataclass(frozen=True)
class GeneratorConfig:
    alphabet_size: int
    chain_length: int    # L: difficulty axis, sequential depth, L in {1,2,3}
    n_distractors: int   # difficulty axis: input length
    n_hard: int           # difficulty axis: retrieval difficulty, per hop
    n_end_decoys: int = MIN_END_DECOYS  # entries with value=END, key unreachable from start_key
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.chain_length < 1:
            raise ValueError("chain_length must be >= 1")
        if self.n_hard * self.chain_length > self.n_distractors:
            raise ValueError(
                f"n_hard*chain_length ({self.n_hard * self.chain_length}) exceeds "
                f"n_distractors budget ({self.n_distractors})"
            )
        if self.alphabet_size < 4:
            raise ValueError(
                "alphabet_size must be >= 4 to support distance-1 and distance-3 negatives"
            )
        if self.n_end_decoys < MIN_END_DECOYS:
            raise ValueError(
                f"n_end_decoys ({self.n_end_decoys}) must be >= {MIN_END_DECOYS}: a single "
                f"reachable END entry lets a model answer by finding 'the' END entry directly, "
                f"never following the chain (§3.1.6)."
            )
        if self.n_hard * self.chain_length + self.n_end_decoys > self.n_distractors:
            raise ValueError(
                f"n_hard*chain_length + n_end_decoys "
                f"({self.n_hard * self.chain_length + self.n_end_decoys}) exceeds "
                f"n_distractors budget ({self.n_distractors})"
            )


@dataclass
class TaskItem:
    memory: list[CacheEntry]
    start_key: Key
    answer: Key | str  # the final key kL, or ABSTAIN
    arm: Arm
    # generator-internal ground truth, for tests/analysis — not model-facing.
    chain_keys: list[Key]        # k1..kL
    chain_values: list[Value]    # k2..kL, END
    hop_j: int | None = None          # 1-indexed corrupted/absent hop, arms B1/B2
    terminal_value: Key | None = None  # arm B2 only: the fake "x"
    rho: int = 0                       # §2.5 integration-count target, len(walk)-1


@dataclass
class TaskFamily:
    """One A/B1/B2/C/D quintuple sharing a chain and distractor pool (§1.4)."""

    A: TaskItem
    B1: TaskItem
    B2: TaskItem
    C: TaskItem
    D: TaskItem


class TaskGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_family(self, hop_j: int | None = None) -> TaskFamily:
        L = self.config.chain_length
        chain_keys, chain_values = self._fresh_chain()
        used_keys: dict[Key, Value] = dict(zip(chain_keys, chain_values))
        distractors = self._build_distractors(chain_keys, chain_values, used_keys)
        chain_entries = [CacheEntry(ck, cv) for ck, cv in zip(chain_keys, chain_values)]

        # --- Arm A: full chain + all distractors, shuffled once. ---
        a_shuffled = self._shuffled(chain_entries + distractors)

        if hop_j is None:
            hop_j = self.rng.randrange(1, L) + 1 if L > 1 else 1
        touched_key = chain_keys[hop_j - 1]
        forbidden_values = set(chain_values)  # incl. END: nothing may continue the chain

        # --- Arm B1 (missing key, deletion-free): hop_j's entry is swapped,
        # in place, for an unrelated decoy — same key count and shuffle
        # position as A, but kj itself is no longer findable. ---
        decoy = self._generate_decoy(used_keys, forbidden_values)
        b1_shuffled = [decoy if e.key == touched_key else e for e in a_shuffled]

        # --- Arm B2 (dead end, thesis arm): hop_j's value is swapped, in
        # place, for a terminal x drawn like a key but present nowhere as
        # one — retrieval at hop_j still succeeds; the next one does not. ---
        terminal_value = self._generate_terminal_value(used_keys)
        b2_shuffled = [
            CacheEntry(touched_key, terminal_value) if e.key == touched_key else e
            for e in a_shuffled
        ]

        # --- Arm C: start_key absent, unrelated pool one entry shorter than A. ---
        arm_c = self._generate_arm_c(target_size=len(a_shuffled) - 1)

        # --- Arm D: full chain, distractors trimmed to match C's token count.
        # B1 and B2 are already token-count-identical to A by construction
        # and need no separate length control; only C is shorter. ---
        d_target_distractors = len(arm_c.memory) - L
        d_shuffled = self._shuffled(chain_entries + distractors[:d_target_distractors])

        max_hops = L + 1  # covers B2 at hop_j==L: L real hops plus one fake step
        arm_a = self._finalize(a_shuffled, chain_keys[0], "A", chain_keys, chain_values, max_hops)
        arm_b1 = self._finalize(
            b1_shuffled, chain_keys[0], "B1", chain_keys, chain_values, max_hops, hop_j=hop_j,
        )
        arm_b2 = self._finalize(
            b2_shuffled, chain_keys[0], "B2", chain_keys, chain_values, max_hops,
            hop_j=hop_j, terminal_value=terminal_value,
        )
        arm_c_item = self._finalize(
            arm_c.memory, arm_c.start_key, "C", [], [], max_hops,
        )
        arm_d = self._finalize(d_shuffled, chain_keys[0], "D", chain_keys, chain_values, max_hops)

        return TaskFamily(A=arm_a, B1=arm_b1, B2=arm_b2, C=arm_c_item, D=arm_d)

    def _finalize(
        self,
        memory: list[CacheEntry],
        start_key: Key,
        arm: Arm,
        chain_keys: list[Key],
        chain_values: list[Value],
        max_hops: int,
        hop_j: int | None = None,
        terminal_value: Key | None = None,
    ) -> TaskItem:
        """Answer and rho are not asserted — they fall out of walking the
        item's own cache, the same way a model would have to."""
        table = {e.key: e.value for e in memory}
        visited, terminal = walk(table, start_key, max_hops)
        answer: Key | str = visited[-1] if terminal == "END" else ABSTAIN
        rho = len(visited) - 1
        return TaskItem(
            memory=memory, start_key=start_key, answer=answer, arm=arm,
            chain_keys=chain_keys, chain_values=chain_values,
            hop_j=hop_j, terminal_value=terminal_value, rho=rho,
        )

    # -- chain construction -------------------------------------------------

    def _fresh_chain(self) -> tuple[list[Key], list[Value]]:
        L = self.config.chain_length
        used: set[Key] = set()
        keys: list[Key] = []
        values: list[Value] = []
        key = self._distinct_tuple(used)
        for i in range(L):
            keys.append(key)
            used.add(key)
            if i < L - 1:
                val = self._distinct_tuple(used)
                used.add(val)
                key = val
            else:
                val = END
            values.append(val)
        return keys, values

    def _distinct_tuple(self, used: set[Key]) -> Key:
        for _ in range(10_000):
            candidate = self._random_tuple()
            if candidate not in used:
                return candidate
        raise RuntimeError("alphabet too small to sample distinct chain symbols")

    # -- distractor construction ---------------------------------------------

    def _build_distractors(
        self,
        chain_keys: list[Key],
        chain_values: list[Value],
        used_keys: dict[Key, Value],
    ) -> list[CacheEntry]:
        """§1.6: per-hop hard negatives (distance 1) filling n_hard*L of the
        budget, `n_end_decoys` unreachable entries whose value is END
        (§3.1.6 — the sole exception to the next rule, deliberately: it is
        what prevents "find the END entry" from being a valid shortcut),
        then easy negatives (distance 3) round-robined over hops filling
        the rest. Everything else: no distractor value ever equals a chain
        value, including END (req 2), applied to both hard and easy
        negatives since it is strictly safer and simpler than restricting
        it to hard negatives only (§8)."""
        L = self.config.chain_length
        forbidden_values = set(chain_values)
        entries: list[CacheEntry] = []

        for j in range(L):
            for _ in range(self.config.n_hard):
                entry = self._generate_negative(chain_keys[j], 1, used_keys, forbidden_values)
                used_keys[entry.key] = entry.value
                entries.append(entry)

        # §3.1.6: the true terminus must not be the only entry whose value
        # is END, or "find the END entry, emit its key" answers every item
        # in one lookup regardless of chain length, without ever following
        # it. Decoy keys are fresh and unreachable from start_key.
        for _ in range(self.config.n_end_decoys):
            entry = self._generate_end_decoy(used_keys)
            used_keys[entry.key] = entry.value
            entries.append(entry)

        remaining = self.config.n_distractors - len(entries)
        for i in range(remaining):
            center = chain_keys[i % L]
            entry = self._generate_negative(center, 3, used_keys, forbidden_values)
            used_keys[entry.key] = entry.value
            entries.append(entry)

        return entries

    def _generate_end_decoy(self, used_keys: dict[Key, Value]) -> CacheEntry:
        for _ in range(10_000):
            key = self._random_tuple()
            if key in used_keys:
                continue
            return CacheEntry(key=key, value=END)
        raise RuntimeError("could not place an END decoy without collision; alphabet too small")

    def _generate_negative(
        self,
        center: Key,
        distance: int,
        used_keys: dict[Key, Value],
        forbidden_values: set[Value],
    ) -> CacheEntry:
        for _ in range(10_000):
            key = self._flip_positions(center, distance)
            if key in used_keys:
                continue  # req 5: no duplicate keys with differing values
            value = self._random_tuple()
            if value in forbidden_values:
                continue  # req 2: distractor must not continue the active chain
            return CacheEntry(key=key, value=value)
        raise RuntimeError("could not place a distractor without collision; alphabet too small")

    def _generate_decoy(
        self, used_keys: dict[Key, Value], forbidden_values: set[Value]
    ) -> CacheEntry:
        """§1.4/§1.6 req 3: a fresh entry whose key collides with no hop
        query (used_keys already contains every chain key) and whose value
        does not map into the chain."""
        for _ in range(10_000):
            key = self._random_tuple()
            if key in used_keys:
                continue
            value = self._random_tuple()
            if value in forbidden_values:
                continue
            return CacheEntry(key=key, value=value)
        raise RuntimeError("could not place a B1 decoy without collision; alphabet too small")

    def _generate_terminal_value(self, used_keys: dict[Key, Value]) -> Key:
        """§1.4/§1.6 req 4: a value drawn like a key (same tuple space) that
        is not itself a key anywhere in the cache — cache-absence is the
        only dead-end cue. Not-a-key-anywhere already implies it cannot
        equal kL, since kL is itself a key in used_keys."""
        for _ in range(10_000):
            value = self._random_tuple()
            if value in used_keys:
                continue
            return value
        raise RuntimeError("could not sample a terminal value; alphabet too small")

    def _generate_arm_c(self, target_size: int) -> TaskFamilyC:
        used: set[Key] = set()
        start_key = self._distinct_tuple(used)
        used.add(start_key)
        used_keys: dict[Key, Value] = {}
        entries: list[CacheEntry] = []
        for _ in range(target_size):
            for _ in range(10_000):
                key = self._random_tuple()
                if key == start_key or key in used_keys:
                    continue
                value = self._random_tuple()
                entries.append(CacheEntry(key, value))
                used_keys[key] = value
                break
            else:
                raise RuntimeError("could not fill arm C distractor pool; alphabet too small")
        return TaskFamilyC(start_key=start_key, memory=self._shuffled(entries))

    # -- primitives -----------------------------------------------------------

    def _random_symbol(self) -> Symbol:
        return self.rng.randrange(self.config.alphabet_size)

    def _random_tuple(self) -> Key:
        return (self._random_symbol(), self._random_symbol(), self._random_symbol())

    def _flip_positions(self, base: Key, n_positions: int) -> Key:
        positions = self.rng.sample(range(3), n_positions)
        result = list(base)
        for p in positions:
            original = result[p]
            new_val = original
            while new_val == original:
                new_val = self._random_symbol()
            result[p] = new_val
        return (result[0], result[1], result[2])

    def _shuffled(self, entries: list[CacheEntry]) -> list[CacheEntry]:
        out = list(entries)
        self.rng.shuffle(out)
        return out


@dataclass(frozen=True)
class TaskFamilyC:
    start_key: Key
    memory: list[CacheEntry]


# -- verification helpers (§1.6 reqs 5 and 7; used by generator_tests) --------


def verify_no_duplicate_keys(entries: list[CacheEntry]) -> bool:
    seen: dict[Key, Value] = {}
    for e in entries:
        if e.key in seen and seen[e.key] != e.value:
            return False
        seen[e.key] = e.value
    return True


def walk(table: dict[Key, Value], start: Key, max_hops: int) -> tuple[list[Key], Terminal]:
    """Follow key->value edges from start_key. Returns the sequence of
    pointer positions visited (including a fake terminal value in the B2
    case, which is treated exactly as a model would: as the current key,
    until it too fails to resolve) and how the walk ended.

    Multiple END-valued entries elsewhere in the cache are expected and
    fine (§1.6 req 7) — this only ever follows the one reachable path."""
    current = start
    visited = [start]
    for _ in range(max_hops):
        if current not in table:
            return visited, "DEAD"
        val = table[current]
        if val == END:
            return visited, "END"
        current = val  # type: ignore[assignment]
        visited.append(current)
    return visited, "TRUNCATED"


def verify_follow_path(item: TaskItem) -> bool:
    """The item's own answer/rho must match what walking its own cache
    produces (§1.6 req 7) — checked independently of how they were set."""
    table: dict[Key, Value] = {e.key: e.value for e in item.memory}
    max_hops = len(item.memory) + 1  # generous upper bound; no arm needs more
    visited, terminal = walk(table, item.start_key, max_hops)
    expected_answer: Key | str = visited[-1] if terminal == "END" else ABSTAIN
    expected_rho = len(visited) - 1
    return item.answer == expected_answer and item.rho == expected_rho
