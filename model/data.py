"""Batches arm-A items (fixed chain_length within a batch) from the
generator into tensors for GatedCacheModel.forward_train /
forward_eval_autoregressive.

Generation is CPU-bound (per-hop hard-negative rejection sampling) and,
at cache sizes near the PREREG.md target (~1024 entries), far slower than
the GPU training step itself. BatchPrefetcher parallelizes generation
across worker processes so training isn't serialized behind it — this
changes nothing about what gets generated, only how fast batches arrive.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import torch

from generator import ABSTAIN, END, GeneratorConfig, TaskGenerator


def make_batch_cpu(
    alphabet_size: int,
    chain_length: int,
    n_distractors: int,
    n_hard: int,
    batch_size: int,
    seed: int | None = None,
    arm: str = "A",
) -> dict:
    """arm: any of "A", "B1", "B2", "C", "D". "A" (default) matches every
    pilot to date; Regime 1 (§2.4) also uses "D"; Regime 2 uses all five.

    Retrieval targets and the abstain label are read mechanically off
    fields the generator already computes (item.rho, item.answer,
    item.chain_keys) — no arm-specific branching needed here, since the
    same formula is correct for every arm:

      n_steps = rho if answer is ABSTAIN else rho + 1

    For an END-terminal arm (A, D), the walk succeeds through its own
    is_end-flagged entry, which IS a valid retrieval target — hence
    rho+1 valid steps (matches item.chain_keys exactly, L of them). For a
    DEAD-terminal arm (B1, B2, C), the walk's last position has no cache
    entry at all by construction (that absence *is* the dead end) — hence
    only the first `rho` positions of chain_keys are valid retrieval
    targets; the abstain head, not a retrieval index, supervises the step
    where the walk actually breaks (docs/phase1.md §2.4)."""
    gen = TaskGenerator(
        GeneratorConfig(
            alphabet_size=alphabet_size, chain_length=chain_length,
            n_distractors=n_distractors, n_hard=n_hard, seed=seed,
        )
    )
    items = [getattr(gen.generate_family(), arm) for _ in range(batch_size)]
    N = len(items[0].memory)
    L = chain_length

    # Build plain Python lists first, one bulk torch.tensor() per field at
    # the end — was one torch.tensor() call per scalar (652,800 calls for
    # a single 64-item batch), profiled as ~18% of total batch time.
    keys_list: list[list[tuple]] = []
    values_list: list[list[tuple]] = []
    is_end_list: list[list[bool]] = []
    start_key_list: list[tuple] = []
    target_idx_list: list[list[int]] = []
    n_steps_list: list[int] = []
    is_abstain_list: list[bool] = []
    rho_list: list[int] = []
    zero_tuple = (0, 0, 0)

    for item in items:
        key_to_idx: dict = {}
        item_keys = []
        item_values = []
        item_is_end = []
        for i, entry in enumerate(item.memory):
            item_keys.append(entry.key)
            if entry.value == END:
                item_is_end.append(True)
                item_values.append(zero_tuple)
            else:
                item_is_end.append(False)
                item_values.append(entry.value)
            key_to_idx[entry.key] = i
        keys_list.append(item_keys)
        values_list.append(item_values)
        is_end_list.append(item_is_end)
        start_key_list.append(item.start_key)

        is_abstain = item.answer == ABSTAIN
        n_steps = item.rho if is_abstain else item.rho + 1
        retrieval_keys = item.chain_keys[:n_steps]
        idxs = [key_to_idx[ck] for ck in retrieval_keys]
        idxs += [0] * (L - len(idxs))  # pad to L with a dummy index; masked out by n_steps downstream
        target_idx_list.append(idxs)
        n_steps_list.append(n_steps)
        is_abstain_list.append(is_abstain)
        rho_list.append(item.rho)

    keys = torch.tensor(keys_list, dtype=torch.long)
    values = torch.tensor(values_list, dtype=torch.long)
    is_end = torch.tensor(is_end_list, dtype=torch.bool)
    start_key = torch.tensor(start_key_list, dtype=torch.long)
    target_idx = torch.tensor(target_idx_list, dtype=torch.long)
    n_steps = torch.tensor(n_steps_list, dtype=torch.long)
    is_abstain_t = torch.tensor(is_abstain_list, dtype=torch.bool)
    rho = torch.tensor(rho_list, dtype=torch.long)

    return {
        "keys": keys, "values": values, "is_end": is_end,
        "start_key": start_key, "target_idx": target_idx, "L": L,
        "n_steps": n_steps, "is_abstain": is_abstain_t, "rho": rho,
    }


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_batch(
    alphabet_size: int, chain_length: int, n_distractors: int, n_hard: int,
    batch_size: int, device: torch.device, seed: int | None = None, arm: str = "A",
) -> dict:
    return to_device(
        make_batch_cpu(alphabet_size, chain_length, n_distractors, n_hard, batch_size, seed, arm),
        device,
    )


class BatchPrefetcher:
    """Keeps `depth` batch-generation jobs in flight across worker
    processes at all times. `next_batch(device)` blocks only if generation
    has fallen behind consumption."""

    def __init__(
        self, alphabet_size: int, n_distractors: int, n_hard: int, batch_size: int,
        chain_lengths: tuple[int, ...], seed_stream_start: int, n_workers: int = 10, depth: int = 20,
        arms: tuple[str, ...] = ("A",),
    ):
        self.alphabet_size = alphabet_size
        self.n_distractors = n_distractors
        self.n_hard = n_hard
        self.batch_size = batch_size
        self.chain_lengths = chain_lengths
        # Cartesian product, not zipped: every (L, arm) combination gets
        # equal representation, cycled in a fixed round-robin. Regime 1
        # (§2.4) is arms=("A","D"); pilots keep the default arms=("A",).
        self.combos = [(L, a) for L in chain_lengths for a in arms]
        self.depth = depth
        self._next_seed = seed_stream_start
        self._step = 0
        # "spawn", not the default "fork": forking a process that has
        # already touched CUDA (the training process has, by the time this
        # is constructed) inherits a broken/partial CUDA context in the
        # children and deadlocks the whole pool on a futex, silently, with
        # no exception raised anywhere. Workers here never touch CUDA at
        # all (make_batch_cpu is pure CPU), so spawn's slower startup is
        # the only cost, not a correctness tradeoff.
        ctx = multiprocessing.get_context("spawn")
        self.pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        self.pending: deque = deque()
        for _ in range(depth):
            self._submit()

    def _submit(self) -> None:
        L, arm = self.combos[self._step % len(self.combos)]
        seed = self._next_seed
        self._next_seed += 1
        self._step += 1
        fut = self.pool.submit(
            make_batch_cpu, self.alphabet_size, L, self.n_distractors, self.n_hard,
            self.batch_size, seed, arm,
        )
        self.pending.append(fut)

    def next_batch(self, device: torch.device) -> dict:
        fut = self.pending.popleft()
        batch = fut.result()
        self._submit()
        return to_device(batch, device)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
