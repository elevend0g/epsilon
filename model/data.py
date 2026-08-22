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

from generator import END, GeneratorConfig, TaskGenerator


def make_batch_cpu(
    alphabet_size: int,
    chain_length: int,
    n_distractors: int,
    n_hard: int,
    batch_size: int,
    seed: int | None = None,
) -> dict:
    gen = TaskGenerator(
        GeneratorConfig(
            alphabet_size=alphabet_size, chain_length=chain_length,
            n_distractors=n_distractors, n_hard=n_hard, seed=seed,
        )
    )
    items = [gen.generate_family().A for _ in range(batch_size)]
    N = len(items[0].memory)
    L = chain_length

    keys = torch.zeros(batch_size, N, 3, dtype=torch.long)
    values = torch.zeros(batch_size, N, 3, dtype=torch.long)
    is_end = torch.zeros(batch_size, N, dtype=torch.bool)
    start_key = torch.zeros(batch_size, 3, dtype=torch.long)
    target_idx = torch.zeros(batch_size, L, dtype=torch.long)

    for b, item in enumerate(items):
        key_to_idx = {}
        for i, entry in enumerate(item.memory):
            keys[b, i] = torch.tensor(entry.key, dtype=torch.long)
            if entry.value == END:
                is_end[b, i] = True
            else:
                values[b, i] = torch.tensor(entry.value, dtype=torch.long)
            key_to_idx[entry.key] = i
        start_key[b] = torch.tensor(item.start_key, dtype=torch.long)
        for t, ck in enumerate(item.chain_keys):
            target_idx[b, t] = key_to_idx[ck]

    return {
        "keys": keys, "values": values, "is_end": is_end,
        "start_key": start_key, "target_idx": target_idx, "L": L,
    }


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_batch(
    alphabet_size: int, chain_length: int, n_distractors: int, n_hard: int,
    batch_size: int, device: torch.device, seed: int | None = None,
) -> dict:
    return to_device(
        make_batch_cpu(alphabet_size, chain_length, n_distractors, n_hard, batch_size, seed),
        device,
    )


class BatchPrefetcher:
    """Keeps `depth` batch-generation jobs in flight across worker
    processes at all times. `next_batch(device)` blocks only if generation
    has fallen behind consumption."""

    def __init__(
        self, alphabet_size: int, n_distractors: int, n_hard: int, batch_size: int,
        chain_lengths: tuple[int, ...], seed_stream_start: int, n_workers: int = 10, depth: int = 20,
    ):
        self.alphabet_size = alphabet_size
        self.n_distractors = n_distractors
        self.n_hard = n_hard
        self.batch_size = batch_size
        self.chain_lengths = chain_lengths
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
        L = self.chain_lengths[self._step % len(self.chain_lengths)]
        seed = self._next_seed
        self._next_seed += 1
        self._step += 1
        fut = self.pool.submit(
            make_batch_cpu, self.alphabet_size, L, self.n_distractors, self.n_hard,
            self.batch_size, seed,
        )
        self.pending.append(fut)

    def next_batch(self, device: torch.device) -> dict:
        fut = self.pending.popleft()
        batch = fut.result()
        self._submit()
        return to_device(batch, device)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
