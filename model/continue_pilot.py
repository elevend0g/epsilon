"""Resume a pilot past its own criterion, to the same total step count
pilot 2 needed (7,600), then re-measure both PRs on the identical
geometry batch. Tests whether pilots 0/1 are pilot 2 slowed down (would
migrate toward its causal-up/query-down profile) or a genuinely
different regime (holds at its own profile regardless of extra steps).

Optimizer state wasn't checkpointed alongside the model, so Adam's
moment estimates restart at zero on resume — a minor deviation, expected
to wash out within the first handful of steps and not to affect where
7,600 steps of training end up."""

from __future__ import annotations

import sys

import torch

from model.data import BatchPrefetcher
from model.geometry import measure_causal_participation_ratio, measure_query_participation_ratio
from model.pilot_train import (
    ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, N_HARD, N_DISTRACTORS,
    LR, CHAIN_LENGTHS, TRAIN_SEED_BASE, GEOM_SEED, N_GEOM, VAL_EVERY,
    build_fixed_batch, validate,
)
from model.task_model import GatedCacheModel


TARGET_STEPS = 7600
N_WORKERS = 5  # reduced: two of these may run concurrently on a 12-core box


def continue_pilot(pilot_idx: int, start_step: int, device, val_batches: dict, geom_batch: dict, log):
    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    model.load_state_dict(torch.load(f"runs/pilot_{pilot_idx}.pt", map_location=device, weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    log(f"=== continuing pilot {pilot_idx} from step {start_step} to {TARGET_STEPS} ===")
    prefetcher = BatchPrefetcher(
        alphabet_size=ALPHABET_SIZE, n_distractors=N_DISTRACTORS, n_hard=N_HARD,
        batch_size=64, chain_lengths=CHAIN_LENGTHS,
        seed_stream_start=TRAIN_SEED_BASE + pilot_idx * 10_000_000 + start_step,
        n_workers=N_WORKERS, depth=16,
    )
    try:
        for step in range(start_step, TARGET_STEPS):
            batch = prefetcher.next_batch(device)
            loss, logs = model.forward_train(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % VAL_EVERY == 0 or step == TARGET_STEPS - 1:
                accs = validate(model, val_batches)
                log(f"pilot {pilot_idx} step {step:6d}  loss={logs['loss']:.4f}  "
                    f"val_acc={ {k: round(v,3) for k,v in accs.items()} }")
    finally:
        prefetcher.close()

    torch.save(model.state_dict(), f"runs/pilot_{pilot_idx}_continued.pt")
    causal_pr, _ = measure_causal_participation_ratio(model, geom_batch)
    query_pr, _ = measure_query_participation_ratio(model, geom_batch)
    log(f"pilot {pilot_idx} @ {TARGET_STEPS} steps: causal_PR={causal_pr:.3f}  query_PR={query_pr:.3f}")
    return causal_pr, query_pr


def main(pilot_idx: int, start_step: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg):
        print(f"[pilot{pilot_idx}] {msg}", flush=True)

    val_batches = {L: build_fixed_batch(L, 128, 900_000 + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)
    causal_pr, query_pr = continue_pilot(pilot_idx, start_step, device, val_batches, geom_batch, log)
    print(f"RESULT pilot={pilot_idx} causal_PR={causal_pr:.4f} query_PR={query_pr:.4f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
