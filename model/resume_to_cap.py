"""Redefines S*: not first touch, but the first step after which criterion
holds at every subsequent evaluation — no window-length parameter, the
transient spike is excluded by construction. Requires training every
pilot to a fixed cap regardless of when criterion is first met, so the
full validation history can be scanned after the fact.

Resumes each pilot from its latest checkpoint to CAP=30,000 (~4x where
pilot 2 originally stabilized), reusing prior compute: parses the
existing log files for the 0->7600 history already on disk instead of
regenerating it, then extends to the cap and combines."""

from __future__ import annotations

import json
import re
import sys

import torch

from model.data import BatchPrefetcher
from model.geometry import measure_causal_participation_ratio, measure_query_participation_ratio
from model.pilot_train import (
    ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, N_HARD, N_DISTRACTORS,
    LR, CHAIN_LENGTHS, TRAIN_SEED_BASE, GEOM_SEED, N_GEOM, VAL_EVERY,
    build_fixed_batch, validate, CRITERION,
)
from model.task_model import GatedCacheModel

CAP = 30_000  # overridden via sys.argv in __main__
N_WORKERS = 4  # three of these run concurrently on a 12-core box

LOG_LINE = re.compile(
    r"step\s+(\d+)\s+loss=[\d.]+\s+val_acc=\{1:\s*([\d.]+),\s*2:\s*([\d.]+),\s*3:\s*([\d.]+)\}"
)


def parse_history(*log_paths: str) -> list[tuple[int, dict]]:
    history: dict[int, dict] = {}
    for path in log_paths:
        try:
            with open(path) as f:
                text = f.read()
        except FileNotFoundError:
            continue
        for m in LOG_LINE.finditer(text):
            step, a1, a2, a3 = m.groups()
            history[int(step)] = {1: float(a1), 2: float(a2), 3: float(a3)}
    return sorted(history.items())


def first_stable_step(history: list[tuple[int, dict]]) -> int | None:
    """S* = first step after which criterion holds at every subsequent
    evaluation. Scan from the end; find the last failing checkpoint; S* is
    whatever comes right after it. None if never stable."""
    last_fail_idx = None
    for i, (step, accs) in enumerate(history):
        if not all(a >= CRITERION for a in accs.values()):
            last_fail_idx = i
    if last_fail_idx is None:
        return history[0][0] if history else None
    if last_fail_idx + 1 >= len(history):
        return None  # last checkpoint itself failed: never stabilized
    return history[last_fail_idx + 1][0]


def resume(pilot_idx: int, checkpoint_path: str, start_step: int, prior_log_paths: list[str], device, val_batches, geom_batch, log):
    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    log(f"=== resuming pilot {pilot_idx} from {checkpoint_path} (step {start_step}) to cap {CAP} ===")
    prefetcher = BatchPrefetcher(
        alphabet_size=ALPHABET_SIZE, n_distractors=N_DISTRACTORS, n_hard=N_HARD,
        batch_size=64, chain_lengths=CHAIN_LENGTHS,
        seed_stream_start=TRAIN_SEED_BASE + pilot_idx * 10_000_000 + start_step,
        n_workers=N_WORKERS, depth=12,
    )
    new_history: list[tuple[int, dict]] = []
    try:
        for step in range(start_step, CAP):
            batch = prefetcher.next_batch(device)
            loss, logs = model.forward_train(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % VAL_EVERY == 0 or step == CAP - 1:
                accs = validate(model, val_batches)
                new_history.append((step, accs))
                log(f"pilot {pilot_idx} step {step:6d}  loss={logs['loss']:.4f}  "
                    f"val_acc={ {k: round(v,3) for k,v in accs.items()} }")
    finally:
        prefetcher.close()

    torch.save(model.state_dict(), f"runs/pilot_{pilot_idx}_capped.pt")

    full_history = parse_history(*prior_log_paths) + new_history
    full_history = sorted(dict(full_history).items())
    s_star = first_stable_step(full_history)

    causal_pr, _ = measure_causal_participation_ratio(model, geom_batch)
    query_pr, _ = measure_query_participation_ratio(model, geom_batch)
    final_accs = full_history[-1][1]

    result = {
        "pilot_idx": pilot_idx, "s_star": s_star, "cap": CAP,
        "reached_stable_criterion": s_star is not None,
        "final_val_acc": final_accs,
        "causal_pr": causal_pr, "query_pr": query_pr,
        "history": full_history,
    }
    with open(f"runs/pilot_{pilot_idx}_capped_result.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"pilot {pilot_idx} DONE: s_star={s_star}  causal_PR={causal_pr:.3f}  query_PR={query_pr:.3f}  "
        f"final_acc={final_accs}")
    print(f"RESULT pilot={pilot_idx} s_star={s_star} causal_PR={causal_pr:.4f} query_PR={query_pr:.4f}", flush=True)


def main(pilot_idx: int, checkpoint_path: str, start_step: int, prior_logs: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg):
        print(f"[pilot{pilot_idx}] {msg}", flush=True)

    val_batches = {L: build_fixed_batch(L, 128, 900_000 + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)
    resume(pilot_idx, checkpoint_path, start_step, prior_logs, device, val_batches, geom_batch, log)


if __name__ == "__main__":
    idx = int(sys.argv[1])
    ckpt = sys.argv[2]
    start = int(sys.argv[3])
    cap_override = int(sys.argv[4])
    logs = sys.argv[5:]
    CAP = cap_override
    main(idx, ckpt, start, logs)
