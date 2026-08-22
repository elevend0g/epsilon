"""§3.1 + §3.1.5: three discarded pilot seeds, full-rank query projection,
cosine LR schedule, trained to a fixed cap (no early stopping) so `S*` —
the onset of stable competence, not first touch — can be computed from
the full validation history. Then measures query PR (rank-setting) and
causal PR (§6.1, informational) on the terminal checkpoint.

Cache size (n_distractors=1021, up to 1024 entries at L=3) and model size
(d_model=64, d_state=8) match PREREG.md's target exactly — both §1.3
bounds hold at this scale (verified by generator/capacity.py).

Optimizer: constant LR=3e-3 was tried first and never held §3.2 criterion
through 8,100 steps on any of three pilots, oscillating 85-97% on a fixed
validation set. A disconfirming test (RECIPE_LOG.md) showed a cosine
decay to 3e-5 eliminates the oscillation entirely, tracking the LR cut.
That is the schedule used here.

Generation at this scale is CPU-bound and far slower than the GPU step
itself, so training batches are produced by a BatchPrefetcher pool
(model/data.py) running across worker processes, not generated serially
in the training loop. Run as `python -m model.pilot_train <idx> [n_workers]`
per pilot (three concurrent processes share the machine's cores) or with
no argument to run all three sequentially.
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch

from model.data import BatchPrefetcher, make_batch
from model.geometry import measure_causal_participation_ratio, measure_query_participation_ratio
from model.task_model import GatedCacheModel

ALPHABET_SIZE = 64
N_DISTRACTORS = 1021  # + L(<=3) chain keys = up to 1024 entries, matching PREREG.md's target
N_HARD = 4
D_MODEL = 64
D_STATE = 8
SYMBOL_DIM = 32
BATCH_SIZE = 64
START_LR = 3e-3
FLOOR_LR = 3e-5
STEP_CAP = 30_000  # §3.1: ~4x where the slowest constant-LR pilot stabilized
VAL_EVERY = 100
N_VAL = 128
N_GEOM = 30
CRITERION = 0.95
CHAIN_LENGTHS = (1, 2, 3)
N_WORKERS = 10
PREFETCH_DEPTH = 24

VAL_SEED_BASE = 900_000       # reserved: validation split, never used for training
GEOM_SEED = 800_003            # reserved: geometry split, never used for training/validation
TRAIN_SEED_BASE = 100_000_000  # per-pilot, per-step: TRAIN_SEED_BASE + pilot_idx*10_000_000 + step


def lr_at(step: int) -> float:
    """Cosine, START_LR -> FLOOR_LR, decaying across the full STEP_CAP."""
    if step >= STEP_CAP:
        return FLOOR_LR
    cos = 0.5 * (1 + math.cos(math.pi * step / STEP_CAP))
    return FLOOR_LR + (START_LR - FLOOR_LR) * cos


def build_fixed_batch(chain_length: int, n_items: int, seed: int, device):
    return make_batch(
        alphabet_size=ALPHABET_SIZE, chain_length=chain_length, n_distractors=N_DISTRACTORS,
        n_hard=N_HARD, batch_size=n_items, device=device, seed=seed,
    )


def validate(model, val_batches: dict) -> dict:
    accs = {}
    for L, batch in val_batches.items():
        correct = model.forward_eval_autoregressive(batch, max_steps=8)
        accs[L] = correct.float().mean().item()
    model.train()
    return accs


def first_stable_step(history: list[tuple[int, dict]]) -> int | None:
    """§3.1: S* = first step after which criterion holds at every
    subsequent evaluation. No window parameter — scan from the end, find
    the last failing checkpoint, S* is whatever comes right after it.
    None if the last checkpoint itself fails (never stabilized)."""
    last_fail_idx = None
    for i, (step, accs) in enumerate(history):
        if not all(a >= CRITERION for a in accs.values()):
            last_fail_idx = i
    if last_fail_idx is None:
        return history[0][0] if history else None
    if last_fail_idx + 1 >= len(history):
        return None
    return history[last_fail_idx + 1][0]


def run_pilot(pilot_idx: int, device, val_batches: dict, geom_batch: dict, log, n_workers: int = N_WORKERS) -> dict:
    torch.manual_seed(1000 + pilot_idx)
    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=START_LR)

    log(f"=== pilot {pilot_idx} start, params={sum(p.numel() for p in model.parameters())}, "
        f"cosine LR {START_LR}->{FLOOR_LR} over {STEP_CAP} steps, no early stopping ===")
    t0 = time.time()
    history: list[tuple[int, dict]] = []

    prefetcher = BatchPrefetcher(
        alphabet_size=ALPHABET_SIZE, n_distractors=N_DISTRACTORS, n_hard=N_HARD,
        batch_size=BATCH_SIZE, chain_lengths=CHAIN_LENGTHS,
        seed_stream_start=TRAIN_SEED_BASE + pilot_idx * 10_000_000,
        n_workers=n_workers, depth=PREFETCH_DEPTH,
    )
    try:
        for step in range(STEP_CAP):
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = prefetcher.next_batch(device)
            loss, logs = model.forward_train(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % VAL_EVERY == 0 or step == STEP_CAP - 1:
                accs = validate(model, val_batches)
                history.append((step, accs))
                elapsed = time.time() - t0
                all_pass = all(a >= CRITERION for a in accs.values())
                log(f"pilot {pilot_idx} step {step:6d}  lr={lr:.2e}  loss={logs['loss']:.4f}  "
                    f"val_acc={ {k: round(v,3) for k,v in accs.items()} }  "
                    f"elapsed={elapsed:.0f}s  {'PASS' if all_pass else ''}")
    finally:
        prefetcher.close()

    s_star = first_stable_step(history)
    final_accs = history[-1][1]
    if s_star is None:
        log(f"pilot {pilot_idx} NEVER reached stable criterion within cap {STEP_CAP}; final_val_acc={final_accs}")
    else:
        log(f"pilot {pilot_idx} stable from step {s_star} onward (terminal check at {STEP_CAP - 1})")

    causal_pr, _ = measure_causal_participation_ratio(model, geom_batch)
    query_pr, _ = measure_query_participation_ratio(model, geom_batch)
    log(f"pilot {pilot_idx} causal_PR={causal_pr:.3f}  query_PR={query_pr:.3f}")

    torch.save(model.state_dict(), f"runs/pilot_{pilot_idx}.pt")
    result = {
        "pilot_idx": pilot_idx, "s_star": s_star, "cap": STEP_CAP,
        "reached_stable_criterion": s_star is not None,
        "final_val_acc": final_accs, "causal_pr": causal_pr, "query_pr": query_pr,
        "wall_time_s": time.time() - t0, "history": history,
    }
    with open(f"runs/pilot_{pilot_idx}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main_single(pilot_idx: int, n_workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        print(f"[pilot{pilot_idx}] {msg}", flush=True)

    val_batches = {L: build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)
    result = run_pilot(pilot_idx, device, val_batches, geom_batch, log, n_workers)
    print(f"RESULT pilot={pilot_idx} s_star={result['s_star']} "
          f"causal_PR={result['causal_pr']:.4f} query_PR={result['query_pr']:.4f}", flush=True)


def main_all() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"device={device}  n_distractors={N_DISTRACTORS} (matches PREREG.md target, ~1024 entries)")

    val_batches = {L: build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)
    log("validation and geometry splits built (fixed, reserved seeds)")

    results = [run_pilot(i, device, val_batches, geom_batch, log) for i in range(3)]
    write_summary(results, log)


def write_summary(results: list[dict], log) -> None:
    s_stars = [r["s_star"] for r in results]
    query_prs = [r["query_pr"] for r in results]
    causal_prs = [r["causal_pr"] for r in results]
    if any(s is None for s in s_stars):
        log("At least one pilot never stabilized — budget/rank not derivable until resolved.")
        return
    budget = min(4 * max(s_stars), 200_000)
    rank = math.ceil(max(query_prs) / 4) * 4

    summary = {
        "pilots": results,
        "s_star_all": s_stars, "s_star_max": max(s_stars), "s_star_spread": max(s_stars) - min(s_stars),
        "training_budget": budget,
        "query_pr_all": query_prs, "query_pr_max": max(query_prs), "query_pr_spread": max(query_prs) - min(query_prs),
        "causal_pr_all": causal_prs, "causal_pr_spread": max(causal_prs) - min(causal_prs),
        "rank": rank,
    }
    log("\n=== SUMMARY ===")
    log(json.dumps({k: v for k, v in summary.items() if k != "pilots"}, indent=2))
    with open("runs/pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("\nwrote runs/pilot_summary.json")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        idx = int(sys.argv[1])
        workers = int(sys.argv[2]) if len(sys.argv) >= 3 else N_WORKERS
        main_single(idx, workers)
    else:
        main_all()
