"""Real-seed training: full budget (102,400 steps), rank=36 (§3.1.5,
confirmed by §3.1.7), Regime 1 (arms A+D, §2.4). Regime 2 (all five arms,
B1/B2/C -> ABSTAIN) needs an explicit abstain output path the model
doesn't have yet — deferred until the second run in the interleaved
sequence actually needs it, not blocking this one.

First run is a canary before the full six-run matrix, per docs/phase1.md
§2.6: S*=15,900 (§3.1.7) came from a 30,000-step pilot; real runs go to
102,400, almost 7x further, at ~100 checkpoints instead of ~30. Disk
growth, resume across that many checkpoints, and terminal-checkpoint
evaluation at full length are all untested at this scale.
"""

from __future__ import annotations

import sys

import torch

from model.pilot_train import (
    ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, N_HARD, N_DISTRACTORS,
    CHAIN_LENGTHS, VAL_SEED_BASE, GEOM_SEED, N_GEOM, N_VAL,
    build_fixed_batch, run_pilot,
)
import model.pilot_train as pt

RANK = 36
REAL_BUDGET = 102_400
REGIME_1_ARMS = ("A", "D")


def main(seed_idx: int, n_workers: int) -> None:
    pt.STEP_CAP = REAL_BUDGET  # module-level override, matches pilot_train's own __main__ pattern

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        print(f"[real_seed{seed_idx}] {msg}", flush=True)

    log(f"=== real seed {seed_idx}, Regime 1 (arms={REGIME_1_ARMS}), rank={RANK}, "
        f"budget={REAL_BUDGET} steps ===")

    val_batches = {L: build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)

    result = run_pilot(
        seed_idx, device, val_batches, geom_batch, log,
        n_workers=n_workers, rank=RANK, checkpoint_name=f"real_seed_r1_{seed_idx}",
        arms=REGIME_1_ARMS,
    )
    print(f"RESULT real_seed={seed_idx} regime=1 s_star={result['s_star']} "
          f"causal_PR={result['causal_pr']:.4f} query_PR={result['query_pr']:.4f}", flush=True)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    main(idx, workers)
