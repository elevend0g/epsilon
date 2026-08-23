"""Real-seed training: full budget (102,400 steps), rank=36, Regime 2
(all five arms, B1/B2/C -> ABSTAIN via a separate trained head, §2.4).

Second run in the interleaved launch order (docs/phase1.md §2.6):
R1seed0, R2seed0, R1seed1, R2seed1, R1seed2, R2seed2 — R1seed0 (the
canary) has already cleared (S*=23,800, causal_PR=25.35, query_PR=26.99,
runs/real_seed_r1_0_result.json). This is the first Regime 2 run and
the first real use of forward_train_regime2 / the abstain head outside
of a tiny dry run (runs/regime2_dryrun*, 16-20 steps, deleted after
confirming the loss decomposition and masking behaved correctly).

Arm sampling and loss weight are pinned in PREREG.md, not left as
dataloader defaults: arms=(A,B1,B2,C,D) at equal frequency (nominal 40%
negative / 60% positive class balance for the abstain head), lambda=1
(abstain BCE added unweighted to retrieval loss + count penalty). Both
must be checked against what this run actually measures, not assumed —
the per-step log line's abstain_class_balance and abstain_loss fields
carry that.
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
REGIME_2_ARMS = ("A", "B1", "B2", "C", "D")


def main(seed_idx: int, n_workers: int) -> None:
    pt.STEP_CAP = REAL_BUDGET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        print(f"[real_seed_r2_{seed_idx}] {msg}", flush=True)

    log(f"=== real seed {seed_idx}, Regime 2 (arms={REGIME_2_ARMS}), rank={RANK}, "
        f"budget={REAL_BUDGET} steps ===")

    val_batches = {L: build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)

    result = run_pilot(
        seed_idx, device, val_batches, geom_batch, log,
        n_workers=n_workers, rank=RANK, checkpoint_name=f"real_seed_r2_{seed_idx}",
        arms=REGIME_2_ARMS, regime2=True,
    )
    print(f"RESULT real_seed={seed_idx} regime=2 s_star={result['s_star']} "
          f"causal_PR={result['causal_pr']:.4f} query_PR={result['query_pr']:.4f}", flush=True)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    main(idx, workers)
