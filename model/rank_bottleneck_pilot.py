"""§3.1.7: one discarded pilot trained with the query projection actually
bottlenecked to rank=36 (§3.1.5's measured value) — the first time in this
project anything has trained with the bottleneck applied at all, since
§3.1.5 mandates full-rank pilots by construction. Same architecture,
generator config, cosine schedule, and §3.2 terminal-checkpoint criterion
as the official pilots; only the query projection's rank differs.

Discarded regardless of outcome. Pass/fail is the finding: if competence
doesn't survive the bottleneck, that's reported before any of the six
real-seed runs (§2.6) are launched at this rank, not discovered during one.
"""

from __future__ import annotations

import sys

import torch

from model.pilot_train import (
    CHAIN_LENGTHS, GEOM_SEED, N_GEOM, N_VAL, VAL_SEED_BASE,
    build_fixed_batch, run_pilot,
)

RANK = 36
PILOT_IDX = 99  # distinct from 0/1/2, and from its own seed stream
N_WORKERS = 10


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"device={device}  rank-bottleneck throwaway pilot, rank={RANK}")
    val_batches = {L: build_fixed_batch(L, N_VAL, VAL_SEED_BASE + L, device) for L in CHAIN_LENGTHS}
    geom_batch = build_fixed_batch(3, N_GEOM, GEOM_SEED, device)

    result = run_pilot(
        PILOT_IDX, device, val_batches, geom_batch, log,
        n_workers=N_WORKERS, rank=RANK, checkpoint_name="rank_bottleneck_pilot",
    )
    verdict = "PASS — bottleneck survivable" if result["reached_stable_criterion"] else "FAIL — bottleneck broke competence"
    log(f"\n=== §3.1.7 VERDICT: {verdict} ===")
    log(f"s_star={result['s_star']}  final_val_acc={result['final_val_acc']}  "
        f"causal_PR={result['causal_pr']:.3f}  query_PR={result['query_pr']:.3f}")
    print("THROWAWAY_PILOT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
