"""Disconfirming test: is the persistent oscillation (pilots 0/1/2 all
failed to hold criterion through step 8,100, including pilot 2 which had
"passed" at 7,600) the optimizer bouncing on a constant LR, or the SSM
driver genuinely not settling?

Takes pilot 2's honest 8,100-step checkpoint (not its peak — its current
state) and continues training with everything identical except a cosine
LR decay, 3e-3 -> 3e-5. Two predictions:
  - recipe: oscillation amplitude damps as LR drops, accuracy crosses
    95% on all three L and holds.
  - architecture: oscillation continues at ~the same amplitude even as
    LR approaches the floor.
Batch size (64) is left untouched deliberately, so this test isolates
one variable."""

from __future__ import annotations

import math

import torch

from model.data import BatchPrefetcher
from model.pilot_train import (
    ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, N_HARD, N_DISTRACTORS,
    CHAIN_LENGTHS, TRAIN_SEED_BASE, VAL_EVERY, build_fixed_batch, validate,
)
from model.task_model import GatedCacheModel

START_STEP = 8100
TOTAL_STEPS = 2500      # ~11-20 min at full worker count, uncontested
DECAY_STEPS = 2000       # cosine 3e-3 -> 3e-5 over this many steps
HOLD_STEPS = TOTAL_STEPS - DECAY_STEPS  # then hold at floor to check settling
START_LR = 3e-3
FLOOR_LR = 3e-5
PILOT_IDX = 2
CHECKPOINT = "runs/pilot_2_capped.pt"
N_WORKERS = 10


def lr_at(step_into_run: int) -> float:
    if step_into_run >= DECAY_STEPS:
        return FLOOR_LR
    cos = 0.5 * (1 + math.cos(math.pi * step_into_run / DECAY_STEPS))
    return FLOOR_LR + (START_LR - FLOOR_LR) * cos


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg):
        print(msg, flush=True)

    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=START_LR)

    val_batches = {L: build_fixed_batch(L, 128, 900_000 + L, device) for L in CHAIN_LENGTHS}

    log(f"=== LR discriminator: pilot {PILOT_IDX} from {CHECKPOINT} (step {START_STEP}), "
        f"cosine {START_LR}->{FLOOR_LR} over {DECAY_STEPS} steps, then hold {HOLD_STEPS} more ===")

    prefetcher = BatchPrefetcher(
        alphabet_size=ALPHABET_SIZE, n_distractors=N_DISTRACTORS, n_hard=N_HARD,
        batch_size=64, chain_lengths=CHAIN_LENGTHS,
        seed_stream_start=TRAIN_SEED_BASE + PILOT_IDX * 10_000_000 + START_STEP,
        n_workers=N_WORKERS, depth=16,
    )
    history = []
    try:
        for i in range(TOTAL_STEPS):
            step = START_STEP + i
            lr = lr_at(i)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = prefetcher.next_batch(device)
            loss, logs = model.forward_train(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % VAL_EVERY == 0 or i == TOTAL_STEPS - 1:
                accs = validate(model, val_batches)
                history.append((step, lr, accs))
                all_pass = all(a >= 0.95 for a in accs.values())
                log(f"step {step:6d}  lr={lr:.2e}  loss={logs['loss']:.4f}  "
                    f"val_acc={ {k: round(v,3) for k,v in accs.items()} }  "
                    f"{'PASS' if all_pass else ''}")
    finally:
        prefetcher.close()

    torch.save(model.state_dict(), "runs/pilot_2_lr_discriminator.pt")

    # summarize: amplitude of L2/L3 accuracy in first third vs last third of the run
    n = len(history)
    early = history[: n // 3]
    late = history[-n // 3 :]
    def spread(seg, L):
        vals = [accs[L] for _, _, accs in seg]
        return max(vals) - min(vals)
    log("\n=== SUMMARY ===")
    for L in (2, 3):
        log(f"L={L}  early third amplitude={spread(early, L):.3f}  "
            f"late third amplitude={spread(late, L):.3f}  "
            f"late-third values={[round(accs[L],3) for _,_,accs in late]}")
    print("DISCRIMINATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
