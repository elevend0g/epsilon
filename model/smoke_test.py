"""Fast pipeline check: tiny config, a few hundred steps, confirm no
crashes and the loss actually moves. Not part of the pilot protocol —
just cheap validation that the architecture, data pipeline, and training
loop are wired correctly before spending real pilot budget."""

from __future__ import annotations

import sys
import time

import torch

from model.data import make_batch
from model.task_model import GatedCacheModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    alphabet_size, d_model, d_state, symbol_dim = 16, 32, 8, 16
    model = GatedCacheModel(alphabet_size, symbol_dim, d_model, d_state).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    print(f"device={device} params={n_params}")

    losses = []
    t0 = time.time()
    n_steps = 300
    for step in range(n_steps):
        L = [1, 2, 3][step % 3]
        batch = make_batch(
            alphabet_size=alphabet_size, chain_length=L, n_distractors=6, n_hard=1,
            batch_size=64, device=device, seed=None,
        )
        loss, logs = model.forward_train(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(logs["loss"])
        if step % 50 == 0 or step == n_steps - 1:
            print(f"step {step:4d}  loss={logs['loss']:.4f}  "
                  f"retr={logs['retrieval_loss']:.4f}  count={logs['count_penalty']:.4f}  "
                  f"tf_acc={logs['teacher_forced_exact_match']:.3f}")

    dt = time.time() - t0
    print(f"\n{n_steps} steps in {dt:.1f}s ({n_steps/dt:.1f} steps/s)")

    first10 = sum(losses[:10]) / 10
    last10 = sum(losses[-10:]) / 10
    print(f"loss: first10 avg={first10:.4f}  last10 avg={last10:.4f}")

    # eval accuracy check, autoregressive, held-out batch
    eval_batch = make_batch(
        alphabet_size=alphabet_size, chain_length=3, n_distractors=6, n_hard=1,
        batch_size=256, device=device, seed=999,
    )
    correct = model.forward_eval_autoregressive(eval_batch, max_steps=8)
    print(f"autoregressive exact-match accuracy (L=3, held-out seed): {correct.float().mean().item():.3f}")

    if last10 >= first10:
        print("\nFAIL: loss did not decrease")
        sys.exit(1)
    print("\nOK: pipeline runs and loss decreases")


if __name__ == "__main__":
    main()
