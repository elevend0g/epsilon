"""Per-step trace: for L=3 items, is the entry retrieved at step 0
(query derived from start_key alone) already the END-holding entry? That
is the direct test — not a rounded sum of soft gate values, and not a
cross-item leakage check (which is chance-level for any model, shortcut
or not, since it isn't measuring what the shortcut would predict)."""

from __future__ import annotations

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.data import make_batch
from model.task_model import GatedCacheModel

ALPHABET_SIZE, N_DISTRACTORS, N_HARD = 64, 1021, 4
D_MODEL, D_STATE, SYMBOL_DIM = 64, 8, 32


@torch.no_grad()
def per_step_trace(model, batch, max_steps=8):
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]

    step_is_end = []  # [step] -> bool tensor [B]: was top-1 the END entry at this step
    step_gate = []     # [step] -> float tensor [B]: raw soft gate value g (pre-END-override)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)

    for step in range(max_steps):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g = model.gate(margin, displacement)

        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        step_is_end.append(is_end_now.clone())
        step_gate.append(g.clone())

        g_eff = torch.where(is_end_now, torch.zeros_like(g), g)
        retrieved_value = cache_values.gather(1, top1.view(-1,1,1).expand(-1,1,cache_values.shape[-1])).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        done = done | is_end_now
        if bool(done.all()):
            break

    return step_is_end, step_gate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    model.load_state_dict(torch.load("runs/pilot_0.pt", map_location=device, weights_only=True))
    model.eval()

    batch = make_batch(ALPHABET_SIZE, 3, N_DISTRACTORS, N_HARD, batch_size=256, device=device, seed=555_003)
    step_is_end, step_gate = per_step_trace(model, batch)

    print("=== L=3: fraction of items where top-1 IS the END entry, by step ===")
    for step, is_end in enumerate(step_is_end):
        frac = is_end.float().mean().item()
        mean_g = step_gate[step].mean().item()
        print(f"step {step}: frac_top1_is_END={frac:.3f}  mean_soft_gate={mean_g:.3f}")

    print("\n=== per-item step at which END was first found (256 items) ===")
    end_step = torch.full((256,), -1, dtype=torch.long)
    for step, is_end in enumerate(step_is_end):
        newly = is_end & (end_step == -1)
        end_step[newly] = step
    print("distribution of first-END-step:", torch.bincount(end_step[end_step >= 0]).tolist())
    print(f"never found within {len(step_is_end)} steps: {(end_step == -1).sum().item()} / 256")


if __name__ == "__main__":
    main()
