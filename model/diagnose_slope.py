"""§3.1.6, the two checks not yet reported: accuracy-vs-L and P(m|L),
gate hardened at inference (threshold 0.5) as §2.2 specifies — not the
soft sum, which can make several partial integrations look like fewer.
Run against all three fixed-generator pilot checkpoints, large N per L
for resolution on whether L=1/L=2 are genuinely tied or just under-sampled."""

from __future__ import annotations

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.data import make_batch
from model.task_model import GatedCacheModel

ALPHABET_SIZE, N_DISTRACTORS, N_HARD = 64, 1021, 4
D_MODEL, D_STATE, SYMBOL_DIM = 64, 8, 32
N_EVAL = 2000
GATE_THRESHOLD = 0.5


@torch.no_grad()
def rollout_hard_gated(model, batch, max_steps=8):
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]
    target_final = batch["target_idx"][:, -1]

    emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    m = torch.zeros(B, dtype=torch.long, device=state.device)  # hard integration count
    last_top1 = None

    for _ in range(max_steps):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        last_top1 = top1
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g_soft = model.gate(margin, displacement)
        g_hard = (g_soft > GATE_THRESHOLD).float()

        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard)
        m = m + torch.where(~done, g_eff.long(), torch.zeros_like(g_eff.long()))

        newly_done = is_end_now & ~done
        emitted = torch.where(newly_done, top1, emitted)
        done = done | newly_done

        retrieved_value = cache_values.gather(1, top1.view(-1,1,1).expand(-1,1,cache_values.shape[-1])).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        if bool(done.all()):
            break

    emitted = torch.where(done, emitted, last_top1)
    correct = emitted == target_final
    return correct, m, done


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gate hardened at threshold {GATE_THRESHOLD}, N={N_EVAL} per L, held-out fixed seeds\n")

    for pilot_idx in range(3):
        model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
        model.load_state_dict(torch.load(f"runs/pilot_{pilot_idx}.pt", map_location=device, weights_only=True))
        model.eval()
        print(f"=== pilot {pilot_idx} ===")
        for L in (1, 2, 3):
            batch = make_batch(ALPHABET_SIZE, L, N_DISTRACTORS, N_HARD, batch_size=N_EVAL,
                                device=device, seed=444_000 + pilot_idx * 10 + L)
            correct, m, done = rollout_hard_gated(model, batch)
            rho = L - 1
            dist = torch.bincount(m, minlength=rho + 3).tolist()
            print(f"  L={L}  rho={rho}  acc={correct.float().mean().item():.4f}  "
                  f"P(m|L)={dist}  mean_m={m.float().mean().item():.3f}  "
                  f"done_rate={done.float().mean().item():.4f}  n_errors={(~correct).sum().item()}/{N_EVAL}")
        print()


if __name__ == "__main__":
    main()
