"""§3.1.6 diagnostic: does pilot 0 actually follow the chain, or has it
found the single-lookup END shortcut? Runs against the already-trained,
discarded pilot checkpoint — contaminates nothing."""

from __future__ import annotations

import torch

from generator import END
from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.data import make_batch
from model.task_model import GatedCacheModel

ALPHABET_SIZE, N_DISTRACTORS, N_HARD = 64, 1021, 4
D_MODEL, D_STATE, SYMBOL_DIM = 64, 8, 32


@torch.no_grad()
def rollout_with_integration_count(model, batch, max_steps=8):
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]
    target_final = batch["target_idx"][:, -1]

    emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    n_integrations = torch.zeros(B, device=state.device)
    steps_to_end = torch.zeros(B, dtype=torch.long, device=state.device)
    last_top1 = None

    for step in range(max_steps):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        last_top1 = top1
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g = model.gate(margin, displacement)

        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g), g)
        n_integrations = n_integrations + torch.where(~done, g_eff, torch.zeros_like(g_eff))

        newly_done = is_end_now & ~done
        emitted = torch.where(newly_done, top1, emitted)
        steps_to_end = torch.where(newly_done, torch.full_like(steps_to_end, step + 1), steps_to_end)
        done = done | newly_done

        retrieved_value = cache_values.gather(1, top1.view(-1,1,1).expand(-1,1,cache_values.shape[-1])).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        if bool(done.all()):
            break

    emitted = torch.where(done, emitted, last_top1)
    correct = emitted == target_final
    return correct, n_integrations, steps_to_end, done


@torch.no_grad()
def leakage_check(model, batch_own, batch_other, max_steps=8):
    """§4.1: pair queries with a DIFFERENT item's cache. Chance = 1/N."""
    mixed = dict(batch_own)
    mixed["keys"] = batch_other["keys"]
    mixed["values"] = batch_other["values"]
    mixed["is_end"] = batch_other["is_end"]
    correct, *_ = rollout_with_integration_count(model, mixed, max_steps)
    return correct.float().mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE).to(device)
    model.load_state_dict(torch.load("runs/pilot_0.pt", map_location=device))
    model.eval()

    print("=== END census ===")
    from generator import GeneratorConfig, TaskGenerator
    gen = TaskGenerator(GeneratorConfig(alphabet_size=ALPHABET_SIZE, chain_length=3,
                                          n_distractors=N_DISTRACTORS, n_hard=N_HARD, seed=42))
    item = gen.generate_family().A
    n_end = sum(1 for e in item.memory if e.value == END)
    print(f"entries with value==END in a {len(item.memory)}-entry cache: {n_end}")

    print("\n=== integration count vs L (held-out, fresh seeds) ===")
    for L in (1, 2, 3):
        batch = make_batch(ALPHABET_SIZE, L, N_DISTRACTORS, N_HARD, batch_size=256, device=device, seed=555_000 + L)
        correct, n_int, steps_to_end, done = rollout_with_integration_count(model, batch)
        rho_target = L - 1
        print(f"L={L}  rho_target={rho_target}  "
              f"acc={correct.float().mean().item():.3f}  "
              f"mean_integrations={n_int.mean().item():.3f}  "
              f"integration_dist={torch.bincount(n_int.round().long().clamp(min=0)).tolist()}  "
              f"mean_steps_to_end={steps_to_end.float().mean().item():.2f}  "
              f"done_rate={done.float().mean().item():.3f}")

    print("\n=== §4.1 leakage check (shuffled cache from a different item) ===")
    for L in (1, 3):
        own = make_batch(ALPHABET_SIZE, L, N_DISTRACTORS, N_HARD, batch_size=256, device=device, seed=666_000 + L)
        other = make_batch(ALPHABET_SIZE, L, N_DISTRACTORS, N_HARD, batch_size=256, device=device, seed=777_000 + L)
        acc = leakage_check(model, own, other)
        chance = 1.0 / len(item.memory)
        print(f"L={L}  shuffled-cache acc={acc:.4f}  chance~{chance:.4f}  "
              f"{'LEAKAGE / SHORTCUT' if acc > chance * 3 else 'looks clean'}")


if __name__ == "__main__":
    main()
