"""Post-hoc diagnostic (§8, §3.1.6 precedent — not a preregistered
criterion, cannot retroactively affect any recorded verdict): is the
top1-minus-top2 retrieval gap on arm A behaviorally sensitive to
n_hard, the generator's retrieval-difficulty axis? If gaps compress as
n_hard rises, the model's confidence tracks difficulty and the
"confident near-miss" species of Q2 (a model settled and wrong, not
just settled and uncertain) is architecturally reachable. If gaps stay
flat, the axis is inert at the settings this project trains at, and
arm B2 (the only arm that manufactures a genuine near-miss structurally,
rather than relying on the model's own uncertainty) carries Q2 alone.

n_hard values chosen for dynamic range, not tuned to a target: {1, 2,
4, 8, 16, 32} at the fixed reference config (n_distractors=1021, L=3,
alphabet_size=64) — 4 is the value used everywhere else in this
project; 1 and 32 bound a wide range around it.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.data import make_batch_cpu
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM
from model.task_model import GatedCacheModel

GAP_SEED_BASE = 820_000_000  # registered in model/seed_registry.py
N_HARD_VALUES = (1, 2, 4, 8, 16, 32)
N_DISTRACTORS_REF = 1021
L_REF = 3
N_ITEMS = 2048
RANK = 36

CHECKPOINT = "runs/real_seed_r1_0.pt"


def load_model(path: str, device) -> GatedCacheModel:
    m = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE, rank=RANK).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


@torch.no_grad()
def per_step_margins(model, batch: dict) -> torch.Tensor:
    """Teacher-forced rollout over all L steps, arm A. Returns margins
    flattened over [B*L]."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    L = batch["L"]

    margins = []
    for t in range(L):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        target = batch["target_idx"][:, t]
        margin = margin_from_logits(logits)
        margins.append(margin)

        is_end_now = batch["is_end"].gather(1, target.unsqueeze(1)).squeeze(1)
        g = model.gate(margin, (state - prev_state).flatten(1).norm(dim=-1))
        g_eff = torch.zeros_like(g) if t == L - 1 else g
        g_eff = g_eff * (1.0 - is_end_now.float())
        retrieved_value = cache_values.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))

    return torch.cat(margins)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    print(f"loaded {CHECKPOINT}", flush=True)

    results = {}
    CHUNK = 512
    for i, n_hard in enumerate(N_HARD_VALUES):
        seed = GAP_SEED_BASE + i
        batch = make_batch_cpu(
            alphabet_size=ALPHABET_SIZE, chain_length=L_REF, n_distractors=N_DISTRACTORS_REF,
            n_hard=n_hard, batch_size=N_ITEMS, seed=seed, arm="A",
        )
        all_margins = []
        for start in range(0, N_ITEMS, CHUNK):
            end = min(start + CHUNK, N_ITEMS)
            chunk = {k: (v[start:end].to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            all_margins.append(per_step_margins(model, chunk).cpu())
        m = torch.cat(all_margins)
        mean_m, median_m, std_m = m.mean().item(), m.median().item(), m.std().item()
        print(f"n_hard={n_hard:3d}: mean_gap={mean_m:.4f}  median_gap={median_m:.4f}  "
              f"std={std_m:.4f}  n={len(m)}", flush=True)
        results[str(n_hard)] = {"mean_gap": mean_m, "median_gap": median_m, "std_gap": std_m, "n": len(m)}

    means = [results[str(n)]["mean_gap"] for n in N_HARD_VALUES]
    is_compressing = all(means[i] >= means[i + 1] - 0.05 for i in range(len(means) - 1)) and means[0] - means[-1] > 0.2
    print(f"\nn_hard=1 mean gap: {means[0]:.4f}  n_hard=32 mean gap: {means[-1]:.4f}  "
          f"delta: {means[0]-means[-1]:.4f}", flush=True)
    print(f"axis appears: {'LIVE (compressing)' if is_compressing else 'FLAT/INERT or non-monotonic'}", flush=True)

    with open("runs/phase3_gap_diagnostic_result.json", "w") as f:
        json.dump({"n_hard_values": list(N_HARD_VALUES), "results": results,
                    "config": {"n_distractors": N_DISTRACTORS_REF, "L": L_REF, "checkpoint": CHECKPOINT}}, f, indent=2)
    print("wrote runs/phase3_gap_diagnostic_result.json", flush=True)


if __name__ == "__main__":
    main()
