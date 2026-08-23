"""§4.3: retrieval margin dynamic range, scale-free criterion. Raw margin
is not preregisterable (scales with similarity temperature); instead:

- margin AUROC for predicting top-1 retrieval correctness >= 0.70
- top-1 retrieval accuracy within [0.60, 0.98] on arm A

Both computed per retrieval step (teacher-forced, not autoregressive —
clean per-step ground truth, no cascading error from earlier mistakes),
pooled across every hop of every held-out arm-A item. This is a *local*
per-retrieval measure, distinct from §3.2's *global* whole-walk exact
match: a model can be at 99% per-step accuracy and still fail the
global criterion if errors land on different items each time.

AUROC computed by the closed-form rank/Mann-Whitney method (no scipy
dependency): AUROC = (sum of ranks of the correct class - n_pos*(n_pos+1)/2)
/ (n_pos * n_neg).
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, CHAIN_LENGTHS, build_fixed_batch
from model.task_model import GatedCacheModel

MARGIN_SEED = 790_000_000  # reserved: §4.3's own split
N_MARGIN = 8192
RANK = 36

CHECKPOINTS = [
    ("real_seed_r1_0", "runs/real_seed_r1_0.pt"),
    ("real_seed_r1_1", "runs/real_seed_r1_1.pt"),
    ("real_seed_r1_2", "runs/real_seed_r1_2.pt"),
    ("real_seed_r2_0", "runs/real_seed_r2_0.pt"),
    ("real_seed_r2_1", "runs/real_seed_r2_1.pt"),
    ("real_seed_r2_2", "runs/real_seed_r2_2.pt"),
]


def load_model(path: str, device) -> GatedCacheModel:
    m = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE, rank=RANK).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def auroc(margins: torch.Tensor, correct: torch.Tensor) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), margin predicting correct=1."""
    n_pos = int(correct.sum().item())
    n_neg = int((~correct).sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(margins)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(margins) + 1, dtype=torch.float64, device=margins.device)
    sum_ranks_pos = ranks[correct].sum().item()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@torch.no_grad()
def per_step_margins_and_correctness(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced rollout over all L steps. Returns (margins, correct)
    flattened over [B*L]."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    L = batch["L"]

    all_margins, all_correct = [], []
    for t in range(L):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        target = batch["target_idx"][:, t]
        top1 = logits.argmax(dim=-1)
        correct = top1 == target
        margin = margin_from_logits(logits)
        all_margins.append(margin)
        all_correct.append(correct)

        g = model.gate(margin, (state - prev_state).flatten(1).norm(dim=-1))
        g_eff = torch.zeros_like(g) if t == L - 1 else g
        retrieved_value = cache_values.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))

    return torch.cat(all_margins), torch.cat(all_correct)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    results = {}

    for name, path in CHECKPOINTS:
        model = load_model(path, device)
        print(f"{name}:", flush=True)
        pooled_margins, pooled_correct = [], []
        entry = {}
        for L in CHAIN_LENGTHS:
            batch = build_fixed_batch(L, N_MARGIN, MARGIN_SEED + L, cpu)
            CHUNK = 1024
            margins_L, correct_L = [], []
            for start in range(0, N_MARGIN, CHUNK):
                end = min(start + CHUNK, N_MARGIN)
                chunk = {k: (v[start:end].to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                m, c = per_step_margins_and_correctness(model, chunk)
                margins_L.append(m.cpu())
                correct_L.append(c.cpu())
            margins_L = torch.cat(margins_L)
            correct_L = torch.cat(correct_L)
            acc_L = correct_L.float().mean().item()
            auroc_L = auroc(margins_L, correct_L)
            n_incorrect = int((~correct_L).sum().item())
            auroc_str = f"{auroc_L:.4f}" if auroc_L is not None else "undefined (0 incorrect)"
            print(f"  L={L}: top1_acc={acc_L:.4f}  margin_AUROC={auroc_str}  "
                  f"n_steps={len(correct_L)}  n_incorrect={n_incorrect}", flush=True)
            entry[f"L{L}"] = {"top1_accuracy": acc_L, "margin_auroc": auroc_L,
                               "n_steps": len(correct_L), "n_incorrect": n_incorrect}
            pooled_margins.append(margins_L)
            pooled_correct.append(correct_L)

        pooled_margins = torch.cat(pooled_margins)
        pooled_correct = torch.cat(pooled_correct)
        pooled_acc = pooled_correct.float().mean().item()
        pooled_auroc = auroc(pooled_margins, pooled_correct)
        pooled_n_incorrect = int((~pooled_correct).sum().item())
        pooled_auroc_str = f"{pooled_auroc:.4f}" if pooled_auroc is not None else "undefined (0 incorrect)"
        print(f"  pooled: top1_acc={pooled_acc:.4f}  margin_AUROC={pooled_auroc_str}  "
              f"n_steps={len(pooled_correct)}  n_incorrect={pooled_n_incorrect}", flush=True)
        entry["pooled"] = {"top1_accuracy": pooled_acc, "margin_auroc": pooled_auroc,
                            "n_steps": len(pooled_correct), "n_incorrect": pooled_n_incorrect}
        results[name] = entry

    with open("runs/phase2_margin_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote runs/phase2_margin_result.json", flush=True)


if __name__ == "__main__":
    main()
