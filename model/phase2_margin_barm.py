"""§4.3, B-arm criterion (supersedes the arm-A form, docs/phase1.md §4.3,
fixed before this was written): margin AUROC discriminating present-key
from absent-key queries, >= 0.70.

Label 1 (present-key): the queried key exists as some cache entry's key
in this item's own memory. Pooled from arm A/D's full teacher-forced
walk (every step, key always present) and from B1/B2's own valid
prefix (steps 0..n_steps-1 — reuses the n_steps/rho fields model/data.py
already computes for Regime 2 training, no new generator work).

Label 0 (absent-key): B1/B2/C's single break-point query per item — the
one query at which the walk genuinely has no matching cache entry
(arm C's very first query, at n_steps=0; B1/B2's query immediately
after their own valid prefix). Exactly one per B1/B2/C item, by
construction — the walk breaks exactly once.

Loop runs t in range(L+1), not range(L): B2's n_steps can equal L
exactly (when the corrupted hop is the chain's last one), so the break
query can land one position past the last valid teacher-forced column.
State is frozen (via torch.where) once an item passes its own n_steps,
so the break query reads the state exactly as it stood after the last
genuine integration, and A/D items (n_steps == L, is_abstain == False)
never contribute a break example at all."""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.data import make_batch_cpu
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, N_DISTRACTORS, N_HARD, SYMBOL_DIM, CHAIN_LENGTHS
from model.task_model import GatedCacheModel

BARM_SEED = 800_000_000  # reserved: §4.3 B-arm's own split
N_PER_ARM = 4096
RANK = 36
ARMS = ("A", "D", "B1", "B2", "C")
ARM_SEED_OFFSET = {"A": 0, "D": 1, "B1": 2, "B2": 3, "C": 4}  # fixed, not hash() -- must be reproducible

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


def auroc(margins: torch.Tensor, labels: torch.Tensor) -> float | None:
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(margins)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(margins) + 1, dtype=torch.float64, device=margins.device)
    sum_ranks_pos = ranks[labels].sum().item()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@torch.no_grad()
def collect_margins_labels(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (margins, labels) pooled over every recorded step for this
    batch (one arm, one L, one chunk)."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    L = batch["L"]
    n_steps = batch["n_steps"]
    is_abstain = batch["is_abstain"]

    margins_out, labels_out = [], []
    for t in range(L + 1):
        active = (t < n_steps)
        is_break = (t == n_steps) & is_abstain

        query = model._query(state)
        logits = retrieve(query, cache_keys)
        margin = margin_from_logits(logits)

        if active.any():
            margins_out.append(margin[active])
            labels_out.append(torch.ones(int(active.sum().item()), dtype=torch.bool, device=margin.device))
        if is_break.any():
            margins_out.append(margin[is_break])
            labels_out.append(torch.zeros(int(is_break.sum().item()), dtype=torch.bool, device=margin.device))

        if t < L:
            target = batch["target_idx"][:, t]
            is_end_now = batch["is_end"].gather(1, target.unsqueeze(1)).squeeze(1)
            g = model.gate(margin, (state - prev_state).flatten(1).norm(dim=-1))
            g_eff = g * active.float() * (1.0 - is_end_now.float())
            retrieved_value = cache_values.gather(
                1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
            ).squeeze(1)
            new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
            prev_state = state
            state = torch.where(active.view(-1, 1, 1), new_state, state)

    if not margins_out:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    return torch.cat(margins_out), torch.cat(labels_out)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    results = {}
    CHUNK = 512

    for name, path in CHECKPOINTS:
        model = load_model(path, device)
        print(f"{name}:", flush=True)
        pooled_margins, pooled_labels = [], []
        entry = {}
        for L in CHAIN_LENGTHS:
            L_margins, L_labels = [], []
            for arm in ARMS:
                batch = make_batch_cpu(
                    alphabet_size=ALPHABET_SIZE, chain_length=L, n_distractors=N_DISTRACTORS,
                    n_hard=N_HARD, batch_size=N_PER_ARM, seed=BARM_SEED + L * 10 + ARM_SEED_OFFSET[arm], arm=arm,
                )
                for start in range(0, N_PER_ARM, CHUNK):
                    end = min(start + CHUNK, N_PER_ARM)
                    chunk = {k: (v[start:end].to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in batch.items()}
                    m, lab = collect_margins_labels(model, chunk)
                    L_margins.append(m.cpu())
                    L_labels.append(lab.cpu())
            L_margins = torch.cat(L_margins)
            L_labels = torch.cat(L_labels)
            n_pos, n_neg = int(L_labels.sum().item()), int((~L_labels).sum().item())
            auroc_L = auroc(L_margins, L_labels)
            auroc_str = f"{auroc_L:.4f}" if auroc_L is not None else "undefined"
            print(f"  L={L}: n_pos={n_pos}  n_neg={n_neg}  AUROC={auroc_str}", flush=True)
            entry[f"L{L}"] = {"n_pos": n_pos, "n_neg": n_neg, "auroc": auroc_L}
            pooled_margins.append(L_margins)
            pooled_labels.append(L_labels)

        pooled_margins = torch.cat(pooled_margins)
        pooled_labels = torch.cat(pooled_labels)
        n_pos, n_neg = int(pooled_labels.sum().item()), int((~pooled_labels).sum().item())
        pooled_auroc = auroc(pooled_margins, pooled_labels)
        pooled_auroc_str = f"{pooled_auroc:.4f}" if pooled_auroc is not None else "undefined"
        verdict = "PASS" if (pooled_auroc is not None and pooled_auroc >= 0.70) else "FAIL"
        print(f"  pooled: n_pos={n_pos}  n_neg={n_neg}  AUROC={pooled_auroc_str}  [{verdict}]", flush=True)
        entry["pooled"] = {"n_pos": n_pos, "n_neg": n_neg, "auroc": pooled_auroc, "verdict": verdict}
        results[name] = entry

    with open("runs/phase2_margin_barm_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote runs/phase2_margin_barm_result.json", flush=True)


if __name__ == "__main__":
    main()
