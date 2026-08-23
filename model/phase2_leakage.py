"""§4.1: shuffled-cache leakage check. Pairs each item's own start_key /
true target with a *different*, independently-generated item's cache —
same distribution and statistics, zero information about this item's
chain. If accuracy against the true target stays at or below chance,
the model's answer genuinely depends on cache content (no leakage
through state alone). Runs on all six real-seed terminal checkpoints;
reported per checkpoint per chain length, never pooled into one verdict
(§4 requires all three Phase 2 checks reported separately).

Two seed streams reserved here and nowhere else: LEAKAGE_SEED_A (the
"true" items) and LEAKAGE_SEED_B (the foreign cache source) — same
generator config, independently drawn, so index i in A and index i in B
share nothing but distribution.

Chance is 1/N (N = cache size for that L), per §4.1's "chance is
computed over the key vocabulary" — not 1/|S|^3. Cache entries are
placed by a uniform random shuffle at generation time, so a true target
index is itself an arbitrary position; "does the model's retrieval index
under a foreign cache match that position" is a well-posed guessing
problem regardless of cache content.

Reported alongside raw accuracy: the END-guessing diagnostic. Every
item's cache carries exactly MIN_END_DECOYS+1 = 9 END-flagged entries
(8 decoys, §1.6 req 7, plus the item's own true terminal). A model that
gives up on an unfamiliar cache and falls back to "guess among the
END-looking entries" would guess uniformly among ~9 positions rather
than uniformly among all N — reported separately so that signature is
never misread as leakage.
"""

from __future__ import annotations

import json
import math

import torch

from model.cache_bank import build_cache, retrieve
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, build_fixed_batch
from model.task_model import GatedCacheModel

LEAKAGE_SEED_A = 700_000_000  # reserved: shuffled-cache "true item" split, never used for train/val/geom/calibration
LEAKAGE_SEED_B = 750_000_000  # reserved: shuffled-cache "foreign cache" split
N_LEAKAGE = 16384  # expected chance-only hits ~16 (N/1024) -- enough for the binomial test to have real power
CHAIN_LENGTHS = (1, 2, 3)
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
    m.load_state_dict(sd, strict=False)  # strict=False: real_seed_r1_0 predates the abstain head
    m.eval()
    return m


def binom_upper_tail(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p), via log-space summation
    (n up to a few thousand: math.comb is exact, this stays numerically
    stable because log_pmf is computed in log-space before exponentiating)."""
    if k <= 0:
        return 1.0
    log_p, log_1mp = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        log_pmf = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_1mp
        total += math.exp(log_pmf)
    return min(1.0, total)


@torch.no_grad()
def shuffled_cache_eval(model, batch_true: dict, batch_foreign: dict, max_steps: int = 8) -> dict:
    """batch_true supplies start_key + the true target index; batch_foreign
    supplies the cache actually retrieved against. Returns per-item
    emitted index, whether it matches the true target, and whether it
    landed on an END-flagged foreign entry."""
    cache_keys, cache_values = build_cache(
        model.encoder, batch_foreign["keys"], batch_foreign["values"], batch_foreign["is_end"]
    )
    start_embed = model.encoder.encode_tuples(batch_true["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    B = cache_keys.shape[0]
    true_target = batch_true["target_idx"][:, -1]
    emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)

    last_top1 = None
    for _ in range(max_steps):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        last_top1 = top1
        margin_val = (torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values)
        margin = margin_val[:, 0] - margin_val[:, 1] if margin_val.shape[-1] > 1 else margin_val[:, 0]
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g_soft = model.gate(margin, displacement)
        g_hard = (g_soft > 0.5).float()

        is_end_now = batch_foreign["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard)

        newly_done = is_end_now & ~done
        emitted = torch.where(newly_done, top1, emitted)
        done = done | newly_done

        retrieved_value = cache_values.gather(
            1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        prev_state = state
        state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        if bool(done.all()):
            break

    emitted = torch.where(done, emitted, last_top1)
    correct = emitted == true_target
    emitted_is_end = batch_foreign["is_end"].gather(1, emitted.unsqueeze(1)).squeeze(1)
    return {"correct": correct, "emitted_is_end": emitted_is_end}


EVAL_CHUNK = 512  # GPU-memory-bound: build_cache materializes [chunk, n_cache, d_model] tensors


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    results = {}

    for L in CHAIN_LENGTHS:
        batch_true = build_fixed_batch(L, N_LEAKAGE, LEAKAGE_SEED_A + L, cpu)
        batch_foreign = build_fixed_batch(L, N_LEAKAGE, LEAKAGE_SEED_B + L, cpu)
        n_cache = batch_true["keys"].shape[1]
        chance = 1.0 / n_cache
        n_end = int(batch_foreign["is_end"][0].sum().item())
        chance_end_guess = n_end / n_cache
        print(f"L={L}: N_leakage={N_LEAKAGE}  cache_size={n_cache}  chance={chance:.6f}  "
              f"n_end_per_cache={n_end}  end_guess_rate={chance_end_guess:.4f}", flush=True)

        for name, path in CHECKPOINTS:
            model = load_model(path, device)
            n_correct = 0
            n_end_guessed = 0
            for start in range(0, N_LEAKAGE, EVAL_CHUNK):
                end = min(start + EVAL_CHUNK, N_LEAKAGE)
                chunk_true = to_device({k: v[start:end] if isinstance(v, torch.Tensor) else v
                                         for k, v in batch_true.items()}, device)
                chunk_foreign = to_device({k: v[start:end] if isinstance(v, torch.Tensor) else v
                                            for k, v in batch_foreign.items()}, device)
                out = shuffled_cache_eval(model, chunk_true, chunk_foreign)
                n_correct += int(out["correct"].sum().item())
                n_end_guessed += int(out["emitted_is_end"].sum().item())

            acc = n_correct / N_LEAKAGE
            end_guess_frac = n_end_guessed / N_LEAKAGE
            p_value = binom_upper_tail(n_correct, N_LEAKAGE, chance)
            verdict = "ABOVE CHANCE (p<0.01) -- HALT" if p_value < 0.01 else "at or below chance"

            print(f"  {name}: correct={n_correct}/{N_LEAKAGE} ({acc:.5f})  "
                  f"chance={chance:.6f}  p={p_value:.4f}  [{verdict}]  "
                  f"end_guess_frac={end_guess_frac:.4f} (nominal END-guess rate {chance_end_guess:.4f})",
                  flush=True)

            results.setdefault(f"L{L}", {})[name] = {
                "n_correct": n_correct, "n_total": N_LEAKAGE, "accuracy": acc,
                "chance": chance, "p_value": p_value, "verdict": verdict,
                "end_guess_frac": end_guess_frac, "nominal_end_guess_rate": chance_end_guess,
                "n_cache": n_cache, "n_end_per_cache": n_end,
            }

    with open("runs/phase2_leakage_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote runs/phase2_leakage_result.json", flush=True)


if __name__ == "__main__":
    main()
