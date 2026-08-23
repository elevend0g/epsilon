"""§4.2: counterfactual content flow, three arms, three predictions.
Tests whether the recurrent pathway carries actual content (the
retrieved value) forward, or merely a generic "something happened"
signal.

Uses chain_length=2 items so there is exactly one hop to intervene on
(hop 1: retrieve k1's entry, integrate its value) and one hop to read
the prediction from (hop 2's retrieval).

- clean: no intervention. Hop 2 should retrieve k2's own cache entry.
- wrong-value: force-integrate a *different* valid key k2' instead of
  the true k2. k2' must be drawn from this item's own cache (a non-END
  entry, not the true k2) so it has a genuine cache slot of its own —
  substituting a well-formed but absent tuple would predict the B2
  dead-end signature instead of substitution, misreading a mis-specified
  intervention as a failed one. k2''s embedding is rescaled to the true
  k2's embedding norm, isolating content from magnitude. Prediction:
  hop 2 retrieves k2''s own cache entry (the substitution signature),
  margin stays high.
- clamped: force g1=0, nothing integrated. Prediction: hop 2 retrieves
  neither the true k2 nor k2' above chance, margin collapses.

clean and wrong-value force g1=1 identically (same integration decision,
different content only) so content is the sole manipulated variable;
clamped is the only arm that also perturbs the trajectory, which is why
it alone is confounded and cannot substitute for the other two.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, SYMBOL_DIM, build_fixed_batch
from model.task_model import GatedCacheModel

COUNTERFACTUAL_SEED = 780_000_000  # reserved: §4.2's own split, distinct from every other seed stream
N_COUNTERFACTUAL = 4096
RANK = 36
L = 2  # one hop to intervene on, one to read the prediction from

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


def pick_wrong_key_idx(batch: dict) -> torch.Tensor:
    """Per item, a uniformly random cache index that is (a) not END-
    flagged (has a genuine key->key entry of its own) and (b) not the
    true k2's own entry — a real distractor from this item's own cache,
    never a merely well-formed but absent tuple."""
    is_end = batch["is_end"]
    true_k2_idx = batch["target_idx"][:, 1]
    B, N = is_end.shape
    eligible = (~is_end).clone()
    eligible.scatter_(1, true_k2_idx.unsqueeze(1), False)
    rand_scores = torch.rand(B, N, device=is_end.device)
    rand_scores = torch.where(eligible, rand_scores, torch.full_like(rand_scores, -1.0))
    return rand_scores.argmax(dim=-1)


@torch.no_grad()
def run_counterfactual(model, batch: dict) -> dict:
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state0 = model.ssm.init_state(start_embed)

    true_k1_idx = batch["target_idx"][:, 0]
    true_k2_idx = batch["target_idx"][:, 1]
    B = cache_keys.shape[0]

    # Hop 1, teacher-forced to the true k1 entry (its own retrieval
    # correctness is Phase 1's concern, not this test's).
    true_retrieved_value = cache_values.gather(
        1, true_k1_idx.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
    ).squeeze(1)

    wrong_idx = pick_wrong_key_idx(batch)
    wrong_key_tuple = batch["keys"].gather(1, wrong_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    wrong_embed = model.encoder.encode_tuples(wrong_key_tuple)
    true_norm = true_retrieved_value.norm(dim=-1, keepdim=True)
    wrong_norm = wrong_embed.norm(dim=-1, keepdim=True)
    wrong_embed_matched = wrong_embed * (true_norm / wrong_norm.clamp(min=1e-9))

    results = {}
    for arm in ("clean", "wrong_value", "clamped"):
        if arm == "clean":
            write = model.value_proj(true_retrieved_value)
        elif arm == "wrong_value":
            write = model.value_proj(wrong_embed_matched)
        else:  # clamped
            write = torch.zeros_like(model.value_proj(true_retrieved_value))
        state1 = model.ssm.step(state0, write)

        query2 = model._query(state1)
        logits2 = retrieve(query2, cache_keys)
        top1_2 = logits2.argmax(dim=-1)
        margin2 = margin_from_logits(logits2)

        results[arm] = {
            "hits_true_k2": (top1_2 == true_k2_idx),
            "hits_wrong_idx": (top1_2 == wrong_idx),
            "margin": margin2,
        }
    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    batch = build_fixed_batch(L, N_COUNTERFACTUAL, COUNTERFACTUAL_SEED, cpu)
    n_cache = batch["keys"].shape[1]
    chance = 1.0 / n_cache
    print(f"N={N_COUNTERFACTUAL}  cache_size={n_cache}  chance={chance:.6f}", flush=True)

    all_results = {}
    CHUNK = 512
    for name, path in CHECKPOINTS:
        model = load_model(path, device)
        acc = {arm: {"hits_true_k2": 0, "hits_wrong_idx": 0, "margin_sum": 0.0} for arm in
               ("clean", "wrong_value", "clamped")}
        for start in range(0, N_COUNTERFACTUAL, CHUNK):
            end = min(start + CHUNK, N_COUNTERFACTUAL)
            chunk = {k: (v[start:end].to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = run_counterfactual(model, chunk)
            for arm in acc:
                acc[arm]["hits_true_k2"] += int(out[arm]["hits_true_k2"].sum().item())
                acc[arm]["hits_wrong_idx"] += int(out[arm]["hits_wrong_idx"].sum().item())
                acc[arm]["margin_sum"] += float(out[arm]["margin"].sum().item())

        print(f"{name}:", flush=True)
        entry = {}
        for arm in ("clean", "wrong_value", "clamped"):
            hits_true = acc[arm]["hits_true_k2"] / N_COUNTERFACTUAL
            hits_wrong = acc[arm]["hits_wrong_idx"] / N_COUNTERFACTUAL
            mean_margin = acc[arm]["margin_sum"] / N_COUNTERFACTUAL
            print(f"  {arm:12s} hits_true_k2={hits_true:.4f}  hits_wrong_idx={hits_wrong:.4f}  "
                  f"mean_margin={mean_margin:.4f}", flush=True)
            entry[arm] = {"hits_true_k2": hits_true, "hits_wrong_idx": hits_wrong, "mean_margin": mean_margin}
        all_results[name] = entry

    all_results["_meta"] = {"n": N_COUNTERFACTUAL, "cache_size": n_cache, "chance": chance}
    with open("runs/phase2_counterfactual_result.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nwrote runs/phase2_counterfactual_result.json", flush=True)


if __name__ == "__main__":
    main()
