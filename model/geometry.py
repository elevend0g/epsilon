"""§3.1.5 / §3.3: two distinct participation-ratio measurements, not one.

**Causal PR** — participation ratio of the retrieval-logit Jacobian
w.r.t. the full flattened state (unconditional, full output vocabulary,
no label reference). Describes output sensitivity across the whole
state; feeds §6.1's causal-subspace-constrained recursion.

**Query PR** — participation ratio of the query projection's *own
output* Jacobian w.r.t. state. Describes the bottleneck map itself —
this is the one that actually detects clipping against `rank` (§3.3).
The two are not interchangeable: causal PR can legitimately exceed
`rank` without the query projection clipping anything, since it draws
on state channels the query projection never touches.

Simplification, flagged per §8: both are taken at the final
(answer-determining) retrieval step of each geometry-split item, not
averaged over every step of the walk. This is the single most
"decision-critical" point in the trajectory and the cheapest to define
precisely; averaging over all steps is a reasonable refinement for later,
not required to produce a usable number today.
"""

from __future__ import annotations

import torch

from model.cache_bank import build_cache, retrieve


def participation_ratio(singular_values: torch.Tensor) -> float:
    s2 = singular_values.pow(2)
    return float((s2.sum() ** 2) / (s2.pow(2).sum() + 1e-12))


@torch.no_grad()
def _run_to_final_state(model, batch: dict) -> torch.Tensor:
    """Teacher-forced rollout (geometry split is arm-A, chain_length known)
    up to the state used for the final (END-retrieving) query. Returns
    state: [B, d_model, d_state], detached except for the flatten used
    downstream by the Jacobian (recomputed with grad in the caller)."""
    cache_keys, cache_values = build_cache(
        model.encoder, batch["keys"], batch["values"], batch["is_end"]
    )
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    L = batch["L"]
    for t in range(L - 1):  # stop one step before the final query
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        target = batch["target_idx"][:, t]
        retrieved_value = cache_values.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        state = model.ssm.step(state, model.value_proj(retrieved_value))
    return state


def _per_item_pr(final_state: torch.Tensor, output_fn) -> list[float]:
    """output_fn(state_b: [1,d_model,d_state]) -> [K] output vector for one
    item; PR is computed on the Jacobian of that output w.r.t. state_b."""
    B = final_state.shape[0]
    prs = []
    for b in range(B):
        state_b = final_state[b : b + 1].clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(
            lambda s, b=b: output_fn(s, b), state_b, create_graph=False
        )
        jac = jac.reshape(jac.shape[0], -1)  # [K, d_model*d_state]
        singular_values = torch.linalg.svdvals(jac)
        prs.append(participation_ratio(singular_values))
    return prs


def measure_causal_participation_ratio(model, batch: dict) -> tuple[float, int]:
    """Returns (mean PR, jacobian output dim) over the batch. Output is the
    retrieval logits over the full cache — "the full output vocabulary."
    batch must be from the geometry split, arm A, fixed chain_length."""
    model.eval()
    final_state = _run_to_final_state(model, batch)
    cache_keys, _ = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])

    def output_fn(s, b):
        q = model._query(s)
        return retrieve(q, cache_keys[b : b + 1]).squeeze(0)

    prs = _per_item_pr(final_state, output_fn)
    return sum(prs) / len(prs), cache_keys.shape[-1]


def measure_query_participation_ratio(model, batch: dict) -> tuple[float, int]:
    """Returns (mean PR, query output dim). Output is the query
    projection's own output — the bottleneck map itself, not the whole
    model's output sensitivity. This is what §3.3 compares against `rank`."""
    model.eval()
    final_state = _run_to_final_state(model, batch)

    def output_fn(s, b):
        return model._query(s).squeeze(0)

    prs = _per_item_pr(final_state, output_fn)
    query_dim = model._query(final_state[:1]).shape[-1]
    return sum(prs) / len(prs), query_dim
