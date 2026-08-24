"""§6.1: causal-subspace-constrained vs. free recursion updates.

Scope fixed in `AMENDMENTS.md` #7, decided after `RESULTS.md` finding
27's diagnostic and before this file was written: Regime 1, arm C (all
`L`) and Regime 1, B1 at `L=1` -- the only cells with real, reproducible
recurse-without-integrate events to intervene on. Arm A/D (finding 26),
B2, and B1 at `L>=2` are vacuous or near-vacuous; Regime 2 collapses
the phenomenon almost entirely everywhere.

Operationalization fixed in `AMENDMENTS.md` #8, before this file was
written: the per-item causal subspace is `model/geometry.py`'s exact
causal-PR object (top singular directions of the Jacobian of retrieval
logits w.r.t. current flattened state), evaluated online at each
RECURSE step -- causally valid, no leakage, using only that item's own
current state and cache. Rank `k = ceil(causal PR)` at that step,
reusing the effective-dimension convention from `AMENDMENTS.md` #3
rather than a fixed architectural number. The constraint applies only
to steps classified RECURSE (gate closed, retrieved entry not
END-flagged, via the identical classification
`model/phase4_recursion_diagnostic.py` already uses) -- not to
integration or END-closing steps, which are a different question.

Both conditions are rolled out independently, step by step, from the
same starting item: the constrained trajectory can diverge from the
free one once a projected update changes state, so which future steps
get classified RECURSE (and therefore intervened on) must be decided
inside each run's own unfolding dynamics, not read off the free run.

Metric: paired outcome-match rate -- this project's analogue of the
prior project's "trajectories preserved" framing (§6.1's own motivation
paragraph). Also reports raw causal PR and the floor-relative increment
over finding 10's untrained baseline (~15-17), per §6.1's own
disclosure requirement, computed at the RECURSE steps actually
intervened on in this population (a different position than finding
10's arm-A terminal-step measurement -- flagged, not silently reused as
if position-matched).

Expensive: a full per-item Jacobian + SVD of the causal-PR object runs
at every RECURSE step of the constrained rollout (unlike
model/geometry.py's single terminal-step measurement at N_GEOM=30).
Batch size kept modest and timed before committing to a run size.
"""

from __future__ import annotations

import json
import math

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.geometry import participation_ratio
from model.phase3_probe import build_combo, load_model
from model.phase4_count import GATE_THRESHOLD, MAX_STEPS

# Untrained/random-init floor, RESULTS.md finding 10 -- measured at arm-A's
# terminal (answer-determining) step, a different position than this
# module's RECURSE steps. Reported as context, not a position-matched control.
UNTRAINED_CAUSAL_PR_FLOOR = (15.0, 17.0)

CHECKPOINTS_R1 = [
    ("R1seed0", "runs/real_seed_r1_0.pt"),
    ("R1seed1", "runs/real_seed_r1_1.pt"),
    ("R1seed2", "runs/real_seed_r1_2.pt"),
]
CELLS = [("C", 1), ("C", 2), ("C", 3), ("B1", 1)]  # AMENDMENTS.md #7
N_DISTRACTORS_REF = 1021


@torch.no_grad()
def _classify(model, state, prev_state, cache_keys, is_end_flags, active, gate_threshold=GATE_THRESHOLD):
    """One step's classification + the components needed to apply it.
    Returns top1, g_eff (hardened, 0 on END or inactive), is_end_now,
    is_recurse (bool [B])."""
    query = model._query(state)
    logits = retrieve(query, cache_keys)
    top1 = logits.argmax(dim=-1)
    margin = margin_from_logits(logits)
    displacement = (state - prev_state).flatten(1).norm(dim=-1)
    g_soft = model.gate(margin, displacement)
    g_hard = (g_soft > gate_threshold).float()
    is_end_now = is_end_flags.gather(1, top1.unsqueeze(1)).squeeze(1)
    g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard) * active.float()
    is_recurse = (~is_end_now) & (g_hard < 0.5) & active
    return top1, g_eff, is_end_now, is_recurse


def _causal_subspace_project(model, state_b: torch.Tensor, cache_keys_b: torch.Tensor,
                              delta_b: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    """state_b: [1,d_model,d_state]. cache_keys_b: [1,N,d_model] (this
    item's own cache). delta_b: [1,d_model,d_state], the proposed
    (unconstrained) update. Returns (projected_delta, k, causal_pr)."""
    state_req = state_b.clone().detach().requires_grad_(True)

    def output_fn(s):
        q = model._query(s)
        return retrieve(q, cache_keys_b).squeeze(0)

    jac = torch.autograd.functional.jacobian(output_fn, state_req, create_graph=False)
    jac = jac.reshape(jac.shape[0], -1)  # [K, D]
    with torch.no_grad():
        _, s_vals, vh = torch.linalg.svd(jac, full_matrices=False)  # vh: [min(K,D), D]
        pr = participation_ratio(s_vals)
        k = max(1, min(vh.shape[0], int(math.ceil(pr))))
        basis = vh[:k]  # [k, D]
        proj = _project_onto_basis(basis, delta_b)
    return proj, k, pr


def _project_onto_basis(basis: torch.Tensor, delta_b: torch.Tensor) -> torch.Tensor:
    delta_flat = delta_b.flatten()
    coeff = basis @ delta_flat
    proj_flat = basis.t() @ coeff
    return proj_flat.view_as(delta_b)


def _random_subspace_project(delta_b: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """Matched-rank NEGATIVE CONTROL (§8: check a suspiciously clean
    number before trusting it -- same discipline as the phase4_count.py
    gate-value sanity check). Same k as the causal subspace at this
    exact step, but a random orthonormal basis instead of the causal
    Jacobian's own top-k right singular vectors. If this preserves
    trajectories just as well as the causal subspace does, the
    preservation rate is about update magnitude, not subspace identity."""
    d = delta_b.numel()
    raw = torch.randn(d, k, generator=generator, device=delta_b.device, dtype=delta_b.dtype)
    q, _ = torch.linalg.qr(raw)  # [D, k], orthonormal columns
    basis = q.t()  # [k, D], matches the causal basis's row convention
    return _project_onto_basis(basis, delta_b)


@torch.no_grad()
def rollout_free(model, batch: dict, max_steps: int = MAX_STEPS) -> dict:
    """Unconstrained, matches model/phase4_recursion_diagnostic.py's own
    rollout exactly -- the free-condition baseline."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    outcome = torch.full((B,), -1, dtype=torch.long, device=state.device)  # cache index if END hit, else -1

    for _ in range(max_steps):
        active = ~done
        top1, g_eff, is_end_now, _ = _classify(model, state, prev_state, cache_keys, batch["is_end"], active)
        newly_done = is_end_now & active
        outcome = torch.where(newly_done, top1, outcome)
        done = done | newly_done

        retrieved_value = cache_values.gather(
            1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        active_mask = active.view(-1, 1, 1)
        prev_state = torch.where(active_mask, state, prev_state)
        state = torch.where(active_mask, new_state, state)
        if bool(done.all()):
            break

    return {"outcome": outcome.cpu(), "exhausted": (~done).cpu()}


def rollout_constrained(model, batch: dict, max_steps: int = MAX_STEPS,
                         mode: str = "causal", random_seed: int = 0) -> dict:
    """Identical to rollout_free except: on every step classified
    RECURSE, the proposed update is projected onto a k-dimensional
    subspace (computed at that item's current, pre-update state) before
    being applied. Non-recurse steps (OPEN, END_CLOSE) are untouched.
    Runs a per-item loop only over the (typically small) subset of
    items classified RECURSE at each step -- everyone else uses the
    ordinary batched update.

    mode="causal": the item's own causal subspace (AMENDMENTS.md #8).
    mode="random": matched-rank negative control (same k, computed the
    same way, but a random orthonormal basis instead of the causal
    Jacobian's own directions) -- checks whether preservation is about
    subspace identity or just update magnitude, before trusting the
    causal result."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    B = cache_keys.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    outcome = torch.full((B,), -1, dtype=torch.long, device=state.device)
    ks_used: list[int] = []
    prs_used: list[float] = []
    generator = torch.Generator(device=state.device).manual_seed(random_seed)

    for _ in range(max_steps):
        active = ~done
        with torch.no_grad():
            top1, g_eff, is_end_now, is_recurse = _classify(
                model, state, prev_state, cache_keys, batch["is_end"], active
            )
            newly_done = is_end_now & active
            outcome = torch.where(newly_done, top1, outcome)
            done = done | newly_done

            retrieved_value = cache_values.gather(
                1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
            ).squeeze(1)
            unconstrained_new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
            new_state = unconstrained_new_state.clone()

        recurse_idx = torch.nonzero(is_recurse, as_tuple=False).flatten().tolist()
        for b in recurse_idx:
            delta_b = (unconstrained_new_state[b:b + 1] - state[b:b + 1]).detach()
            # Always compute the causal subspace + its rank k -- the random
            # control uses the same k for a fair matched-rank comparison,
            # so mode="random" doesn't skip this, only discards the basis.
            causal_proj, k, pr = _causal_subspace_project(
                model, state[b:b + 1], cache_keys[b:b + 1], delta_b
            )
            if mode == "causal":
                proj_delta = causal_proj
            elif mode == "random":
                proj_delta = _random_subspace_project(delta_b, k, generator)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            new_state[b:b + 1] = state[b:b + 1] + proj_delta
            ks_used.append(k)
            prs_used.append(pr)

        with torch.no_grad():
            active_mask = active.view(-1, 1, 1)
            prev_state = torch.where(active_mask, state, prev_state)
            state = torch.where(active_mask, new_state, state)
        if bool(done.all()):
            break

    return {"outcome": outcome.cpu(), "exhausted": (~done).cpu(), "ks_used": ks_used, "prs_used": prs_used}


def report_cell(name: str, path: str, arm: str, L: int, n: int, device) -> dict:
    model = load_model(path, device)
    batch = build_combo(N_DISTRACTORS_REF, L, arm, n=n, device=device)

    free = rollout_free(model, batch)
    causal = rollout_constrained(model, batch, mode="causal")
    random_ctrl = rollout_constrained(model, batch, mode="random", random_seed=0)

    preserved_causal = (free["outcome"] == causal["outcome"]).float().mean().item()
    preserved_random = (free["outcome"] == random_ctrl["outcome"]).float().mean().item()
    free_exhaustion = free["exhausted"].float().mean().item()
    causal_exhaustion = causal["exhausted"].float().mean().item()
    random_exhaustion = random_ctrl["exhausted"].float().mean().item()
    n_recurse_events = len(causal["ks_used"])
    mean_k = sum(causal["ks_used"]) / n_recurse_events if n_recurse_events else float("nan")
    mean_pr = sum(causal["prs_used"]) / n_recurse_events if n_recurse_events else float("nan")
    floor_lo, floor_hi = UNTRAINED_CAUSAL_PR_FLOOR
    floor_relative = mean_pr - (floor_lo + floor_hi) / 2 if n_recurse_events else float("nan")

    print(f"  {name} arm={arm} L={L} n={n}: preserved_causal={preserved_causal:.4f}  "
          f"preserved_random={preserved_random:.4f}  free_exhaustion={free_exhaustion:.4f}  "
          f"causal_exhaustion={causal_exhaustion:.4f}  random_exhaustion={random_exhaustion:.4f}  "
          f"recurse_events={n_recurse_events}  mean_k={mean_k:.2f}  mean_causal_PR={mean_pr:.2f}  "
          f"floor_relative={floor_relative:+.2f}", flush=True)

    return {
        "checkpoint": name, "arm": arm, "L": L, "n": n,
        "preserved_causal": preserved_causal, "preserved_random": preserved_random,
        "free_exhaustion": free_exhaustion, "causal_exhaustion": causal_exhaustion,
        "random_exhaustion": random_exhaustion,
        "n_recurse_events": n_recurse_events, "mean_k": mean_k, "mean_causal_pr": mean_pr,
        "floor_relative": floor_relative,
    }


def main(n: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, path in CHECKPOINTS_R1:
        print(f"=== {name}: {path} ===", flush=True)
        for arm, L in CELLS:
            results.append(report_cell(name, path, arm, L, n, device))

    with open("runs/phase4_causal_subspace_result.json", "w") as f:
        json.dump({"n_per_cell": n, "max_steps": MAX_STEPS, "results": results}, f, indent=2)


if __name__ == "__main__":
    import sys
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    main(n_arg)
