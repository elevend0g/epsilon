"""§5.1-§5.4: the Q2 coverage probe. Builds the probe corpus, captures
frozen state at P1/P2 in one instrumentation pass per checkpoint, and
runs the linear probe with margin regression built in from the start
(not a follow-up robustness check) — §4.3's B-arm result already showed
margin alone nearly perfectly separates present-key from absent-key
retrieval (AUROC ~0.999), so a raw state-probe AUROC could just be
re-deriving margin from state, which is circular for Q2's actual
question. The residualized AUROC (state probe controlling for margin)
is the number that means something; raw AUROC is reported alongside it
for completeness, never in place of it.

Position construction, uniform across arms: pos1 = frozen state right
before the item's own decisive query (the break query for B1/B2/C, the
END-retrieving query for A/D) — computed via a per-item teacher-forced
prefix that freezes at a per-item stop count (n_steps for B1/B2/C,
n_steps-1 for A/D, since A/D's decisive query IS step n_steps-1, not
one beyond it). pos2 = one further, natural (autoregressive, not
teacher-forced -- there is no valid target at a genuinely absent-key
query) step from pos1: the model's own retrieval, margin, gate decision,
and (if the gate fires) integration.

Label: True (answerable) for arms A/D, False (not answerable) for
B1/B2/C -- matching §5.4's arm A vs B2 framing, generalized to every
arm the corpus contains.

Split by generated graph, not by item (§5.2): cross-condition combos
(different L or different n_distractors) are drawn from independently
seeded generation runs and share no graph by construction, so the
pinned transfer combos already satisfy this without an additional
same-condition holdout.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.data import make_batch_cpu
from model.pilot_train import ALPHABET_SIZE, D_MODEL, D_STATE, N_HARD, SYMBOL_DIM
from model.task_model import GatedCacheModel

PROBE_SEED_BASE = 810_000_000  # registered in model/seed_registry.py, verified non-overlapping
BAND_OFFSET = {1021: 0, 256: 1}
ARM_OFFSET = {"A": 0, "B1": 1, "B2": 2, "C": 3, "D": 4}
ANSWERABLE_ARMS = {"A", "D"}

N_PER_COMBO = 2048
RANK = 36


def load_model(path: str, device) -> GatedCacheModel:
    m = GatedCacheModel(ALPHABET_SIZE, SYMBOL_DIM, D_MODEL, D_STATE, rank=RANK).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def build_combo(n_distractors: int, L: int, arm: str, n: int = N_PER_COMBO, device=None) -> dict:
    seed = PROBE_SEED_BASE + BAND_OFFSET[n_distractors] * 100 + L * 10 + ARM_OFFSET[arm]
    batch = make_batch_cpu(
        alphabet_size=ALPHABET_SIZE, chain_length=L, n_distractors=n_distractors,
        n_hard=N_HARD, batch_size=n, seed=seed, arm=arm,
    )
    if device is not None:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return batch


@torch.no_grad()
def capture_pos1_pos2(model, batch: dict, arm: str) -> dict:
    """Returns per-item pos1 state, pos2 state, pos2 margin — all [B, ...]."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state
    pos1_prev_state = state.clone()  # tracks prev_state at the moment each item last updated, for pos2's displacement

    L = batch["L"]
    n_steps = batch["n_steps"]
    is_abstain = batch["is_abstain"]
    # A/D's decisive query IS step n_steps-1 (the END-retrieval); B1/B2/C's
    # decisive (break) query is step n_steps, one past their valid prefix.
    pos1_count = torch.where(is_abstain, n_steps, (n_steps - 1).clamp(min=0))

    integration_count = torch.zeros(state.shape[0], device=state.device)  # sum of g_eff through pos1 -- the actual
                                                                            # gate-fire count, not just the step index
    for t in range(L):
        active = (t < pos1_count)
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        target = batch["target_idx"][:, t]
        is_end_now = batch["is_end"].gather(1, target.unsqueeze(1)).squeeze(1)
        margin = margin_from_logits(logits)
        g = model.gate(margin, (state - prev_state).flatten(1).norm(dim=-1))
        g_eff = g * active.float() * (1.0 - is_end_now.float())
        integration_count = integration_count + g_eff
        retrieved_value = cache_values.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        active_mask = active.view(-1, 1, 1)
        pos1_prev_state = torch.where(active_mask, state, pos1_prev_state)  # last prev_state while still active
        prev_state = state
        state = torch.where(active_mask, new_state, state)

    pos1_state = state.clone()
    pos1_integration_count = integration_count.clone()

    # pos2: one further, natural (autoregressive) step from pos1. Displacement
    # uses the same prev-state convention as the main loop (the state one
    # step before the current query), not zero -- tracked above since
    # per-item freezing means the loop's own final prev_state doesn't equal
    # "the state right before pos1" for items that froze early.
    query2 = model._query(pos1_state)
    logits2 = retrieve(query2, cache_keys)
    top1_2 = logits2.argmax(dim=-1)
    margin2 = margin_from_logits(logits2)
    displacement2 = (pos1_state - pos1_prev_state).flatten(1).norm(dim=-1)
    g2 = model.gate(margin2, displacement2)
    g2_hard = (g2 > 0.5).float()
    is_end_2 = batch["is_end"].gather(1, top1_2.unsqueeze(1)).squeeze(1)
    g2_eff = g2_hard * (1.0 - is_end_2.float())
    retrieved_value2 = cache_values.gather(
        1, top1_2.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
    ).squeeze(1)
    pos2_state = model.ssm.step(pos1_state, g2_eff.unsqueeze(-1) * model.value_proj(retrieved_value2))
    pos2_integration_count = pos1_integration_count + g2_eff  # arm A's is deterministically == pos1's
                                                                # (END forces g2_eff=0); B2's may or may not
                                                                # increment depending on whether the gate wrongly
                                                                # fires on the dead-end query -- itself a possible
                                                                # confound (§2.5's rho asymmetry, not just at P1)

    return {
        "pos1": pos1_state.flatten(1).cpu(),
        "pos2": pos2_state.flatten(1).cpu(),
        "pos2_margin": margin2.cpu(),
        "pos1_integration_count": pos1_integration_count.cpu(),
        "pos2_integration_count": pos2_integration_count.cpu(),
        "pos2_is_end": is_end_2.cpu(),  # structural: arm A's pos2 is definitionally its own correct
                                         # END-retrieval, not a Regime-2-specific artifact (see finding 19)
        "label": torch.full((pos1_state.shape[0],), arm in ANSWERABLE_ARMS, dtype=torch.bool),
    }


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    sum_ranks_pos = ranks[labels].sum().item()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def fit_linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, epochs: int = 300, lr: float = 0.05) -> torch.nn.Linear:
    """Plain logistic regression, no sklearn dependency. train_x: [N,D],
    train_y: [N] bool."""
    d = train_x.shape[1]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x = (train_x - mean) / std
    y = train_y.float()

    probe = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        logit = probe(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    probe.eval()
    probe._feature_mean = mean
    probe._feature_std = std
    return probe


@torch.no_grad()
def probe_score(probe: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    x = (x - probe._feature_mean) / probe._feature_std
    return probe(x).squeeze(-1)


def bootstrap_auroc_ci(scores: torch.Tensor, labels: torch.Tensor, n_boot: int = 2000, ci: float = 0.95,
                        seed: int = 0) -> tuple[float, float, float] | None:
    """Item-resampling bootstrap CI for AUROC — item counts here are in the
    thousands, not large enough that a 0.004 gap to an unrelated threshold
    should be read as a real difference without checking. Returns
    (point_estimate, lo, hi) or None if the point estimate is undefined."""
    point = auroc(scores, labels)
    if point is None:
        return None
    n = len(scores)
    g = torch.Generator().manual_seed(seed)
    boot_vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        v = auroc(scores[idx], labels[idx])
        if v is not None:
            boot_vals.append(v)
    boot_vals = torch.tensor(boot_vals)
    alpha = (1 - ci) / 2
    lo = torch.quantile(boot_vals, alpha).item()
    hi = torch.quantile(boot_vals, 1 - alpha).item()
    return point, lo, hi


def residualize(score: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    """OLS residual of score ~ margin (closed form, 1D predictor)."""
    m = margin - margin.mean()
    s = score - score.mean()
    coef = (m * s).sum() / (m * m).sum().clamp(min=1e-9)
    predicted = score.mean() + coef * (margin - margin.mean())
    return score - predicted


def residualize_multi(score: torch.Tensor, covariates: list[torch.Tensor]) -> torch.Tensor:
    """OLS residual of score ~ covariates, jointly (closed-form least
    squares via the normal equations) -- not each covariate regressed out
    separately in sequence. Used for the P2 three-way control (margin,
    END-flag identity, integration count): the question is whether any
    signal survives once all three known structural confounds are removed
    at once, not just each one in isolation."""
    n = score.shape[0]
    X = torch.stack([c.float() for c in covariates] + [torch.ones(n)], dim=1)  # design matrix, +1 for intercept
    beta = torch.linalg.lstsq(X, score.unsqueeze(1)).solution.squeeze(1)
    predicted = X @ beta
    return score - predicted
