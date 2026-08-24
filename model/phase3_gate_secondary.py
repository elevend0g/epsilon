"""§5.4 secondary: "the gate, not the probe." Measures gate behavior at
the terminal step, arm A vs. arm B2, matched items, all six real seeds.

Terminal step, uniform formula for both arms: t = n_steps - 1 (0-indexed).
For arm A this is the step retrieving the item's own true END entry
(is_end_now=True by construction). For arm B2 this is the step
retrieving kj's own entry -- kj's *key* is untouched by the B2
corruption (only its *value* was swapped to the fake terminal x), so
this is a genuine, confident key match, structurally identical in kind
to any other hop; is_end_now is False here (x is a fake-terminal value,
never literally END). The gate's *next* query (on x itself) is P2,
already measured in model/phase3_canary.py -- this file measures one
step earlier, the actual "does it open on x" moment §2.4 asks about.

Captured at the terminal step: the raw (pre-hard-threshold, pre-END-
override) soft gate value, margin, displacement, and the integration
count accumulated *before* this step (matching the covariate convention
already used for P1/P2). Reported: raw AUROC (gate value predicting
answerable), then residualized jointly against margin + END-flag
identity (is_end_now, itself confounded with arm by construction here --
expected, not a bug) + integration count, reusing the same OLS/bootstrap
path as model/phase3_canary.py. No threshold on the residualized number
-- descriptive, per instruction.

Also computes §2.2's quadrant occupancy (margin-quantile x displacement-
quantile) as a 2x2 per arm, not collapsed to a scalar. No calibration-
split CDF exists yet (PREREG.md: PENDING) -- quantiles are the empirical
rank within the pooled A+B2 terminal-step population for this checkpoint,
logged as the §8 simplest-version stand-in.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, retrieve, margin_from_logits
from model.data import make_batch_cpu
from model.phase3_probe import (
    ARM_OFFSET, BAND_OFFSET, PROBE_SEED_BASE,
    auroc, bootstrap_auroc_ci, load_model, residualize_multi,
)
from model.pilot_train import ALPHABET_SIZE, N_HARD

N_PER_ARM = 2048
L_REF = 2
N_DISTRACTORS_REF = 1021  # training config, matching the P2 probe's own combo1 train side

CHECKPOINTS = [
    ("R1seed0", "runs/real_seed_r1_0.pt", "R1"),
    ("R1seed1", "runs/real_seed_r1_1.pt", "R1"),
    ("R1seed2", "runs/real_seed_r1_2.pt", "R1"),
    ("R2seed0", "runs/real_seed_r2_0.pt", "R2"),
    ("R2seed1", "runs/real_seed_r2_1.pt", "R2"),
    ("R2seed2", "runs/real_seed_r2_2.pt", "R2"),
]


def build_arm_batch(arm: str, device) -> dict:
    seed = PROBE_SEED_BASE + BAND_OFFSET[N_DISTRACTORS_REF] * 100 + L_REF * 10 + ARM_OFFSET[arm]
    batch = make_batch_cpu(
        alphabet_size=ALPHABET_SIZE, chain_length=L_REF, n_distractors=N_DISTRACTORS_REF,
        n_hard=N_HARD, batch_size=N_PER_ARM, seed=seed, arm=arm,
    )
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def capture_terminal_gate(model, batch: dict) -> dict:
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    L = batch["L"]
    n_steps = batch["n_steps"]
    B = state.shape[0]

    term_g_soft = torch.zeros(B, device=state.device)
    term_margin = torch.zeros(B, device=state.device)
    term_displacement = torch.zeros(B, device=state.device)
    term_is_end = torch.zeros(B, dtype=torch.bool, device=state.device)
    term_count_before = torch.zeros(B, device=state.device)
    integration_count = torch.zeros(B, device=state.device)

    for t in range(L):
        active = (t < n_steps)
        is_terminal = (t == n_steps - 1)
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        target = batch["target_idx"][:, t]
        is_end_now = batch["is_end"].gather(1, target.unsqueeze(1)).squeeze(1)
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g = model.gate(margin, displacement)
        g_eff = g * active.float() * (1.0 - is_end_now.float())

        term_g_soft = torch.where(is_terminal, g, term_g_soft)
        term_margin = torch.where(is_terminal, margin, term_margin)
        term_displacement = torch.where(is_terminal, displacement, term_displacement)
        term_is_end = torch.where(is_terminal, is_end_now, term_is_end)
        term_count_before = torch.where(is_terminal, integration_count, term_count_before)

        integration_count = integration_count + g_eff
        retrieved_value = cache_values.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        prev_state = state
        state = torch.where(active.view(-1, 1, 1), new_state, state)

    return {
        "term_g_soft": term_g_soft.cpu(), "term_margin": term_margin.cpu(),
        "term_displacement": term_displacement.cpu(), "term_is_end": term_is_end.cpu(),
        "term_count_before": term_count_before.cpu(),
    }


def quadrant_table(margin: torch.Tensor, displacement: torch.Tensor) -> dict:
    """Empirical-rank quantile transform (no calibration-split CDF exists
    yet), split at the median -- §8 simplest-version stand-in, logged."""
    m_q = margin.argsort().argsort().float() / (len(margin) - 1)
    d_q = displacement.argsort().argsort().float() / (len(displacement) - 1)
    high_m, high_d = m_q >= 0.5, d_q >= 0.5
    n = len(margin)
    return {
        "integrate (high m, high d)": float((high_m & high_d).sum()) / n,
        "recurse (low m, high d)": float((~high_m & high_d).sum()) / n,
        "emit (high m, low d)": float((high_m & ~high_d).sum()) / n,
        "abstain (low m, low d)": float((~high_m & ~high_d).sum()) / n,
    }


def report_gate(name: str, a_data: dict, b2_data: dict) -> dict:
    g = torch.cat([a_data["term_g_soft"], b2_data["term_g_soft"]])
    margin = torch.cat([a_data["term_margin"], b2_data["term_margin"]])
    is_end = torch.cat([a_data["term_is_end"], b2_data["term_is_end"]])
    count = torch.cat([a_data["term_count_before"], b2_data["term_count_before"]])
    label = torch.cat([torch.ones(len(a_data["term_g_soft"]), dtype=torch.bool),
                        torch.zeros(len(b2_data["term_g_soft"]), dtype=torch.bool)])

    raw_auroc = auroc(g, label)
    covs = [margin, is_end.float(), count]
    joint_resid = residualize_multi(g, covs)
    joint_auroc, joint_lo, joint_hi = bootstrap_auroc_ci(joint_resid, label)

    a_mean_g, b2_mean_g = a_data["term_g_soft"].mean().item(), b2_data["term_g_soft"].mean().item()
    a_open_frac = (a_data["term_g_soft"] > 0.5).float().mean().item()
    b2_open_frac = (b2_data["term_g_soft"] > 0.5).float().mean().item()

    print(f"{name}: A_mean_g={a_mean_g:.4f}  B2_mean_g={b2_mean_g:.4f}  "
          f"A_open_frac={a_open_frac:.4f}  B2_open_frac={b2_open_frac:.4f}  "
          f"raw_AUROC={raw_auroc:.4f}  JOINT_resid={joint_auroc:.4f}[{joint_lo:.3f},{joint_hi:.3f}]", flush=True)

    a_quad = quadrant_table(a_data["term_margin"], a_data["term_displacement"])
    b2_quad = quadrant_table(b2_data["term_margin"], b2_data["term_displacement"])
    print(f"  A quadrant occupancy:  {a_quad}", flush=True)
    print(f"  B2 quadrant occupancy: {b2_quad}", flush=True)

    return {
        "a_mean_g": a_mean_g, "b2_mean_g": b2_mean_g,
        "a_open_frac": a_open_frac, "b2_open_frac": b2_open_frac,
        "raw_auroc": raw_auroc, "joint_residual": {"resid_auroc": joint_auroc, "resid_ci": (joint_lo, joint_hi)},
        "a_quadrant": a_quad, "b2_quadrant": b2_quad,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_batch = build_arm_batch("A", device)
    b2_batch = build_arm_batch("B2", device)

    results = {}
    for name, path, regime in CHECKPOINTS:
        model = load_model(path, device)
        a_data = capture_terminal_gate(model, a_batch)
        b2_data = capture_terminal_gate(model, b2_batch)
        r = report_gate(name, a_data, b2_data)
        r["regime"] = regime
        results[name] = r

    with open("runs/phase3_gate_secondary_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote runs/phase3_gate_secondary_result.json", flush=True)


if __name__ == "__main__":
    main()
