"""§6.2: integration count distribution, built as a GATE on §6.1's
constrained-vs-free comparison, not as a warm-up alongside it. §6.1's
constrained/free intervention modifies the recursion update rule;
measuring under that intervention would confound "does the gate
integrate rho times" with "does the intervention change how often it
does." §6.2 must run clean first, on the unmodified model, so §6.1
knows in advance whether its own premise ("recursion steps between
integrations") holds.

§6.2, verbatim: "Record P(m integrations | generator chain length L) --
conditioned on the generator's L, not any depth label, which no longer
exists in the input. Preregistered expectation: mode at rho, with
P(m = rho) >= 0.80 on held-out arm A. Below that, the gate did not
learn the follow-until-END rule and §6.2 reports a gate failure rather
than a count finding." Deviations (m > rho: over-fire toward
always-integrate; m < rho: skipped a hop) are reported separately, never
filtered, and m < rho is NOT interpreted as internal synthesis here --
that requires the §1.3 arithmetic (already weakened, RESULTS.md finding
21) plus a §4.1 shuffled-cache check on the specific under-firing items,
neither of which this module runs unless m < rho actually appears at a
rate worth chasing.

Rollout: fixed step budget MAX_STEPS=8 -- §6.1's own formula,
4*(L_max-1) at L_max=3, "one value, not a sweep" -- the same budget
§6.1 will run under, so this gate result is informative about the
actual conditions §6.1 uses. (Also, independently, pilot_train.py's own
held-out accuracy check already uses max_steps=8 -- same value, same
justification, arrived at separately.) Hard-gated (threshold 0.5),
autoregressive, no teacher forcing -- matches
GatedCacheModel.forward_eval_autoregressive's inference path, extended
to freeze each item's state and prev_state the step after its own
END-retrieval, so post-completion steps cannot inflate its integration
count (forward_eval_autoregressive doesn't need this freeze since it
only ever reads out final-emission correctness, never a running count).

Held-out: a fresh seed stream, COUNT_SEED_BASE, never used for
train/val/geom/calibration/Phase 2/Phase 3 -- reserved and checked in
model/seed_registry.py, per this project's per-phase seed-isolation
discipline, rather than reusing training's own VAL_SEED_BASE split.

Simplification, flagged per §8: the P(m=rho)>=0.80 pass/fail threshold
is evaluated POOLED across all three chain lengths into one number, one
verdict. §6.2's text conditions the *reported distribution* on L
explicitly, but does not say whether the preregistered threshold itself
is per-L or pooled. Pooling is the simplest version that still reports
everything the per-L conditioning requires (the full breakdown is
printed and logged regardless) without inventing three separate,
unstated per-L thresholds. Logged in AMENDMENTS.md, not decided
silently.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.data import make_batch_cpu
from model.pilot_train import ALPHABET_SIZE, N_HARD, SYMBOL_DIM
from model.phase3_probe import load_model

COUNT_SEED_BASE = 830_000_000  # registered in model/seed_registry.py, verified non-overlapping
CHAIN_LENGTHS = (1, 2, 3)
N_DISTRACTORS = 1021  # training config, matches other Phase 3/4 reference passes
N_EVAL = 1024  # matches N_VAL convention (pilot_train.py, finding 10's 128->1024 correction)
MAX_STEPS = 8  # §6.1's own formula: 4 * (L_max - 1), L_max = 3, fixed, not swept
GATE_THRESHOLD = 0.5
PASS_THRESHOLD = 0.80

CHECKPOINTS = [
    ("R1seed0", "runs/real_seed_r1_0.pt", "R1"),
    ("R1seed1", "runs/real_seed_r1_1.pt", "R1"),
    ("R1seed2", "runs/real_seed_r1_2.pt", "R1"),
    ("R2seed0", "runs/real_seed_r2_0.pt", "R2"),
    ("R2seed1", "runs/real_seed_r2_1.pt", "R2"),
    ("R2seed2", "runs/real_seed_r2_2.pt", "R2"),
]


def build_arm_a_batch(L: int, n: int = N_EVAL, device=None) -> dict:
    seed = COUNT_SEED_BASE + L
    batch = make_batch_cpu(
        alphabet_size=ALPHABET_SIZE, chain_length=L, n_distractors=N_DISTRACTORS,
        n_hard=N_HARD, batch_size=n, seed=seed, arm="A",
    )
    if device is not None:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return batch


@torch.no_grad()
def capture_integration_counts(model, batch: dict, max_steps: int = MAX_STEPS,
                                gate_threshold: float = GATE_THRESHOLD) -> dict:
    """Autoregressive, hard-gated, fixed-budget rollout. Returns per-item
    m (integration count actually taken), rho (generator ground truth),
    done (reached its own true END within budget), correct (emitted the
    true final key). Items are frozen -- both state and integration
    accounting -- the step after they retrieve their own END, exactly
    the same freeze convention model/task_model.py's regime2 path uses
    for padding, applied here for post-completion steps instead."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    B = cache_keys.shape[0]
    target_final = batch["target_idx"][:, -1]
    emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    m = torch.zeros(B, device=state.device)

    for _ in range(max_steps):
        active = ~done
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g_soft = model.gate(margin, displacement)
        g_hard = (g_soft > gate_threshold).float()

        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard)

        newly_done = is_end_now & active
        emitted = torch.where(newly_done, top1, emitted)
        m = m + g_eff * active.float()

        retrieved_value = cache_values.gather(
            1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
        ).squeeze(1)
        new_state = model.ssm.step(state, g_eff.unsqueeze(-1) * model.value_proj(retrieved_value))
        active_mask = active.view(-1, 1, 1)
        prev_state = torch.where(active_mask, state, prev_state)
        state = torch.where(active_mask, new_state, state)
        done = done | newly_done

        if bool(done.all()):
            break

    return {
        "m": m.cpu(),
        "rho": batch["rho"].float().cpu(),
        "done": done.cpu(),
        "correct": (emitted == target_final).cpu(),
    }


def report_checkpoint(name: str, path: str, regime: str, device) -> dict:
    model = load_model(path, device)
    print(f"=== {name} ({regime}): {path} ===", flush=True)

    per_L = {}
    pooled_m, pooled_rho, pooled_done, pooled_correct = [], [], [], []
    for L in CHAIN_LENGTHS:
        batch = build_arm_a_batch(L, device=device)
        out = capture_integration_counts(model, batch)
        m, rho, done, correct = out["m"], out["rho"], out["done"], out["correct"]

        p_eq = (m == rho).float().mean().item()
        p_over = (m > rho).float().mean().item()
        p_under = (m < rho).float().mean().item()
        exhaustion_rate = (~done).float().mean().item()
        acc = correct.float().mean().item()

        per_L[L] = {
            "n": m.shape[0], "rho": L - 1, "P(m=rho)": p_eq, "P(m>rho)": p_over,
            "P(m<rho)": p_under, "exhaustion_rate": exhaustion_rate, "accuracy": acc,
        }
        print(f"  L={L}  rho={L-1}  P(m=rho)={p_eq:.4f}  P(m>rho)={p_over:.4f}  "
              f"P(m<rho)={p_under:.4f}  exhaustion={exhaustion_rate:.4f}  acc={acc:.4f}", flush=True)

        pooled_m.append(m); pooled_rho.append(rho)
        pooled_done.append(done); pooled_correct.append(correct)

    pooled_m = torch.cat(pooled_m); pooled_rho = torch.cat(pooled_rho)
    pooled_done = torch.cat(pooled_done); pooled_correct = torch.cat(pooled_correct)
    pooled_p_eq = (pooled_m == pooled_rho).float().mean().item()
    pooled_p_over = (pooled_m > pooled_rho).float().mean().item()
    pooled_p_under = (pooled_m < pooled_rho).float().mean().item()
    pooled_exhaustion = (~pooled_done).float().mean().item()
    pooled_acc = pooled_correct.float().mean().item()
    passed = pooled_p_eq >= PASS_THRESHOLD

    verdict = "PASS" if passed else "GATE FAILURE"
    print(f"  POOLED (all L, n={pooled_m.shape[0]}): P(m=rho)={pooled_p_eq:.4f}  "
          f"P(m>rho)={pooled_p_over:.4f}  P(m<rho)={pooled_p_under:.4f}  "
          f"exhaustion={pooled_exhaustion:.4f}  acc={pooled_acc:.4f}  "
          f"threshold=0.80  verdict={verdict}", flush=True)

    return {
        "checkpoint": path, "regime": regime, "per_L": per_L,
        "pooled": {"P(m=rho)": pooled_p_eq, "P(m>rho)": pooled_p_over, "P(m<rho)": pooled_p_under,
                   "exhaustion_rate": pooled_exhaustion, "accuracy": pooled_acc},
        "passed": passed,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, path, regime in CHECKPOINTS:
        results.append({"name": name, **report_checkpoint(name, path, regime, device)})

    all_passed = all(r["passed"] for r in results)
    print(f"\n=== §6.2 overall: {'ALL PASS' if all_passed else 'AT LEAST ONE GATE FAILURE'} "
          f"-- {'§6.1 may proceed' if all_passed else '§6.1 inherits this as a bound, per the preregistered text'} ===",
          flush=True)

    with open("runs/phase4_count_result.json", "w") as f:
        json.dump({"max_steps": MAX_STEPS, "pass_threshold": PASS_THRESHOLD,
                    "all_passed": all_passed, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
