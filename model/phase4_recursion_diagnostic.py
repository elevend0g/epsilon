"""Post-hoc diagnostic (§8 precedent, not a preregistered criterion --
same status as model/phase3_gap_diagnostic.py), built to answer one
direct question before any §6.1 code is written: does a genuine
recurse-without-integrate step -- gate closed, and NOT because the
retrieved entry happens to be END-flagged -- occur ANYWHERE in this
project's data, or does arm A simply never use the step budget's slack?

model/phase4_count.py established P(m=rho)=1.0000 on held-out arm A:
every trajectory takes exactly rho integrations plus one forced-closed
END step, zero extra recursion, always. That result is about arm A
only -- §6.2 conditions on arm A specifically, §6.1 does not name an
arm. B1/B2/C were never checked. Their post-break queries are
genuinely low-margin by construction (the queried key, or B2's fake
terminal, has no correct cache entry) -- there is no structural reason
the gate must resolve those as cleanly as arm A's always-present-key,
always-high-margin queries do. Regime 2 in particular trained directly
on B1/B2/C's structure (all five arms, equal frequency); Regime 1 never
saw them at training time but is still run through this diagnostic
autoregressively, same as every other Phase 2/3 integrity check in this
project.

Reuses model/phase3_probe.py's existing, already-registered probe
corpus (`build_combo`, `PROBE_SEED_BASE`) rather than minting a new
seed range -- "existing capture path," per the request that triggered
this module. All five arms, all three chain lengths, the primary/
training band (`n_distractors=1021`), `N_PER_COMBO=2048` per
(checkpoint, arm, L) cell, all six checkpoints.

Rollout mechanics extend model/phase4_count.py's
`capture_integration_counts` (fixed budget `MAX_STEPS=8`, hard-gated
threshold `0.5`, autoregressive, no teacher forcing, freezes each
item's state/counting the step after any is_end-flagged retrieval --
genuine or a spurious decoy hit, since a real deployment halts there
either way, per §4.1's END-guessing diagnostic) to classify EVERY
step taken, not just tally integrations:
  OPEN       -- gate fires, retrieved entry not END-flagged (a real
                integration, matches phase4_count.py's `m`)
  END_CLOSE  -- retrieved entry IS END-flagged, gate forced shut
                regardless of g_soft (structural, not a gate decision)
  RECURSE    -- gate closed, retrieved entry NOT END-flagged -- the
                literal event §6.1's narrow reading needs to exist at
                all, anywhere, to have something to constrain.
"""

from __future__ import annotations

import json

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.phase3_probe import ARM_OFFSET, N_PER_COMBO, build_combo, load_model
from model.phase4_count import CHECKPOINTS, GATE_THRESHOLD, MAX_STEPS

CHAIN_LENGTHS = (1, 2, 3)
N_DISTRACTORS_REF = 1021  # primary/training band, matches other Phase 3/4 reference passes
ARMS = tuple(ARM_OFFSET.keys())  # A, B1, B2, C, D


@torch.no_grad()
def classify_steps(model, batch: dict, max_steps: int = MAX_STEPS,
                    gate_threshold: float = GATE_THRESHOLD) -> dict:
    """Same freeze convention as model/phase4_count.py's
    capture_integration_counts, extended to tally OPEN / END_CLOSE /
    RECURSE per item instead of only OPEN. Returns per-item totals plus
    a bool "any_recurse" flag."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    B = cache_keys.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    n_open = torch.zeros(B, device=state.device)
    n_end_close = torch.zeros(B, device=state.device)
    n_recurse = torch.zeros(B, device=state.device)

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

        is_open = (g_eff > 0.5) & active
        is_end_close = is_end_now & active
        is_recurse = (~is_end_now) & (g_hard < 0.5) & active

        n_open = n_open + is_open.float()
        n_end_close = n_end_close + is_end_close.float()
        n_recurse = n_recurse + is_recurse.float()

        newly_done = is_end_now & active
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

    return {
        "n_open": n_open.cpu(), "n_end_close": n_end_close.cpu(), "n_recurse": n_recurse.cpu(),
        "any_recurse": (n_recurse > 0).cpu(), "exhausted": (~done).cpu(),
    }


def report_checkpoint(name: str, path: str, regime: str, device) -> dict:
    model = load_model(path, device)
    print(f"=== {name} ({regime}): {path} ===", flush=True)

    per_cell = {}
    total_recurse_events = 0
    total_items_with_recurse = 0
    total_items = 0
    for arm in ARMS:
        for L in CHAIN_LENGTHS:
            batch = build_combo(N_DISTRACTORS_REF, L, arm, n=N_PER_COMBO, device=device)
            out = classify_steps(model, batch)
            n = out["n_recurse"].shape[0]
            recurse_events = int(out["n_recurse"].sum().item())
            items_with_recurse = int(out["any_recurse"].sum().item())
            exhaustion = float(out["exhausted"].float().mean().item())
            mean_open = float(out["n_open"].mean().item())
            mean_end_close = float(out["n_end_close"].mean().item())
            mean_recurse = float(out["n_recurse"].mean().item())

            per_cell[f"{arm}_L{L}"] = {
                "n": n, "recurse_events": recurse_events, "items_with_recurse": items_with_recurse,
                "frac_items_with_recurse": items_with_recurse / n,
                "mean_open": mean_open, "mean_end_close": mean_end_close, "mean_recurse": mean_recurse,
                "exhaustion_rate": exhaustion,
            }
            flag = "  <-- RECURSE FOUND" if recurse_events > 0 else ""
            print(f"  arm={arm:<3} L={L}  mean_open={mean_open:.3f}  mean_end_close={mean_end_close:.3f}  "
                  f"mean_recurse={mean_recurse:.4f}  frac_items_any_recurse={items_with_recurse/n:.4f}  "
                  f"exhaustion={exhaustion:.4f}{flag}", flush=True)

            total_recurse_events += recurse_events
            total_items_with_recurse += items_with_recurse
            total_items += n

    print(f"  TOTAL: {total_recurse_events} recurse-steps across {total_items} items "
          f"({total_items_with_recurse} items with >=1 recurse step, "
          f"{total_items_with_recurse/total_items:.4f})", flush=True)

    return {
        "checkpoint": path, "regime": regime, "per_cell": per_cell,
        "total_recurse_events": total_recurse_events, "total_items": total_items,
        "total_items_with_recurse": total_items_with_recurse,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, path, regime in CHECKPOINTS:
        results.append({"name": name, **report_checkpoint(name, path, regime, device)})

    grand_total_events = sum(r["total_recurse_events"] for r in results)
    grand_total_items = sum(r["total_items"] for r in results)
    print(f"\n=== §6.1 premise check, all six checkpoints, all five arms, all three L: "
          f"{grand_total_events} genuine recurse-without-integrate steps "
          f"across {grand_total_items} item-cells sampled ===", flush=True)
    if grand_total_events == 0:
        print("=== VERDICT: recursion-without-integration NEVER occurs anywhere in this data. "
              "The narrow §6.1 reading is globally vacuous, not just on arm A. ===", flush=True)
    else:
        print("=== VERDICT: recursion-without-integration DOES occur -- see per-cell breakdown "
              "for where. The narrow §6.1 reading has real steps to bite on in that scope. ===", flush=True)

    with open("runs/phase4_recursion_diagnostic_result.json", "w") as f:
        json.dump({"max_steps": MAX_STEPS, "results": results,
                    "grand_total_recurse_events": grand_total_events,
                    "grand_total_items": grand_total_items}, f, indent=2)


if __name__ == "__main__":
    main()
