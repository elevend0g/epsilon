"""§7 raw/*.jsonl — resolved scope: evaluation and instrumentation passes
only (held-out §3.2 terminal check, calibration §4.4/§5.3, geometry
§3.1.5/§3.3), never the training loop itself. §7 originally said
"per-item, per-step" without saying which passes — per-step records
*during training* would be enormous (up to 102,400 steps x 64 items x a
recursion budget, per run, x six runs) and analytically useless; nobody
reads mid-optimization per-step traces for a task at this scale. This
module is deliberately only wired into evaluation-time rollouts.

One JSON object per line: per-step records while an item is still
active (gate state, margin, displacement, integration count, retrieval
target), plus one final record per item (prediction, target, correctness).
Keyed by item_id/split so records can be joined back to a specific
held-out/calibration/geometry item without re-running the model.
"""

from __future__ import annotations

import json
import os

import torch

from model.cache_bank import build_cache, margin_from_logits, retrieve


class RawRecordWriter:
    def __init__(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(path, "w")
        self.n_records = 0
        self.n_bytes = 0

    def write(self, record: dict) -> None:
        line = json.dumps(record) + "\n"
        self._f.write(line)
        self.n_records += 1
        self.n_bytes += len(line)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def manifest_entry(self, path: str, split: str) -> dict:
        return {"path": path, "split": split, "n_records": self.n_records, "n_bytes": self.n_bytes}


def write_manifest(path: str, entries: list[dict], run_id: str) -> None:
    """A small, git-tracked summary of what the (gitignored) raw/*.jsonl
    files contain — paths, per-split record/byte counts, run identity —
    so the raw data's existence and shape is visible in the repo even
    though the data itself isn't committed."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"run_id": run_id, "files": entries,
                    "total_records": sum(e["n_records"] for e in entries),
                    "total_bytes": sum(e["n_bytes"] for e in entries)}, f, indent=2)


@torch.no_grad()
def forward_eval_with_trace(
    model, batch: dict, max_steps: int, writer: RawRecordWriter,
    item_ids: list[str], split: str, gate_threshold: float = 0.5,
) -> torch.Tensor:
    """Same rollout as GatedCacheModel.forward_eval_autoregressive, but
    emits one JSONL record per active item per step, plus a final record
    per item. item_ids: one identifier per batch element (e.g.
    f"{split}_L{L}_{index}"), used to key records back to a specific item
    without needing to re-run the model. Returns the same bool tensor
    forward_eval_autoregressive does: emitted key == kL."""
    cache_keys, cache_values = build_cache(model.encoder, batch["keys"], batch["values"], batch["is_end"])
    start_embed = model.encoder.encode_tuples(batch["start_key"])
    state = model.ssm.init_state(start_embed)
    prev_state = state

    B = cache_keys.shape[0]
    target_final = batch["target_idx"][:, -1]
    emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
    done = torch.zeros(B, dtype=torch.bool, device=state.device)
    integration_count = torch.zeros(B, device=state.device)

    model.eval()
    last_top1 = None
    for step in range(max_steps):
        query = model._query(state)
        logits = retrieve(query, cache_keys)
        top1 = logits.argmax(dim=-1)
        last_top1 = top1
        margin = margin_from_logits(logits)
        displacement = (state - prev_state).flatten(1).norm(dim=-1)
        g_soft = model.gate(margin, displacement)
        g_hard = (g_soft > gate_threshold).float()

        is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
        g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard)
        integration_count = integration_count + torch.where(
            ~done, g_eff, torch.zeros_like(g_eff)
        )

        active_idx = (~done).nonzero(as_tuple=True)[0].tolist()
        for b in active_idx:
            writer.write({
                "item_id": item_ids[b], "split": split, "step": step,
                "gate_open": bool(g_hard[b].item() > 0.5),
                "margin": round(float(margin[b].item()), 5),
                "displacement": round(float(displacement[b].item()), 5),
                "integration_count": float(integration_count[b].item()),
                "is_end_retrieved": bool(is_end_now[b].item()),
            })

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
    correct = emitted == target_final

    for b in range(B):
        writer.write({
            "item_id": item_ids[b], "split": split, "step": "final",
            "predicted_idx": int(emitted[b].item()), "target_idx": int(target_final[b].item()),
            "correct": bool(correct[b].item()),
        })

    return correct
