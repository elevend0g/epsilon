"""Top-level model: embed -> pointer-tracking SSM state -> content-
addressable cache retrieval -> gated integration -> (train: teacher-forced
per-step retrieval loss + count penalty; eval: autoregressive rollout).

Pilot-scope simplifications (§8: implement the simplest version, log it):
  - Trained on arm A only (Regime 1 is A+D; D is deferred to real-seed
    training, since pilots are discarded and exist only for S*/rank).
  - Query projection defaults to full-rank (§3.1.5: pilots must not be
    rank-limited); pass `rank=` to bottleneck it, for the §3.1.7
    throwaway pilot and real seeds. Factored as state -> rank -> d_model
    (two linear layers) rather than one low-rank-constrained matrix, so
    the bottleneck dimension is architecturally exact, not just
    encouraged.
  - Displacement is the raw state-delta norm, not a causal-subspace
    projection (no causal subspace is defined yet at pilot time — that is
    exactly what §3.1.5 measures, from these very pilots).
  - Gate quantile inputs use running batch statistics, not a frozen
    calibration-split CDF (calibration splits are a real-seed protocol
    element; pilots are discarded before any calibration would matter).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cache_bank import build_cache, margin_from_logits, retrieve
from model.embeddings import TupleEncoder
from model.gate import QuantileGate
from model.ssm import SSMCell


class GatedCacheModel(nn.Module):
    def __init__(
        self, alphabet_size: int, symbol_dim: int, d_model: int, d_state: int,
        rank: int | None = None,
    ):
        super().__init__()
        self.encoder = TupleEncoder(alphabet_size, symbol_dim, d_model)
        self.ssm = SSMCell(d_model, d_state)
        self.gate = QuantileGate()
        self.rank = rank
        if rank is None:
            self.query_proj = nn.Linear(d_model * d_state, d_model)  # full-rank (pilots)
        else:
            self.query_proj = nn.Sequential(
                nn.Linear(d_model * d_state, rank),
                nn.Linear(rank, d_model),
            )
        self.value_proj = nn.Linear(d_model, d_model)  # W_v, §2.1's additive write

    def _query(self, state: torch.Tensor) -> torch.Tensor:
        return self.query_proj(state.flatten(1))

    def forward_train(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Teacher-forced over arm A's L retrieval steps. batch tensors are
        already on the model's device. Returns (loss, logs)."""
        cache_keys, cache_values = build_cache(
            self.encoder, batch["keys"], batch["values"], batch["is_end"]
        )
        start_embed = self.encoder.encode_tuples(batch["start_key"])
        state = self.ssm.init_state(start_embed)
        prev_state = state

        L = batch["L"]
        B = cache_keys.shape[0]
        step_losses = []
        step_correct = []
        gate_sum = state.new_zeros(B)

        for t in range(L):
            query = self._query(state)
            logits = retrieve(query, cache_keys)
            target = batch["target_idx"][:, t]
            step_losses.append(F.cross_entropy(logits, target, reduction="none"))
            step_correct.append((logits.argmax(dim=-1) == target).float())

            margin = margin_from_logits(logits)
            displacement = (state - prev_state).flatten(1).norm(dim=-1)
            g = self.gate(margin, displacement)
            g_eff = torch.zeros_like(g) if t == L - 1 else g  # END never opens the gate

            gate_sum = gate_sum + g_eff
            retrieved_value = cache_values.gather(
                1, target.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
            ).squeeze(1)
            prev_state = state
            state = self.ssm.step(state, g_eff.unsqueeze(-1) * self.value_proj(retrieved_value))

        retrieval_loss = torch.stack(step_losses, dim=1).mean()
        rho = L - 1
        count_penalty = ((gate_sum - rho) ** 2).mean()
        loss = retrieval_loss + count_penalty

        exact_match = torch.stack(step_correct, dim=1).all(dim=1).float().mean()
        logs = {
            "loss": loss.item(),
            "retrieval_loss": retrieval_loss.item(),
            "count_penalty": count_penalty.item(),
            "teacher_forced_exact_match": exact_match.item(),
        }
        return loss, logs

    @torch.no_grad()
    def forward_eval_autoregressive(
        self, batch: dict, max_steps: int, gate_threshold: float = 0.5
    ) -> torch.Tensor:
        """Runs the gated loop using the model's own retrieval decisions (no
        teacher forcing). Emit rule is §1.1's literal deterministic rule —
        "on retrieving END, emit the current key" — not a learned
        margin/displacement threshold: the 2x2 table's emit cell describes
        why this naturally coincides with low displacement/high margin, it
        is not an independent decision the model must additionally learn
        for arm-A-only pilots.

        Integration is HARD-gated here: §2.2 says the gate is "trained
        soft, hardened at inference," and this is the inference path — an
        earlier version left it soft, which let a soft-blending failure
        mode pass as ~99% accuracy on one pilot that scored 77% once
        actually hardened (§3.1.6). `forward_train` stays soft; only this
        function thresholds. Returns a bool tensor [B]: emitted key == kL."""
        cache_keys, cache_values = build_cache(
            self.encoder, batch["keys"], batch["values"], batch["is_end"]
        )
        start_embed = self.encoder.encode_tuples(batch["start_key"])
        state = self.ssm.init_state(start_embed)
        prev_state = state

        B = cache_keys.shape[0]
        target_final = batch["target_idx"][:, -1]  # index of the (kL, END) entry
        emitted = torch.full((B,), -1, dtype=torch.long, device=state.device)
        done = torch.zeros(B, dtype=torch.bool, device=state.device)

        self.eval()
        last_top1 = None
        for _ in range(max_steps):
            query = self._query(state)
            logits = retrieve(query, cache_keys)
            top1 = logits.argmax(dim=-1)
            last_top1 = top1
            margin = margin_from_logits(logits)
            displacement = (state - prev_state).flatten(1).norm(dim=-1)
            g_soft = self.gate(margin, displacement)
            g_hard = (g_soft > gate_threshold).float()

            is_end_now = batch["is_end"].gather(1, top1.unsqueeze(1)).squeeze(1)
            g_eff = torch.where(is_end_now, torch.zeros_like(g_hard), g_hard)

            newly_done = is_end_now & ~done
            emitted = torch.where(newly_done, top1, emitted)
            done = done | newly_done

            retrieved_value = cache_values.gather(
                1, top1.view(-1, 1, 1).expand(-1, 1, cache_values.shape[-1])
            ).squeeze(1)
            prev_state = state
            state = self.ssm.step(state, g_eff.unsqueeze(-1) * self.value_proj(retrieved_value))

            if bool(done.all()):
                break

        emitted = torch.where(done, emitted, last_top1)  # budget exhaustion: report as incomplete via mismatch
        return emitted == target_final
