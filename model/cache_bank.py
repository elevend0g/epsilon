"""Content-addressable cache bank. Built once per item (a feedforward
embedding of every entry, not something ingested through the SSM's
recurrence — the recurrent state's job is the pointer trajectory, not
memorizing the cache; see docs/phase1.md §2.1's "pointer hazard" note)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.embeddings import TupleEncoder


def build_cache(encoder: TupleEncoder, keys: torch.Tensor, values: torch.Tensor, is_end: torch.Tensor):
    """keys: [B,N,3] long, values: [B,N,3] long (ignored where is_end),
    is_end: [B,N] bool. Returns (cache_keys, cache_values): [B,N,d_model] each."""
    cache_keys = encoder.encode_tuples(keys)
    cache_values = encoder.encode_tuples(values)
    end_vec = encoder.end_vector().view(1, 1, -1).expand_as(cache_values)
    cache_values = torch.where(is_end.unsqueeze(-1), end_vec, cache_values)
    return cache_keys, cache_values


def retrieve(query: torch.Tensor, cache_keys: torch.Tensor):
    """query: [B,d_model], cache_keys: [B,N,d_model] -> similarity logits [B,N]."""
    q = F.normalize(query, dim=-1)
    k = F.normalize(cache_keys, dim=-1)
    scale = query.shape[-1] ** 0.5
    return torch.einsum("bd,bnd->bn", q, k) * scale


def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """[B,N] -> top1-minus-top2 margin [B]."""
    top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]
