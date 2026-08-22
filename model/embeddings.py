"""Symbol/key/value embedding. Keys and non-END values live in the same
tuple space and must embed identically (§4.2's wrong-value counterfactual
requires a valid key's embedding to be usable as a retrieved value)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TupleEncoder(nn.Module):
    def __init__(self, alphabet_size: int, symbol_dim: int, d_model: int):
        super().__init__()
        self.symbol_embed = nn.Embedding(alphabet_size, symbol_dim)
        self.combine = nn.Linear(3 * symbol_dim, d_model)
        self.end_embed = nn.Parameter(torch.randn(d_model) * 0.02)

    def encode_tuples(self, symbols: torch.Tensor) -> torch.Tensor:
        """symbols: [..., 3] long -> [..., d_model]"""
        e = self.symbol_embed(symbols)  # [..., 3, symbol_dim]
        e = e.flatten(-2, -1)  # [..., 3*symbol_dim]
        return self.combine(e)

    def end_vector(self) -> torch.Tensor:
        return self.end_embed
