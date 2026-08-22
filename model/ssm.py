"""Minimal diagonal SSM cell. Per-channel learned decay (stable via
-softplus), input-driven injection expanded into the d_state dimension.
This is a simplified stand-in for a full selective-scan SSM (Mamba/S4)
— justified per §8 ("implement the simplest version, log it"): sequences
here are short (a handful of recursion steps), so there is no need for
the hardware-efficient parallel-scan machinery real SSM implementations
exist for. Flagged as a simplification, not a claim about SSMs generally."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMCell(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_decay = nn.Parameter(torch.randn(d_model, d_state) * 0.5 - 1.0)
        self.input_proj = nn.Linear(d_model, d_model * d_state)

    def decay(self) -> torch.Tensor:
        return torch.exp(-F.softplus(self.log_decay))  # (0,1) per element

    def step(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """state: [B, d_model, d_state], x: [B, d_model] -> [B, d_model, d_state]"""
        b = x.shape[0]
        injected = self.input_proj(x).view(b, self.d_model, self.d_state)
        return state * self.decay() + injected

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] -> initial state seeded from the query embedding."""
        b = x.shape[0]
        return self.input_proj(x).view(b, self.d_model, self.d_state)
