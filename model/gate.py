"""Integration gate. §2.2: quantile inputs, sigmoid soft gate, hardened at
inference. Pilots use running batch statistics (BatchNorm-style) as a
stand-in for the frozen calibration-split empirical CDF — the real CDF
calibration belongs to the full Phase 1-4 protocol on real seeds, not to
pilots, which exist only to produce S* and a rank measurement and are
discarded (§3.1). Flagged as a pilot-specific simplification."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantileGate(nn.Module):
    def __init__(self, momentum: float = 0.05):
        super().__init__()
        self.margin_norm = nn.BatchNorm1d(1, momentum=momentum, affine=False)
        self.disp_norm = nn.BatchNorm1d(1, momentum=momentum, affine=False)
        self.combine = nn.Linear(3, 1)

    def forward(self, margin: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """margin, displacement: [B] raw values -> soft gate g in (0,1): [B]."""
        m = self.margin_norm(margin.unsqueeze(-1)).squeeze(-1)
        d = self.disp_norm(displacement.unsqueeze(-1)).squeeze(-1)
        feats = torch.stack([m, d, m * d], dim=-1)
        return torch.sigmoid(self.combine(feats).squeeze(-1))
