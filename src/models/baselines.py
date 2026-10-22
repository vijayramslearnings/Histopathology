"""
baselines.py
Baseline MIL Aggregators for comparison:
- MeanPoolingMIL
- MaxPoolingMIL
- ABMIL (Attention-based MIL, Ilse et al.)
- TransMIL (Transformer-based MIL, Shao et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.multi_subtype_mil import TASKS, GatedAttention

class MeanPoolingMIL(nn.Module):
    def __init__(self, in_dim=1024, dim=256, tasks=TASKS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.heads = nn.ModuleDict({t: nn.Linear(dim, len(c)) for t, c in tasks.items()})

    def forward(self, f20, f5=None, parent=None):
        h = self.proj(f20)
        z = h.mean(dim=0)
        logits = {t: head(z) for t, head in self.heads.items()}
        # Uniform attention for interface compatibility
        n = f20.shape[0]
        attn = {t: torch.full((n,), 1.0 / n, device=f20.device) for t in self.heads}
        return logits, attn


class MaxPoolingMIL(nn.Module):
    def __init__(self, in_dim=1024, dim=256, tasks=TASKS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.heads = nn.ModuleDict({t: nn.Linear(dim, len(c)) for t, c in tasks.items()})

    def forward(self, f20, f5=None, parent=None):
        h = self.proj(f20)
        z, _ = h.max(dim=0)
        logits = {t: head(z) for t, head in self.heads.items()}
        n = f20.shape[0]
        attn = {t: torch.full((n,), 1.0 / n, device=f20.device) for t in self.heads}
        return logits, attn


class ABMIL(nn.Module):
    def __init__(self, in_dim=1024, dim=256, tasks=TASKS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.attn = nn.ModuleDict({t: GatedAttention(dim) for t in tasks})
        self.heads = nn.ModuleDict({t: nn.Linear(dim, len(c)) for t, c in tasks.items()})

    def forward(self, f20, f5=None, parent=None):
        h = self.proj(f20)
        logits, attn = {}, {}
        for t, head in self.heads.items():
            a, z = self.attn[t](h)
            logits[t] = head(z)
            attn[t] = a
        return logits, attn


class TransMIL(nn.Module):
    """Compact TransMIL architecture with self-attention across patch tokens."""
    def __init__(self, in_dim=1024, dim=256, tasks=TASKS, n_layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, dropout=0.25, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.heads = nn.ModuleDict({t: nn.Linear(dim, len(c)) for t, c in tasks.items()})

    def forward(self, f20, f5=None, parent=None):
        h = self.proj(f20).unsqueeze(0)  # [1, N, dim]
        cls = self.cls_token             # [1, 1, dim]
        tokens = torch.cat([cls, h], dim=1)  # [1, N+1, dim]
        out = self.transformer(tokens)
        z = out[0, 0]  # representation of CLS token
        logits = {t: head(z) for t, head in self.heads.items()}
        n = f20.shape[0]
        attn = {t: torch.full((n,), 1.0 / n, device=f20.device) for t in self.heads}
        return logits, attn
