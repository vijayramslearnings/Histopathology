"""
models/multi_scale_model.py
Upgraded Multi-Scale Computational Pathology Model:
- ScaleAttention, PatchAttention, and MILPooling modules
- Contrastive Projection Head (MLP: 512 -> 128) with L2 normalization for SupCon
- Hierarchical Cancer Head and Subtype Classifier Heads
- Post-hoc Temperature Scaling calibration parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ScaleAttention(nn.Module):
    """Attention over multiple optical resolution scales (e.g. 20x, 10x, 5x)."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, num_scales, dim]
        scores = self.net(x)                 # [B, num_scales, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)    # [B, dim]
        return pooled, weights.squeeze(-1)


class PatchAttention(nn.Module):
    """Gated Attention Mechanism over patches (ABMIL style)."""
    def __init__(self, dim: int, hid: int = 128, dropout: float = 0.25):
        super().__init__()
        self.V = nn.Linear(dim, hid)
        self.U = nn.Linear(dim, hid)
        self.w = nn.Linear(hid, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [N, dim] or [B, N, dim]
        if h.ndim == 2:
            a = self.w(self.drop(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))))
            a = torch.softmax(a.squeeze(-1), dim=0)  # [N]
            z = (a.unsqueeze(-1) * h).sum(dim=0)     # [dim]
            return z, a
        else:
            # Batch dimension [B, N, dim]
            a = self.w(self.drop(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))))
            a = torch.softmax(a, dim=1)              # [B, N, 1]
            z = (a * h).sum(dim=1)                   # [B, dim]
            return z, a.squeeze(-1)


class MultiScalePathologyModel(nn.Module):
    """
    Multi-Scale Pathology Model with Contrastive Feature Projection Head.
    """
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        num_cancer_classes: int = 3,  # BRCA, LUAD, GBM (or customizable)
        num_subtype_classes: int = 7, # e.g. IDC_LumA, IDC_LumB, TNBC, LUAD_Acinar, LUAD_Papillary, Mesenchymal, Proneural
        dropout: float = 0.30,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Input patch projection
        self.patch_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention pooling
        self.scale_attention = ScaleAttention(hidden_dim)
        self.patch_attention = PatchAttention(hidden_dim, hid=128, dropout=dropout)

        # Primary Cancer Classification Head
        self.cancer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_cancer_classes),
        )

        # Subtype Classification Head
        self.subtype_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subtype_classes),
        )

        # Contrastive Learning Projection Head (MLP: 512 -> 128)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim),
        )

        # Temperature calibration parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        patch_features: torch.Tensor,
        scale_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        patch_features: [N, in_dim] or [B, N, in_dim]
        scale_features: optional multi-scale tensor [B, S, in_dim]
        """
        # Encode patch features
        h = self.patch_encoder(patch_features)  # [N, hidden_dim]

        # Aggregate across patches
        if h.ndim == 2:
            slide_emb, patch_attn = self.patch_attention(h)  # [hidden_dim]
            slide_emb = slide_emb.unsqueeze(0)               # [1, hidden_dim]
            patch_attn = patch_attn.unsqueeze(0)
        else:
            slide_emb, patch_attn = self.patch_attention(h)  # [B, hidden_dim]

        # Optional multi-scale attention fusion
        scale_attn = None
        if scale_features is not None:
            s_enc = self.patch_encoder(scale_features)
            scale_pooled, scale_attn = self.scale_attention(s_enc)
            slide_emb = 0.5 * slide_emb + 0.5 * scale_pooled

        # Classification logits
        cancer_logits = self.cancer_head(slide_emb)
        subtype_logits = self.subtype_head(slide_emb)

        # Contrastive projection head with L2 normalization
        projected = self.projection_head(slide_emb)
        z_norm = F.normalize(projected, p=2, dim=-1)

        return {
            "slide_embedding": slide_emb,
            "cancer_logits": cancer_logits,
            "subtype_logits": subtype_logits,
            "projected_features": z_norm,
            "patch_attention": patch_attn,
            "scale_attention": scale_attn,
        }

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by learned temperature parameter T."""
        t = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / t

    @torch.no_grad()
    def predict_calibrated(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference with temperature calibration and OOD energy score."""
        self.eval()
        out = self.forward(patch_features)
        
        cal_cancer = self.calibrate_logits(out["cancer_logits"])
        cal_subtype = self.calibrate_logits(out["subtype_logits"])

        cancer_probs = F.softmax(cal_cancer, dim=-1)
        subtype_probs = F.softmax(cal_subtype, dim=-1)

        # Negative free energy score for Out-of-Distribution detection
        # E(x) = -T * log(sum(exp(logits / T)))
        t = torch.clamp(self.temperature, min=0.1, max=10.0)
        energy = -t * torch.logsumexp(out["cancer_logits"] / t, dim=-1)

        return {
            "cancer_probs": cancer_probs,
            "subtype_probs": subtype_probs,
            "energy_score": energy,
            "patch_attention": out["patch_attention"],
        }
