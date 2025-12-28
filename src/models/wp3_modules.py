"""
WP3 modules: domain-stage adapter, morphological MoE branch, and stage/domain generalization branch.
All modules assume a frozen backbone upstream; only these components are trainable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# -----------------------------
# Utilities
# -----------------------------

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFn.apply(x, self.lambda_)


class StyleInvariantNorm(nn.Module):
    """
    Simple SIN: convex mix of InstanceNorm and BatchNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(dim, affine=False, eps=eps)
        self.bnorm = nn.BatchNorm1d(dim, eps=eps, momentum=momentum, affine=False)
        self.alpha = nn.Parameter(torch.zeros(1))  # learnable mix; 0 -> BN, 1 -> IN

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C) or (B, C)
        orig_dim = x.dim()
        if orig_dim == 2:
            x = x.unsqueeze(1)  # (B,1,C)
        x_perm = x.transpose(1, 2)  # (B, C, L)
        nin = self.inorm(x_perm)
        nbn = self.bnorm(x_perm)
        mix = torch.sigmoid(self.alpha)
        mixed = mix * nin + (1 - mix) * nbn
        out = mixed.transpose(1, 2)
        return out if orig_dim == 3 else out.squeeze(1)


def _flatten_tokens(feats: Tensor) -> Tensor:
    """Convert (B,C,H,W) -> (B, HW, C); leave (B,L,C) unchanged."""
    if feats.dim() == 4:
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
    return feats


# -----------------------------
# WP3.1 Domain-Stage Adapter
# -----------------------------

class DomainStageAdapter(nn.Module):
    """
    Lightweight adapter + domain/stage token injection.

    Args:
        in_dim: backbone feature dim
        token_dim: output token dim
        use_adapter: if True, applies linear projection; else expects in_dim == token_dim
        use_domain_token / use_stage_token: ablation toggles
        stage_vocab: mapping str->id (e.g., {"D3":0,"D5":1,"ED1":2,...,"UNK":K})
        domain_vocab: mapping str->id (e.g., {"roboflow_ssl":0,"hv_kaggle":1,"hv_clinical":2})
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        use_adapter: bool = True,
        use_domain_token: bool = True,
        use_stage_token: bool = True,
        stage_vocab: Optional[Dict[str, int]] = None,
        domain_vocab: Optional[Dict[str, int]] = None,
        adapter_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_adapter = use_adapter
        self.use_domain_token = use_domain_token
        self.use_stage_token = use_stage_token
        self.stage_vocab = stage_vocab or {"D3": 0, "D5": 1, "ED1": 2, "ED2": 3, "ED3": 4, "ED4": 5, "UNK": 6}
        self.domain_vocab = domain_vocab or {"roboflow_ssl": 0, "hv_kaggle": 1, "hv_clinical": 2}
        hid = adapter_hidden or token_dim
        if use_adapter:
            self.adapter = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, token_dim))
        else:
            self.adapter = nn.Identity()
        self.domain_embed = nn.Embedding(len(self.domain_vocab), token_dim)
        self.stage_embed = nn.Embedding(len(self.stage_vocab), token_dim)

    def forward(
        self,
        feats: Tensor,
        domain_ids: Optional[Tensor] = None,
        stage_labels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        tokens = _flatten_tokens(feats)
        tokens = self.adapter(tokens)

        batch_size = tokens.size(0)
        domain_token: Optional[Tensor] = None
        stage_token: Optional[Tensor] = None

        if self.use_domain_token:
            if domain_ids is None:
                raise ValueError("domain_ids required when use_domain_token=True")
            domain_token = self.domain_embed(domain_ids.clamp(min=0, max=self.domain_embed.num_embeddings - 1))
            domain_token = domain_token.unsqueeze(1)  # (B,1,D)
            tokens = torch.cat([domain_token, tokens], dim=1)

        if self.use_stage_token:
            ids = []
            unk_id = self.stage_vocab.get("UNK", len(self.stage_vocab) - 1)
            for lbl in stage_labels or [None] * batch_size:
                if lbl is None:
                    ids.append(unk_id)
                else:
                    ids.append(self.stage_vocab.get(str(lbl).upper(), unk_id))
            stage_ids = torch.tensor(ids, device=tokens.device, dtype=torch.long)
            stage_token = self.stage_embed(stage_ids).unsqueeze(1)  # (B,1,D)
            tokens = torch.cat([stage_token, tokens], dim=1)

        return {"tokens": tokens, "domain_token": domain_token, "stage_token": stage_token}


# -----------------------------
# WP3.2 Morphological MoE Branch
# -----------------------------

class MorphologicalMoEBranch(nn.Module):
    """
    Morphology-focused MoE with cross-attention to shared tokens and optional SIN.
    """

    def __init__(
        self,
        token_dim: int,
        num_experts: int = 4,
        num_heads: int = 4,
        use_sin: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = max(1, num_experts)
        self.router = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, self.num_experts))
        self.expert_queries = nn.Parameter(torch.randn(self.num_experts, token_dim))
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)) for _ in range(self.num_experts)])
        self.use_sin = use_sin
        self.sin = StyleInvariantNorm(token_dim) if use_sin else nn.Identity()

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, L, D)
        pooled = tokens.mean(dim=1)
        logits = self.router(pooled)  # (B, E)
        gates = torch.softmax(logits, dim=-1)  # soft routing

        b = tokens.size(0)
        q = self.expert_queries.unsqueeze(0).expand(b, -1, -1)  # (B,E,D)
        attn_out, _ = self.attn(q, tokens, tokens)  # (B,E,D)

        expert_outs = []
        for i, expert in enumerate(self.experts):
            expert_outs.append(expert(attn_out[:, i, :]))  # (B,D)
        expert_stack = torch.stack(expert_outs, dim=1)  # (B,E,D)

        mixed = torch.einsum("be,bed->bd", gates, expert_stack)  # (B,D)
        mixed = self.sin(mixed)
        return mixed  # z_m


# -----------------------------
# WP3.3 Stage / Domain Generalization Branch
# -----------------------------

class StageDomainGeneralizationBranch(nn.Module):
    """
    Transformer encoder with optional stage cross-attention and domain alignment (GRL classifier or CORAL).
    """

    def __init__(
        self,
        token_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        dim_ff: int = 512,
        use_stage_cross_attn: bool = True,
        use_domain_align: bool = True,
        domain_align_type: str = "grl",  # "grl", "coral", "none"
        num_domains: int = 3,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.use_stage_cross_attn = use_stage_cross_attn
        self.stage_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.use_domain_align = use_domain_align
        self.domain_align_type = domain_align_type
        self.domain_classifier = nn.Sequential(nn.Linear(token_dim, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, num_domains))
        self.grl = GradientReversal(lambda_=grl_lambda)

    def forward(
        self,
        tokens: Tensor,
        stage_token: Optional[Tensor] = None,
        domain_ids: Optional[Tensor] = None,
        align: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # tokens: (B,L,D)
        x = self.encoder(tokens)

        if self.use_stage_cross_attn and stage_token is not None:
            stage_ctx, _ = self.stage_attn(stage_token, x, x)  # (B,1,D)
            x = torch.cat([stage_ctx, x], dim=1)

        z = x.mean(dim=1)  # pooled z_g

        aux: Dict[str, Tensor] = {}
        if self.use_domain_align and align and domain_ids is not None:
            if self.domain_align_type == "grl":
                logits = self.domain_classifier(self.grl(z))
                aux["domain_logits"] = logits
            elif self.domain_align_type == "coral":
                # Return features for external CORAL loss computation
                aux["coral_features"] = z
        return z, aux


# -----------------------------
# Fusion Wrapper
# -----------------------------

class WP3FeatureComposer(nn.Module):
    """
    End-to-end WP3 feature pipeline:
        frozen backbone feats -> DomainStageAdapter -> MorphologicalMoEBranch -> StageDomainGeneralizationBranch -> concat.
    """

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        adapter_cfg: Dict,
        morph_cfg: Dict,
        gen_cfg: Dict,
    ) -> None:
        super().__init__()
        self.adapter = DomainStageAdapter(in_dim=in_dim, token_dim=token_dim, **adapter_cfg)
        self.morph_branch = MorphologicalMoEBranch(token_dim=token_dim, **morph_cfg)
        self.gen_branch = StageDomainGeneralizationBranch(token_dim=token_dim, **gen_cfg)

    def forward(
        self,
        feats: Tensor,
        domain_ids: Optional[Tensor] = None,
        stage_labels: Optional[List[str]] = None,
        align: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        adapted = self.adapter(feats, domain_ids=domain_ids, stage_labels=stage_labels)
        tokens = adapted["tokens"]
        z_m = self.morph_branch(tokens)
        z_g, aux = self.gen_branch(tokens, stage_token=adapted.get("stage_token"), domain_ids=domain_ids, align=align)
        fused = torch.cat([z_m, z_g], dim=1)
        aux.update({"z_m": z_m, "z_g": z_g})
        return fused, aux
