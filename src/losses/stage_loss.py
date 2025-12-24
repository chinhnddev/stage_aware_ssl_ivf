"""
Stage-aware multi-positive loss.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


def stage_loss(
    embeddings: torch.Tensor,
    stage_ids: Sequence[str],
    num_stage_positives: int = 2,
) -> torch.Tensor:
    """
    embeddings: [B, D] target embeddings (detached).
    stage_ids: list of stage_id strings length B.
    """
    B = embeddings.size(0)
    loss_terms = []
    for i in range(B):
        sid = stage_ids[i]
        # indices with same stage and not self
        pos_indices = [j for j in range(B) if j != i and stage_ids[j] == sid]
        if not pos_indices:
            continue
        if len(pos_indices) > num_stage_positives:
            pos_indices = pos_indices[:num_stage_positives]
        anchor = F.normalize(embeddings[i], dim=-1)
        for j in pos_indices:
            pos = F.normalize(embeddings[j], dim=-1)
            loss_terms.append(1 - torch.dot(anchor, pos))

    if not loss_terms:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
    return torch.stack(loss_terms).mean()
