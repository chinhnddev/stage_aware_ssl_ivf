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
    Legacy batch-only stage loss (positives must be in the same batch).
    embeddings: [B, D] target embeddings (detached).
    stage_ids: list of stage_id strings length B.
    This is kept for backward compatibility; new code should use
    stage_loss_from_pairs with positives sampled across the dataset.
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


def stage_loss_from_pairs(
    anchor_embeddings: torch.Tensor,
    pos_emb_dict: dict[int, torch.Tensor],
    pos_lists: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, int]:
    """
    Stage loss when positives are sampled across the dataset.
    anchor_embeddings: [B, D] target embeddings for anchors (already detached).
    pos_emb_dict: mapping from dataset index -> target embedding for that positive.
    pos_lists: list per anchor of dataset indices to treat as positives.
    Returns: (mean loss, anchors_with_pos_count)
    """
    if len(pos_emb_dict) == 0:
        zero = torch.tensor(0.0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
        return zero, 0

    anchor_t = torch.nn.functional.normalize(anchor_embeddings, dim=-1)
    loss_terms = []
    anchors_with_pos = 0
    for anchor_emb, pos_idx_list in zip(anchor_t, pos_lists):
        valid = [pi for pi in pos_idx_list if pi in pos_emb_dict]
        if not valid:
            continue
        anchors_with_pos += 1
        for pi in valid:
            loss_terms.append(1 - torch.dot(anchor_emb, pos_emb_dict[pi]))

    if not loss_terms:
        zero = torch.tensor(0.0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
        return zero, anchors_with_pos
    return torch.stack(loss_terms).mean(), anchors_with_pos
