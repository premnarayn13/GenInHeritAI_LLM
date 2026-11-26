"""
GenInherit Training Loss Functions (Pillar 2 Specification)
L_total = L_LM + lambda_inv * L_inv + lambda_z * L_z
Computed strictly in float32 to prevent fp16 underflow/overflow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_weighted_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    # Ensure float32 for stable cross-entropy
    logits_f32 = logits.float()
    B, T, V = logits_f32.shape
    flat_logits = logits_f32.view(-1, V)
    flat_targets = targets.view(-1)
    flat_weights = weights.view(-1).float()
    flat_mask = target_mask.view(-1).float()

    effective_weights = flat_weights * flat_mask
    weight_sum = effective_weights.sum().clamp(min=1e-6)

    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    loss = (ce * effective_weights).sum() / weight_sum
    return loss


def compute_z_loss(logits: torch.Tensor, lambda_z: float = 1e-4) -> torch.Tensor:
    # Strictly compute logsumexp in float32 to prevent fp16 overflow
    logits_f32 = logits.float()
    lse = torch.logsumexp(logits_f32, dim=-1)
    z_loss = (lse ** 2).mean()
    return lambda_z * z_loss


def bio_composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    target_mask: torch.Tensor,
    lambda_z: float = 1e-4
) -> torch.Tensor:
    lm_loss = compute_weighted_lm_loss(logits, targets, weights, target_mask)
    z_term = compute_z_loss(logits, lambda_z)
    return lm_loss + z_term
