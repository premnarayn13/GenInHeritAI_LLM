"""
GenInherit-Base Model Architecture (Pillar 2 Specification)
Decoder-only Transformer with RMSNorm, RoPE, Grouped-Query Attention (GQA),
QK-norm, SwiGLU, multi-stream structural embeddings, and Fourier numeric projection.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenInheritConfig:
    vocab_size: int = 4096
    d_model: int = 1024
    n_layers: int = 7
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2752
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    n_slots: int = 128
    n_gens: int = 16
    n_fields: int = 64
    n_fourier_frequencies: int = 8
    qk_norm: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).type_as(x)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    c = cos[pos][:, None, :, :].to(x.dtype)
    s = sin[pos][:, None, :, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: GenInheritConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                pos: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        c = self.cfg

        q = self.q(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin, pos)
        k = apply_rope(self.k_norm(k), cos, sin, pos)

        k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(B, T, c.n_heads * c.head_dim)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GenInheritConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GenInheritConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                pos: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, pos, mask=mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FourierNumericProjector(nn.Module):
    def __init__(self, d_model: int, n_frequencies: int = 8):
        super().__init__()
        self.n_frequencies = n_frequencies
        in_dim = 1 + 2 * n_frequencies
        self.proj = nn.Linear(in_dim, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(u)
        u_clean = torch.nan_to_num(u, nan=0.0).unsqueeze(-1)

        feats = [u_clean]
        for k in range(self.n_frequencies):
            freq = 2.0 ** k
            feats.append(torch.sin(2.0 * math.pi * freq * u_clean))
            feats.append(torch.cos(2.0 * math.pi * freq * u_clean))

        psi = torch.cat(feats, dim=-1)
        projected = self.proj(psi)
        return projected * mask.unsqueeze(-1).to(projected.dtype)


class GenInheritBase(nn.Module):
    def __init__(self, cfg: GenInheritConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.slot_emb = nn.Embedding(cfg.n_slots, cfg.d_model)
        self.gen_emb = nn.Embedding(cfg.n_gens, cfg.d_model)
        self.field_emb = nn.Embedding(cfg.n_fields, cfg.d_model)
        self.num_proj = FourierNumericProjector(cfg.d_model, cfg.n_fourier_frequencies)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor,
                slots: torch.Tensor = None, gens: torch.Tensor = None,
                fields: torch.Tensor = None, num: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_emb(tokens)

        if slots is not None:
            x = x + self.slot_emb(slots)
        if gens is not None:
            x = x + self.gen_emb(gens)
        if fields is not None:
            x = x + self.field_emb(fields)
        if num is not None:
            x = x + self.num_proj(num)

        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin, pos, mask=mask)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits


def infer_config_from_state_dict(sd: dict) -> GenInheritConfig:
    """Infer GenInheritConfig dimensions directly from checkpoint state_dict tensors."""
    vocab_size, d_model = sd["token_emb.weight"].shape
    n_layers = len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".attn_norm.weight")])
    d_ff = sd["blocks.0.mlp.w1.weight"].shape[0]
    qk_norm = "blocks.0.attn.q_norm.weight" in sd
    head_dim = sd["blocks.0.attn.q_norm.weight"].shape[0] if qk_norm else 64
    n_heads = sd["blocks.0.attn.q.weight"].shape[0] // head_dim
    n_kv_heads = sd["blocks.0.attn.k.weight"].shape[0] // head_dim
    return GenInheritConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        n_slots=sd["slot_emb.weight"].shape[0],
        n_gens=sd["gen_emb.weight"].shape[0],
        n_fields=sd["field_emb.weight"].shape[0],
        n_fourier_frequencies=(sd["num_proj.proj.weight"].shape[1] - 1) // 2,
        qk_norm=qk_norm,
    )


