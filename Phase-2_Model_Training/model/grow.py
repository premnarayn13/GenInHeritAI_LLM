"""
GenInherit Progressive Depth Growth Operator (Pillar 2 Specification)
Expands model depth (e.g. 7 layers -> 10 layers -> 22 layers) with
function-preserving identity initialization: f_grown(x) == f_base(x).
"""

import copy
import torch
import torch.nn as nn
from .model import GenInheritConfig, GenInheritBase


def grow_model_depth(
    base_model: GenInheritBase,
    target_config: GenInheritConfig,
    verify_identity: bool = True
) -> GenInheritBase:
    old_n_layers = len(base_model.blocks)
    new_n_layers = target_config.n_layers

    if new_n_layers <= old_n_layers:
        raise ValueError(f"Target layers ({new_n_layers}) must exceed base layers ({old_n_layers})")

    grown_model = GenInheritBase(target_config)
    grown_model.token_emb.load_state_dict(base_model.token_emb.state_dict())
    grown_model.slot_emb.load_state_dict(base_model.slot_emb.state_dict())
    grown_model.gen_emb.load_state_dict(base_model.gen_emb.state_dict())
    grown_model.field_emb.load_state_dict(base_model.field_emb.state_dict())
    grown_model.num_proj.load_state_dict(base_model.num_proj.state_dict())
    grown_model.final_norm.load_state_dict(base_model.final_norm.state_dict())

    # Map old layers to new layers
    # Distribute original layers evenly across the new depth
    stride = (new_n_layers - 1) / max(1, old_n_layers - 1)
    mapped_indices = [int(round(i * stride)) for i in range(old_n_layers)]

    for old_idx, new_idx in enumerate(mapped_indices):
        grown_model.blocks[new_idx].load_state_dict(base_model.blocks[old_idx].state_dict())

    # Newly inserted layers: copy nearest mapped layer, but ZERO-INITIALIZE output projections
    # (attn.o and mlp.w2) so the residual addition adds EXACTLY 0.0 at step 0!
    for new_idx in range(new_n_layers):
        if new_idx not in mapped_indices:
            nearest_old = min(mapped_indices, key=lambda x: abs(x - new_idx))
            grown_model.blocks[new_idx].load_state_dict(grown_model.blocks[nearest_old].state_dict())
            
            # Identity zeroing:
            nn.init.zeros_(grown_model.blocks[new_idx].attn.o.weight)
            nn.init.zeros_(grown_model.blocks[new_idx].mlp.w2.weight)

    if verify_identity:
        base_model.eval()
        grown_model.eval()
        with torch.no_grad():
            B, T = 2, 32
            toks = torch.randint(0, target_config.vocab_size, (B, T))
            pos = torch.arange(T).unsqueeze(0).repeat(B, 1)
            out_base = base_model(toks, pos)
            out_grown = grown_model(toks, pos)
            max_diff = (out_base - out_grown).abs().max().item()
            assert max_diff < 1e-4, f"Identity growth verification failed! Max output difference: {max_diff}"
            print(f"✓ Function-Preserving Identity Growth Verified! Max diff = {max_diff:.2e}")

    return grown_model

