"""
GenInherit - Phase 2: Model Training
CLI Entrypoint: Progressive Depth Growth from existing checkpoint to higher rungs.
"""

import os
import sys
import yaml
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.model import GenInheritConfig, GenInheritBase, infer_config_from_state_dict
from model.grow import grow_model_depth
from train.trainer import train_loop, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="GenInherit Progressive Depth Growth")
    parser.add_argument("--source_ckpt", type=str, required=True, help="Path to source checkpoint (e.g. runs/r01_82m/r01_82m_end.pt)")
    parser.add_argument("--target_config", type=str, required=True, help="Path to target YAML config (e.g. configs/r02_115m.yaml)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing binary .pt shards")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for new rung (e.g. runs/r02_115m)")
    parser.add_argument("--verify_identity", action="store_true", default=True, help="Verify zero-diff identity before training")

    args = parser.parse_args()

    with open(args.target_config, "r", encoding="utf-8") as f:
        t_cfg_dict = yaml.safe_load(f)

    print("=" * 80)
    print(f"  GENINHERIT PROGRESSIVE GROWTH: Growing to {t_cfg_dict['run']['name'].upper()}")
    print("=" * 80)

    # Load source model
    ckpt = torch.load(args.source_ckpt, map_location="cpu", weights_only=False)
    base_cfg = infer_config_from_state_dict(ckpt["model"])
    base_model = GenInheritBase(base_cfg)
    base_model.load_state_dict(ckpt["model"])
    print(f"✓ Loaded Base Model ({base_cfg.n_layers} layers, d_model={base_cfg.d_model})")

    # Grow model
    m_dict = t_cfg_dict["model"]
    target_cfg = GenInheritConfig(
        vocab_size=m_dict.get("vocab_size", base_cfg.vocab_size),
        d_model=m_dict["d_model"],
        n_layers=m_dict["n_layers"],
        n_heads=m_dict["n_heads"],
        n_kv_heads=m_dict["n_kv_heads"],
        head_dim=m_dict.get("head_dim", base_cfg.head_dim),
        d_ff=m_dict.get("d_ff", base_cfg.d_ff),
        max_seq_len=m_dict.get("max_seq_len", base_cfg.max_seq_len),
        rope_base=m_dict.get("rope_base", base_cfg.rope_base),
        qk_norm=m_dict.get("qk_norm", base_cfg.qk_norm),
    )
    grown_model = grow_model_depth(base_model, target_cfg, verify_identity=args.verify_identity)
    print(f"✓ Grown Model Initialized ({target_cfg.n_layers} layers)")

    # Save as latest.pt in target out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    initial_latest = os.path.join(args.out_dir, "latest.pt")
    opt = torch.optim.AdamW(grown_model.parameters(), lr=float(t_cfg_dict["optim"]["peak_lr"]))
    initial_state = {"step": ckpt["state"]["step"], "tokens": ckpt["state"]["tokens"], "epoch": 0}
    save_checkpoint(initial_latest, grown_model, opt, None, initial_state)

    # Launch training on the grown model
    print("\nProceeding directly into training on the grown model...")
    train_loop(args.target_config, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()

