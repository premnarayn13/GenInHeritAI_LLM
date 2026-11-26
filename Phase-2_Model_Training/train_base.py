"""
GenInherit - Phase 2: Model Training
CLI Entrypoint: Pre-trains GenInherit-Base from scratch on Google Colab / local GPU.
"""

import os
import sys
import argparse

# Add package root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.trainer import train_loop


def main():
    parser = argparse.ArgumentParser(description="GenInherit-Base Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration (e.g. configs/r01_82m.yaml)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing binary .pt shards from Phase 1")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save checkpoints (on Google Drive)")
    parser.add_argument("--session_hours", type=float, default=11.0, help="Max session duration before clean checkpoint & exit")

    args = parser.parse_args()
    train_loop(args.config, args.data_dir, args.out_dir, args.session_hours)


if __name__ == "__main__":
    main()

