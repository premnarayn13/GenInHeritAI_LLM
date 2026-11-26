"""
GenInherit - Phase 1: Data Preparation
Master Pipeline Runner: Executes Steps 1 through 5 end-to-end.
"""

import os
import sys
import subprocess
import time


def run_command(cmd, desc):
    print("\n" + "=" * 80)
    print(f"▶ EXECUTING: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n❌ Pipeline failed at step: {desc} (Exit Code: {res.returncode})")
        sys.exit(res.returncode)
    print(f"✓ Step completed in {elapsed:.2f}s\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = sys.argv[1] if len(sys.argv) > 1 else "/content"
    
    raw_dir = os.path.join(workspace_root, "raw_data")
    vocab_file = os.path.join(workspace_root, "bioDSL_vocab.json")
    synthetic_file = os.path.join(raw_dir, "synthetic_pedigrees.json")
    shards_dir = os.path.join(workspace_root, "shards")

    print("=" * 80)
    print("  GENINHERIT: PHASE 1 END-TO-END MASTER EXECUTION PIPELINE")
    print(f"  Root Destination Workspace: {workspace_root}")
    print("=" * 80)

    # Step 1: Download Raw Datasets
    run_command([sys.executable, os.path.join(base_dir, "01_download_raw_data.py"), raw_dir],
                "Step 1: Download Public Genomic Datasets (ClinVar, HPO, MANE, ClinGen)")

    # Step 2: Generate Multigenerational Synthetic Pedigree Cohort
    run_command([sys.executable, os.path.join(base_dir, "02_synthetic_pedigree_generator.py"), synthetic_file, "15000"],
                "Step 2: Generate 15,000 Multigenerational Synthetic Family Pedigrees")

    # Step 3: Build BioDSL 4,096 Vocabulary
    run_command([sys.executable, os.path.join(base_dir, "03_build_biodsl_tokenizer.py"), vocab_file],
                "Step 3: Construct & Sanity-Check BioDSL 4,096-Token Vocabulary")

    # Step 4: Preprocess and Shard into PyTorch Tensors
    run_command([sys.executable, os.path.join(base_dir, "04_preprocess_and_shard.py"), synthetic_file, vocab_file, shards_dir, "2500"],
                "Step 4: Preprocess, Encode Fourier Numeric Channels, and Save Binary Shards")

    # Step 5: Audit and Verify Shards
    run_command([sys.executable, os.path.join(base_dir, "05_verify_phase1_shards.py"), shards_dir],
                "Step 5: Audit Shard Structural Integrity & Tensor Shapes")

    print("\n" + "=" * 80)
    print("  🎉 PHASE 1 PIPELINE EXECUTION 100% COMPLETE & VERIFIED!")
    print(f"  All data shards and vocabulary are ready at: {shards_dir}")
    print("  You are now ready to proceed to Phase 2: Base Model Pre-Training!")
    print("=" * 80)


if __name__ == "__main__":
    main()

