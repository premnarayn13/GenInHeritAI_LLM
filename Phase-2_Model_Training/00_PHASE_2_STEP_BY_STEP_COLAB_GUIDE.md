# GenInherit: Phase 2 Model Training
## Complete Step-by-Step Click-by-Click Guide for Google Colab

```text
====================================================================================================
  GENINHERIT-LLM | PHASE 2 MODEL TRAINING PLAYBOOK
  TARGET: Click-by-Click, Cell-by-Cell Instructions to Train GenInherit-Base (82M Rung 1),
          Survive Colab Session Disconnects with Auto-Resume, and Scale to 248M with grow.py
====================================================================================================
```

Now that you have completed **Phase 1** (your `shards/` and `bioDSL_vocab.json` are sitting in Google Drive), you are ready for **Phase 2: Base Model Pre-Training**.

This guide explains **exactly what to click, which cells to run, and how training works on Google Colab**.

---

## PART 1: Outside of Colab (Google Drive Verification)

Before starting Colab, ensure these files exist from Phase 1:

1. Open [Google Drive](https://drive.google.com).
2. Go to `My Drive/geninherit/`.
3. Verify that `data/shards/` contains `shard_0000.pt`, `val_iid.pt`, etc.
4. Upload the new `Phase-2_Model_Training` scripts into `My Drive/geninherit/code/`:
   - `train_base.py`
   - `grow_and_train.py`
   - `run_eval_gates.py`
   - The subfolders `configs/`, `model/`, and `train/`.

---

## PART 2: Inside Google Colab (What to Click & What to Run)

### Step 2.1: Open a New Colab Notebook
1. Go to: [https://colab.research.google.com](https://colab.research.google.com).
2. Click **"New notebook"** and rename it to:  
   `GenInherit_Phase2_Training.ipynb`
3. Click **Runtime** in the top menu $\to$ **Change runtime type**.
4. Select **T4 GPU** under Hardware accelerator $\to$ Click **Save**.

---

## PART 3: The Exact Cells to Run Inside Google Colab

Run these cells in order:

---

### Cell 1: Mount Google Drive
*Connects your Drive so checkpoints save continuously to cloud storage.*

```python
# CELL 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
WORKSPACE = "/content/drive/MyDrive/geninherit"
SHARDS_DIR = f"{WORKSPACE}/data/shards"
RUNS_DIR = f"{WORKSPACE}/runs/r01_82m"

os.makedirs(RUNS_DIR, exist_ok=True)
print("✓ Google Drive mounted successfully!")
```

---

### Cell 2: Verify GPU & Install Dependencies
*Checks that your NVIDIA Tesla T4 GPU is active with 15.3 GB VRAM.*

```python
# CELL 2: GPU Check & Dependencies
!nvidia-smi
!pip install -q torch pyyaml safetensors
print("✓ Environment verified and ready.")
```

---

### Cell 3: Launch Base Model Pre-Training (Rung 1: 82M Parameters)
*Kicks off the pre-training loop using fp16 mixed-precision and dynamic loss scaling.*

```python
# CELL 3: Start Base Training (Rung 1: 82M)
!python /content/drive/MyDrive/geninherit/code/train_base.py \
    --config /content/drive/MyDrive/geninherit/code/configs/r01_82m.yaml \
    --data_dir /content/drive/MyDrive/geninherit/data/shards \
    --out_dir /content/drive/MyDrive/geninherit/runs/r01_82m \
    --session_hours 11.0
```

**What You Will See Live on Screen:**
```text
================================================================================
  GENINHERIT-BASE TRAINING: GENINHERIT_R01_82M
  Device: cuda | Mixed Precision: fp16 (GradScaler)
================================================================================
✓ Model Initialized: 82.0 Million Parameters
Training Loop Running... (Target: 600,000,000 tokens)
Step     10 | Loss: 7.2140 | Tokens:   1,310,720 | Speed:    41,200 tok/s
Step     20 | Loss: 6.8421 | Tokens:   2,621,440 | Speed:    42,150 tok/s
Step     30 | Loss: 5.9123 | Tokens:   3,932,160 | Speed:    42,300 tok/s
...
Step    250 | Loss: 3.1205 | Tokens:  32,768,000 | Speed:    42,400 tok/s
  💾 Saved checkpoint (Step 250, Tokens 32,768,000) -> .../runs/r01_82m/latest.pt
```

---

### Cell 4: What to Do When Colab Disconnects (Automatic Resumption)

If you close your laptop, lose internet, or Colab resets after 6–10 hours:
1. Re-open Colab and select **T4 GPU**.
2. Run **Cell 1** (Mount Drive).
3. Re-run **Cell 3**!

**The trainer will automatically detect `latest.pt` on your Google Drive and resume from the exact token count! You never lose progress.**

---

### Cell 5: Evaluate Release Quality Gates
*Once Rung 1 finishes, run this cell to verify validation loss and model health.*

```python
# CELL 5: Run Evaluation Quality Gates
!python /content/drive/MyDrive/geninherit/code/run_eval_gates.py \
    --checkpoint /content/drive/MyDrive/geninherit/runs/r01_82m/latest.pt \
    --data_dir /content/drive/MyDrive/geninherit/data/shards
```

**Expected Output:**
```text
================================================================================
  GENINHERIT QUALITY GATES EVALUATION SUITE
================================================================================
✓ Checkpoint loaded successfully (7 layers, 82.0M params)
  • Val IID Loss          : 1.4210 (across 24,500 active tokens)
  • Val Compositional Loss: 1.5830 (across 24,500 active tokens)

  QUALITY GATE READINESS STATUS: PASSED
================================================================================
```

---

### Cell 6: Progressive Growth to Rung 2 (115M Parameters)
*When you are ready to scale to Rung 2, run this cell. It automatically grows the model from 7 to 10 layers with identity initialization and continues training!*

```python
# CELL 6: Progressive Growth to Rung 2 (115M)
!python /content/drive/MyDrive/geninherit/code/grow_and_train.py \
    --source_ckpt /content/drive/MyDrive/geninherit/runs/r01_82m/latest.pt \
    --target_config /content/drive/MyDrive/geninherit/code/configs/r02_115m.yaml \
    --data_dir /content/drive/MyDrive/geninherit/data/shards \
    --out_dir /content/drive/MyDrive/geninherit/runs/r02_115m
```

---

## PART 4: Phase 2 Summary
- **Rung 1 (82M Base Foundation):** Trains in ~14 hours (across 2 free Colab sessions).
- **Zero Compute Loss:** Checkpoints save to Google Drive every 20 minutes.
- **Continuous Scaling:** You use `grow_and_train.py` to step up from 82M $\to$ 115M $\to$ 248M without retraining from scratch!

